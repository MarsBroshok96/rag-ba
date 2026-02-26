from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import chromadb
from dotenv import load_dotenv
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.storage.storage_context import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.common.project_paths import CHROMA_RAG_DIR, PROJECT_ROOT
from src.index.citations import format_sources, load_manifest

load_dotenv()

DEFAULT_EMBED_MODEL = "intfloat/multilingual-e5-base"
DEFAULT_OLLAMA_MODEL = "qwen2.5:14b-instruct"
DEFAULT_OLLAMA_TIMEOUT = 180.0
DEFAULT_SIMILARITY_TOP_K = 8
_CITATION_RE = re.compile(r"\[(\d+)\]")
_CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")
_LATIN_RE = re.compile(r"[A-Za-z]")


@dataclass(slots=True)
class RagRuntimeSettings:
    embed_model_name: str = DEFAULT_EMBED_MODEL
    llm_model_name: str = DEFAULT_OLLAMA_MODEL
    llm_timeout_sec: float = DEFAULT_OLLAMA_TIMEOUT
    similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K
    max_sources: int = 6

    @classmethod
    def from_env(cls) -> RagRuntimeSettings:
        max_sources = int(os.getenv("RAG_BA_MAX_SOURCES", "6"))
        return cls(
            embed_model_name=os.getenv("RAG_BA_EMBED_MODEL", DEFAULT_EMBED_MODEL),
            llm_model_name=os.getenv("RAG_BA_OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
            llm_timeout_sec=float(
                os.getenv("RAG_BA_OLLAMA_TIMEOUT", str(DEFAULT_OLLAMA_TIMEOUT))
            ),
            similarity_top_k=int(
                os.getenv("RAG_BA_SIMILARITY_TOP_K", str(DEFAULT_SIMILARITY_TOP_K))
            ),
            max_sources=max(0, min(6, max_sources)),
        )


@dataclass(slots=True)
class RagReply:
    answer: str
    sources_text: str
    context_messages_used: int


class RagChatBackend:
    def __init__(
        self,
        *,
        project_root: Path = PROJECT_ROOT,
        chroma_dir: Path = CHROMA_RAG_DIR,
        settings: RagRuntimeSettings | None = None,
    ) -> None:
        self.project_root = project_root
        self.chroma_dir = chroma_dir
        self.settings = settings or RagRuntimeSettings.from_env()
        self._manifest: dict | None = None
        self._retriever = None
        self._configured = False

    def _configure_runtime(self) -> None:
        if self._configured:
            return
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=self.settings.embed_model_name
        )
        Settings.llm = Ollama(
            model=self.settings.llm_model_name,
            request_timeout=self.settings.llm_timeout_sec,
        )
        self._configured = True

    def _load_manifest(self) -> dict:
        if self._manifest is None:
            self._manifest = load_manifest(self.project_root)
        return self._manifest

    def _build_index(self) -> VectorStoreIndex:
        client = chromadb.PersistentClient(path=str(self.chroma_dir))
        collection = client.get_collection(name="rag")
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        return VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context,
        )

    def _get_retriever(self):
        if self._retriever is None:
            self._configure_runtime()
            index = self._build_index()
            self._retriever = index.as_retriever(
                similarity_top_k=self.settings.similarity_top_k
            )
        return self._retriever

    @staticmethod
    def _prepare_context(messages: list[dict[str, str]], memory_k: int) -> list[dict[str, str]]:
        filtered = [
            {"role": m.get("role", ""), "content": m.get("content", "")}
            for m in messages
            if m.get("role") in {"user", "assistant", "system"} and m.get("content")
        ]
        if memory_k <= 0:
            return []
        return filtered[-memory_k:]

    @staticmethod
    def _extract_citation_ids(text: str) -> set[int]:
        return {int(m.group(1)) for m in _CITATION_RE.finditer(text or "")}

    @staticmethod
    def _strip_invalid_citations(text: str, *, max_valid_id: int) -> str:
        if max_valid_id <= 0:
            return _CITATION_RE.sub("", text or "").strip()

        def _replace(match: re.Match[str]) -> str:
            n = int(match.group(1))
            return match.group(0) if 1 <= n <= max_valid_id else ""

        cleaned = _CITATION_RE.sub(_replace, text or "")
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        cleaned = re.sub(r" +([,.;:!?])", r"\1", cleaned)
        return cleaned.strip()

    @staticmethod
    def _detect_query_lang(text: str) -> str:
        has_cyr = bool(_CYRILLIC_RE.search(text or ""))
        has_lat = bool(_LATIN_RE.search(text or ""))
        if has_cyr and not has_lat:
            return "ru"
        if has_lat and not has_cyr:
            return "en"
        if has_cyr:
            return "ru"
        return "unknown"

    @staticmethod
    def _normalize_query_text(text: str) -> str:
        return " ".join((text or "").split()).strip()

    def _translate_query_for_retrieval(self, user_text: str, *, target_lang: str) -> str | None:
        if target_lang not in {"en", "ru"}:
            return None
        lang_name = "English" if target_lang == "en" else "Russian"
        prompt = (
            "Translate this user question for semantic document retrieval.\n"
            f"Target language: {lang_name}\n"
            "Rules:\n"
            "- Preserve meaning.\n"
            "- Keep it as a search query/question.\n"
            "- Return ONLY the translated text, one line.\n\n"
            f"User question:\n{user_text}\n"
        )
        try:
            translated = Settings.llm.complete(prompt).text
        except Exception:
            return None
        translated = self._normalize_query_text(translated)
        if not translated:
            return None
        if translated.lower() == self._normalize_query_text(user_text).lower():
            return None
        return translated

    def _build_retrieval_queries(self, user_text: str) -> list[str]:
        user_text = self._normalize_query_text(user_text)
        if not user_text:
            return []

        queries = [user_text]
        lang = self._detect_query_lang(user_text)
        target_lang = "en" if lang == "ru" else "ru" if lang == "en" else None
        if target_lang:
            translated = self._translate_query_for_retrieval(user_text, target_lang=target_lang)
            if translated and translated not in queries:
                queries.append(translated)
        return queries

    @staticmethod
    def _source_node_key(node_with_score) -> str:
        md = getattr(getattr(node_with_score, "node", None), "metadata", {}) or {}
        for k in ("chunk_id", "id", "region_id"):
            v = md.get(k)
            if v:
                return f"{k}:{v}"
        text = ""
        try:
            text = (node_with_score.node.get_text() or "").strip()
        except Exception:
            text = ""
        return f"text:{hash(text)}"

    @staticmethod
    def _merge_source_nodes(node_lists: list[list], *, limit: int) -> list:
        merged: dict[str, object] = {}
        for nodes in node_lists:
            for nws in nodes:
                key = RagChatBackend._source_node_key(nws)
                prev = merged.get(key)
                prev_score = getattr(prev, "score", None) if prev is not None else None
                cur_score = getattr(nws, "score", None)
                if prev is None:
                    merged[key] = nws
                elif isinstance(cur_score, (int, float)) and (
                    not isinstance(prev_score, (int, float)) or cur_score > prev_score
                ):
                    merged[key] = nws

        def _score(nws):
            s = getattr(nws, "score", None)
            return s if isinstance(s, (int, float)) else float("-inf")

        return sorted(merged.values(), key=_score, reverse=True)[:limit]

    def generate_reply(
        self,
        *,
        user_text: str,
        messages: list[dict[str, str]],
        memory_k: int,
    ) -> RagReply:
        retriever = self._get_retriever()
        manifest = self._load_manifest()

        context_messages = self._prepare_context(messages, memory_k)
        retrieval_queries = self._build_retrieval_queries(user_text)
        node_lists = [retriever.retrieve(q) for q in retrieval_queries] or [[]]
        source_nodes = self._merge_source_nodes(
            node_lists,
            limit=max(self.settings.max_sources, self.settings.similarity_top_k),
        )
        resp = SimpleNamespace(source_nodes=source_nodes)
        available_source_count = min(
            len(getattr(resp, "source_nodes", []) or []),
            self.settings.max_sources,
        )

        sources_block_llm = format_sources(
            resp,
            project_root=self.project_root,
            manifest=manifest,
            max_sources=self.settings.max_sources,
            snippet_chars=700,
            include_paths=False,
        )

        conversation_context = "\n".join(
            f"{m['role']}: {m['content']}" for m in context_messages
        )

        prompt = f"""
You are answering using RAG.

Current user question:
{user_text}

Conversation history:
{conversation_context}

Rules:
- Detect the language of the CURRENT USER QUESTION and answer in the same language.
- Prefer information from the SNIPPET texts below.
- Every claim grounded in snippets must end with citations like [1] or [2][3].
- Use ONLY citation IDs that exist in the Sources list below (from [1] to [{available_source_count}] if any sources exist).
- You MUST choose exactly one of these 3 response modes:
  A) Source answer is sufficient:
     - Answer using the sources only (can rephrase, but do not distort facts).
     - Use citations for source-grounded claims.
     - Do NOT add any extra header text.
  B) Sources partially answer the question:
     - First provide the source-grounded part with citations.
     - Then add one blank line.
     - Then write EXACTLY this header: "ДОПОЛНЕНИЕ, не основанное на источниках:"
     - Then provide your own hypothesis/additional reasoning WITHOUT citations.
  C) Sources do not answer the question:
     - Write EXACTLY this prefix: "Информация в источниках отсутствует. ОТВЕТ от LLM:"
     - Then answer using your own reasoning/general knowledge WITHOUT citations.
- Never mix the three modes in one response.
- Do not fabricate citations.

Sources:
{sources_block_llm}

Answer:
"""

        answer = Settings.llm.complete(prompt).text
        cited_ids = self._extract_citation_ids(answer)
        if cited_ids and (not available_source_count or max(cited_ids) > available_source_count):
            answer = self._strip_invalid_citations(answer, max_valid_id=available_source_count)
        sources_text = format_sources(
            resp,
            project_root=self.project_root,
            manifest=manifest,
            max_sources=self.settings.max_sources,
            snippet_chars=200,
            include_paths=True,
        )
        return RagReply(
            answer=answer,
            sources_text=sources_text,
            context_messages_used=len(context_messages),
        )
