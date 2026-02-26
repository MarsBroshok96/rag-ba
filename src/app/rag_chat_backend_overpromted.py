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
_SUPPLEMENT_HEADER = "ДОПОЛНЕНИЕ, не основанное на источниках:"
_NO_SOURCES_PREFIX = "Информация в источниках отсутствует. ОТВЕТ от LLM:"
_NO_SOURCE_INFO_TOKEN = "__NO_SOURCE_INFO__"
_NO_SUPPLEMENT_TOKEN = "__NO_SUPPLEMENT__"


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
    def _strip_all_citations(text: str) -> str:
        cleaned = _CITATION_RE.sub("", text or "")
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        cleaned = re.sub(r" +([,.;:!?])", r"\1", cleaned)
        return cleaned.strip()

    @staticmethod
    def _split_answer_for_modes(answer: str) -> tuple[str, str | None]:
        if not answer:
            return "", None
        if _SUPPLEMENT_HEADER in answer:
            head, tail = answer.split(_SUPPLEMENT_HEADER, maxsplit=1)
            return head, _SUPPLEMENT_HEADER + tail
        return answer, None

    def _enforce_mode_citation_rules(self, answer: str) -> str:
        if not answer:
            return answer

        # Mode C must not contain citations at all.
        if _NO_SOURCES_PREFIX in answer:
            return self._strip_all_citations(answer)

        # In mode B, supplementary section is explicitly not source-grounded.
        grounded, supplement = self._split_answer_for_modes(answer)
        if supplement is None:
            return answer
        supplement_clean = self._strip_all_citations(supplement)
        return grounded.rstrip() + "\n\n" + supplement_clean.lstrip()

    @staticmethod
    def _is_mode_c_answer(answer: str) -> bool:
        return (answer or "").lstrip().startswith(_NO_SOURCES_PREFIX)

    @staticmethod
    def _grounded_section_for_citations(answer: str) -> str:
        grounded, _ = RagChatBackend._split_answer_for_modes(answer)
        return grounded or ""

    @staticmethod
    def _is_sparse_citation_distribution(answer: str, *, available_source_count: int) -> bool:
        if available_source_count < 2 or not answer:
            return False
        if _NO_SOURCES_PREFIX in answer:
            return False

        grounded = RagChatBackend._grounded_section_for_citations(answer)
        grounded_plain = RagChatBackend._strip_all_citations(grounded)
        if len(grounded_plain) < 120:
            return False

        lines = [ln.strip() for ln in grounded.splitlines() if ln.strip()]
        if len(lines) < 2:
            return False
        content_lines = [
            ln for ln in lines if len(RagChatBackend._strip_all_citations(ln)) >= 18
        ]
        if len(content_lines) < 2:
            return False

        unique_citations = RagChatBackend._extract_citation_ids(grounded)
        if len(unique_citations) <= 1:
            return True

        cited_content_lines = sum(1 for ln in content_lines if _CITATION_RE.search(ln))
        if len(content_lines) >= 4 and cited_content_lines <= max(1, len(content_lines) // 2):
            return True
        return False

    @staticmethod
    def _needs_citation_repair(answer: str, *, available_source_count: int) -> bool:
        if available_source_count <= 0 or not answer:
            return False
        if _NO_SOURCES_PREFIX in answer:
            return False

        grounded = RagChatBackend._grounded_section_for_citations(answer)
        if not grounded.strip():
            return False

        lines = [ln.strip() for ln in grounded.splitlines() if ln.strip()]
        bullet_lines = [ln for ln in lines if ln.startswith(("-", "*", "1.", "2.", "3.", "4.", "5.", "6."))]
        missing_bullet_citations = [ln for ln in bullet_lines if not _CITATION_RE.search(ln)]
        if missing_bullet_citations:
            return True

        citation_count = len(_CITATION_RE.findall(grounded))
        if len(lines) >= 4 and citation_count <= 1:
            return True
        if RagChatBackend._is_sparse_citation_distribution(
            answer,
            available_source_count=available_source_count,
        ):
            return True
        return False

    def _repair_answer_citations(
        self,
        *,
        answer: str,
        sources_block_llm: str,
        available_source_count: int,
    ) -> str:
        if not self._needs_citation_repair(answer, available_source_count=available_source_count):
            return answer

        repair_prompt = f"""
You are editing an existing answer. Preserve the wording and structure as much as possible.

Task:
- Add missing source citations [n] to source-grounded statements in the answer draft.
- Use ONLY citation IDs that exist in Sources ([1]..[{available_source_count}] if available).
- Do NOT invent facts or new text unless needed to place citations.
- Keep the answer language unchanged.
- Re-check every substantive bullet/paragraph line; do not leave grounded lines without citations.
- Do not lazily reuse only [1] across the whole answer if different statements are supported by other snippets.
- If one source truly supports all grounded statements, keeping only that source is allowed.
- Keep the exact header "{_SUPPLEMENT_HEADER}" if present.
- Do NOT add citations inside the section after "{_SUPPLEMENT_HEADER}".
- If the answer starts with "{_NO_SOURCES_PREFIX}", return the answer unchanged.

Sources:
{sources_block_llm}

Answer draft:
{answer}

Revised answer:
"""
        try:
            repaired = Settings.llm.complete(repair_prompt).text
        except Exception:
            return answer
        repaired = (repaired or "").strip() or answer
        if not self._is_sparse_citation_distribution(
            repaired,
            available_source_count=available_source_count,
        ):
            return repaired

        # Some models still collapse all grounded claims to [1]. Run a stricter line-by-line pass.
        strict_prompt = f"""
You are performing a strict citation audit for a RAG answer.

Task:
        - Keep the answer wording and structure as close to the draft as possible.
- For the source-grounded part, audit each non-empty line/bullet and add missing [n] citations.
- If different lines are supported by different snippets, reflect that with different citation IDs.
- Do NOT overuse only [1] when other snippets clearly match the line better.
- Use ONLY source IDs from [1]..[{available_source_count}].
- Do NOT add citations after the header "{_SUPPLEMENT_HEADER}".
- If the answer starts with "{_NO_SOURCES_PREFIX}", return it unchanged.
- Return only the revised answer text.

Sources:
{sources_block_llm}

Answer draft:
{repaired}
"""
        try:
            strict_repaired = Settings.llm.complete(strict_prompt).text
        except Exception:
            return repaired
        return (strict_repaired or "").strip() or repaired

    def _retry_if_suspicious_mode_c(
        self,
        *,
        answer: str,
        user_text: str,
        conversation_context: str,
        sources_block_llm: str,
        available_source_count: int,
    ) -> str:
        # If the model says "no info in sources" while we actually have retrieved snippets,
        # force one retry and explicitly disallow mode C.
        if not self._is_mode_c_answer(answer):
            return answer
        if available_source_count <= 0:
            return answer

        retry_prompt = f"""
You previously answered in mode C ("{_NO_SOURCES_PREFIX}").
That was likely incorrect because relevant snippets were retrieved.

Re-answer the SAME user question using the provided snippets.

Rules:
- Answer in the same language as the user question.
- You MUST choose mode A or B (mode C is forbidden for this retry).
- If snippets answer only part of the question, use mode B with exact header "{_SUPPLEMENT_HEADER}".
- If snippets answer the question sufficiently, use mode A.
- Use citations [n] for source-grounded statements and bullet points.
- Use ONLY citation IDs from [1]..[{available_source_count}].
- Do not add citations in the "{_SUPPLEMENT_HEADER}" section.
- Prefer generic concept definitions and standard terminology when user asks a generic question.

Current user question:
{user_text}

Conversation history:
{conversation_context}

Sources:
{sources_block_llm}

Revised answer:
"""
        try:
            retried = Settings.llm.complete(retry_prompt).text
        except Exception:
            return answer
        retried = (retried or "").strip()
        if not retried:
            return answer
        return retried

    def _generate_grounded_answer(
        self,
        *,
        user_text: str,
        conversation_context: str,
        sources_block_llm: str,
        available_source_count: int,
    ) -> str:
        prompt = f"""
You are a senior business analyst answering with RAG evidence.

Task:
- Answer the user question using ONLY the provided snippets.
- Snippets may be in a different language than the user question; this is still valid evidence.
- Answer in the SAME language as the user question.
- Use a structured, business-oriented style (clear bullets/sections, concise wording, practical framing).
- For broad generic questions, prefer general definitions and canonical approaches over narrow domain-specific examples unless the user asks for a domain.
- Every source-grounded claim/bullet must include citations like [1] or [2][3].
- Use ONLY citation IDs from [1]..[{available_source_count}].
- If snippets answer even PART of the question, provide that partial answer with citations.
- Return EXACTLY {_NO_SOURCE_INFO_TOKEN} ONLY when snippets contain no relevant information at all.
- Do NOT add the headers "{_SUPPLEMENT_HEADER}" or "{_NO_SOURCES_PREFIX}" in this step.

Current user question:
{user_text}

Conversation history:
{conversation_context}

Sources:
{sources_block_llm}

Answer:
"""
        return (Settings.llm.complete(prompt).text or "").strip()

    def _classify_coverage(
        self,
        *,
        user_text: str,
        grounded_answer: str,
        sources_block_llm: str,
        available_source_count: int,
    ) -> str:
        if not grounded_answer or grounded_answer.strip() == _NO_SOURCE_INFO_TOKEN:
            return "NONE"
        if available_source_count <= 0:
            return "NONE"

        prompt = f"""
Classify whether the source-grounded answer covers the user's question.

Return EXACTLY one token: FULL or PARTIAL or NONE

Rules:
- FULL: the source-grounded answer covers all essential parts of the user's question.
- PARTIAL: sources answer only part of the question, or answer lacks essential requested details.
- NONE: sources do not provide relevant information.
- If in doubt between FULL and PARTIAL, choose PARTIAL.

User question:
{user_text}

Source-grounded answer:
{grounded_answer}

Sources (for reference):
{sources_block_llm}

Classification:
"""
        out = (Settings.llm.complete(prompt).text or "").strip().upper()
        if "FULL" in out:
            return "FULL"
        if "PARTIAL" in out:
            return "PARTIAL"
        if "NONE" in out:
            return "NONE"
        return "PARTIAL"

    def _generate_supplement(
        self,
        *,
        user_text: str,
        grounded_answer: str,
    ) -> str:
        prompt = f"""
You are a senior business analyst.

Task:
- Write ONLY the missing part of the answer that is NOT covered by the source-grounded answer.
- Use the same language as the user question.
- Keep business-analytic style: structured, concise, practical.
- This is NOT source-grounded text, so DO NOT use citations.
- If there is no meaningful missing part, return EXACTLY: {_NO_SUPPLEMENT_TOKEN}
- Do NOT repeat the grounded answer.
- Do NOT include header "{_SUPPLEMENT_HEADER}" (the caller will add it).

User question:
{user_text}

Source-grounded answer already provided:
{grounded_answer}

Missing-part supplement:
"""
        return (Settings.llm.complete(prompt).text or "").strip()

    def _generate_llm_only_answer(
        self,
        *,
        user_text: str,
        conversation_context: str,
    ) -> str:
        prompt = f"""
You are a senior business analyst.

Task:
- Answer the user question using your general knowledge (no source grounding available).
- Use the SAME language as the user question.
- Use a structured, business-oriented style (clear bullets/sections, concise wording, practical framing).
- Do NOT use citations like [1].
- Do NOT include the prefix "{_NO_SOURCES_PREFIX}" (the caller will add it).

Current user question:
{user_text}

Conversation history:
{conversation_context}

Answer:
"""
        body = (Settings.llm.complete(prompt).text or "").strip()
        body = self._strip_all_citations(body)
        return f"{_NO_SOURCES_PREFIX}\n\n{body}" if body else _NO_SOURCES_PREFIX

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
        if not node_lists:
            return []

        merged: list[object] = []
        seen: set[str] = set()
        per_list_idx = [0 for _ in node_lists]

        def _take_next_from_list(list_idx: int) -> bool:
            nodes = node_lists[list_idx]
            while per_list_idx[list_idx] < len(nodes):
                nws = nodes[per_list_idx[list_idx]]
                per_list_idx[list_idx] += 1
                key = RagChatBackend._source_node_key(nws)
                if key in seen:
                    continue
                seen.add(key)
                merged.append(nws)
                return True
            return False

        # Stage 1: keep top primary results first, but don't let them fully crowd out translated retrieval.
        primary_budget = max(1, min(len(node_lists[0]), limit // 2 + 1))
        while len(merged) < min(primary_budget, limit) and _take_next_from_list(0):
            pass
        if len(merged) >= limit:
            return merged[:limit]

        # Stage 2: round-robin across supplementary retrieval queries (e.g., translated query)
        # to improve cross-language coverage in the top-k prompt sources.
        progress = True
        while len(merged) < limit and progress:
            progress = False
            for list_idx in range(1, len(node_lists)):
                if len(merged) >= limit:
                    break
                added = _take_next_from_list(list_idx)
                progress = progress or added

        # Stage 3: fill remaining slots by round-robin over all queries, keeping dedupe.
        progress = True
        while len(merged) < limit and progress:
            progress = False
            for list_idx in range(len(node_lists)):
                if len(merged) >= limit:
                    break
                added = _take_next_from_list(list_idx)
                progress = progress or added

        return merged[:limit]

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

        grounded_answer = self._generate_grounded_answer(
            user_text=user_text,
            conversation_context=conversation_context,
            sources_block_llm=sources_block_llm,
            available_source_count=available_source_count,
        )
        # Some models may ignore the instruction for this stage and emit the mode-C prefix.
        # Keep the content instead of discarding it.
        if grounded_answer.startswith(_NO_SOURCES_PREFIX):
            grounded_answer = grounded_answer[len(_NO_SOURCES_PREFIX) :].lstrip(":\n\r\t ")
        if grounded_answer.strip() == _NO_SOURCE_INFO_TOKEN or available_source_count <= 0:
            answer = self._generate_llm_only_answer(
                user_text=user_text,
                conversation_context=conversation_context,
            )
        else:
            grounded_answer = self._strip_invalid_citations(
                grounded_answer,
                max_valid_id=available_source_count,
            )
            grounded_answer = self._repair_answer_citations(
                answer=grounded_answer,
                sources_block_llm=sources_block_llm,
                available_source_count=available_source_count,
            )
            grounded_answer = self._strip_invalid_citations(
                grounded_answer,
                max_valid_id=available_source_count,
            )
            grounded_cited_ids = self._extract_citation_ids(grounded_answer)

            coverage = self._classify_coverage(
                user_text=user_text,
                grounded_answer=grounded_answer,
                sources_block_llm=sources_block_llm,
                available_source_count=available_source_count,
            )
            # Guard against classifier false negatives: if we already produced a
            # non-empty grounded answer (especially with citations), do not
            # downgrade to mode C.
            if coverage == "NONE" and grounded_answer.strip():
                if grounded_cited_ids or len(self._strip_all_citations(grounded_answer).strip()) >= 40:
                    coverage = "PARTIAL"

            if coverage == "FULL":
                answer = grounded_answer
            elif coverage == "PARTIAL":
                supplement = self._generate_supplement(
                    user_text=user_text,
                    grounded_answer=grounded_answer,
                )
                supplement = self._strip_all_citations(supplement)
                if not supplement or supplement == _NO_SUPPLEMENT_TOKEN:
                    answer = grounded_answer
                else:
                    answer = grounded_answer.rstrip() + f"\n\n{_SUPPLEMENT_HEADER}\n" + supplement
            else:
                answer = self._generate_llm_only_answer(
                    user_text=user_text,
                    conversation_context=conversation_context,
                )

        answer = self._retry_if_suspicious_mode_c(
            answer=answer,
            user_text=user_text,
            conversation_context=conversation_context,
            sources_block_llm=sources_block_llm,
            available_source_count=available_source_count,
        )
        if not self._is_mode_c_answer(answer) and available_source_count > 0:
            answer = self._strip_invalid_citations(answer, max_valid_id=available_source_count)
            answer = self._repair_answer_citations(
                answer=answer,
                sources_block_llm=sources_block_llm,
                available_source_count=available_source_count,
            )
            answer = self._strip_invalid_citations(answer, max_valid_id=available_source_count)
        answer = self._enforce_mode_citation_rules(answer)
        cited_ids = self._extract_citation_ids(answer)
        if cited_ids and (not available_source_count or max(cited_ids) > available_source_count):
            answer = self._strip_invalid_citations(answer, max_valid_id=available_source_count)
            cited_ids = self._extract_citation_ids(answer)

        cited_ids_in_range = {
            n for n in cited_ids if 1 <= n <= available_source_count
        } or None
        only_source_ids = cited_ids_in_range
        if self._is_sparse_citation_distribution(
            answer,
            available_source_count=available_source_count,
        ):
            only_source_ids = None
        sources_text = format_sources(
            resp,
            project_root=self.project_root,
            manifest=manifest,
            max_sources=self.settings.max_sources,
            snippet_chars=200,
            include_paths=True,
            only_source_ids=only_source_ids,
        )
        return RagReply(
            answer=answer,
            sources_text=sources_text,
            context_messages_used=len(context_messages),
        )
