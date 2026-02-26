#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
import traceback
from pathlib import Path


def _silence_stdio() -> None:
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
    finally:
        os.close(devnull_fd)


def _worker(queue: mp.Queue, question: str, memory_k: int, messages: list[dict[str, str]]) -> None:
    try:
        _silence_stdio()
        from src.app.rag_chat_backend import RagChatBackend

        backend = RagChatBackend()
        reply = backend.generate_reply(
            user_text=question,
            messages=messages,
            memory_k=memory_k,
        )
        queue.put(
            {
                "ok": True,
                "answer": reply.answer,
                "sources_text": reply.sources_text,
                "context_messages_used": reply.context_messages_used,
            }
        )
    except Exception as exc:  # pragma: no cover - best effort diagnostics for local probing
        queue.put(
            {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )


def _load_messages(messages_json: str | None) -> list[dict[str, str]]:
    if not messages_json:
        return []
    data = json.loads(messages_json)
    if not isinstance(data, list):
        raise ValueError("--messages-json must be a JSON array")
    out: list[dict[str, str]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip()
        content = str(item.get("content", "")).strip()
        if role in {"user", "assistant", "system"} and content:
            out.append({"role": role, "content": content})
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Local RAG chat probe with hard timeout and JSON output.")
    parser.add_argument("--question", required=True, help="User question to send into RagChatBackend")
    parser.add_argument("--memory-k", type=int, default=6)
    parser.add_argument("--timeout-sec", type=int, default=180)
    parser.add_argument("--output", default="/tmp/rag_chat_probe_result.json")
    parser.add_argument(
        "--messages-json",
        default=None,
        help='Optional conversation history JSON, e.g. \'[{"role":"user","content":"..."}]\'',
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = time.time()

    try:
        messages = _load_messages(args.messages_json)
    except Exception as exc:
        payload = {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "question": args.question,
            "memory_k": args.memory_k,
            "duration_sec": round(time.time() - started_at, 3),
        }
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(str(output_path))
        return 2

    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue(maxsize=1)
    proc = ctx.Process(target=_worker, args=(queue, args.question, args.memory_k, messages))
    proc.start()
    proc.join(timeout=max(1, args.timeout_sec))

    payload: dict[str, object]
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=5)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=2)
        payload = {
            "ok": False,
            "error_type": "TimeoutError",
            "error": f"Probe timed out after {args.timeout_sec} seconds",
            "question": args.question,
            "memory_k": args.memory_k,
        }
        exit_code = 124
    else:
        try:
            payload = queue.get_nowait()
            exit_code = 0 if payload.get("ok") else 1
        except Exception as exc:  # pragma: no cover
            payload = {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": f"Worker exited without result: {exc}",
                "question": args.question,
                "memory_k": args.memory_k,
                "worker_exitcode": proc.exitcode,
            }
            exit_code = 1

    payload.update(
        {
            "question": args.question,
            "memory_k": args.memory_k,
            "duration_sec": round(time.time() - started_at, 3),
            "worker_exitcode": proc.exitcode,
        }
    )

    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(output_path)
    print(str(output_path))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
