from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from src.common.project_paths import (
    CHROMA_RAG_DIR,
    DATA_DIR,
    LAYOUT_ROOT,
    MANIFEST_PATH,
    MODELS_DIR,
    OCR_EXPORT_ROOT,
    OCR_LAYOUT_ROOT,
    PDF_INBOX_DIR,
    PROJECT_ROOT,
    RAG_BA_OCR_DIR,
    WORD_INBOX_DIR,
)


@dataclass(slots=True)
class CheckResult:
    status: str  # ok | warn | fail
    label: str
    detail: str


def _exists(path: Path, *, kind: str = "any") -> bool:
    if kind == "dir":
        return path.is_dir()
    if kind == "file":
        return path.is_file()
    return path.exists()


def run_checks() -> list[CheckResult]:
    results: list[CheckResult] = []

    required_dirs = [
        ("repo root", PROJECT_ROOT, "dir"),
        ("data dir", DATA_DIR, "dir"),
        ("ocr app dir", RAG_BA_OCR_DIR, "dir"),
    ]
    required_files = [
        ("root pyproject", PROJECT_ROOT / "pyproject.toml", "file"),
        ("Makefile", PROJECT_ROOT / "Makefile", "file"),
        ("ocr pyproject", RAG_BA_OCR_DIR / "pyproject.toml", "file"),
        ("layout module", PROJECT_ROOT / "src" / "ingestion" / "layout_export_and_viz_pdf.py", "file"),
        ("index module", PROJECT_ROOT / "src" / "index" / "build_vector_index_from_chunks.py", "file"),
        ("ocr wrapper", RAG_BA_OCR_DIR / "ocr_regions.py", "file"),
        ("ocr pipeline impl", RAG_BA_OCR_DIR / "src" / "pipeline" / "ocr_regions.py", "file"),
    ]

    for label, path, kind in [*required_dirs, *required_files]:
        if _exists(path, kind=kind):
            results.append(CheckResult("ok", label, str(path)))
        else:
            results.append(CheckResult("fail", label, f"missing {kind}: {path}"))

    runtime_dirs = [
        ("pdf inbox", PDF_INBOX_DIR),
        ("word inbox", WORD_INBOX_DIR),
        ("layout output", LAYOUT_ROOT),
        ("ocr layout output", OCR_LAYOUT_ROOT),
        ("ocr export output", OCR_EXPORT_ROOT),
        ("vectorstore", CHROMA_RAG_DIR),
    ]
    for label, path in runtime_dirs:
        status = "ok" if path.exists() else "warn"
        msg = str(path) if path.exists() else f"not created yet: {path}"
        results.append(CheckResult(status, label, msg))

    model_candidates = [
        MODELS_DIR / "publaynet_frcnn_model_final.pth",
        MODELS_DIR / "d2_configs" / "COCO-Detection" / "faster_rcnn_R_50_FPN_3x.yaml",
    ]
    for path in model_candidates:
        status = "ok" if path.exists() else "warn"
        msg = str(path) if path.exists() else f"optional until `make layout`: {path}"
        results.append(CheckResult(status, f"model asset {path.name}", msg))

    manifest_status = "ok" if MANIFEST_PATH.exists() else "warn"
    manifest_msg = str(MANIFEST_PATH) if MANIFEST_PATH.exists() else f"generated after `make index`: {MANIFEST_PATH}"
    results.append(CheckResult(manifest_status, "manifest", manifest_msg))

    return results


def _print_group(title: str, items: Iterable[CheckResult]) -> None:
    print(title)
    for item in items:
        print(f"  - {item.label}: {item.detail}")


def main() -> None:
    results = run_checks()
    fails = [r for r in results if r.status == "fail"]
    warns = [r for r in results if r.status == "warn"]
    oks = [r for r in results if r.status == "ok"]

    print("Smoke check: rag-ba-agents")
    _print_group("OK", oks)
    if warns:
        _print_group("WARN", warns)
    if fails:
        _print_group("FAIL", fails)
        raise SystemExit(1)

    print(f"Summary: ok={len(oks)} warn={len(warns)} fail=0")


if __name__ == "__main__":
    main()
