# Data Contracts

This file documents the JSON formats produced/consumed by the pipeline.

## 1. Layout JSON (PDF) — `layout/<doc_id>/pageXXX_layout.json`
Top-level:
- `doc_id`: str
- `source_pdf`: str (absolute path)
- `page`: int
- `regions`: list[region]

Region:
- `id`: str (stable)
- `doc_id`: str
- `source_pdf`: str
- `type`: "text"|"title"|"table"|"figure"|"other" (list normalized to text upstream)
- `page`: int
- `bbox_px`: [x0,y0,x1,y1]
- `bbox_norm`: [x0,y0,x1,y1] in [0..1]
- `score`: float
- `page_image_path`: str (absolute)

## 2. Layout OCR JSON — `layout_ocr/<doc_id>/pageXXX_layout_ocr.json`
Top-level:
- `doc_id`
- `source_pdf`
- `page`
- `image`: page image path
- `size_px`: [W,H]
- `regions`: list[region+ocr]

Each region adds:
- `crop_path`
- `ocr_lines`: list[{text, score, bbox_px}]
- `ocr_line_count`

## 3. Canonical page — `canon/<doc_id>/pageXXX_canonical.json`
- `page`: int
- `blocks`: list[{region_id,type,bbox_px,text,(optional paths)}]

## 4. Full document — `export/<doc_id>/full_document.json`
(Used for both PDF and DOCX)
- `doc_id`
- `source_file`
- `source_path` (preferred; pdf/docx)
- `page_count` (pdf) OR absent for docx 1-page
- `pages`: list[{page:int, blocks:[...]}]  (PDF)
OR for DOCX simplified:
- `page`: 1
- `blocks`: [...]

## 5. Chunks — `export/<doc_id>/chunks.json`
List[chunk]:
- `chunk_id`: f"{doc_id}__cN"
- `doc_id`
- `source_file`
- `source_path`
- `page`: int|None
- `page_image_path`: str|None
- `type`: e.g. title/text/table
- `region_ids`: list[str] or compact form upstream
- `crop_paths`: list[str]
- `text`: str
- `char_len`: int

## 6. Manifest — `data/manifest.json`
- `docs`: dict[doc_id -> {pdf_path/source_path, page_image_tpl, crop_dir_tpl, ...}]
Used for resolving best clickable path for sources.