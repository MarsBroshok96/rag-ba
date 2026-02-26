# Web Chat Data Contracts (`rag_ba/web`)

## Chat file (`data/chats/<chat_id>.json`)
- `chat_id`: `str` (UUID)
- `title`: `str`
- `created_at`: ISO-8601 UTC string
- `updated_at`: ISO-8601 UTC string
- `messages`: array of:
  - `role`: `"user" | "assistant" | "system"`
  - `content`: `str`
  - `ts`: ISO-8601 UTC string
- `settings`:
  - `memory_k`: `int` (0..100)

## API

### `GET /api/chats`
Returns list of chat summaries:
- `chat_id`
- `title`
- `created_at`
- `updated_at`
- `message_count`

### `POST /api/chats`
Request (optional):
- `settings.memory_k`

Response:
- full chat object (same shape as chat file)

### `GET /api/chats/{chat_id}`
Response:
- full chat object

### `PATCH /api/chats/{chat_id}/settings`
Request:
- `memory_k` (optional, int)

Response:
- updated full chat object

### `DELETE /api/chats/{chat_id}`
Response:
- `{ "ok": true }`

### `POST /api/chat/send`
Request:
- `chat_id`
- `message`

Success response:
- `chat`: updated full chat
- `assistant_message`: last assistant message object
- `sources_text`: formatted sources for UI display
- `context_messages_used`: number of messages passed to LLM context after `memory_k` trimming

Error response:
- `{ "error": "chat_send_failed", "detail": "..." }` with HTTP 500

### `GET /api/source/file?path=...`
Purpose:
- Open/render a source file (crop/page/pdf/etc.) in browser via local web server

Rules:
- `path` must resolve to a file inside repo `PROJECT_ROOT`
- Paths outside project root are rejected

Response:
- `FileResponse` (inline)
