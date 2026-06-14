# GeomTok v1.0 API — Conformance & Status

Status of the managed API (`geomtok/server/app.py`) + core façade (`geomtok/api.py`) against [PRD.md](PRD.md) §7. Reviewed and hardened via a 4-dimension adversarial pass (correctness · PRD-conformance · security/design · test coverage) + adversarial re-verification (50 real icons, partial-chunk streaming, hostile inputs). All findings P0–P2 are resolved; see `git log feat/v1-api`.

## Endpoint conformance (PRD §7.1)

| Endpoint | Status | Notes |
|---|---|---|
| `POST /v1/tokenize` | ✅ production | size guard, unsupported-element reject, deterministic |
| `POST /v1/detokenize` | ✅ production | `tokenizer_version` required; **FSA-validated** (`{valid, repaired}`); auto-unmerges L2 ids at any level |
| `POST /v1/eval` | ✅ production | builtin + remote; echoes version/vocab |
| `POST /v1/batch` | ✅ production | ≤1000 items, ≤32MB (svg+token bytes), op validated, `on_error∈{skip,fail_fast}` |
| `POST /v1/batch/jobs` | ⚠️ **stub** | correct 202 contract; processes in-process synchronously — swap for a real queue/bucket backend in production |
| `GET /v1/batch/jobs/{id}` | ⚠️ **stub** | correct status/progress/partial shape; backed by in-memory `app.state.jobs` |
| `POST /v1/stream` | ✅ functional | NDJSON, ordered, bounded memory, per-line error isolation; verified over real chunked HTTP |
| `GET /v1/vocab/{id}` | ✅ production | immutable manifest + `content_hash` |
| `GET /v1/healthz` | ✅ production | version, vocab, has_l2, manifest hash |

## Guarantees enforced

- **Determinism (PRD §9):** same `(tokenizer_version, vocab_id, config)` → bit-identical tokens. Compression ratio is now deterministic `round(len(svg)/n_tokens, 4)` (the retracted cl100k strawman and its non-deterministic `0.0` fallback were removed; the honest 3.54× vs domain-BPE is a *corpus* metric reported by `/v1/eval`).
- **Validity (PRD G3):** `detokenize` runs the torch-free `GrammarFSA`; invalid streams are repaired and flagged. No more hardcoded `valid:true`.
- **No silent geometry loss:** L2 `token_ids` decoded with default `level="L1"` auto-unmerge correctly (the bundled manifest ships 3000 L2 merges, so `default()` has L2 on).
- **Version echo (PRD §7.6):** every endpoint echoes `tokenizer_version` + `vocab_id`.
- **Error codes (PRD §7.6):** `PARSE_ERROR`, `PARSE_UNSUPPORTED_ELEMENT`, `PAYLOAD_TOO_LARGE` (413), `VOCAB_MISMATCH`, `VERSION_REQUIRED`, `INVALID_REQUEST` (400), `JOB_NOT_FOUND` (404) — serialized as `{error:{code,message,detail?}}`.
- **DoS surface:** 256KB single / 32MB batch byte limits; unsupported elements rejected before parse.

## Known limitations / next for production

1. **Async jobs & stream are single-node stubs.** `/v1/batch/jobs` processes synchronously and tracks jobs in process memory; a polling client never observes `running`. Production needs a real queue (e.g. Redis/SQS) + object-store in/out + webhooks (PRD §8).
2. **No auth / rate-limit / multi-tenancy yet** — these are the managed-tier wedge (PRD §6); the OSS core deliberately ships without them.
3. **Domain-BPE compression baseline** (the honest 3.54×) is computed by `/v1/eval` over a corpus, not per-call; per-call `compression_ratio` is chars-per-token.
4. **L3 spatial** tokens are experimental (emit L1 substrate with a warning).

## Tests

Core 145 + API 77 + review-fix regressions 12 = **234 assertions, 0 regressions**. Run all via the subprocess meta-runner: `pytest tests/test_all_scripts.py` (17/17). New API regression tests: `tests/test_api_review_fixes.py`.
