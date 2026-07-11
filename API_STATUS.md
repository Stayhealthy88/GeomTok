# GeomTok v1.1 API — Conformance & Status

Status of the managed API (`geomtok/server/app.py`) + core façade (`geomtok/api.py`) against [PRD.md](PRD.md) §7. Reviewed and hardened via a 4-dimension adversarial pass (correctness · PRD-conformance · security/design · test coverage) + adversarial re-verification (50 real icons, partial-chunk streaming, hostile inputs). All findings P0–P2 are resolved; see `git log feat/v1-api`.

## Endpoint conformance (PRD §7.1)

| Endpoint | Status | Notes |
|---|---|---|
| `POST /v1/tokenize` | ✅ production | size guard, unsupported-element reject, deterministic; **`lean:true`** emits marker-free lean L1 (~23% shorter, PAPER §5.1) |
| `POST /v1/detokenize` | ✅ production | `tokenizer_version` required; **FSA-validated** (`{valid, repaired}`); auto-unmerges L2 ids at any level; full and lean streams decode to byte-identical SVG |
| `POST /v1/eval` | ✅ production | builtin + remote; echoes version/vocab; **GeomTok-Eval/1.1** adds coord-error tail (p50/p90/p95/p99 + `perceptible_err_rate` @2px) and `lean` |
| `POST /v1/batch` | ✅ production | ≤1000 items, ≤32MB (svg+token bytes), op validated, `on_error∈{skip,fail_fast}`; tokenize rows echo `lean` |
| `POST /v1/batch/jobs` | ✅ **async** | real background worker (ThreadPoolExecutor); inline items or `input_uri` NDJSON blob; `output_uri`, `webhook_url`; **v1.1:** job-level `level`/`lean` are merged into every item as defaults (per-NDJSON-line keys override) — in v1.0, job-level `level` was silently ignored on all paths |
| `GET /v1/batch/jobs/{id}` | ✅ **async** | live status (client observes `running` + climbing progress); `results_url` for blob output |
| `DELETE /v1/batch/jobs/{id}` | ✅ **async** | cooperative cancellation at the next item boundary |
| `POST /v1/stream` | ✅ functional | NDJSON, ordered, bounded memory, per-line error isolation; verified over real chunked HTTP |
| `GET /v1/vocab/{id}` | ✅ production | immutable manifest + `content_hash` |
| `GET /v1/healthz` | ✅ production | version, vocab, has_l2, manifest hash |

## Guarantees enforced

- **Determinism (PRD §9):** same `(tokenizer_version, vocab_id, config)` → bit-identical tokens. `lean` is part of the request config and is echoed in every tokenize payload; the **default (`lean:false`) stream is bit-identical to v1.0**, so `tokenizer_version` stays `geomtok-1.0.0`. Compression ratio is now deterministic `round(len(svg)/n_tokens, 4)` (the retracted cl100k strawman and its non-deterministic `0.0` fallback were removed; the honest 3.54× vs domain-BPE is a *corpus* metric reported by `/v1/eval`).
- **Validity (PRD G3):** `detokenize` runs the torch-free `GrammarFSA`; invalid streams are repaired and flagged. No more hardcoded `valid:true`.
- **No silent geometry loss:** L2 `token_ids` decoded with default `level="L1"` auto-unmerge correctly (the bundled manifest ships 3000 L2 merges, so `default()` has L2 on).
- **Version echo (PRD §7.6):** every endpoint echoes `tokenizer_version` + `vocab_id`.
- **Error codes (PRD §7.6):** `PARSE_ERROR`, `PARSE_UNSUPPORTED_ELEMENT`, `PAYLOAD_TOO_LARGE` (413), `VOCAB_MISMATCH`, `VERSION_REQUIRED`, `INVALID_REQUEST` (400), `JOB_NOT_FOUND` (404) — serialized as `{error:{code,message,detail?}}`.
- **DoS surface:** 256KB single / 32MB batch byte limits; unsupported elements rejected before parse.
- **Renderer honesty (Eval/1.1):** without a renderer (cairosvg/resvg) `render_ssim_mean` is `null` and `render_scenes: 0` — a missing renderer is no longer reported as SSIM `0.0` (which reads as "fidelity collapsed"). Render *failures* still score 0.0, preserving game-resistance; the degenerate-tokenizer control also trips on value fidelity, which needs no renderer.

## Known limitations / next for production

1. **Async jobs are genuinely async but single-node.** `/v1/batch/jobs` runs a real ThreadPoolExecutor worker — submit returns immediately, the client observes `running` + climbing progress, cooperative cancel works, and webhooks fire on completion. The `JobStore`/`BlobStore` are pluggable interfaces: the defaults are in-memory + local-filesystem (single node); production drops in Redis/DB + S3/GCS without touching routes. (`geomtok/server/jobs.py`.)
2. **No auth / rate-limit / multi-tenancy yet** — these are the managed-tier wedge (PRD §6); the OSS core deliberately ships without them.
3. **Domain-BPE compression baseline** (the honest 3.54×) is computed by `/v1/eval` over a corpus, not per-call; per-call `compression_ratio` is chars-per-token.
4. **L3 spatial** tokens are experimental (emit L1 substrate with a warning).

## Tests

Core 145 + API 77 + review-fix regressions 12 + async-jobs 17 + lean-L1/eval-tail 30 = **281 assertions, 0 regressions**. Run all via the subprocess meta-runner: `pytest tests/test_all_scripts.py` (19/19). New in v1.1: `tests/test_lean_l1.py` (19 checks: marker-free streams are FSA-valid, decode byte-identical to full streams on all fixtures, and survive the module/server routes — including job-level `lean`/`level` defaults on inline *and* NDJSON job inputs, per-line override precedence, and `lean` echo in batch/job rows), plus tail-metric and renderer-honesty checks in `tests/test_eval_protocol.py` (17 → 28 checks with a renderer; renderer-less environments skip the 1 SSIM game-resistance check → 280 total).
