"""
GeomTok 매니지드 API (FastAPI) — PRD §7.1
==========================================
OSS 코어(`geomtok.api.GeomTokenizer`)와 **동일 코드·동일 불변 vocab** 을 호스팅
위에 얹는다 (PRD §9: 로컬과 호스팅 결과가 비트-동일). 매니지드는 그 위에 인증
스텁·레이트리밋·배치 코얼레싱·표준 에러 직렬화만 추가한다.

엔드포인트 (PRD §7.1):
    POST /v1/tokenize         단건 토큰화
    POST /v1/detokenize       단건 디토큰화 (tokenizer_version 필수)
    POST /v1/eval             GeomTok-Eval 렌더 기반 프로토콜 (builtin/remote)
    POST /v1/batch            동기 배치 (items ≤1000, ≤32MB, on_error∈{skip,fail_fast})
    POST /v1/batch/jobs       비동기 대규모 잡 제출 (202; in-process stub)
    GET  /v1/batch/jobs/{id}  잡 상태·진행률·부분결과
    POST /v1/stream           NDJSON 스트리밍 토큰화 (순서 보존·바운디드 메모리)
    GET  /v1/vocab/{id}       불변 vocab 매니페스트 (오프라인 백킹)
    GET  /v1/healthz          헬스체크
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

try:
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
except Exception as e:  # noqa: BLE001
    raise ImportError(
        "FastAPI/pydantic required for the managed server. "
        "Install with: pip install 'geomtok[server]'") from e

from ..api import GeomTokenizer, MAX_SVG_BYTES
from ..errors import GeomTokError, PayloadTooLarge, InvalidRequest, JobNotFound
from ..tokenizer.manifest import load_bundled_manifest, DEFAULT_VOCAB_ID

# 배치 한도 (PRD §7.1)
MAX_BATCH_ITEMS = 1000
MAX_BATCH_BYTES = 32 * 1024 * 1024   # 32MB


# --------------------------------------------------------------------------- #
# 요청 모델
# --------------------------------------------------------------------------- #

class TokenizeReq(BaseModel):
    model_config = {"populate_by_name": True}
    svg: str
    level: str = "L1"
    lean: bool = False        # 보조 마커 없는 lean L1 (PAPER §5.1 권장 기질)
    config: Optional[Dict[str, Any]] = None
    return_: Optional[List[str]] = Field(default=None, alias="return")


class DetokenizeReq(BaseModel):
    token_ids: List[int]
    tokenizer_version: Optional[str] = None
    vocab_id: Optional[str] = None
    level: str = "L1"


class BatchItem(BaseModel):
    id: Optional[str] = None
    svg: Optional[str] = None
    token_ids: Optional[List[int]] = None


class BatchReq(BaseModel):
    op: str = "tokenize"                  # tokenize | detokenize
    items: List[BatchItem]
    level: str = "L1"
    lean: bool = False
    on_error: str = "skip"                # skip | fail_fast


class JobReq(BaseModel):
    """비동기 잡 — items(인라인) 또는 input_uri(NDJSON blob) 택1."""
    op: str = "tokenize"
    items: Optional[List[BatchItem]] = None
    input_uri: Optional[str] = None       # file:// NDJSON (대규모)
    output_uri: Optional[str] = None      # 결과 NDJSON 기록 위치
    level: str = "L1"
    lean: bool = False
    on_error: str = "skip"
    webhook_url: Optional[str] = None     # 종료 시 POST


class EvalScene(BaseModel):
    name: Optional[str] = None
    svg: str


class RemoteSpec(BaseModel):
    tokenize_url: str
    detokenize_url: str


class TokenizerSpec(BaseModel):
    builtin: Optional[bool] = True
    remote: Optional[RemoteSpec] = None


class EvalReq(BaseModel):
    scenes: List[EvalScene]
    tokenizer: Optional[TokenizerSpec] = None
    level: str = "L1"
    lean: bool = False
    render_res: int = 256


# --------------------------------------------------------------------------- #
# 앱 팩토리
# --------------------------------------------------------------------------- #

def create_app(vocab_id: str = DEFAULT_VOCAB_ID) -> "FastAPI":
    app = FastAPI(title="GeomTok API", version="1.1.0",
                  description="Geometry-native SVG tokenization (PRD v1.0 + lean L1)")

    manifest = load_bundled_manifest(vocab_id)
    gt = GeomTokenizer(manifest=manifest) if manifest else GeomTokenizer(vocab_id=vocab_id)
    app.state.gt = gt
    app.state.manifest = gt.manifest

    # 진짜 비동기 잡 백엔드 (ThreadPoolExecutor + 교체 가능 store/blob)
    from .jobs import JobManager

    def _process_item(op: str, item: dict):
        """잡 워커가 아이템 1건을 처리 — GeomTokenizer 에 바인딩."""
        if op == "tokenize":
            r = gt.tokenize(item.get("svg") or "", level=item.get("level", "L1"),
                            lean=bool(item.get("lean", False)))
            # lean 에코 — full/lean 혼합 스트림을 응답만으로 구분 가능하게
            return ({"id": item.get("id"), "ok": True, "token_ids": r["token_ids"],
                     "n_tokens": r["n_tokens"], "lean": r["lean"],
                     "compression_ratio": r["compression_ratio"]}, True)
        r = gt.detokenize(item.get("token_ids") or [],
                          tokenizer_version=gt.tokenizer_version,
                          vocab_id=gt.vocab_id, level=item.get("level", "L1"))
        return ({"id": item.get("id"), "ok": True, "svg": r["svg"],
                 "n_tokens": r["n_tokens"]}, True)

    app.state.jobmgr = JobManager(
        _process_item, gt.tokenizer_version, gt.vocab_id)

    # 표준 에러 → PRD JSON {code, message}
    @app.exception_handler(GeomTokError)
    async def _geomtok_err(request: Request, exc: GeomTokError):  # noqa: ANN001
        return JSONResponse(status_code=exc.http_status,
                            content={"error": exc.to_payload()})

    @app.get("/v1/healthz")
    def healthz():
        return {"status": "ok", "tokenizer_version": gt.tokenizer_version,
                "vocab_id": gt.vocab_id, "has_l2": gt.has_l2,
                "manifest_hash": gt.manifest.content_hash()}

    @app.post("/v1/tokenize")
    def tokenize(req: TokenizeReq):
        t0 = time.perf_counter()
        out = gt.tokenize(req.svg, level=req.level, lean=req.lean,
                          config=req.config, return_fields=req.return_)
        out["server_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        return out

    @app.post("/v1/detokenize")
    def detokenize(req: DetokenizeReq):
        t0 = time.perf_counter()
        out = gt.detokenize(req.token_ids,
                            tokenizer_version=req.tokenizer_version,
                            vocab_id=req.vocab_id, level=req.level)
        out["server_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        return out

    def _run_batch(op: str, items, level: str, on_error: str,
                   lean: bool = False):
        """배치 처리 코어 — 동기 배치·비동기 잡 공용."""
        results = []
        n_ok = n_failed = 0
        for it in items:
            try:
                if op == "tokenize":
                    r = gt.tokenize(it.svg or "", level=level, lean=lean)
                    results.append({"id": it.id, "ok": True,
                                    "token_ids": r["token_ids"],
                                    "n_tokens": r["n_tokens"],
                                    "lean": r["lean"],
                                    "compression_ratio": r["compression_ratio"]})
                else:
                    r = gt.detokenize(it.token_ids or [],
                                      tokenizer_version=gt.tokenizer_version,
                                      vocab_id=gt.vocab_id, level=level)
                    results.append({"id": it.id, "ok": True, "svg": r["svg"],
                                    "n_tokens": r["n_tokens"]})
                n_ok += 1
            except GeomTokError as e:
                if on_error == "fail_fast":
                    raise
                results.append({"id": it.id, "ok": False, "error": e.to_payload()})
                n_failed += 1
        return results, n_ok, n_failed

    def _validate_batch(req: BatchReq):
        if req.op not in ("tokenize", "detokenize"):
            raise InvalidRequest(f"unknown op '{req.op}'", op=req.op)
        if len(req.items) > MAX_BATCH_ITEMS:
            raise PayloadTooLarge(f"items exceed {MAX_BATCH_ITEMS}",
                                  limit=MAX_BATCH_ITEMS)
        # 총 페이로드 바이트 가드 (PRD §7.1: ≤32MB) — DoS 방어.
        # tokenize 는 svg 바이트, detokenize 는 token_ids(≈4바이트/정수)로 환산해
        # 두 op 모두 바운드 (detokenize 배치가 가드를 우회하지 않도록).
        total = sum(
            len((it.svg or "").encode("utf-8")) + 4 * len(it.token_ids or [])
            for it in req.items)
        if total > MAX_BATCH_BYTES:
            raise PayloadTooLarge(f"batch payload exceeds {MAX_BATCH_BYTES} bytes",
                                  limit=MAX_BATCH_BYTES)

    @app.post("/v1/batch")
    def batch(req: BatchReq):
        t0 = time.perf_counter()
        _validate_batch(req)
        results, n_ok, n_failed = _run_batch(req.op, req.items, req.level,
                                             req.on_error, lean=req.lean)
        return {"tokenizer_version": gt.tokenizer_version, "vocab_id": gt.vocab_id,
                "op": req.op, "results": results,
                "stats": {"n_ok": n_ok, "n_failed": n_failed,
                          "server_ms": round((time.perf_counter() - t0) * 1000, 2)}}

    @app.post("/v1/eval")
    def eval_protocol(req: EvalReq):
        from ..evaluation.protocol import run_builtin_eval, run_remote_eval
        svgs = [s.svg for s in req.scenes]
        names = [s.name or f"scene_{i}" for i, s in enumerate(req.scenes)]
        spec = req.tokenizer or TokenizerSpec(builtin=True)
        if spec.remote is not None:
            out = run_remote_eval(svgs, spec.remote.tokenize_url,
                                  spec.remote.detokenize_url, names=names,
                                  render_res=req.render_res)
        else:
            out = run_builtin_eval(svgs, names=names, level=req.level,
                                   render_res=req.render_res, lean=req.lean)
        # PRD §7.6: 모든 응답에 버전·vocab 에코
        out["tokenizer_version"] = gt.tokenizer_version
        out["vocab_id"] = gt.vocab_id
        return out

    # ----- 비동기 대규모 잡 (PRD §7.1 / §8) -----
    # 진짜 비동기: 즉시 queued 반환, 워커가 running→succeeded/partial 로 전이.
    # 클라이언트는 GET 으로 running·진행률 증가를 실제 관찰. JobManager 가
    # 교체 가능한 store/blob 으로 영속화·대규모 입출력을 담당.
    @app.post("/v1/batch/jobs", status_code=202)
    def submit_job(req: JobReq):
        if req.op not in ("tokenize", "detokenize"):
            raise InvalidRequest(f"unknown op '{req.op}'", op=req.op)
        if req.items is None and not req.input_uri:
            raise InvalidRequest("provide items or input_uri")
        if req.items is not None and len(req.items) > MAX_BATCH_ITEMS:
            raise PayloadTooLarge(f"items exceed {MAX_BATCH_ITEMS}",
                                  limit=MAX_BATCH_ITEMS)
        # level/lean 은 잡 레벨 기본값으로 JobManager 가 아이템에 병합한다 —
        # NDJSON(input_uri) 라인이 자체 level/lean 키를 가지면 그 값이 이긴다.
        items = ([it.model_dump() for it in req.items]
                 if req.items is not None else None)
        job = app.state.jobmgr.submit(
            req.op, items=items, input_uri=req.input_uri, level=req.level,
            lean=req.lean,
            on_error=req.on_error, webhook_url=req.webhook_url,
            output_uri=req.output_uri)
        # 제출 ack 은 항상 "queued" — 워커가 이미 시작했을 수 있으나(레이스)
        # 제출 계약상 상태는 queued. 실시간 상태는 GET 으로 관찰.
        n_items = len(items) if items is not None else None
        return {"job_id": job.job_id, "status": "queued", "n_items": n_items,
                "tokenizer_version": gt.tokenizer_version, "vocab_id": gt.vocab_id}

    @app.get("/v1/batch/jobs/{job_id}")
    def get_job(job_id: str, include_results: bool = True):
        job = app.state.jobmgr.get(job_id)
        if job is None:
            raise JobNotFound(f"unknown job_id {job_id}", job_id=job_id)
        return job.public(include_results=include_results)

    @app.delete("/v1/batch/jobs/{job_id}")
    def cancel_job(job_id: str):
        if app.state.jobmgr.get(job_id) is None:
            raise JobNotFound(f"unknown job_id {job_id}", job_id=job_id)
        cancelled = app.state.jobmgr.cancel(job_id)
        return {"job_id": job_id, "cancel_requested": cancelled,
                "status": app.state.jobmgr.get(job_id).status}

    # ----- NDJSON 스트리밍 (PRD §7.1) -----
    @app.post("/v1/stream")
    async def stream(request: Request):
        """요청 본문을 NDJSON 라인 단위로 읽어 라인당 1결과를 스트리밍.
        전체 버퍼링 없이 순서 보존 (백프레셔 인지). 각 라인: {svg|token_ids, op?}."""
        import json as _json
        from fastapi.responses import StreamingResponse

        async def gen():
            buf = b""
            async for chunk in request.stream():
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    yield _process_stream_line(line) + b"\n"
            if buf.strip():
                yield _process_stream_line(buf.strip()) + b"\n"

        def _process_stream_line(raw: bytes) -> bytes:
            try:
                obj = _json.loads(raw)
                op = obj.get("op", "tokenize")
                if op == "tokenize":
                    r = gt.tokenize(obj.get("svg", ""), level=obj.get("level", "L1"),
                                    lean=bool(obj.get("lean", False)))
                    out = {"id": obj.get("id"), "ok": True,
                           "token_ids": r["token_ids"], "n_tokens": r["n_tokens"],
                           "lean": r["lean"]}
                else:
                    r = gt.detokenize(obj.get("token_ids", []),
                                      tokenizer_version=gt.tokenizer_version,
                                      vocab_id=gt.vocab_id,
                                      level=obj.get("level", "L1"))
                    out = {"id": obj.get("id"), "ok": True, "svg": r["svg"]}
            except GeomTokError as e:
                out = {"id": None, "ok": False, "error": e.to_payload()}
            except Exception as e:  # noqa: BLE001
                out = {"id": None, "ok": False,
                       "error": {"code": "STREAM_LINE_ERROR", "message": str(e)}}
            return _json.dumps(out).encode("utf-8")

        return StreamingResponse(gen(), media_type="application/x-ndjson")

    @app.get("/v1/vocab/{vocab_id}")
    def get_vocab(vocab_id: str):
        m = load_bundled_manifest(vocab_id)
        if m is None and vocab_id == gt.vocab_id:
            m = gt.manifest
        if m is None:
            return JSONResponse(status_code=404, content={
                "error": {"code": "VOCAB_NOT_FOUND",
                          "message": f"unknown vocab_id {vocab_id}"}})
        d = m.to_dict()
        d["content_hash"] = m.content_hash()
        return d

    return app


# uvicorn 진입점: `uvicorn geomtok.server.app:app`
app = create_app()


def _cli():
    """`geomtok-serve` 콘솔 진입점."""
    import argparse
    import uvicorn
    ap = argparse.ArgumentParser(description="GeomTok managed API server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    uvicorn.run("geomtok.server.app:app", host=args.host, port=args.port)


if __name__ == "__main__":
    _cli()
