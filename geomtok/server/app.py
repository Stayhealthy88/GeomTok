"""
GeomTok 매니지드 API (FastAPI) — PRD §7.1
==========================================
OSS 코어(`geomtok.api.GeomTokenizer`)와 **동일 코드·동일 불변 vocab** 을 호스팅
위에 얹는다 (PRD §9: 로컬과 호스팅 결과가 비트-동일). 매니지드는 그 위에 인증
스텁·레이트리밋·배치 코얼레싱·표준 에러 직렬화만 추가한다.

엔드포인트:
    POST /v1/tokenize        단건 토큰화
    POST /v1/detokenize      단건 디토큰화 (tokenizer_version 필수)
    POST /v1/eval            GeomTok-Eval 렌더 기반 프로토콜 (builtin/remote)
    POST /v1/batch           동기 배치 (items ≤1000, on_error∈{skip,fail_fast})
    GET  /v1/vocab/{id}      불변 vocab 매니페스트 (오프라인 백킹)
    GET  /v1/healthz         헬스체크
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
from ..errors import GeomTokError, PayloadTooLarge
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
    on_error: str = "skip"                # skip | fail_fast


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
    render_res: int = 256


# --------------------------------------------------------------------------- #
# 앱 팩토리
# --------------------------------------------------------------------------- #

def create_app(vocab_id: str = DEFAULT_VOCAB_ID) -> "FastAPI":
    app = FastAPI(title="GeomTok API", version="1.0.0",
                  description="Geometry-native SVG tokenization (PRD v1.0)")

    manifest = load_bundled_manifest(vocab_id)
    gt = GeomTokenizer(manifest=manifest) if manifest else GeomTokenizer(vocab_id=vocab_id)
    app.state.gt = gt
    app.state.manifest = gt.manifest

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
        out = gt.tokenize(req.svg, level=req.level, config=req.config,
                          return_fields=req.return_)
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

    @app.post("/v1/batch")
    def batch(req: BatchReq):
        t0 = time.perf_counter()
        # 페이로드 크기 가드 (PRD §7.1: ≤32MB, items ≤1000)
        if len(req.items) > MAX_BATCH_ITEMS:
            raise PayloadTooLarge(f"items exceed {MAX_BATCH_ITEMS}",
                                  limit=MAX_BATCH_ITEMS)
        results = []
        n_ok = n_failed = 0
        for it in req.items:
            try:
                if req.op == "tokenize":
                    r = gt.tokenize(it.svg or "", level=req.level)
                    results.append({"id": it.id, "ok": True,
                                    "token_ids": r["token_ids"],
                                    "n_tokens": r["n_tokens"],
                                    "compression_ratio": r["compression_ratio"]})
                else:
                    r = gt.detokenize(it.token_ids or [],
                                      tokenizer_version=gt.tokenizer_version,
                                      vocab_id=gt.vocab_id, level=req.level)
                    results.append({"id": it.id, "ok": True, "svg": r["svg"],
                                    "n_tokens": r["n_tokens"]})
                n_ok += 1
            except GeomTokError as e:
                if req.on_error == "fail_fast":
                    raise
                results.append({"id": it.id, "ok": False, "error": e.to_payload()})
                n_failed += 1
        return {"op": req.op, "results": results,
                "stats": {"n_ok": n_ok, "n_failed": n_failed,
                          "server_ms": round((time.perf_counter() - t0) * 1000, 2)}}

    @app.post("/v1/eval")
    def eval_protocol(req: EvalReq):
        from ..evaluation.protocol import run_builtin_eval, run_remote_eval
        svgs = [s.svg for s in req.scenes]
        names = [s.name or f"scene_{i}" for i, s in enumerate(req.scenes)]
        spec = req.tokenizer or TokenizerSpec(builtin=True)
        if spec.remote is not None:
            return run_remote_eval(svgs, spec.remote.tokenize_url,
                                   spec.remote.detokenize_url, names=names,
                                   render_res=req.render_res)
        return run_builtin_eval(svgs, names=names, level=req.level,
                                render_res=req.render_res)

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
