"""
매니지드 API 서버 테스트 (PRD §7.1)
=====================================
모든 /v1 엔드포인트와 표준 에러 코드를 TestClient 로 검증.
fastapi/httpx 부재 시 스킵(통과)한다 — OSS 코어는 서버 의존이 없다.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

passed = 0
failed = 0


def check(cond, msg):
    global passed, failed
    if cond:
        print(f"  [PASS] {msg}"); passed += 1
    else:
        print(f"  [FAIL] {msg}"); failed += 1


def main():
    try:
        from fastapi.testclient import TestClient
        from geomtok.server.app import create_app
    except Exception as e:  # noqa: BLE001
        print(f"  [SKIP] server deps unavailable ({type(e).__name__}); OSS core needs none")
        sys.exit(0)

    c = TestClient(create_app())
    SVG = "<svg viewBox='0 0 24 24'><path d='M4 4 L20 4 L20 20 Z'/><circle cx='12' cy='12' r='6'/></svg>"

    h = c.get("/v1/healthz").json()
    check(h["status"] == "ok" and h["vocab_id"] == "geom-5561-v1", "healthz ok")

    j = c.post("/v1/tokenize", json={"svg": SVG, "level": "L1"}).json()
    check(j["n_tokens"] > 2 and j["tokenizer_version"] == "geomtok-1.0.0", "tokenize")
    ids = j["token_ids"]

    j2 = c.post("/v1/tokenize", json={"svg": SVG, "level": "L2"}).json()
    check(j2["n_tokens"] <= j["n_tokens"], "L2 tokenize fewer tokens")

    r0 = c.post("/v1/detokenize", json={"token_ids": ids})
    check(r0.status_code == 400 and r0.json()["error"]["code"] == "VERSION_REQUIRED",
          "detokenize without version -> VERSION_REQUIRED")

    rd = c.post("/v1/detokenize", json={"token_ids": ids,
                "tokenizer_version": "geomtok-1.0.0", "vocab_id": "geom-5561-v1"}).json()
    check(rd["valid"] and "<svg" in rd["svg"], "detokenize valid")

    rb = c.post("/v1/batch", json={"op": "tokenize", "on_error": "skip", "items": [
        {"id": "ic_001", "svg": SVG},
        {"id": "ic_002", "svg": "<svg><filter id='b'/><path d='M0 0'/></svg>"}]}).json()
    check(rb["stats"]["n_ok"] == 1 and rb["stats"]["n_failed"] == 1, "batch partial success")
    check(rb["results"][1]["error"]["code"] == "PARSE_UNSUPPORTED_ELEMENT",
          "batch surfaces unsupported error")

    re_ = c.post("/v1/eval", json={"scenes": [{"name": "a", "svg": SVG}],
                 "tokenizer": {"builtin": True}, "render_res": 128}).json()
    check(re_["protocol"] == "GeomTok-Eval/1.0", "eval protocol id")
    check("render_ssim_mean" in re_["summary"], "eval summary has ssim")

    v = c.get("/v1/vocab/geom-5561-v1").json()
    check(v["vocab_size"] == 5561 and v.get("merges"), "vocab manifest with merges")
    check(c.get("/v1/vocab/nope").status_code == 404, "unknown vocab 404")

    big = "<svg>" + "x" * 270000 + "</svg>"
    check(c.post("/v1/tokenize", json={"svg": big}).status_code == 413, "payload too large 413")

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
