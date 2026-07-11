"""
Lean L1 (보조 마커 미방출) 모드 테스트 — v1.1
==============================================
PAPER §5.1 ablation 의 제품화 검증:
  (1) lean 스트림엔 연속성[30,34)·곡률[40,56) 토큰이 없다
  (2) lean 스트림도 FSA 문법 유효 (마커는 문법상 옵션)
  (3) detokenize(full) == detokenize(lean) 바이트 동일 — 마커는 파생 가능
  (4) 기본값(full)은 v1.0 과 동일: 곡선 아이콘엔 마커가 존재
  (5) fixture 코퍼스에서 lean 이 시퀀스를 유의미하게 단축
  (6) 모듈 함수·서버 라우트 passthrough
"""

import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import geomtok
from geomtok.api import GeomTokenizer
from geomtok.tokenizer.grammar import GrammarFSA

passed = 0
failed = 0
_FIX = os.path.join(os.path.dirname(__file__), "fixtures", "icons")

_CONT = set(range(30, 34))
_CURV = set(range(40, 56))
_MARKERS = _CONT | _CURV


def check(cond, msg):
    global passed, failed
    if cond:
        print(f"  [PASS] {msg}"); passed += 1
    else:
        print(f"  [FAIL] {msg}"); failed += 1


def main():
    gt = GeomTokenizer.default()
    fsa = GrammarFSA(gt.vocab)
    files = sorted(glob.glob(os.path.join(_FIX, "*.svg")))
    svgs = [open(f).read() for f in files]
    check(len(svgs) >= 20, f"fixture corpus loaded ({len(svgs)} icons)")

    total_full = total_lean = 0
    lean_has_marker = False
    lean_all_valid = True
    recon_all_identical = True
    any_full_marker = False

    for svg in svgs:
        full = gt.tokenize(svg)
        lean = gt.tokenize(svg, lean=True)
        total_full += full["n_tokens"]
        total_lean += lean["n_tokens"]

        if any(t in _MARKERS for t in lean["token_ids"]):
            lean_has_marker = True
        if any(t in _MARKERS for t in full["token_ids"]):
            any_full_marker = True
        if not fsa.is_valid(lean["token_ids"]):
            lean_all_valid = False

        svg_full = gt.detokenize(full["token_ids"],
                                 tokenizer_version=gt.tokenizer_version,
                                 vocab_id=gt.vocab_id)["svg"]
        svg_lean = gt.detokenize(lean["token_ids"],
                                 tokenizer_version=gt.tokenizer_version,
                                 vocab_id=gt.vocab_id)["svg"]
        if svg_full != svg_lean:
            recon_all_identical = False

    check(not lean_has_marker, "lean streams contain no marker tokens")
    check(lean_all_valid, "all lean streams are FSA-valid")
    check(recon_all_identical,
          "detokenize(full) == detokenize(lean) byte-identical on all fixtures")
    check(any_full_marker,
          "default (full) streams still carry markers — v1.0 behavior preserved")
    check(total_lean < total_full,
          f"lean shortens sequences ({total_full} -> {total_lean} tokens)")
    reduction = 1.0 - total_lean / total_full
    check(reduction > 0.10,
          f"lean saves >10% tokens on fixtures (actual {reduction:.1%})")

    # 응답 계약: lean 필드 에코, return_fields 화이트리스트에도 생존
    out = gt.tokenize(svgs[0], lean=True, return_fields=["token_ids"])
    check(out.get("lean") is True, "payload echoes lean=True")
    check(gt.tokenize(svgs[0]).get("lean") is False, "payload echoes lean=False by default")

    # 결정성: 동일 입력 → 동일 lean 스트림
    check(gt.tokenize(svgs[0], lean=True)["token_ids"]
          == gt.tokenize(svgs[0], lean=True)["token_ids"],
          "lean tokenization is deterministic")

    # 모듈 함수 passthrough
    m = geomtok.tokenize(svgs[0], lean=True)
    check(m["lean"] is True and not any(t in _MARKERS for t in m["token_ids"]),
          "geomtok.tokenize(lean=True) passthrough")

    # 서버 라우트 passthrough (fastapi 설치 시)
    try:
        from fastapi.testclient import TestClient
        from geomtok.server.app import create_app
        client = TestClient(create_app())
        r_full = client.post("/v1/tokenize", json={"svg": svgs[0]}).json()
        r_lean = client.post("/v1/tokenize",
                             json={"svg": svgs[0], "lean": True}).json()
        check(r_lean["n_tokens"] <= r_full["n_tokens"]
              and not any(t in _MARKERS for t in r_lean["token_ids"]),
              "/v1/tokenize honors lean")
        rb = client.post("/v1/batch", json={
            "op": "tokenize", "lean": True,
            "items": [{"id": "a", "svg": svgs[0]}]}).json()
        check(not any(t in _MARKERS for t in rb["results"][0]["token_ids"]),
              "/v1/batch honors lean")
        check(rb["results"][0].get("lean") is True,
              "/v1/batch result rows echo lean")

        # 비동기 잡: 잡 레벨 lean/level 이 아이템 기본값으로 병합된다 (v1.1 수정
        # — v1.0 은 잡 레벨 level 을 죽은 파라미터로 무시했다)
        import time as _t

        def _run_job(payload):
            job = client.post("/v1/batch/jobs", json=payload).json()
            for _ in range(300):
                got = client.get(f"/v1/batch/jobs/{job['job_id']}").json()
                if got["status"] in ("succeeded", "partial", "failed"):
                    return got
                _t.sleep(0.01)
            return got

        jl = _run_job({"op": "tokenize", "lean": True,
                       "items": [{"id": "a", "svg": svgs[0]}]})
        check(jl["status"] == "succeeded"
              and not any(t in _MARKERS for t in jl["results"][0]["token_ids"])
              and jl["results"][0].get("lean") is True,
              "/v1/batch/jobs honors job-level lean (inline items)")

        want_l2 = gt.tokenize(svgs[0], level="L2")["token_ids"]
        j2 = _run_job({"op": "tokenize", "level": "L2",
                       "items": [{"id": "a", "svg": svgs[0]}]})
        check(j2["results"][0]["token_ids"] == want_l2,
              "/v1/batch/jobs honors job-level level (was dead in v1.0)")

        # NDJSON(input_uri) 잡: 잡 레벨 lean 이 라인 기본값으로 병합되고,
        # 라인 자체 lean 키가 있으면 그 값이 이긴다 (v1.1 수정 — v1.0 은
        # input_uri 경로에서 잡 레벨 설정이 통째로 무시됐다)
        import json as _json
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".ndjson",
                                         delete=False) as tf:
            tf.write(_json.dumps({"id": "d", "svg": svgs[0]}) + "\n")
            tf.write(_json.dumps({"id": "o", "svg": svgs[0],
                                  "lean": False}) + "\n")
            ndjson_uri = "file://" + tf.name
        jn = _run_job({"op": "tokenize", "lean": True,
                       "input_uri": ndjson_uri})
        rows = {r["id"]: r for r in jn["results"]}
        check(jn["status"] == "succeeded"
              and rows["d"]["lean"] is True
              and not any(t in _MARKERS for t in rows["d"]["token_ids"]),
              "/v1/batch/jobs input_uri: job-level lean reaches NDJSON lines")
        check(rows["o"]["lean"] is False
              and any(t in _MARKERS for t in rows["o"]["token_ids"]),
              "/v1/batch/jobs input_uri: per-line lean overrides job default")
        os.unlink(ndjson_uri[len("file://"):])

        # /v1/eval lean=true — lean 스트림 평가 (충실도 동일, 토큰만 절감)
        ev_full = client.post("/v1/eval", json={
            "scenes": [{"name": "a", "svg": svgs[0]}], "render_res": 64}).json()
        ev_lean = client.post("/v1/eval", json={
            "scenes": [{"name": "a", "svg": svgs[0]}], "lean": True,
            "render_res": 64}).json()
        check(ev_lean["summary"]["tokens"] <= ev_full["summary"]["tokens"]
              and ev_lean["summary"]["attr_max_err"]
              == ev_full["summary"]["attr_max_err"],
              "/v1/eval lean: fewer tokens, identical fidelity")
    except ImportError:
        print("  [SKIP] server routes (fastapi not installed)")

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
