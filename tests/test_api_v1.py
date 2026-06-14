"""
GeomTok 파사드 (api.py) v1.0 테스트
====================================
단건 tokenize/detokenize, 결정성, 표준 에러 코드, 미지원 거부, 페이로드 한도,
L1/L2 동등 SVG, 좌표 충실도 게이트(fixture)를 검증.
"""

import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import geomtok
from geomtok.api import GeomTokenizer, MAX_SVG_BYTES
from geomtok.errors import (
    VersionRequired, VocabMismatch, ParseUnsupportedElement, PayloadTooLarge,
)

passed = 0
failed = 0
_FIX = os.path.join(os.path.dirname(__file__), "fixtures", "icons")
SVG = "<svg viewBox='0 0 24 24'><path d='M4 4 L20 4 L20 20 Z'/><circle cx='12' cy='12' r='6'/></svg>"


def check(cond, msg):
    global passed, failed
    if cond:
        print(f"  [PASS] {msg}"); passed += 1
    else:
        print(f"  [FAIL] {msg}"); failed += 1


def expect_error(fn, exc, label):
    try:
        fn(); check(False, f"{label}: no error raised")
    except exc:
        check(True, f"{label} raised")
    except Exception as e:  # noqa: BLE001
        check(False, f"{label}: wrong error {type(e).__name__}")


def main():
    gt = GeomTokenizer.default()

    out = gt.tokenize(SVG)
    check(out["tokenizer_version"] == "geomtok-1.0.0", "version echoed")
    check(out["vocab_id"] == "geom-5561-v1", "vocab_id echoed")
    check(out["token_ids"][0] == 1 and out["token_ids"][-1] == 2, "BOS/EOS bracket")
    check(out["n_elements"] == 2, "2 elements via SEP")
    check(out["normalization"]["viewBox_applied"], "viewBox normalization flagged")

    # 결정성
    check(gt.tokenize(SVG)["token_ids"] == out["token_ids"], "tokenize deterministic")

    # 디토크나이즈
    back = gt.detokenize(out["token_ids"], tokenizer_version="geomtok-1.0.0",
                         vocab_id="geom-5561-v1")
    check(back["valid"] and "<svg" in back["svg"], "detokenize valid SVG")
    check(back["fidelity"]["coord_max_px"] <= 3.32, "coord max within bound")

    # 에러 코드
    expect_error(lambda: gt.detokenize(out["token_ids"]), VersionRequired, "VERSION_REQUIRED")
    expect_error(lambda: gt.detokenize(out["token_ids"], tokenizer_version="geomtok-1.0.0",
                                       vocab_id="x"), VocabMismatch, "VOCAB_MISMATCH")
    expect_error(lambda: gt.tokenize("<svg><filter id='b'/><path d='M0 0'/></svg>"),
                 ParseUnsupportedElement, "PARSE_UNSUPPORTED_ELEMENT")
    expect_error(lambda: gt.tokenize("<svg>" + "x" * (MAX_SVG_BYTES + 10) + "</svg>"),
                 PayloadTooLarge, "PAYLOAD_TOO_LARGE")

    # L1/L2 같은 SVG 로 디코드
    o1 = gt.tokenize(SVG, level="L1")
    o2 = gt.tokenize(SVG, level="L2")
    s1 = gt.detokenize(o1["token_ids"], tokenizer_version="geomtok-1.0.0", level="L1")["svg"]
    s2 = gt.detokenize(o2["token_ids"], tokenizer_version="geomtok-1.0.0", level="L2")["svg"]
    check(s1 == s2 and o2["n_tokens"] <= o1["n_tokens"], "L2 decodes to same SVG, fewer tokens")

    # 모듈 함수형 인터페이스
    mo = geomtok.tokenize(SVG)
    check(geomtok.detokenize(mo["token_ids"])["valid"], "module-level tokenize/detokenize")

    # fixture 좌표 게이트
    files = sorted(glob.glob(os.path.join(_FIX, "*.svg")))
    within = 0
    for f in files:
        pts = gt.encoded_points(open(f).read())
        if pts and gt.arcs.roundtrip_fidelity(pts)["within_bound"]:
            within += 1
    check(within == len(files), f"fixture coords within bound: {within}/{len(files)}")

    # 미지원 거부가 parse 100% 를 깨지 않음 (지원 요소만 100%)
    parse_ok = sum(1 for f in files if gt.tokenize(open(f).read())["n_tokens"] > 2)
    check(parse_ok == len(files), f"fixture parse+tokenize: {parse_ok}/{len(files)}")

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
