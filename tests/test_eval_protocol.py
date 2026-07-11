"""
GeomTok-Eval / 1.0 프로토콜 테스트 (PRD G4)
============================================
SSIM 자기동일성, 게임-내성(랜덤 토크나이저 탄로), summary 키, builtin 러너 확인.
렌더러(cairosvg) 부재 시 SSIM 항목은 건너뛴다.
"""

import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from geomtok.api import GeomTokenizer
from geomtok.evaluation.protocol import (
    GeomTokEvalProtocol, ssim, _render_gray, run_builtin_eval,
    renderer_available,
)

passed = 0
failed = 0
_FIX = os.path.join(os.path.dirname(__file__), "fixtures", "icons")


def check(cond, msg):
    global passed, failed
    if cond:
        print(f"  [PASS] {msg}"); passed += 1
    else:
        print(f"  [FAIL] {msg}"); failed += 1


def main():
    # SSIM 기본 성질
    a = np.zeros((32, 32)); a[8:24, 8:24] = 1.0
    check(abs(ssim(a, a) - 1.0) < 1e-6, "ssim(x,x) == 1")
    b = 1.0 - a
    check(ssim(a, b) < 0.5, "ssim of inverted image is low")

    gt = GeomTokenizer.default()
    files = sorted(glob.glob(os.path.join(_FIX, "*.svg")))[:20]
    svgs = [open(f).read() for f in files]

    summary = run_builtin_eval(svgs, render_res=128)["summary"]
    for k in ("scenes", "attr_mean_err", "attr_max_err", "count_acc",
              "render_ssim_mean", "render_scenes", "tokens", "tokens_bpe",
              "compression_vs_baseline", "bits_per_icon"):
        check(k in summary, f"summary has '{k}'")
    check(summary["scenes"] == len(svgs), "all scenes evaluated")
    check(summary["count_acc"] == 1.0, "count accuracy 1.0 on path icons")
    check(summary["attr_max_err"] <= 4.0, "attr max err within bound")
    check(summary["compression_vs_baseline"] > 1.5, "compression beats cl100k")

    # 좌표 오차 꼬리 (v1.1 / E6): 분위수 + 인지 가능(>2px) 비율
    for k in ("coords_measured", "coord_err_p50", "coord_err_p95",
              "coord_err_p99", "perceptible_px", "perceptible_err_rate"):
        check(k in summary, f"summary has tail key '{k}'")
    check(summary["coord_err_p50"] <= summary["coord_err_p95"]
          <= summary["coord_err_p99"] <= summary["attr_max_err"] + 1e-9,
          "tail percentiles are monotone and bounded by max")
    check(0.0 <= summary["perceptible_err_rate"] <= 1.0,
          "perceptible rate in [0,1]")

    # 렌더 가능하면 SSIM 합리적; 렌더러 부재 시 None (0.0 오염 금지)
    if renderer_available():
        check(summary["render_ssim_mean"] > 0.7, "render ssim reasonable")
        check(summary["render_scenes"] == summary["scenes"],
              "all scenes rendered when renderer available")
    else:
        check(summary["render_ssim_mean"] is None,
              "render ssim is None (not 0.0) without a renderer")
        check(summary["render_scenes"] == 0, "no scenes rendered without renderer")

    # 게임-내성: 무작위(상수) 토크나이저는 충실도에서 탄로.
    # SSIM 축은 렌더러가 있어야 측정 가능; 없으면 값-충실도 축으로 검증한다.
    proto = GeomTokEvalProtocol(
        tokenize_fn=lambda s: [1, 2],                 # 빈 스트림
        detokenize_fn=lambda ids: "<svg viewBox='0 0 300 300'></svg>",
        gt_norm_fn=lambda s: gt.normalized_svg(s), render_res=128)
    for i, s in enumerate(svgs[:8]):
        proto.eval_scene(f"s{i}", s)
    bad = proto.summary()["summary"]
    if renderer_available():
        check(bad["render_ssim_mean"] < summary["render_ssim_mean"],
              "degenerate tokenizer scores worse SSIM (game-resistant)")
    else:
        print("  [SKIP] SSIM game-resistance (no renderer installed)")
    check(bad["attr_mean_err"] > summary["attr_mean_err"] + 10.0,
          "degenerate tokenizer exposed by value fidelity (game-resistant)")

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
