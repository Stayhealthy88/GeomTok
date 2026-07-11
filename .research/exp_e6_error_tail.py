"""실험 E6-a (v1.1): 좌표 오차 꼬리 + lean L1 코퍼스 실측.

PAPER §10 E6 잔여 항목 "coordinate-error tail (>2px perceptible rate)" 과
lean L1 제품화(§5.1 ablation 의 shipped 구현)의 정직한 숫자를 만든다.

측정 대상: 번들 검증 코퍼스 corpus/icons (2,726 실세계 단색 path 아이콘,
Tabler/Feather/Heroicons/Lucide — ATTRIBUTION.md). HF 다운로드·외부 경로
없이 저장소만으로 재현된다: `python .research/exp_e6_error_tail.py`.

지표:
  - 좌표 오차: gt.normalized_svg(원본) ↔ detokenize(tokenize(원본)) 를
    protocol._attr_errors 로 점열 비교 (GeomTok-Eval/1.1 과 동일 경로)
  - 꼬리: p50/p90/p95/p99/max + 인지 가능(>2px) 비율
  - lean: tokens/icon 절감률, 마커 점유율, detok 바이트 동일률, FSA 유효율
"""
import glob
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import numpy as np

from geomtok.api import GeomTokenizer
from geomtok.evaluation.protocol import _attr_errors
from geomtok.tokenizer.grammar import GrammarFSA

_CONT = set(range(30, 34))
_CURV = set(range(40, 56))
_MARKERS = _CONT | _CURV
PERCEPTIBLE_PX = 2.0
# uniform-L6 셀 반대각(≈3.315px)에 검출·출력 반올림 여유를 더한 운용 한계.
# PAPER §5.1 이 보고한 실측 max 3.37px 과 일치 — 이를 넘는 오차는 양자화가
# 아니라 다른 원인(예: viewBox 밖 클램핑)이어야 한다.
QUANT_BOUND_PX = 3.40


def main():
    gt = GeomTokenizer.default()
    fsa = GrammarFSA(gt.vocab)
    files = sorted(glob.glob(os.path.join(_ROOT, "corpus", "icons", "*.svg")))
    print(f"corpus: {len(files)} icons (corpus/icons, real-world monochrome)")

    errs_all = []
    n_parse_fail = 0
    n_full = n_lean = n_marker_tokens = 0
    n_icons = 0
    n_byte_identical = 0
    n_lean_fsa_valid = 0
    n_count_match = 0
    # 한계 초과 꼬리 귀속: 양자화 한계를 넘는 좌표가 clamp 경고 아이콘에서
    # 나오는지 추적 — "bounded-error" 주장의 반례인지 클램핑 산물인지 판별.
    n_over_bound = n_over_bound_clamped = 0
    worst = []

    for f in files:
        svg = open(f).read()
        try:
            full = gt.tokenize(svg)
            lean = gt.tokenize(svg, lean=True)
            recon_full = gt.detokenize(
                full["token_ids"], tokenizer_version=gt.tokenizer_version,
                vocab_id=gt.vocab_id)["svg"]
            recon_lean = gt.detokenize(
                lean["token_ids"], tokenizer_version=gt.tokenizer_version,
                vocab_id=gt.vocab_id)["svg"]
            norm = gt.normalized_svg(svg)
        except Exception as e:  # noqa: BLE001 — 아이콘 격리, 실패는 집계
            n_parse_fail += 1
            print(f"  [parse-fail] {os.path.basename(f)}: {e}")
            continue

        n_icons += 1
        n_full += full["n_tokens"]
        n_lean += lean["n_tokens"]
        n_marker_tokens += sum(1 for t in full["token_ids"] if t in _MARKERS)
        n_byte_identical += int(recon_full == recon_lean)
        n_lean_fsa_valid += int(fsa.is_valid(lean["token_ids"]))

        errs, count_match = _attr_errors(norm, recon_full)
        errs_all.extend(errs)
        n_count_match += int(count_match)

        n_over = sum(1 for e in errs if e > QUANT_BOUND_PX)
        if n_over:
            clamped = any("clamped" in w for w in full["warnings"])
            n_over_bound += n_over
            n_over_bound_clamped += n_over if clamped else 0
            worst.append((max(errs), os.path.basename(f), clamped))

    arr = np.asarray(errs_all, dtype=np.float64)
    print(f"\n== corpus round-trip ({n_icons} icons, {n_parse_fail} parse-fail) ==")
    print(f"coords measured      : {arr.size}")
    print(f"count-match rate     : {n_count_match / n_icons:.4f}")
    print(f"coord err mean       : {arr.mean():.4f} px")
    print(f"coord err p50        : {np.percentile(arr, 50):.4f} px")
    print(f"coord err p90        : {np.percentile(arr, 90):.4f} px")
    print(f"coord err p95        : {np.percentile(arr, 95):.4f} px")
    print(f"coord err p99        : {np.percentile(arr, 99):.4f} px")
    print(f"coord err max        : {arr.max():.4f} px")
    print(f"perceptible (>{PERCEPTIBLE_PX:.0f}px)  : "
          f"{(arr > PERCEPTIBLE_PX).mean():.4f}")

    print(f"\n== beyond-quantization-bound tail (> {QUANT_BOUND_PX}px) ==")
    print(f"coords over bound    : {n_over_bound} / {arr.size} "
          f"({n_over_bound / arr.size:.6f})")
    print(f"  in clamp-warned icons: {n_over_bound_clamped} "
          f"(clamp warning = tokenizer's own honest signal)")
    for mx, name, cl in sorted(worst, reverse=True)[:5]:
        print(f"  worst {mx:7.2f}px  {name}  clamp_warning={cl}")

    print(f"\n== lean L1 vs full L1 ==")
    print(f"tokens/icon full     : {n_full / n_icons:.1f}")
    print(f"tokens/icon lean     : {n_lean / n_icons:.1f}")
    print(f"lean reduction       : {1 - n_lean / n_full:.4f}")
    print(f"marker share (full)  : {n_marker_tokens / n_full:.4f}")
    print(f"detok byte-identical : {n_byte_identical}/{n_icons}")
    print(f"lean FSA-valid       : {n_lean_fsa_valid}/{n_icons}")

    ok = (n_byte_identical == n_icons and n_lean_fsa_valid == n_icons
          and n_parse_fail == 0)
    print(f"\nverdict: {'PASS' if ok else 'CHECK'} — lean is a free win iff "
          f"byte-identical & FSA-valid are 100%")


if __name__ == "__main__":
    main()
