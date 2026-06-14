#!/usr/bin/env python3
"""
코퍼스 라운드트립 검증 게이트 (PRD §10)
========================================
실세계 아이콘 코퍼스 전체를 GeomTok 파이프라인에 통과시켜 PRD 게이트를 측정한다:

  G1 파싱+라운드트립 100% (지원 요소), 좌표 max ≤ 이론 상한(3.32px), 평균 ~1.79px
  G2 결정성: 동일 입력 → 비트-동일; 매니페스트 리로드 시 해시 동일
  G3 FSA 유효성: 모든 토큰 스트림 well-formed; 미지원 요소 명시 거부
  L2 무손실: decode(encode(L1)) == L1, 토큰 감소율 측정
  SSIM: 렌더 기반 재구성 품질 (샘플)

산출: corpus/validation_report.json + 콘솔 요약.
"""

import argparse
import glob
import json
import os
import random
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from geomtok.api import GeomTokenizer
from geomtok.tokenizer.grammar import GrammarFSA
from geomtok.tokenizer.manifest import load_bundled_manifest
from geomtok.errors import ParseUnsupportedElement, GeomTokError


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus/icons")
    ap.add_argument("--ssim-sample", type=int, default=250)
    ap.add_argument("--out", default="corpus/validation_report.json")
    args = ap.parse_args()

    gt = GeomTokenizer.default()
    fsa = GrammarFSA(gt.vocab)
    files = sorted(glob.glob(os.path.join(args.corpus, "*.svg")))
    print(f"corpus: {len(files)} icons | vocab_id={gt.vocab_id} "
          f"ver={gt.tokenizer_version} hash={gt.manifest.content_hash()}")

    n = 0
    parse_ok = rt_ok = fsa_ok = within_bound = l2_lossless = 0
    coord_means, coord_maxes = [], []
    l1_lens, l2_lens = [], []
    determinism_ok = True
    failures = []

    t0 = time.time()
    for f in files:
        svg = open(f, encoding="utf-8").read()
        try:
            o1 = gt.tokenize(svg, level="L1")
        except GeomTokError as e:
            failures.append((os.path.basename(f), e.code))
            continue
        n += 1
        parse_ok += 1
        l1_ids = o1["token_ids"]

        # 결정성 (동일 입력 2회) — 앞 300개 표본
        if n <= 300 and gt.tokenize(svg, level="L1")["token_ids"] != l1_ids:
            determinism_ok = False

        # FSA 유효성
        if fsa.validate(l1_ids).valid:
            fsa_ok += 1

        # 라운드트립 + 좌표 충실도 (encoded_points 1회 파싱)
        pts = gt.encoded_points(svg)
        if pts:
            fid = gt.arcs.roundtrip_fidelity(pts)
            coord_means.append(fid["mean_error"])
            coord_maxes.append(fid["max_error"])
            if fid["within_bound"]:
                within_bound += 1
        back = gt.detokenize(l1_ids, tokenizer_version=gt.tokenizer_version,
                             vocab_id=gt.vocab_id, measure=False)
        if back["valid"] and back["svg"]:
            rt_ok += 1
        l1_lens.append(o1["n_tokens"])

        # L2 무손실 + 길이 (재파싱 없이 머지 코덱 적용)
        l2_ids = gt._merge_codec.encode(l1_ids)
        if gt._merge_codec.decode(l2_ids) == l1_ids:
            l2_lossless += 1
        l2_lens.append(len(l2_ids))

    # 미지원 요소 명시 거부
    unsupported_cases = [
        ("filter", "<svg><filter id='b'/><path d='M0 0 L1 1'/></svg>"),
        ("gradient", "<svg><linearGradient id='g'/><path d='M0 0 L1 1'/></svg>"),
        ("image", "<svg><image href='x.png'/></svg>"),
        ("text", "<svg><text x='0' y='0'>hi</text></svg>"),
    ]
    rejected = 0
    for _, s in unsupported_cases:
        try:
            gt.tokenize(s)
        except ParseUnsupportedElement:
            rejected += 1
        except GeomTokError:
            pass

    # 매니페스트 리로드 결정성 (해시 동일)
    reloaded = load_bundled_manifest(gt.vocab_id)
    manifest_stable = (reloaded is not None
                       and reloaded.content_hash() == gt.manifest.content_hash())

    # SSIM (샘플)
    print(f"measuring SSIM on {args.ssim_sample} sample ...")
    from geomtok.evaluation.protocol import GeomTokEvalProtocol
    rnd = random.Random(7)
    sample = files[:]
    rnd.shuffle(sample)
    proto = GeomTokEvalProtocol(
        tokenize_fn=lambda s: gt.tokenize(s)["token_ids"],
        detokenize_fn=lambda ids: gt.detokenize(
            ids, tokenizer_version=gt.tokenizer_version, vocab_id=gt.vocab_id)["svg"],
        gt_norm_fn=lambda s: gt.normalized_svg(s))
    for f in sample[: args.ssim_sample]:
        try:
            proto.eval_scene(os.path.basename(f), open(f, encoding="utf-8").read())
        except Exception:
            continue
    esum = proto.summary()["summary"]

    report = {
        "corpus_size": len(files),
        "tokenizer_version": gt.tokenizer_version,
        "vocab_id": gt.vocab_id,
        "manifest_hash": gt.manifest.content_hash(),
        "gates": {
            "parse_rate": round(parse_ok / len(files), 4),
            "roundtrip_rate": round(rt_ok / n, 4) if n else 0,
            "fsa_valid_rate": round(fsa_ok / n, 4) if n else 0,
            "coord_within_bound_rate": round(within_bound / max(len(coord_maxes), 1), 4),
            "coord_mean_px": round(statistics.mean(coord_means), 4) if coord_means else None,
            "coord_max_px": round(max(coord_maxes), 4) if coord_maxes else None,
            "coord_p99_px": round(float(np.percentile(coord_maxes, 99)), 4) if coord_maxes else None,
            "determinism_ok": determinism_ok,
            "manifest_reload_stable": manifest_stable,
            "unsupported_rejected": f"{rejected}/{len(unsupported_cases)}",
        },
        "l2": {
            "lossless_rate": round(l2_lossless / n, 4) if n else 0,
            "mean_l1_tokens": round(statistics.mean(l1_lens), 2) if l1_lens else None,
            "mean_l2_tokens": round(statistics.mean(l2_lens), 2) if l2_lens else None,
            "reduction_pct": round(100 * (1 - statistics.mean(l2_lens) / statistics.mean(l1_lens)), 2) if l1_lens else None,
        },
        "eval_sample": esum,
        "failures": failures[:50],
        "elapsed_s": round(time.time() - t0, 1),
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    print("\n===== VALIDATION REPORT =====")
    print(json.dumps(report["gates"], indent=2, ensure_ascii=False))
    print("L2:", json.dumps(report["l2"], ensure_ascii=False))
    print("EVAL:", json.dumps(report["eval_sample"], ensure_ascii=False))
    print(f"\nwrote {args.out}  ({report['elapsed_s']}s)")


if __name__ == "__main__":
    main()
