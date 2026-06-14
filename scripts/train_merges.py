#!/usr/bin/env python3
"""
L2 머지 학습 스크립트 (BPE-on-L1)
==================================
실세계 아이콘 코퍼스를 L1 로 토큰화한 뒤 BPE 머지를 학습하고, 머지를 포함한
불변 vocab 매니페스트를 저장한다.

사용:
    python scripts/train_merges.py --corpus /tmp/corpus --num-merges 1500 \
        --out geomtok/data/manifest_geom-5561-v1.json
"""

import argparse
import glob
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geomtok.api import GeomTokenizer
from geomtok.tokenizer.merge_tokenizer import learn_merges, MergeCodec
from geomtok.tokenizer.manifest import VocabManifest, DEFAULT_VOCAB_ID


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="/tmp/corpus")
    ap.add_argument("--num-merges", type=int, default=1500)
    ap.add_argument("--limit", type=int, default=2000, help="max icons for training")
    ap.add_argument("--out", default="geomtok/data/manifest_geom-5561-v1.json")
    args = ap.parse_args()

    gt = GeomTokenizer()  # L1 only
    files = sorted(glob.glob(os.path.join(args.corpus, "*.svg")))[: args.limit]

    print(f"[1/4] tokenizing {len(files)} icons to L1 ...")
    streams = []
    l1_lens = []
    for f in files:
        try:
            out = gt.tokenize(open(f, encoding="utf-8").read())
        except Exception:
            continue
        streams.append(out["token_ids"])
        l1_lens.append(out["n_tokens"])
    print(f"      {len(streams)} streams, mean L1 = {statistics.mean(l1_lens):.1f} tok/icon")

    print(f"[2/4] learning {args.num_merges} merges ...")
    t0 = time.time()
    merges = learn_merges(streams, num_merges=args.num_merges,
                          vocab_size=gt.vocab.vocab_size, min_freq=2)
    print(f"      learned {len(merges)} merges in {time.time()-t0:.1f}s")

    print("[3/4] verifying lossless roundtrip + measuring L2 ...")
    codec = MergeCodec(merges, gt.vocab.vocab_size)
    l2_lens = []
    lossless = True
    for s in streams:
        enc = codec.encode(s)
        dec = codec.decode(enc)
        if dec != s:
            lossless = False
        l2_lens.append(len(enc))
    mean_l1 = statistics.mean(l1_lens)
    mean_l2 = statistics.mean(l2_lens)
    print(f"      lossless={lossless}  L1={mean_l1:.1f}  L2={mean_l2:.1f}  "
          f"reduction={100*(1-mean_l2/mean_l1):.1f}%")

    print(f"[4/4] writing manifest with merges -> {args.out}")
    manifest = VocabManifest.build(vocab_id=DEFAULT_VOCAB_ID, merges=merges)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(manifest.to_json())
    print(f"      vocab_id={manifest.vocab_id}  hash={manifest.content_hash()}  "
          f"merge_base_id={manifest.merge_base_id}")

    stats = {
        "n_streams": len(streams), "mean_l1": mean_l1, "mean_l2": mean_l2,
        "reduction_pct": 100 * (1 - mean_l2 / mean_l1), "n_merges": len(merges),
        "lossless": lossless,
    }
    print("STATS " + json.dumps(stats))


if __name__ == "__main__":
    main()
