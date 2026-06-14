"""
BPE-on-L1 머지 코덱 테스트 (L2, PRD §7.3)
==========================================
머지 학습·인코드/디코드 무손실·토큰 감소·결정성 확인.
"""

import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geomtok.api import GeomTokenizer
from geomtok.tokenizer.merge_tokenizer import learn_merges, MergeCodec

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
    gt = GeomTokenizer.default()

    # 1) 동봉 코덱 무손실 + 감소 (fixture)
    files = sorted(glob.glob(os.path.join(_FIX, "*.svg")))
    lossless = True
    l1t = l2t = 0
    for f in files:
        o1 = gt.tokenize(open(f).read(), level="L1")
        o2 = gt.tokenize(open(f).read(), level="L2")
        if gt._merge_codec.decode(o2["token_ids"]) != o1["token_ids"]:
            lossless = False
        l1t += o1["n_tokens"]; l2t += o2["n_tokens"]
    check(lossless, "bundled L2 decode(encode(L1)) == L1 (lossless)")
    check(l2t < l1t, f"L2 reduces tokens: {l1t} -> {l2t}")

    # 2) 보호 토큰(BOS/EOS/SEP)은 머지 경계
    streams = [gt.tokenize(open(f).read())["token_ids"] for f in files]
    merges = learn_merges(streams, num_merges=50, vocab_size=gt.vocab.vocab_size)
    for a, b, _ in merges:
        check_ok = a not in (1, 2, 3) and b not in (1, 2, 3)
        if not check_ok:
            check(False, "protected token leaked into a merge"); break
    else:
        check(True, "no merge crosses BOS/EOS/SEP")

    # 3) 결정성: 같은 코퍼스 → 같은 머지
    merges2 = learn_merges(streams, num_merges=50, vocab_size=gt.vocab.vocab_size)
    check(merges == merges2, "merge learning is deterministic")

    # 4) 코덱 라운드트립 on synthetic
    codec = MergeCodec(merges, gt.vocab.vocab_size)
    sample = streams[0]
    check(codec.decode(codec.encode(sample)) == sample, "codec roundtrip lossless")
    check(len(codec.encode(sample)) <= len(sample), "encode does not expand")

    # 5) merge ids >= vocab_size
    check(all(m[2] >= gt.vocab.vocab_size for m in merges), "merge ids above base vocab")

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
