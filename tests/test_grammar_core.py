"""
torch-free FSA 문법 코어 테스트 (PRD G3)
==========================================
GrammarFSA 의 검증·마스킹·repair 가 well-formed 스트림을 통과시키고 위반을
정확히 잡는지, 그리고 fixture 아이콘 토큰 스트림이 100% 유효한지 확인.
"""

import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geomtok.api import GeomTokenizer
from geomtok.tokenizer.grammar import GrammarFSA

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
    fsa = GrammarFSA(gt.vocab)

    # 1) 실제 아이콘 스트림 100% 유효
    files = sorted(glob.glob(os.path.join(_FIX, "*.svg")))
    ok = sum(fsa.validate(gt.tokenize(open(f).read())["token_ids"]).valid
             for f in files)
    check(ok == len(files), f"fixture streams valid: {ok}/{len(files)}")

    # 2) 위반 케이스 거부
    check(not fsa.validate([10, 140, 2]).valid, "missing BOS rejected")
    check(not fsa.validate([1, 14, 140, 2]).valid, "CUBIC short args rejected")
    check(not fsa.validate([1, 2, 10]).valid, "tokens after EOS rejected")
    check(not fsa.validate([1, 10, 11, 2]).valid, "non-coord where coord required rejected")

    # 3) 정상 케이스 통과
    check(fsa.validate([1, 10, 140, 11, 141, 2]).valid, "MOVE+LINE valid")
    check(fsa.is_valid([1, 2]), "empty BOS->EOS valid")

    # 4) 마스크: coord 위치에서는 coord 만, boundary 에서는 명령
    mask_after_move = fsa.allowed_ids([1, 10])     # MOVE → expects coord
    check(all(gt.vocab.id_to_coord(t) is not None for t in mask_after_move),
          "after MOVE only coords allowed")
    boundary = fsa.allowed_ids([1])                # after BOS
    check(2 in boundary and 10 in boundary and 140 not in boundary,
          "after BOS: EOS/CMD allowed, coord not")

    # 5) repair: 망가진 스트림 → 유효
    rep = fsa.repair([1, 14, 140, 140, 10, 17, 14, 140])
    check(fsa.is_valid(rep), "repair produces valid stream")
    rep2 = fsa.repair([14, 140, 999, 2, 2])         # no BOS, junk, extra EOS
    check(fsa.is_valid(rep2) and rep2[0] == 1, "repair adds BOS and closes")

    # 6) numpy 마스크 길이 = vocab
    m = fsa.allowed_mask([1])
    check(m.shape[0] == gt.vocab.vocab_size, "mask length == vocab_size")

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
