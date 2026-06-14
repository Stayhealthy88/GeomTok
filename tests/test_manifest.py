"""
Vocab 매니페스트 · 결정성 테스트 (PRD G2)
==========================================
매니페스트 구축·직렬화 왕복·내용 해시 안정성·어휘 크기(5561) 확인.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geomtok.tokenizer.manifest import (
    VocabManifest, TOKENIZER_VERSION, DEFAULT_VOCAB_ID, load_bundled_manifest,
)

passed = 0
failed = 0


def check(cond, msg):
    global passed, failed
    if cond:
        print(f"  [PASS] {msg}"); passed += 1
    else:
        print(f"  [FAIL] {msg}"); failed += 1


def main():
    m = VocabManifest.build()
    check(m.vocab_size == 5561, f"vocab_size == 5561 (got {m.vocab_size})")
    check(m.vocab_id == DEFAULT_VOCAB_ID, "default vocab_id")
    check(m.tokenizer_version == TOKENIZER_VERSION, "tokenizer_version pinned")
    check(m.coord_token_base == 100, "coord base 100")
    check(len(m.level_layout) == 7, "7 coord levels (0..6)")
    check(m.level_layout[-1]["grid"] == 64, "deepest grid 64x64")

    # 직렬화 왕복
    m2 = VocabManifest.from_json(m.to_json())
    check(m2.vocab_size == m.vocab_size and m2.special == m.special,
          "json roundtrip preserves fields")
    check(m2.content_hash() == m.content_hash(), "hash stable across roundtrip")

    # 결정성: 같은 빌드 → 같은 해시
    check(VocabManifest.build().content_hash() == m.content_hash(),
          "rebuild is bit-identical (deterministic)")

    # 다른 어휘 → 다른 해시
    m3 = VocabManifest.build(max_coord_level=5)
    check(m3.vocab_size != m.vocab_size and m3.content_hash() != m.content_hash(),
          "different config -> different hash")

    # 동봉 매니페스트 로드 (L2 머지 포함)
    b = load_bundled_manifest()
    check(b is not None, "bundled manifest loads")
    if b is not None:
        check(b.has_merges and b.merge_base_id == 5561,
              f"bundled has merges, base {b.merge_base_id}")
        check(b.content_hash() == load_bundled_manifest().content_hash(),
              "bundled reload stable")

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
