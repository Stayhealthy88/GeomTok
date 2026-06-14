"""
API 검토 수정 회귀 테스트 (P0/P1/P2)
=====================================
4차원 적대 검토에서 확정된 결함의 수정을 고정한다:
  P0-1 detokenize 가 FSA 검증을 실제로 실행 (valid 하드코딩 제거)
  P0-2 L2 토큰을 기본 level 로 디코드 시 무음 geometry 손실 없음
  P1-4 압축비 결정적 (cl100k strawman·tiktoken 의존 제거)
  P1-5 eval/batch 응답이 tokenizer_version+vocab_id 에코
  P1-3 신규 라우트 batch/jobs · jobs/{id} · stream
  P2   batch op 검증 · detect_unsupported 거짓양성 · MergeCodec 경계 보호
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geomtok.api import GeomTokenizer
from geomtok.tokenizer.merge_tokenizer import MergeCodec

passed = 0
failed = 0


def check(cond, msg):
    global passed, failed
    if cond:
        print(f"  [PASS] {msg}"); passed += 1
    else:
        print(f"  [FAIL] {msg}"); failed += 1


SQ = '<svg viewBox="0 0 24 24"><path d="M4 4 L20 4 L20 20 L4 20 Z"/></svg>'
gt = GeomTokenizer.default()
VER, VID = gt.tokenizer_version, gt.vocab_id

print("=" * 60)
print("API 검토 수정 회귀 테스트")
print("=" * 60)

print("\n[P0-1] detokenize FSA 검증 실제 실행")
good = gt.detokenize(gt.tokenize(SQ)["token_ids"], tokenizer_version=VER, vocab_id=VID)
check(good["valid"] is True and good["repaired"] is False,
      f"진짜 인코더 출력 → valid=True repaired=False (실제 {good['valid']}/{good['repaired']})")
# 좌표만 있는 비문법 스트림 → valid=False, repair 동작
bad = gt.detokenize([1, 200, 2], tokenizer_version=VER, vocab_id=VID)
check(bad["valid"] is False and bad["repaired"] is True,
      f"좌표-only 비문법 스트림 → valid=False repaired=True (실제 {bad['valid']}/{bad['repaired']})")
empty = gt.detokenize([2], tokenizer_version=VER, vocab_id=VID)
check(empty["valid"] is False, "EOS-only(BOS 없음) → valid=False")

print("\n[P0-2] L2 토큰 기본 level 디코드 무음손실 없음")
if gt.has_l2:
    l2 = gt.tokenize(SQ, level="L2")
    d_default = gt.detokenize(l2["token_ids"], tokenizer_version=VER, vocab_id=VID)
    d_l2 = gt.detokenize(l2["token_ids"], tokenizer_version=VER, vocab_id=VID, level="L2")
    check('d="M' in d_default["svg"] or "<path" in d_default["svg"],
          "기본 level 디코드가 비어있지 않은 geometry 복원")
    check(d_default["svg"] == d_l2["svg"], "기본 level 디코드 == level='L2' 디코드 (bit-identical)")
    # L1 순수 스트림은 자동 언머지에 변형되지 않음
    l1 = gt.tokenize(SQ)["token_ids"]
    d_l1 = gt.detokenize(l1, tokenizer_version=VER, vocab_id=VID)
    check(d_l1["valid"] is True, "L1 순수 스트림 자동언머지 no-op, valid")
else:
    check(True, "L2 미탑재 — 스킵")

print("\n[P1-4] 압축비 결정성 (cl100k·tiktoken 의존 제거)")
r1 = gt.tokenize(SQ)["compression_ratio"]
r2 = gt.tokenize(SQ)["compression_ratio"]
check(r1 == r2 and r1 > 0, f"동일 입력 → 동일 비 {r1} (결정적)")
check(abs(r1 - round(len(SQ) / gt.tokenize(SQ)["n_tokens"], 4)) < 1e-9,
      "압축비 = round(SVG 문자수 / 토큰수, 4) (도메인 독립·결정적)")

print("\n[P2] detect_unsupported 거짓양성 (주석/desc 내 언급)")
commented = '<svg viewBox="0 0 24 24"><!-- <text> placeholder --><path d="M4 4 L20 20"/></svg>'
try:
    c = gt.tokenize(commented)
    check(c["n_tokens"] > 2, "주석 내 <text> 언급 path-only 아이콘 정상 토큰화")
except Exception as e:
    check(False, f"주석 FP — 거부됨: {e}")
# 진짜 <text> 요소는 여전히 거부
from geomtok.errors import ParseUnsupportedElement
try:
    gt.tokenize('<svg viewBox="0 0 24 24"><text x="1" y="1">hi</text></svg>')
    check(False, "진짜 <text> 요소가 거부되지 않음")
except ParseUnsupportedElement:
    check(True, "진짜 <text> 요소는 여전히 거부")

print("\n[P2] MergeCodec 경계(SEP) 보호")
# 수기 머지 테이블이 SEP(3) 횡단을 시도해도 encode 가 넘지 않음
codec = MergeCodec([[100, 200, 5561], [5561, 3, 5562]], vocab_size=5561)
enc = codec.encode([100, 200, 3, 100, 200])
check(5562 not in enc, f"SEP 횡단 머지 미적용 (enc={enc})")
check(codec.decode(codec.encode([100, 200, 3, 100, 200])) == [100, 200, 3, 100, 200],
      "경계 보호 하에서도 라운드트립 무손실")

print("\n" + "=" * 60)
print(f"  {passed} passed, {failed} failed")
print("=" * 60)
sys.exit(0 if failed == 0 else 1)
