"""
v0.6 스칼라 고정소수점 코덱 테스트
===================================
E1: 스칼라 (v,v) 2D 양자화 파탄 수정 검증.

회귀 기준 (v0.5.1에서 확인된 실제 결함):
    - r=20 원 → r=37.5 복원 (87% 오차)
    - 7개 원 REPEAT → 38개 복원
    - 예약 ID 5-9, 71-79 decode_token_id ValueError

v0.6 목표:
    - 스칼라 오차 ≤ canvas/(4096-1)/2 ≈ 0.037px (목표 2px의 ~50배 정밀)
    - 카운트 0..4095 왕복 무손실
    - 예약 ID 디코드 무크래시
"""

import sys
import os
import re

sys.path.insert(0, '..')

from geomtok.parser.svg_parser import SVGParser
from geomtok.tokenizer.arcs import ARCS
from geomtok.tokenizer.vocabulary import GPLVocabulary
from geomtok.tokenizer.composite_tokenizer import CompositeTokenizer
from geomtok.tokenizer.spatial_tokenizer import SpatialTokenizer
from geomtok.tokenizer.detokenizer import Detokenizer

passed = 0
failed = 0


def check(condition: bool, message: str):
    global passed, failed
    if condition:
        print(f"  [PASS] {message}")
        passed += 1
    else:
        print(f"  [FAIL] {message}")
        failed += 1


def attr(svg: str, name: str, tag: str = '(circle|rect|ellipse)') -> float:
    m = re.search(rf'<{tag}[^>]*[ "]{name}="([\d.]+)"', svg)
    return float(m.group(m.lastindex)) if m else float("nan")


print("=" * 60)
print("v0.6 스칼라 코덱 테스트")
print("=" * 60)

arcs = ARCS(canvas_size=300.0, max_level=6)
vocab = GPLVocabulary(max_coord_level=6)
parser = SVGParser()

print("\n[1] 코덱 단위 — 스칼라 왕복")
bound = arcs.scalar_max_error()
check(bound < 0.05, f"이론 오차 한계 {bound:.4f}px < 0.05px")
worst = max(abs(arcs.dequantize_scalar(arcs.quantize_scalar(v)) - v)
            for v in [0.0, 0.07, 1.0, 20.0, 37.5, 50.0, 149.97, 299.9])
check(worst <= bound + 1e-9, f"실측 최대 오차 {worst:.4f}px ≤ 한계")

print("\n[2] 코덱 단위 — 카운트 무손실")
check(all(arcs.dequantize_count(arcs.quantize_count(n)) == n
          for n in list(range(0, 200)) + [4095]), "0..199, 4095 왕복 정확")

print("\n[3] 회귀 — r=20 원 (v0.5.1: 37.5 복원)")
doc = parser.parse_string('<svg viewBox="0 0 300 300"><circle cx="150" cy="150" r="20"/></svg>')
ct = CompositeTokenizer()
res = ct.tokenize(doc.elements[0].commands)
svg_out = Detokenizer(ct.vocab, ct.arcs).to_svg_document(res.token_ids)
r_err = abs(attr(svg_out, 'r') - 20)
check(r_err <= bound + 1e-6, f"반지름 오차 {r_err:.4f}px (이전 17.5px)")
check(len(res.token_ids) <= 5 + 2, f"토큰 수 유지: {len(res.token_ids)}")

print("\n[4] 회귀 — 7개 원 REPEAT (v0.5.1: 38개 복원)")
svg7 = '<svg viewBox="0 0 300 300">' + "".join(
    f'<circle cx="{30 + i * 40}" cy="150" r="15"/>' for i in range(7)) + '</svg>'
doc7 = parser.parse_string(svg7)
st = SpatialTokenizer()
res7 = st.tokenize_multi([e.commands for e in doc7.elements])
out7 = Detokenizer(st.vocab, st.arcs).to_svg_document(res7.token_ids)
n_circles = out7.count('<circle')
check(n_circles == 7, f"원 개수 7 == {n_circles} (이전 38)")
radii = [float(v) for v in re.findall(r'r="([\d.]+)"', out7)]
check(radii and all(abs(r - 15) <= bound + 1e-6 for r in radii),
      f"모든 반지름 ≈15 (실제 {radii[:3]}...)")
cxs = sorted(float(v) for v in re.findall(r'cx="([\d.]+)"', out7))
sp_err = max(abs((cxs[i + 1] - cxs[i]) - 40) for i in range(len(cxs) - 1)) if len(cxs) >= 2 else 99
check(sp_err <= 2 * bound + 2.0, f"간격 40 오차 {sp_err:.3f}px (이전 ~12.5px)")

print("\n[5] 회귀 — rect/ellipse 크기")
doc_r = parser.parse_string('<svg viewBox="0 0 300 300"><rect x="50" y="50" width="80" height="40"/></svg>')
ct_r = CompositeTokenizer()
out_r = Detokenizer(ct_r.vocab, ct_r.arcs).to_svg_document(ct_r.tokenize(doc_r.elements[0].commands).token_ids)
check(abs(attr(out_r, 'width') - 80) <= bound + 1e-6 and abs(attr(out_r, 'height') - 40) <= bound + 1e-6,
      f"rect 80x40 → {attr(out_r, 'width')}x{attr(out_r, 'height')}")
doc_e = parser.parse_string('<svg viewBox="0 0 300 300"><ellipse cx="150" cy="150" rx="60" ry="25"/></svg>')
ct_e = CompositeTokenizer()
out_e = Detokenizer(ct_e.vocab, ct_e.arcs).to_svg_document(ct_e.tokenize(doc_e.elements[0].commands).token_ids)
check(abs(attr(out_e, 'rx') - 60) <= bound + 1e-6 and abs(attr(out_e, 'ry') - 25) <= bound + 1e-6,
      f"ellipse 60/25 → {attr(out_e, 'rx')}/{attr(out_e, 'ry')}")

print("\n[6] 예약 ID 디코드 무크래시 (v0.5.1: ValueError)")
ok = True
for tid in [5, 9, 18, 25, 35, 57, 71, 79, 99]:
    try:
        d = vocab.decode_token_id(tid)
        ok = ok and d["type"] == "unknown"
    except ValueError:
        ok = False
check(ok, "예약 ID 9종 → unknown 무크래시")

print("\n" + "=" * 60)
print(f"  Total: {passed + failed} | Passed: {passed} | Failed: {failed}")
print("=" * 60)
sys.exit(0 if failed == 0 else 1)
