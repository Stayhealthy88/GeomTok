"""
v0.6 / E2 — transform 평탄화 + defs 제외 + viewBox 정규화 테스트
=================================================================
회귀 기준 (v0.5.1 확인 결함):
    - <g transform="translate(100,100)"><rect/></g> → bbox (0,0,10,10), 변환 무시
    - <defs><rect/></defs> → 가시 요소로 수집
    - 50x50 viewBox 아이콘 → 절대 px 공차로 공간 분석 붕괴
"""

import sys
import math

sys.path.insert(0, '..')

from geomtok.parser.svg_parser import SVGParser

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


def bbox(elem):
    xs, ys = [], []
    for c in elem.commands:
        if c.end_point:
            xs.append(c.end_point[0])
            ys.append(c.end_point[1])
    return (min(xs), min(ys), max(xs), max(ys)) if xs else None


print("=" * 60)
print("E2: transform 평탄화 테스트")
print("=" * 60)

P = SVGParser()

print("\n[1] g translate 상속 (v0.5.1: 무시)")
doc = P.parse_string('<svg viewBox="0 0 300 300"><g transform="translate(100,100)">'
                     '<rect x="0" y="0" width="10" height="10"/></g></svg>')
b = bbox(doc.elements[0])
check(abs(b[0] - 100) < 1e-6 and abs(b[2] - 110) < 1e-6, f"translate 적용: bbox {b}")

print("\n[2] 중첩 그룹 누적")
doc = P.parse_string('<svg viewBox="0 0 300 300"><g transform="translate(50,0)">'
                     '<g transform="scale(2)"><rect x="10" y="10" width="10" height="10"/></g></g></svg>')
b = bbox(doc.elements[0])
check(abs(b[0] - 70) < 1e-6 and abs(b[2] - 90) < 1e-6, f"translate∘scale: bbox {b}")

print("\n[3] rotate(90) 원점 회전")
doc = P.parse_string('<svg viewBox="0 0 300 300"><path transform="rotate(90)" d="M 10 0 L 20 0"/></svg>')
ep = doc.elements[0].commands[1].end_point
check(abs(ep[0] - 0) < 1e-6 and abs(ep[1] - 20) < 1e-6, f"(20,0)→{tuple(round(v, 3) for v in ep)}")

print("\n[4] matrix() 직접 지정")
doc = P.parse_string('<svg viewBox="0 0 300 300"><path transform="matrix(1 0 0 1 30 40)" d="M 0 0 L 10 10"/></svg>')
ep = doc.elements[0].commands[1].end_point
check(abs(ep[0] - 40) < 1e-6 and abs(ep[1] - 50) < 1e-6, f"(10,10)→{tuple(round(v, 3) for v in ep)}")

print("\n[5] 요소 자체 transform + 그룹 동시")
doc = P.parse_string('<svg viewBox="0 0 300 300"><g transform="translate(100,0)">'
                     '<rect transform="translate(0,100)" x="0" y="0" width="10" height="10"/></g></svg>')
b = bbox(doc.elements[0])
check(abs(b[0] - 100) < 1e-6 and abs(b[1] - 100) < 1e-6, f"합성: bbox {b}")

print("\n[6] defs/clipPath 제외 (v0.5.1: 누수)")
doc = P.parse_string('<svg viewBox="0 0 300 300"><defs><rect width="10" height="10"/></defs>'
                     '<clipPath><circle cx="5" cy="5" r="5"/></clipPath>'
                     '<circle cx="150" cy="150" r="20"/></svg>')
check(len(doc.elements) == 1 and doc.elements[0].tag == "circle",
      f"가시 요소 1개만 수집 (실제 {len(doc.elements)})")

print("\n[7] viewBox 정규화 (50x50 → 300 캔버스)")
Pn = SVGParser(normalize_canvas=300.0)
doc = Pn.parse_string('<svg viewBox="0 0 50 50"><circle cx="25" cy="25" r="10"/></svg>')
b = bbox(doc.elements[0])
cx = (b[0] + b[2]) / 2
check(abs(cx - 150) < 1.0, f"중심 25→{cx:.1f} (스케일 6x)")
check(doc.viewbox == (0.0, 0.0, 300.0, 300.0), "viewbox 갱신")

print("\n[8] 기본값은 무정규화 (기존 동작 보존)")
doc = P.parse_string('<svg viewBox="0 0 50 50"><circle cx="25" cy="25" r="10"/></svg>')
b = bbox(doc.elements[0])
check(abs((b[0] + b[2]) / 2 - 25) < 1e-6, "정규화 off → 좌표 불변")

print("\n" + "=" * 60)
print(f"  Total: {passed + failed} | Passed: {passed} | Failed: {failed}")
print("=" * 60)
sys.exit(0 if failed == 0 else 1)
