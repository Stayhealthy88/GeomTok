"""Figure 2: 압축–모델가능성 산점도 (SVG 출력).
x = tokens/icon (압축, 왼쪽이 더 압축), y = held-out NLL bits/icon (낮을수록 모델링 우수).
비단조 관계: 가장 압축된 점이 가장 나쁜 모델링.
데이터: F5 §5.3 (char/l1/l1_bpe) + E1 budget sweep (L1 family)."""
W, H = 720, 460
ML, MR, MT, MB = 92, 30, 50, 70
PW, PH = W - ML - MR, H - MT - MB

# (label, tok/icon, NLL bits, color, family)
F5 = [("char-BPE", 159, 775, "#ff5d9e"), ("L1+BPE(1k)", 59, 666, "#ffc857"), ("L1", 104, 576, "#37e6d4")]
E1 = [(105, 608), (71, 635), (70, 649), (68, 668), (65, 680)]  # L1 budget 0,250,500,1k,2k

xs = [p[1] for p in F5] + [p[0] for p in E1]
ys = [p[2] for p in F5] + [p[1] for p in E1]
xmin, xmax = 40, 175
ymin, ymax = 560, 800

def X(v): return ML + (v - xmin) / (xmax - xmin) * PW
def Y(v): return MT + (1 - (v - ymin) / (ymax - ymin)) * PH

s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Inter,Arial,sans-serif">']
s.append(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')
# axes
s.append(f'<line x1="{ML}" y1="{MT}" x2="{ML}" y2="{MT+PH}" stroke="#333" stroke-width="1.2"/>')
s.append(f'<line x1="{ML}" y1="{MT+PH}" x2="{ML+PW}" y2="{MT+PH}" stroke="#333" stroke-width="1.2"/>')
# gridlines + y ticks
for yv in range(560, 801, 40):
    yy = Y(yv)
    s.append(f'<line x1="{ML}" y1="{yy:.1f}" x2="{ML+PW}" y2="{yy:.1f}" stroke="#eee" stroke-width="1"/>')
    s.append(f'<text x="{ML-10}" y="{yy+4:.1f}" font-size="12" fill="#555" text-anchor="end">{yv}</text>')
for xv in range(40, 176, 20):
    xx = X(xv)
    s.append(f'<text x="{xx:.1f}" y="{MT+PH+20}" font-size="12" fill="#555" text-anchor="middle">{xv}</text>')
# axis labels
s.append(f'<text x="{ML+PW/2:.0f}" y="{H-22}" font-size="14" fill="#222" text-anchor="middle" font-weight="600">tokens / icon  (← more compressed)</text>')
s.append(f'<text x="22" y="{MT+PH/2:.0f}" font-size="14" fill="#222" text-anchor="middle" font-weight="600" transform="rotate(-90 22 {MT+PH/2:.0f})">held-out NLL bits / icon  (↓ better modeling)</text>')
# E1 L1-family trend line (budget sweep) — connect in budget order
pts = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in E1)
s.append(f'<polyline points="{pts}" fill="none" stroke="#37e6d4" stroke-width="1.6" stroke-dasharray="5 4" opacity="0.7"/>')
for x, y in E1:
    s.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="4" fill="#37e6d4" opacity="0.55"/>')
s.append(f'<text x="{X(65):.1f}" y="{Y(680)-10:.1f}" font-size="10.5" fill="#1a9b8e" text-anchor="middle">L1 + more merges →</text>')
# F5 main points (bigger, labeled)
for label, x, y, col in F5:
    s.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="8" fill="{col}" stroke="#fff" stroke-width="2"/>')
    # place labels to avoid collisions: char-BPE below-left, L1+BPE below, L1 below
    if label == "char-BPE":
        s.append(f'<text x="{X(x)-12:.1f}" y="{Y(y)+22:.1f}" font-size="13" fill="#111" text-anchor="end" font-weight="700">{label}</text>')
    else:
        dy = -16 if label == "L1" else 24
        s.append(f'<text x="{X(x):.1f}" y="{Y(y)+dy:.1f}" font-size="13" fill="#111" text-anchor="middle" font-weight="700">{label}</text>')
# annotation: non-monotonic — upper-left, away from char-BPE point
s.append(f'<text x="{ML+14}" y="{MT+18}" font-size="13.5" fill="#a23" text-anchor="start" font-weight="700">Most compressed ≠ best model</text>')
s.append(f'<text x="{ML+14}" y="{MT+36}" font-size="11.5" fill="#888" text-anchor="start">L1 (104 tok) models better than L1+BPE (59 tok) and char-BPE (159 tok)</text>')
s.append('</svg>')
open("/Users/limit/Projects/gpl-tokenizer/assets/fig2_compression_modelability.svg", "w").write("\n".join(s))
print("wrote assets/fig2_compression_modelability.svg")
