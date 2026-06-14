"""Figure 4: 실제 생성 샘플 그리드 — GeomTok 렌더 가능 vs char-BPE 0% (깨진 XML)."""
import numpy as np
d = np.load("/tmp/fig4_gen.npy", allow_pickle=True).item()
l1 = [x for x in d["l1"] if x is not None][:12]      # 렌더 가능 12개
char_total = len(d["char"]); char_ok = sum(1 for x in d["char"] if x is not None)

cell, pad, top = 86, 10, 64
ncol = 6
nrow = 2
W = pad + ncol * (cell + pad)
Hgeom = top + nrow * (cell + pad)
# char 패널: 빈 셀 그리드 (0% 렌더)
Hchar = 56 + 1 * (cell + pad)
H = Hgeom + Hchar + 20

s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Inter,Arial,sans-serif">']
s.append(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')
# GeomTok 패널
s.append(f'<text x="{pad}" y="28" font-size="14" font-weight="700" fill="#1a9b8e">GeomTok-L1 — generated samples</text>')
s.append(f'<text x="{pad}" y="48" font-size="11.5" fill="#666">48 / 60 renderable (plain top-k sampling, no FSA). Rough but valid icon-like geometry.</text>')
for i, b in enumerate(l1):
    r, c = divmod(i, ncol)
    x = pad + c * (cell + pad); y = top + r * (cell + pad)
    s.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" rx="7" fill="#fafffe" stroke="#cdeee9"/>')
    s.append(f'<image x="{x+4}" y="{y+4}" width="{cell-8}" height="{cell-8}" href="data:image/png;base64,{b}"/>')
# char-BPE 패널
cy = Hgeom + 10
s.append(f'<text x="{pad}" y="{cy+18}" font-size="14" font-weight="700" fill="#a23">text BPE — generated samples</text>')
s.append(f'<text x="{pad}" y="{cy+38}" font-size="11.5" fill="#666">0 / 60 renderable — same sampling produces malformed XML. (illustrative empty cells)</text>')
for c in range(ncol):
    x = pad + c * (cell + pad); y = cy + 50
    s.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" rx="7" fill="#fdf6f8" stroke="#f3d4de"/>')
    s.append(f'<text x="{x+cell/2:.0f}" y="{y+cell/2+5:.0f}" font-size="22" fill="#e3b8c4" text-anchor="middle">⌀</text>')
s.append('</svg>')
open("/Users/limit/Projects/gpl-tokenizer/assets/fig4_gen_samples.svg", "w").write("\n".join(s))
print(f"wrote assets/fig4_gen_samples.svg  (l1 shown {len(l1)}, char {char_ok}/{char_total})")
