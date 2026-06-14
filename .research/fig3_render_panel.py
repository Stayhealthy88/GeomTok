"""Figure 3: 정성 렌더 패널 — 원본 vs uniform 왕복 vs adaptive 왕복 (SSIM 캡션).
uniform이 adaptive보다 시각적으로 우수함을 보임 (§3.3/§6 부정결과 뒷받침)."""
import numpy as np
data = np.load("/tmp/fig3_data.npy", allow_pickle=True)   # [(ref,uni,ssim_u,adp,ssim_a)]

cell, pad, lab, top = 130, 18, 26, 56
ncol = 3
W = pad + ncol * (cell + pad)
H = top + len(data) * (cell + lab + pad)
s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Inter,Arial,sans-serif">']
s.append(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')
heads = [("Original", "#111"), ("GeomTok uniform-L6", "#1a9b8e"), ("adaptive quadtree", "#a23")]
for c, (t, col) in enumerate(heads):
    x = pad + c * (cell + pad) + cell / 2
    s.append(f'<text x="{x:.0f}" y="34" font-size="13" font-weight="700" fill="{col}" text-anchor="middle">{t}</text>')

for r, row in enumerate(data):
    ref, uni, su, adp, sa = row
    y = top + r * (cell + lab + pad)
    imgs = [(ref, None, "#111"), (uni, float(su), "#1a9b8e"), (adp, float(sa), "#a23")]
    for c, (b64, sval, col) in enumerate(imgs):
        x = pad + c * (cell + pad)
        s.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" rx="8" fill="#fafafa" stroke="#e5e5e5"/>')
        s.append(f'<image x="{x+4}" y="{y+4}" width="{cell-8}" height="{cell-8}" href="data:image/png;base64,{b64}"/>')
        if sval is not None:
            s.append(f'<text x="{x+cell/2:.0f}" y="{y+cell+18}" font-size="12" font-weight="600" fill="{col}" text-anchor="middle">SSIM {sval:.3f}</text>')
s.append(f'<text x="{pad}" y="{H-8}" font-size="10.5" fill="#888">Uniform grid preserves straight content the adaptive tree starves (§6).</text>')
s.append(f'<text x="{W-pad}" y="{H-8}" font-size="10.5" fill="#888" text-anchor="end">corpus SSIM 0.929 vs 0.846</text>')
s.append('</svg>')
open("/Users/limit/Projects/gpl-tokenizer/assets/fig3_render_panel.svg", "w").write("\n".join(s))
print("wrote assets/fig3_render_panel.svg")
