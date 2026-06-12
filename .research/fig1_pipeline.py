"""Figure 1: 파이프라인 + 토큰 스트림 대조 (SVG)."""
W, H = 760, 430
s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Inter,Arial,sans-serif">']
s.append(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')

# ── 상단: 좌표 한 개가 두 토크나이저에서 어떻게 처리되는가 ──
s.append('<text x="30" y="34" font-size="15" font-weight="700" fill="#111">A coordinate `150.5` — two tokenizations</text>')
# char-BPE row
s.append('<text x="30" y="66" font-size="12.5" fill="#a23" font-weight="600">text BPE</text>')
frag = ["1", "50", ".", "5"]
x = 120
for f in frag:
    w = 30
    s.append(f'<rect x="{x}" y="52" width="{w}" height="22" rx="4" fill="#fdeaf1" stroke="#ff5d9e" stroke-width="1"/>')
    s.append(f'<text x="{x+w/2}" y="67" font-size="12" font-family="monospace" fill="#a23" text-anchor="middle">{f}</text>')
    x += w + 6
s.append(f'<text x="{x+8}" y="67" font-size="11.5" fill="#999">4 fragments · no notion of a point</text>')
# GeomTok row
s.append('<text x="30" y="100" font-size="12.5" fill="#1a9b8e" font-weight="600">GeomTok</text>')
s.append('<rect x="120" y="86" width="118" height="22" rx="4" fill="#e3faf6" stroke="#37e6d4" stroke-width="1.2"/>')
s.append('<text x="179" y="101" font-size="12" font-family="monospace" fill="#1a9b8e" text-anchor="middle">COORD@cell(32,17)</text>')
s.append('<text x="250" y="101" font-size="11.5" fill="#999">1 token · a point in space (fixed-point codec, §3.2)</text>')

# ── 하단: 4단계 파이프라인 ──
sy = 175
stages = [
    ("PARSE", "flatten transforms,\nnorm viewBox,\ndrop defs", "#8b6cff"),
    ("QUANTIZE", "uniform 64×64 grid\n+ fixed-point codec\n(0.04px, §3.2)", "#37e6d4"),
    ("TOKENIZE", "commands · shapes ·\ncoordinates →\n5,561 vocab", "#ffc857"),
    ("DECODE", "FSA-constrained →\nvalid SVG\n(bounded-error)", "#ff5d9e"),
]
bw, gap = 158, 24
x0 = 30
for i, (title, body, col) in enumerate(stages):
    x = x0 + i * (bw + gap)
    s.append(f'<rect x="{x}" y="{sy}" width="{bw}" height="96" rx="12" fill="#fff" stroke="{col}" stroke-width="1.6"/>')
    s.append(f'<rect x="{x}" y="{sy}" width="{bw}" height="26" rx="12" fill="{col}" opacity="0.16"/>')
    s.append(f'<text x="{x+12}" y="{sy+18}" font-size="12.5" font-weight="800" fill="{col}" letter-spacing="0.05em">{title}</text>')
    for j, line in enumerate(body.split("\n")):
        s.append(f'<text x="{x+12}" y="{sy+44+j*16}" font-size="11" fill="#444">{line}</text>')
    if i < 3:
        ax = x + bw + 4
        s.append(f'<path d="M{ax} {sy+48} l14 0 m-5 -5 l5 5 l-5 5" stroke="#bbb" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"/>')

# ── 하단 캡션: 토큰 스트림 대조 (실측 카운트는 코퍼스 평균이 대표적) ──
cy = 310
s.append(f'<text x="30" y="{cy}" font-size="13" font-weight="700" fill="#111">Same icon, two token streams</text>')
s.append(f'<text x="30" y="{cy+24}" font-size="12" fill="#a23" font-family="monospace">text BPE:  &lt; s v g _ v i e w B o x = " 0 …  ›  meaningless character fragments</text>')
s.append(f'<text x="30" y="{cy+46}" font-size="12" fill="#1a9b8e" font-family="monospace">GeomTok:  ⟨BOS⟩ MOVE ● LINE ● ● ⟨G0⟩ CLOSE  CIRCLE ●center ●r  ⟨EOS⟩</text>')
s.append(f'<text x="30" y="{cy+70}" font-size="11.5" fill="#666">● = one coordinate/scalar token. Across 400 real icons: 935 → 264 tokens/icon (3.54× at matched 5,561 vocab, §5.1).</text>')
s.append('</svg>')
open("/Users/limit/Projects/gpl-tokenizer/assets/fig1_pipeline.svg", "w").write("\n".join(s))
print("wrote assets/fig1_pipeline.svg")
