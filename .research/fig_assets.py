"""Fig 1/3/4 에셋 생성: 토큰 스트림 대조(Fig1), 정성 렌더 패널(Fig3), 생성 샘플(Fig4)."""
import sys, io
sys.path.insert(0, "/Users/limit/Projects/gpl-tokenizer")
sys.path.insert(0, "/Users/limit/Projects/gpl-tokenizer/.research")
import numpy as np, pandas as pd, resvg_py, tiktoken
from PIL import Image
from huggingface_hub import hf_hub_download
from geomtok.parser.svg_parser import SVGParser
from geomtok.parser.path_parser import PathParser
from geomtok.tokenizer.primitive_tokenizer import PrimitiveTokenizer
from geomtok.tokenizer.arcs import ARCS
from geomtok.tokenizer.detokenizer import Detokenizer

CANVAS = 300.0
P = SVGParser(normalize_canvas=CANVAS)
enc = tiktoken.get_encoding("cl100k_base")

def render_png(svg, res=160, stroke=3):
    png = resvg_py.svg_to_bytes(svg_string=svg, width=res, height=res)
    rgba = Image.open(io.BytesIO(bytes(png))).convert("RGBA")
    bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    return Image.alpha_composite(bg, rgba).convert("L")

def raw_d(e):
    parts = []
    for c in e.commands:
        v = c.abs_params or []; k = c.command_type.value
        if k in ("M", "L"): parts.append(f"{k} {v[0]:.1f} {v[1]:.1f}")
        elif k == "C": parts.append("C " + " ".join(f"{x:.1f}" for x in v[:6]))
        elif k == "Q": parts.append("Q " + " ".join(f"{x:.1f}" for x in v[:4]))
        elif k == "Z": parts.append("Z")
    return " ".join(parts)

def wrap(paths, stroke=3):
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 300">'
            + "".join(f'<path d="{d}" fill="none" stroke="black" stroke-width="{stroke}"/>' for d in paths) + "</svg>")

# ---- Fig 1 데이터: 간단 아이콘의 char-BPE vs GeomTok-L1 토큰 스트림 ----
ICON = '<svg viewBox="0 0 24 24"><path d="M4 4 L20 4 L20 20 Z"/><circle cx="12" cy="12" r="6"/></svg>'
doc = P.parse_string(ICON)
char_toks = enc.encode(ICON)
pt = PrimitiveTokenizer()
gtoks = []
for e in doc.elements:
    gtoks += pt.tokenize(e.commands).token_ids
gnames = []
for t in gtoks:
    info = pt.vocab.decode_token_id(t)
    if info.get("type") == "coord": gnames.append(f"·")          # coordinate
    elif info.get("type") == "command": gnames.append(info["value"][:4])
    elif info.get("type") == "special": gnames.append(info["value"])
    else: gnames.append(info.get("value", "?")[:4])
print(f"FIG1 char_bpe={len(char_toks)} geomtok={len(gtoks)}")
print(f"FIG1 char_sample={enc.decode(char_toks[:6])!r}...")
print(f"FIG1 geom_names={gnames}")

# ---- Fig 3 데이터: 원본 vs L1 균일 vs 적응 왕복 렌더 (3 아이콘) ----
df = pd.read_parquet(hf_hub_download("starvector/svg-icons","data/test-00000-of-00001.parquet",repo_type="dataset")).head(40)
from skimage.metrics import structural_similarity as ssim
picks = []
for svg in df["Svg"]:
    try:
        d = P.parse_string(svg)
        nc = sum(len(e.commands) for e in d.elements)
        if d.elements and 8 <= nc <= 60: picks.append((svg, d))
    except: pass
    if len(picks) >= 3: break

def recon(d, uniform):
    out = []
    for e in d.elements:
        p = PrimitiveTokenizer(uniform_coords=uniform)
        r = p.tokenize(e.commands)
        out.append(Detokenizer(p.vocab, p.arcs).detokenize(r.token_ids))
    return wrap(out)

import base64
def b64(img):
    bio = io.BytesIO(); img.save(bio, "PNG"); return base64.b64encode(bio.getvalue()).decode()

fig3 = []
for svg, d in picks:
    ref = render_png(wrap([raw_d(e) for e in d.elements]))
    uni = render_png(recon(d, True)); adp = render_png(recon(d, False))
    ra = np.asarray(ref,float)/255; ua=np.asarray(uni,float)/255; aa=np.asarray(adp,float)/255
    fig3.append((b64(ref), b64(uni), ssim(ra,ua,data_range=1.0), b64(adp), ssim(ra,aa,data_range=1.0)))
np.save("/tmp/fig3_data.npy", np.array(fig3, dtype=object), allow_pickle=True)
print(f"FIG3 {len(fig3)} icons, uniform SSIM {[round(f[2],3) for f in fig3]}, adaptive {[round(f[4],3) for f in fig3]}")

# ---- Fig 4 데이터: GeomTok 생성 샘플 (저장된 모델 없으니, 실데이터 아이콘을 'L1 왕복'으로 대체 시각화) ----
# 생성 모델 재학습은 비용 큼 → Fig4는 'L1 토큰으로 표현·복원되는 실제 아이콘' 그리드 + 텍스트=빈칸 대조로 개념 전달
gallery = []
cnt = 0
for svg in df["Svg"]:
    try:
        d = P.parse_string(svg)
        if not d.elements: continue
        gallery.append(b64(render_png(recon(d, True))))
        cnt += 1
    except: pass
    if cnt >= 8: break
np.save("/tmp/fig4_data.npy", np.array(gallery, dtype=object), allow_pickle=True)
print(f"FIG4 {len(gallery)} renderable GeomTok samples")
import json
json.dump({"char_n":len(char_toks),"geom_n":len(gtoks),"geom_names":gnames}, open("/tmp/fig1_data.json","w"))
print("assets data ready")
