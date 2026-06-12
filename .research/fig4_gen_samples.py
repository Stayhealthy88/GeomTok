"""Figure 4 데이터: 실제 생성 샘플 — GeomTok-L1 모델 vs char-BPE 모델, 동일 plain 샘플링.
GeomTok은 렌더 가능 아이콘 생성, char-BPE는 0% 렌더(깨진 XML)를 시각적으로 대조."""
import sys, io, base64
sys.path.insert(0, "/Users/limit/Projects/gpl-tokenizer")
sys.path.insert(0, "/Users/limit/Projects/gpl-tokenizer/.research")
import numpy as np, torch
import resvg_py
from PIL import Image
import f5_data, f5_gen
from f5_run import VanillaLM, train

torch.manual_seed(0); np.random.seed(0)
RES = 128

def render_or_blank(svg):
    try:
        if not svg or "<" not in svg: return None
        png = resvg_py.svg_to_bytes(svg_string=svg, width=RES, height=RES)
        rgba = Image.open(io.BytesIO(bytes(png))).convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255,255,255,255))
        im = Image.alpha_composite(bg, rgba).convert("L")
        if (np.asarray(im) < 230).sum() < 12: return None   # 백지
        return im
    except Exception:
        return None

def b64(img):
    bio = io.BytesIO(); img.save(bio, "PNG"); return base64.b64encode(bio.getvalue()).decode()

(trd, ted) = f5_data.load(n_train=1500, n_test=300, max_len=160)
samples = {}
for arm in ("l1", "char_bpe"):
    seqs = trd[arm]; V = f5_data.VOCAB[arm]
    torch.manual_seed(0); np.random.seed(0)
    m = train(VanillaLM(V).to("cpu"), seqs, epochs=25, max_len=200)
    gens = f5_gen.gen(m, 60)
    imgs = []
    for g in gens:
        svg = f5_gen.decode("char" if arm == "char_bpe" else "l1", g)
        im = render_or_blank(svg)
        imgs.append(b64(im) if im is not None else None)
    ok = sum(1 for i in imgs if i is not None)
    samples[arm] = imgs
    print(f"{arm}: {ok}/{len(imgs)} renderable", flush=True)

np.save("/tmp/fig4_gen.npy", {"l1": samples["l1"], "char": samples["char_bpe"]}, allow_pickle=True)
print("fig4 gen data saved")
