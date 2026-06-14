"""E4 (main-short): 이산 좌표 토큰 vs 연속 회귀 헤드 — CNM(Ogezi et al. 2026) 응수.

동일 백본·동일 데이터·동일 좌표 입력(연속 linear). 출력 헤드만 변경:
  - discrete : 다음 좌표를 4096 셀 분류(CE) → 셀 중심 → px
  - continuous: 다음 좌표를 (nx,ny) 회귀(MSE) → px
공정 화폐: held-out teacher-forced 다음-좌표 예측 픽셀 오차.
이산이 연속에 밀리지 않으면 '이산 토큰 기질'이 정당화됨.
"""
import sys, math, statistics
sys.path.insert(0, "/Users/limit/Projects/gpl-tokenizer")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import pandas as pd
from huggingface_hub import hf_hub_download
from geomtok.parser.svg_parser import SVGParser
from geomtok.tokenizer.primitive_tokenizer import PrimitiveTokenizer
from geomtok.tokenizer.vocabulary import GPLVocabulary

DEV, CANVAS, GRID = "cpu", 300.0, 64
parser = SVGParser(normalize_canvas=CANVAS)
VOCAB = GPLVocabulary(max_coord_level=6)
# 구조 토큰 ID(좌표 제외) → 0..K 리맵
STRUCT_IDS = sorted({t for t in range(100)})   # 0-99 구조/특수 영역
s2i = {sid: i for i, sid in enumerate(STRUCT_IDS)}
N_STRUCT = len(STRUCT_IDS)
BOS, EOS = 1, 2


def to_items(svg):
    """아이콘 → [(type,val)]. type 'S'=구조 토큰 id, 'C'=(nx,ny)."""
    doc = parser.parse_string(svg)
    items = [("S", BOS)]
    for e in doc.elements:
        for tid in PrimitiveTokenizer().tokenize(e.commands).token_ids:
            info = VOCAB.decode_token_id(tid)
            if info.get("type") == "coord":
                g = 2 ** info["level"]
                items.append(("C", ((info["qx"] + 0.5) / g, (info["qy"] + 0.5) / g)))
            elif tid not in (BOS, EOS) and tid in s2i:
                items.append(("S", tid))
    items.append(("S", EOS))
    return items


def load(n_train=900, n_test=200, max_len=180):
    df_tr = pd.read_parquet(hf_hub_download("starvector/svg-icons", "data/train-00000-of-00001.parquet", repo_type="dataset")).head(n_train * 2)
    df_te = pd.read_parquet(hf_hub_download("starvector/svg-icons", "data/test-00000-of-00001.parquet", repo_type="dataset")).head(n_test * 2)
    def build(df, n):
        out = []
        for svg in df["Svg"]:
            try:
                it = to_items(svg)
            except Exception:
                continue
            if 3 < len(it) <= max_len:
                out.append(it)
            if len(out) >= n:
                break
        return out
    return build(df_tr, n_train), build(df_te, n_test)


class CoordLM(nn.Module):
    def __init__(self, head, d=128, nl=3, nh=4):
        super().__init__()
        self.head_mode = head
        self.s_emb = nn.Embedding(N_STRUCT, d)
        self.c_in = nn.Linear(2, d)            # 연속 좌표 입력 (양 arm 동일)
        self.pos = nn.Embedding(256, d)
        layer = nn.TransformerEncoderLayer(d, nh, d * 4, dropout=0.1, batch_first=True, activation="gelu", norm_first=True)
        self.enc = nn.TransformerEncoder(layer, nl)
        self.ln = nn.LayerNorm(d)
        self.s_head = nn.Linear(d, N_STRUCT)   # 다음 구조 토큰 (양 arm 동일)
        if head == "discrete":
            self.c_head = nn.Linear(d, GRID * GRID)
        else:
            self.c_head = nn.Linear(d, 2)

    def embed(self, items):
        vecs = []
        for typ, val in items:
            if typ == "S":
                vecs.append(self.s_emb(torch.tensor(s2i[val], device=DEV)))
            else:
                vecs.append(self.c_in(torch.tensor(val, dtype=torch.float32, device=DEV)))
        return torch.stack(vecs).unsqueeze(0)   # (1,L,d)

    def forward(self, items):
        L = len(items)
        h = self.embed(items) + self.pos(torch.arange(L, device=DEV))
        mask = torch.triu(torch.ones(L, L, device=DEV), 1).bool()
        h = self.enc(h, mask=mask)
        return self.ln(h)[0]   # (L,d)


def train(model, data, epochs=12, lr=3e-4):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for ep in range(epochs):
        np.random.shuffle(data)
        for items in data:
            h = model(items)                       # (L,d)
            loss = 0.0; ns = nc = 0
            for i in range(len(items) - 1):
                typ, val = items[i + 1]
                if typ == "S":
                    loss = loss + F.cross_entropy(model.s_head(h[i:i+1]), torch.tensor([s2i[val]], device=DEV)); ns += 1
                else:
                    if model.head_mode == "discrete":
                        qx = min(GRID-1, int(val[0]*GRID)); qy = min(GRID-1, int(val[1]*GRID))
                        loss = loss + F.cross_entropy(model.c_head(h[i:i+1]), torch.tensor([qx*GRID+qy], device=DEV))
                    else:
                        pred = model.c_head(h[i:i+1])[0]
                        loss = loss + F.mse_loss(pred, torch.tensor(val, dtype=torch.float32, device=DEV))
                    nc += 1
            loss = loss / max(ns + nc, 1)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    return model


@torch.no_grad()
def eval_px(model, data):
    model.eval(); errs = []
    for items in data:
        h = model(items)
        for i in range(len(items) - 1):
            typ, val = items[i + 1]
            if typ != "C":
                continue
            if model.head_mode == "discrete":
                cell = int(model.c_head(h[i:i+1]).argmax()); qx, qy = cell // GRID, cell % GRID
                px, py = (qx + 0.5) / GRID, (qy + 0.5) / GRID
            else:
                p = model.c_head(h[i:i+1])[0]; px, py = float(p[0]), float(p[1])
            errs.append(math.hypot((px - val[0]) * CANVAS, (py - val[1]) * CANVAS))
    return errs


def main():
    tr, te = load()
    print(f"train {len(tr)}, test {len(te)} 아이콘")
    print(f"{'head':<12}{'mean px↓':>10}{'median px↓':>12}{'<2px %':>9}")
    res = {}
    for head in ("discrete", "continuous"):
        runs = []
        for seed in (0, 1):
            torch.manual_seed(seed); np.random.seed(seed)
            m = train(CoordLM(head), list(tr), epochs=12)
            runs.append(eval_px(m, te))
        e = np.array(runs[0] + runs[1])
        res[head] = e
        print(f"{head:<12}{e.mean():>10.2f}{np.median(e):>12.2f}{(e<2).mean()*100:>8.0f}%", flush=True)
    print(f"\n해석: discrete 평균 px ≤ continuous 면 '이산 좌표 토큰이 연속 회귀에 밀리지 않음'(CNM 응수).")
    print(f"  이산은 격자 양자화 바닥(평균~1.78px) 존재; 연속은 바닥 없으나 소데이터서 회귀가 더 어려울 수 있음.")
    print(f"  discrete {res['discrete'].mean():.2f} vs continuous {res['continuous'].mean():.2f} px")


if __name__ == "__main__":
    main()
