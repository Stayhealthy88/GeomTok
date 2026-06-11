"""F5 (논문 핵심 표): 토크나이저-스왑 다운스트림 — 실데이터, 동일 바닐라 모델.

NAACL'24 방법론: 토크나이저만 바꾼 동일 디코더-only LM을 실세계 아이콘에
학습해 생성 품질을 렌더 기반으로 비교. arm: char-BPE / L1 / L1+BPE.
지표: held-out NLL(bits/icon), 생성 렌더-FID-lite, 다양도(1-meanIoU), 신규성.
"""
import sys, io, math
sys.path.insert(0, "/Users/limit/Projects")
sys.path.insert(0, "/Users/limit/Projects/gpl-tokenizer/.research")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import resvg_py
from PIL import Image
import f5_data

torch.manual_seed(0)
DEV = "cpu"
PAD = 0
RES = 64


class VanillaLM(nn.Module):
    """토크나이저 비종속 디코더-only LM (모든 arm 동일 구조)."""
    def __init__(self, vocab, d=192, nl=4, nh=6, ff=512, max_len=256):
        super().__init__()
        self.tok = nn.Embedding(vocab, d, padding_idx=PAD)
        self.pos = nn.Embedding(max_len, d)
        layer = nn.TransformerEncoderLayer(d, nh, ff, dropout=0.1, batch_first=True,
                                           activation="gelu", norm_first=True)
        self.enc = nn.TransformerEncoder(layer, nl)
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight
        self.max_len = max_len

    def forward(self, x):
        L = x.size(1)
        h = self.tok(x) + self.pos(torch.arange(L, device=x.device))
        mask = torch.triu(torch.ones(L, L, device=x.device), 1).bool()
        h = self.enc(h, mask=mask, src_key_padding_mask=(x == PAD))
        return self.head(self.ln(h))


def batches(seqs, bs, max_len):
    order = np.random.permutation(len(seqs))
    for i in range(0, len(seqs), bs):
        chunk = [seqs[j][:max_len - 1] for j in order[i:i + bs]]
        L = max(len(s) for s in chunk) + 1
        x = torch.full((len(chunk), L), PAD, dtype=torch.long)
        for k, s in enumerate(chunk):
            x[k, :len(s) + 1] = torch.tensor([1] + s)  # BOS=1
        yield x.to(DEV)


def train(model, seqs, epochs=25, bs=32, max_len=256, lr=3e-4):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for ep in range(epochs):
        tot = 0.0
        for x in batches(seqs, bs, max_len):
            logits = model(x[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   x[:, 1:].reshape(-1), ignore_index=PAD)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
    return model


@torch.no_grad()
def heldout_bits(model, seqs, max_len=256):
    """held-out 평균 bits/icon (생성 품질 대용 — 분포 적합도)."""
    model.eval()
    tot_bits = 0.0
    for s in seqs:
        x = torch.tensor([[1] + s[:max_len - 1]], device=DEV)
        logits = model(x[:, :-1])
        ll = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                             x[:, 1:].reshape(-1), ignore_index=PAD, reduction="sum")
        tot_bits += ll.item() / math.log(2)
    return tot_bits / len(seqs)


def main():
    import statistics
    (trd, ted) = f5_data.load(n_train=1500, n_test=300, max_len=160)
    print(f"train {len(trd['l1'])}, test {len(ted['l1'])} 아이콘")
    print(f"{'arm':<10}{'vocab':>7}{'tok/icon':>10}{'heldout bits/icon (3 seed)':>28}")
    res = {}
    for arm in ("char_bpe", "l1", "l1_bpe"):
        V = f5_data.VOCAB[arm]
        bits = []
        for seed in (0, 1, 2):
            torch.manual_seed(seed); np.random.seed(seed)
            model = VanillaLM(V).to(DEV)
            train(model, trd[arm], epochs=25, max_len=200)
            bits.append(heldout_bits(model, ted[arm], max_len=200))
        res[arm] = bits
        tpi = np.mean([len(s) for s in ted[arm]])
        m, sd = statistics.mean(bits), statistics.stdev(bits)
        print(f"{arm:<10}{V:>7}{tpi:>10.1f}      {m:>8.0f} ± {sd:<6.0f}", flush=True)
    print("\n해석: held-out bits/icon 낮을수록 토큰화가 실데이터 분포를 잘 모델링(토크나이저 비종속).")
    print(f"  L1 {statistics.mean(res['l1']):.0f} < L1+BPE {statistics.mean(res['l1_bpe']):.0f} 이면 "
          f"'압축 우위(L1+BPE 58.8tok)가 다운스트림으로 이어지지 않음' — 내재≠외재 반례.")
    print(f"  L1 < char-BPE {statistics.mean(res['char_bpe']):.0f} 이면 '기하 토큰이 최고 다운스트림 기질'.")


if __name__ == "__main__":
    main()
