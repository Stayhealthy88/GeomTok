"""main-short E1-E3 공용: 코퍼스 로더 + 자체 BPE(정확·예산제어) + 학습/평가 재사용."""
import sys, collections, math, os
sys.path.insert(0, "/Users/limit/Projects/gpl-tokenizer")
sys.path.insert(0, "/Users/limit/Projects/gpl-tokenizer/.research")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
from huggingface_hub import hf_hub_download
from geomtok.parser.svg_parser import SVGParser

PAD, BOS, EOS = 0, 1, 2
_parser = SVGParser(normalize_canvas=300.0)


def _l1(svg):
    from geomtok.tokenizer.primitive_tokenizer import PrimitiveTokenizer
    doc = _parser.parse_string(svg)
    out = []
    for e in doc.elements:
        out += PrimitiveTokenizer().tokenize(e.commands).token_ids
    return out


def load_corpus(dataset, n_train=1000, n_test=200, max_len=160):
    """dataset HF id → {train:{l1,svg}, test:{l1,svg}}. svg 텍스트도 보존(char-BPE용)."""
    def grab(split, mult):
        # train 분할이 샤딩된 경우 첫 샤드만
        fn = f"data/{split}-00000-of-00001.parquet"
        try:
            p = hf_hub_download(dataset, fn, repo_type="dataset")
        except Exception:
            p = hf_hub_download(dataset, f"data/{split}-00000-of-00009.parquet", repo_type="dataset")
        return pd.read_parquet(p).head(mult)
    tr_df = grab("train", n_train * 3)
    te_df = grab("test", n_test * 3)

    def build(df, n):
        out = {"l1": [], "svg": []}
        for svg in df["Svg"]:
            try:
                ids = _l1(svg)
            except Exception:
                continue
            if not ids or len(ids) > max_len:
                continue
            out["l1"].append(ids); out["svg"].append(svg)
            if len(out["l1"]) >= n:
                break
        return out
    return {"train": build(tr_df, n_train), "test": build(te_df, n_test)}


# ---- 자체 BPE (정확 역매핑, 예산 N 제어) ----
def bpe_train(corpus, n):
    seqs = [list(s) for s in corpus]
    pairs = collections.Counter()
    for s in seqs:
        for i in range(len(s) - 1):
            pairs[(s[i], s[i + 1])] += 1
    merges = []
    for _ in range(n):
        if not pairs:
            break
        (a, b), c = max(pairs.items(), key=lambda kv: kv[1])
        if c < 2:
            break
        new = ("M", len(merges)); merges.append((a, b, new))
        for s in seqs:
            i = 0
            while i < len(s) - 1:
                if s[i] == a and s[i + 1] == b:
                    if i > 0: pairs[(s[i - 1], a)] -= 1
                    if i + 2 < len(s): pairs[(b, s[i + 2])] -= 1
                    s[i] = new; del s[i + 1]
                    if i > 0: pairs[(s[i - 1], new)] += 1
                    if i + 1 < len(s): pairs[(new, s[i + 1])] += 1
                else:
                    i += 1
        pairs.pop((a, b), None)
    return merges


def bpe_apply(seq, merges):
    s = list(seq)
    for a, b, new in merges:
        i = 0
        while i < len(s) - 1:
            if s[i] == a and s[i + 1] == b:
                s[i:i + 2] = [new]
            else:
                i += 1
    return s


def remap(seqs, base=5561):
    """심볼(int 또는 머지 튜플) → 연속 id. (remapped_seqs, vocab_size) 반환."""
    sym2id = {}
    out = []
    for s in seqs:
        r = []
        for x in s:
            if isinstance(x, int):
                r.append(x)
            else:
                if x not in sym2id:
                    sym2id[x] = base + len(sym2id)
                r.append(sym2id[x])
        out.append(r)
    return out, base + len(sym2id)


def train_sp_text(svgs, vocab_size, prefix):
    """corpus SVG 텍스트로 SentencePiece BPE 학습 → encode 함수 반환."""
    model = f"{prefix}.model"
    if not os.path.exists(model):
        txt = f"{prefix}_train.txt"
        with open(txt, "w") as f:
            for s in svgs:
                f.write(s.replace("\n", " ") + "\n")
        spm.SentencePieceTrainer.train(input=txt, model_prefix=prefix, vocab_size=vocab_size,
                                       model_type="bpe", character_coverage=1.0,
                                       max_sentence_length=100000)
    sp = spm.SentencePieceProcessor(model_file=model)
    return sp


# ---- 바닐라 LM (토크나이저 비종속) ----
class VanillaLM(nn.Module):
    def __init__(self, vocab, d=192, nl=4, nh=6, ff=512, max_len=256):
        super().__init__()
        self.tok = nn.Embedding(vocab, d, padding_idx=PAD)
        self.pos = nn.Embedding(max_len, d)
        layer = nn.TransformerEncoderLayer(d, nh, ff, dropout=0.1, batch_first=True,
                                           activation="gelu", norm_first=True)
        self.enc = nn.TransformerEncoder(layer, nl)
        self.ln = nn.LayerNorm(d); self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight; self.max_len = max_len

    def forward(self, x):
        L = x.size(1)
        h = self.tok(x) + self.pos(torch.arange(L, device=x.device))
        m = torch.triu(torch.ones(L, L, device=x.device), 1).bool()
        return self.head(self.ln(self.enc(h, mask=m, src_key_padding_mask=(x == PAD))))


def _batches(seqs, bs, max_len, seed):
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(seqs))
    for i in range(0, len(seqs), bs):
        chunk = [seqs[j][:max_len - 1] for j in order[i:i + bs]]
        L = max(len(s) for s in chunk) + 1
        x = torch.full((len(chunk), L), PAD, dtype=torch.long)
        for k, s in enumerate(chunk):
            x[k, :len(s) + 1] = torch.tensor([BOS] + list(s))
        yield x


def train(model, seqs, epochs=22, bs=32, max_len=200, lr=3e-4, seed=0):
    opt = torch.optim.AdamW(model.parameters(), lr=lr); model.train()
    for ep in range(epochs):
        for x in _batches(seqs, bs, max_len, seed * 1000 + ep):
            logits = model(x[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1), ignore_index=PAD)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    return model


@torch.no_grad()
def heldout_bits(model, seqs, max_len=200):
    model.eval(); tot = 0.0
    for s in seqs:
        x = torch.tensor([[BOS] + list(s[:max_len - 1])])
        logits = model(x[:, :-1])
        ll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1),
                             ignore_index=PAD, reduction="sum")
        tot += ll.item() / math.log(2)
    return tot / len(seqs)


def count_params(m):
    return sum(p.numel() for p in m.parameters())
