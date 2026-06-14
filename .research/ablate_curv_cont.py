"""Ablation: does L1 NEED curvature(16)+continuity(4) tokens?
Reuses f5_run's VanillaLM + train + heldout_bits EXACTLY. Adds an arm 'l1_strip'
that removes curvature/continuity token ids from the L1 stream.
"""
import sys, math, statistics
sys.path.insert(0, "/Users/limit/Projects/gpl-tokenizer")
sys.path.insert(0, "/Users/limit/Projects/gpl-tokenizer/.research")
import numpy as np, torch
import pandas as pd
from huggingface_hub import hf_hub_download
from geomtok.parser.svg_parser import SVGParser
from geomtok.tokenizer.primitive_tokenizer import PrimitiveTokenizer
from geomtok.tokenizer.vocabulary import CURVATURE_TOKEN_BASE, N_CURVATURE_BINS, ContinuityToken
from f5_run import VanillaLM, train, heldout_bits

CANVAS = 300.0
CURV = set(range(CURVATURE_TOKEN_BASE, CURVATURE_TOKEN_BASE + N_CURVATURE_BINS))
CONT = set(int(c) for c in ContinuityToken)
parser = SVGParser(normalize_canvas=CANVAS)
pt = PrimitiveTokenizer()


def l1_ids(svg):
    doc = parser.parse_string(svg)
    out = []
    for e in doc.elements:
        out += pt.tokenize(e.commands).token_ids
    return out


def load(n_train=1045, n_test=213, max_len=200):
    tr = pd.read_parquet(hf_hub_download("starvector/svg-icons", "data/train-00000-of-00001.parquet",
                                         repo_type="dataset")).head(n_train * 2)
    te = pd.read_parquet(hf_hub_download("starvector/svg-icons", "data/test-00000-of-00001.parquet",
                                         repo_type="dataset")).head(n_test * 2)

    def build(df, n):
        arms = {"l1": [], "l1_strip": []}
        for svg in df["Svg"]:
            try:
                ids = l1_ids(svg)
            except Exception:
                continue
            if not ids or len(ids) > max_len:
                continue
            arms["l1"].append(ids)
            arms["l1_strip"].append([i for i in ids if i not in CURV and i not in CONT])
            if len(arms["l1"]) >= n:
                break
        return arms

    return build(tr, n_train), build(te, n_test)


def main():
    trd, ted = load()
    print(f"train {len(trd['l1'])}, test {len(ted['l1'])} icons")
    print(f"{'arm':<10}{'vocab':>7}{'tok/icon':>10}{'heldout bits/icon (3 seed)':>30}")
    res = {}
    for arm in ("l1", "l1_strip"):
        V = 5561
        bits = []
        for seed in (0, 1, 2):
            torch.manual_seed(seed); np.random.seed(seed)
            model = VanillaLM(V).to("cpu")
            train(model, trd[arm], epochs=25, max_len=200)
            bits.append(heldout_bits(model, ted[arm], max_len=200))
        res[arm] = bits
        tpi = np.mean([len(s) for s in ted[arm]])
        m, sd = statistics.mean(bits), statistics.stdev(bits)
        print(f"{arm:<10}{V:>7}{tpi:>10.1f}      {m:>8.1f} +/- {sd:<6.1f}", flush=True)
    dl = statistics.mean(res['l1']); ds = statistics.mean(res['l1_strip'])
    print(f"\nL1-full {dl:.1f} vs L1-strip {ds:.1f}  (delta {ds-dl:+.1f} bits/icon)")
    print("If L1-strip <= L1-full within noise: curvature+continuity are UNJUSTIFIED overhead for modelability.")
    print("If L1-full clearly lower: the tokens earn their cost.")


if __name__ == "__main__":
    main()
