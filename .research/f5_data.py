"""F5 데이터: 실세계 아이콘을 arm별 토큰 스트림으로 (char-BPE / L1 / L1+BPE)."""
import sys
sys.path.insert(0, "/Users/limit/Projects")

import pandas as pd
import sentencepiece as spm
from huggingface_hub import hf_hub_download
from gpl_tokenizer.parser.svg_parser import SVGParser
from gpl_tokenizer.tokenizer.primitive_tokenizer import PrimitiveTokenizer

CANVAS = 300.0
_parser = SVGParser(normalize_canvas=CANVAS)


def _l1(svg):
    doc = _parser.parse_string(svg)
    out = []
    for e in doc.elements:
        out += PrimitiveTokenizer().tokenize(e.commands).token_ids
    return out


def load(n_train=1500, n_test=300, max_len=200):
    tr = pd.read_parquet(hf_hub_download("starvector/svg-icons", "data/train-00000-of-00001.parquet",
                                         repo_type="dataset")).head(n_train * 2)
    te = pd.read_parquet(hf_hub_download("starvector/svg-icons", "data/test-00000-of-00001.parquet",
                                         repo_type="dataset")).head(n_test * 2)
    sp_text = spm.SentencePieceProcessor(model_file="/tmp/svg_bpe_5561.model")
    sp_l1 = spm.SentencePieceProcessor(model_file="/tmp/l1_bpe.model")

    def build(df, n):
        arms = {"char_bpe": [], "l1": [], "l1_bpe": [], "svg": []}
        for svg in df["Svg"]:
            try:
                ids = _l1(svg)
            except Exception:
                continue
            if not ids or len(ids) > max_len:
                continue
            arms["l1"].append(ids)
            arms["char_bpe"].append(sp_text.encode(svg)[:max_len])
            arms["l1_bpe"].append(sp_l1.encode("".join(chr(0x4E00 + i) for i in ids)))
            arms["svg"].append(svg)
            if len(arms["l1"]) >= n:
                break
        return arms

    return build(tr, n_train), build(te, n_test)


VOCAB = {"char_bpe": 5561, "l1": 5561, "l1_bpe": 7000}
