"""카메라레디: 각 실험이 실제로 사용한 아이콘 split manifest 생성.
각 실험의 정확한 로더(head 배수·필터)를 복제해 Filename 목록을 기록한다:
  - F5/§5.1·5.3·5.4 : f5_data.load → head(n*2), 0<L1<=160 (n=1500/300)
  - E1/E2           : e_common.load_corpus svg-icons → head(n*3), 0<L1<=160 (n=1000/200)
  - E3              : e_common.load_corpus svg-emoji → head(n*3), 0<L1<=160 (n=1000/200)
  - A               : exp_A 직접 head(N), parse 성공만 (N=1200/400)
"""
import sys, os
sys.path.insert(0, "/Users/limit/Projects/gpl-tokenizer")
import pandas as pd
from huggingface_hub import hf_hub_download
from geomtok.parser.svg_parser import SVGParser
from geomtok.tokenizer.primitive_tokenizer import PrimitiveTokenizer

OUT = "/Users/limit/Projects/gpl-tokenizer/.research/splits"
os.makedirs(OUT, exist_ok=True)
P = SVGParser(normalize_canvas=300.0)


def l1len(svg):
    doc = P.parse_string(svg)
    return sum(len(PrimitiveTokenizer().tokenize(e.commands).token_ids) for e in doc.elements), len(doc.elements)


def grab(ds, split, mult):
    return pd.read_parquet(hf_hub_download(ds, f"data/{split}-00000-of-00001.parquet", repo_type="dataset")).head(mult)


def split_lenfilter(ds, mult_fn, n_train, n_test, max_len):
    """head(n*mult) 후 0<L1<=max_len 필터, n개 수집. mult_fn(n)→head."""
    res = {}
    for split, n in (("train", n_train), ("test", n_test)):
        df = grab(ds, split, mult_fn(n)); names = []
        for fn, svg in zip(df["Filename"], df["Svg"]):
            try:
                ln, ne = l1len(svg)
            except Exception:
                continue
            if ne and 0 < ln <= max_len:
                names.append(fn)
            if len(names) >= n:
                break
        res[split] = names
    return res


def split_parseonly(ds, n_train, n_test):
    """exp_A: 직접 head(N), parse 성공(L1>0)만."""
    res = {}
    for split, n in (("train", n_train), ("test", n_test)):
        df = grab(ds, split, n); names = []
        for fn, svg in zip(df["Filename"], df["Svg"]):
            try:
                ln, ne = l1len(svg)
            except Exception:
                continue
            if ne and ln:
                names.append(fn)
        res[split] = names
    return res


def write(name, splits):
    for split, names in splits.items():
        open(f"{OUT}/{name}_{split}.txt", "w").write("\n".join(names) + "\n")
        print(f"{name}_{split}: {len(names)} icons")


write("f5_icons",  split_lenfilter("starvector/svg-icons", lambda n: n * 2, 1500, 300, 160))   # F5 → 1045/213
write("e12_icons", split_lenfilter("starvector/svg-icons", lambda n: n * 3, 1000, 200, 160))   # E1/E2
write("e3_emoji",  split_lenfilter("starvector/svg-emoji", lambda n: n * 3, 1000, 200, 160))   # E3 → 401/48
write("expA_icons", split_parseonly("starvector/svg-icons", 1200, 400))                         # A → 1200/400
print("\nsplit manifests → .research/splits/  (counts must match the experiments' archived results)")
