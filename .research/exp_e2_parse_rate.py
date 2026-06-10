"""실험 4 (E2): 실세계 SVG 코퍼스 파싱 성공률 + 토큰화/왕복 성공률.

코퍼스: starvector/svg-icons test split (SVG-Bench의 SVG-Icons, 2,682개 실세계 아이콘).
성공 기준: parse rate >= 95% (E2 목표).
"""
import sys, traceback, collections
sys.path.insert(0, "/Users/limit/Projects")

import pandas as pd
from huggingface_hub import hf_hub_download
from gpl_tokenizer.parser.svg_parser import SVGParser
from gpl_tokenizer.tokenizer.spatial_tokenizer import SpatialTokenizer
from gpl_tokenizer.tokenizer.detokenizer import Detokenizer

p = hf_hub_download("starvector/svg-icons", "data/test-00000-of-00001.parquet", repo_type="dataset")
df = pd.read_parquet(p)
print(f"코퍼스: {len(df)}개")

parser = SVGParser(normalize_canvas=300.0)
errors = collections.Counter()
parsed = empty = tok_ok = rt_ok = 0
n_cmds_total = n_tok_total = 0

for i, svg in enumerate(df["Svg"]):
    try:
        doc = parser.parse_string(svg)
        cmds = sum(len(e.commands) for e in doc.elements)
        if not doc.elements or cmds == 0:
            empty += 1
            errors["EMPTY"] += 1
            continue
        parsed += 1
        n_cmds_total += cmds
    except Exception as ex:
        errors[type(ex).__name__] += 1
        continue
    try:
        st = SpatialTokenizer()
        res = st.tokenize_multi([e.commands for e in doc.elements])
        tok_ok += 1
        n_tok_total += len(res.token_ids)
        svg_out = Detokenizer(st.vocab, st.arcs).to_svg_document(res.token_ids)
        if svg_out and ("<path" in svg_out or "<circle" in svg_out or "<rect" in svg_out or "<ellipse" in svg_out):
            rt_ok += 1
    except Exception as ex:
        errors["TOK_" + type(ex).__name__] += 1

n = len(df)
print(f"\n파싱 성공: {parsed}/{n} = {parsed/n:.1%}  (목표 95%)")
print(f"토큰화 성공: {tok_ok}/{n} = {tok_ok/n:.1%}")
print(f"왕복(비어있지 않은 SVG): {rt_ok}/{n} = {rt_ok/n:.1%}")
if parsed:
    print(f"평균 명령 수: {n_cmds_total/parsed:.1f}, 평균 토큰 수: {n_tok_total/max(tok_ok,1):.1f}")
print(f"오류 상위: {errors.most_common(8)}")
