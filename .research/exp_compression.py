"""실험 1: 실제 BPE 토크나이저 대비 GPL 압축률 측정.

기존 주장 (README): GID 2-3x vs BPE, L2 5.6x, L3 최대 15x.
기존 측정 방식: bpe_est = len(svg_text)/4 (실제 토크나이저 미사용).
이 실험: tiktoken cl100k_base(GPT-4) / o200k_base(GPT-4o)로 실측 비교.
"""
import sys
sys.path.insert(0, "/Users/limit/Projects")

import tiktoken
from gpl_tokenizer.parser.svg_parser import SVGParser
from gpl_tokenizer.tokenizer.primitive_tokenizer import PrimitiveTokenizer
from gpl_tokenizer.tokenizer.composite_tokenizer import CompositeTokenizer
from gpl_tokenizer.tokenizer.spatial_tokenizer import SpatialTokenizer

CASES = {
    "circle_1": '<svg viewBox="0 0 300 300"><circle cx="150" cy="150" r="50"/></svg>',
    "rect_1": '<svg viewBox="0 0 300 300"><rect x="50" y="50" width="200" height="100"/></svg>',
    "circles_5_row": '<svg viewBox="0 0 300 300">' + "".join(
        f'<circle cx="{50 + i * 50}" cy="150" r="20"/>' for i in range(5)) + "</svg>",
    "circles_7_row": '<svg viewBox="0 0 300 300">' + "".join(
        f'<circle cx="{30 + i * 40}" cy="150" r="15"/>' for i in range(7)) + "</svg>",
    "icon_path": ('<svg viewBox="0 0 300 300"><path d="M 50 150 C 50 94.77 94.77 50 150 50 '
                  'C 205.23 50 250 94.77 250 150 C 250 205.23 205.23 250 150 250 '
                  'C 94.77 250 50 205.23 50 150 Z"/></svg>'),
    "mixed_scene": ('<svg viewBox="0 0 300 300"><rect x="20" y="20" width="60" height="60"/>'
                    '<rect x="120" y="20" width="60" height="60"/>'
                    '<rect x="220" y="20" width="60" height="60"/>'
                    '<circle cx="150" cy="200" r="40"/></svg>'),
}

enc4 = tiktoken.get_encoding("cl100k_base")
enc4o = tiktoken.get_encoding("o200k_base")
parser = SVGParser()

print(f"{'case':<16}{'chars':>6}{'cl100k':>8}{'o200k':>7}{'old_est':>8}{'L1':>5}{'L2':>5}{'L3':>5}"
      f"{'L3 vs cl100k':>13}{'L3 vs old_est':>14}")
rows = []
for name, svg in CASES.items():
    doc = parser.parse_string(svg)
    l1 = sum(len(PrimitiveTokenizer().tokenize(e.commands).token_ids) for e in doc.elements)
    l2 = sum(len(CompositeTokenizer().tokenize(e.commands).token_ids) for e in doc.elements)
    try:
        l3 = len(SpatialTokenizer().tokenize_multi([e.commands for e in doc.elements]).token_ids)
    except Exception:
        l3 = -1
    n4, n4o, est = len(enc4.encode(svg)), len(enc4o.encode(svg)), len(svg) // 4
    r_real = n4 / l3 if l3 > 0 else float("nan")
    r_old = est / l3 if l3 > 0 else float("nan")
    rows.append((n4, l3))
    print(f"{name:<16}{len(svg):>6}{n4:>8}{n4o:>7}{est:>8}{l1:>5}{l2:>5}{l3:>5}{r_real:>12.2f}x{r_old:>13.2f}x")

tot4 = sum(r[0] for r in rows)
tot3 = sum(r[1] for r in rows if r[1] > 0)
print(f"\n전체: cl100k {tot4} tok vs GPL-L3 {tot3} tok → 실측 압축 {tot4 / tot3:.2f}x")
