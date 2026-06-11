"""실험 3: GeomTok-Eval 프로토콜로 v0.5.1 vs v0.6 값-수준 충실도 비교."""
import sys
sys.path.insert(0, "/Users/limit/Projects/gpl-tokenizer")

from geomtok.parser.svg_parser import SVGParser
from geomtok.tokenizer.spatial_tokenizer import SpatialTokenizer
from geomtok.tokenizer.detokenizer import Detokenizer
from geomtok.evaluation import GeomTokEval

SCENES = {
    "circle_r20": '<svg viewBox="0 0 300 300"><circle cx="150" cy="150" r="20"/></svg>',
    "circles_5": '<svg viewBox="0 0 300 300">' + "".join(
        f'<circle cx="{50 + i * 50}" cy="150" r="20"/>' for i in range(5)) + '</svg>',
    "circles_7": '<svg viewBox="0 0 300 300">' + "".join(
        f'<circle cx="{30 + i * 40}" cy="150" r="15"/>' for i in range(7)) + '</svg>',
    "rect_80x40": '<svg viewBox="0 0 300 300"><rect x="50" y="50" width="80" height="40"/></svg>',
    "ellipse": '<svg viewBox="0 0 300 300"><ellipse cx="150" cy="150" rx="60" ry="25"/></svg>',
    "mixed": ('<svg viewBox="0 0 300 300"><rect x="20" y="20" width="60" height="60"/>'
              '<rect x="120" y="20" width="60" height="60"/>'
              '<rect x="220" y="20" width="60" height="60"/>'
              '<circle cx="150" cy="200" r="40"/></svg>'),
}

parser = SVGParser()
st = SpatialTokenizer()
detok = Detokenizer(st.vocab, st.arcs)

ev = GeomTokEval(
    tokenize_fn=lambda svg: st.tokenize_multi(
        [e.commands for e in parser.parse_string(svg).elements]).token_ids,
    detokenize_fn=lambda ids: detok.to_svg_document(ids),
)

print(f"{'scene':<12}{'max_err':>9}{'mean_err':>9}{'count':>7}{'tok':>5}{'bpe':>5}")
for name, svg in SCENES.items():
    r = ev.eval_scene(name, svg)
    print(f"{name:<12}{r.attr_max_err:>9.3f}{r.attr_mean_err:>9.3f}"
          f"{'OK' if r.count_match else 'FAIL':>7}{r.n_tokens:>5}{r.n_tokens_bpe:>5}")

s = ev.summary()
print(f"\nGeomTok-Eval 요약: attr_max_err={s['attr_max_err']:.3f}px "
      f"mean={s['attr_mean_err']:.3f}px count_acc={s['count_acc']:.0%} "
      f"압축 {s['compression']:.2f}x (BPE {s['tokens_bpe']} → GPL {s['tokens']})")
