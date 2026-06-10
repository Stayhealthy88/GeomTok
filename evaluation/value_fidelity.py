"""
GeomTok-Eval — 기하 토크나이저 내재 평가 프로토콜 (v0.6 시드)
=============================================================

기하 토크나이저 분야엔 표준 내재 지표가 없다 (압축률 단독은 ACL 2024,
arXiv:2403.06265 기준 불충분). 이 모듈은 토크나이저-비종속 프로토콜의
첫 두 축을 구현한다:

  1. 값-수준 충실도 (Value-level Fidelity, VF)
     렌더링 없이, SVG 속성 값(중심·반지름·크기·요소 수)의 왕복 오차를 직접
     측정한다. 좌표만 보는 기존 fidelity 와 달리 스칼라·카운트 경로 손실
     (v0.5.1 의 r=20→37.5, 7개→38개 파탄)을 정확히 잡아낸다.

  2. 토큰 효율 (tokens/scene)
     실제 BPE(tiktoken cl100k)와 동일 장면 비교. chars/4 휴리스틱 금지.

프로토콜 출력 (장면 집합 평균):
  - attr_max_err  : 속성값 최대 절대 오차 (px) — 무손실 주장 시 0
  - attr_mean_err : 속성값 평균 오차
  - count_acc     : 요소 개수 정확도 (생성 개수 == 원본 개수)
  - tokens        : 토큰 수 (tokenizer)
  - tokens_bpe    : 토큰 수 (cl100k 참조)
  - compression   : tokens_bpe / tokens
"""

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

_NUM = r'([\d.]+)'
_SHAPE_TAGS = ("circle", "ellipse", "rect")
_ATTRS = ("cx", "cy", "r", "rx", "ry", "x", "y", "width", "height")


def _extract_values(svg: str) -> List[Tuple[str, str, float]]:
    """SVG 문자열에서 (태그, 속성, 값) 트리플 추출 — 비교 가능한 정규형."""
    values = []
    for tag in _SHAPE_TAGS:
        for elem in re.findall(rf'<{tag}\b[^>]*>', svg):
            for a in _ATTRS:
                m = re.search(rf'[ "]{a}="{_NUM}"', elem)
                if m:
                    values.append((tag, a, float(m.group(1))))
    return values


@dataclass
class SceneResult:
    name: str
    attr_max_err: float
    attr_mean_err: float
    count_match: bool
    n_tokens: int
    n_tokens_bpe: Optional[int]


@dataclass
class GeomTokEval:
    """tokenize_fn: svg → token_ids, detokenize_fn: token_ids → svg."""
    tokenize_fn: Callable[[str], List[int]]
    detokenize_fn: Callable[[List[int]], str]
    results: List[SceneResult] = field(default_factory=list)

    def eval_scene(self, name: str, svg: str) -> SceneResult:
        token_ids = self.tokenize_fn(svg)
        recon = self.detokenize_fn(token_ids)

        orig = _extract_values(svg)
        rec = _extract_values(recon)

        # 태그·속성별 정렬 매칭 (요소 순서 보존 가정, 개수 불일치는 별도 지표)
        errs = []
        by_key_o: Dict[Tuple[str, str], List[float]] = {}
        by_key_r: Dict[Tuple[str, str], List[float]] = {}
        for t, a, v in orig:
            by_key_o.setdefault((t, a), []).append(v)
        for t, a, v in rec:
            by_key_r.setdefault((t, a), []).append(v)
        for key, ov in by_key_o.items():
            rv = sorted(by_key_r.get(key, []))
            for o, r in zip(sorted(ov), rv):
                errs.append(abs(o - r))

        n_orig = sum(svg.count(f"<{t}") for t in _SHAPE_TAGS)
        n_rec = sum(recon.count(f"<{t}") for t in _SHAPE_TAGS)

        try:
            import tiktoken
            n_bpe = len(tiktoken.get_encoding("cl100k_base").encode(svg))
        except ImportError:
            n_bpe = None

        res = SceneResult(
            name=name,
            attr_max_err=max(errs) if errs else float("inf"),
            attr_mean_err=sum(errs) / len(errs) if errs else float("inf"),
            count_match=(n_orig == n_rec),
            n_tokens=len(token_ids),
            n_tokens_bpe=n_bpe,
        )
        self.results.append(res)
        return res

    def summary(self) -> Dict[str, float]:
        rs = self.results
        n_bpe = sum(r.n_tokens_bpe or 0 for r in rs)
        n_tok = sum(r.n_tokens for r in rs)
        return {
            "scenes": len(rs),
            "attr_max_err": max(r.attr_max_err for r in rs),
            "attr_mean_err": sum(r.attr_mean_err for r in rs) / len(rs),
            "count_acc": sum(r.count_match for r in rs) / len(rs),
            "tokens": n_tok,
            "tokens_bpe": n_bpe,
            "compression": n_bpe / n_tok if n_tok else 0.0,
        }
