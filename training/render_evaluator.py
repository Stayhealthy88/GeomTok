"""
렌더 기반 생성 품질 평가기 (v0.6 / E4)
=========================================
기존 GPLEvaluator의 구조/기하 점수는 무작위 토큰열에 1.000을 부여하는
게임 가능한 지표였다(2026-06 감사). 본 평가기는 분야 표준(HiVG·OmniSVG·
StarVector)인 렌더 기반 측정으로 교체한다.

지표:
    grammar_valid : FSA 문법 + PathParser 파싱 통과율
    renderable    : resvg 렌더 성공률
    ink_ratio     : 평균 잉크 비율 (빈/포화 출력 검출, 적정 0.01~0.6)
    non_degenerate: 잉크 0.5%~60% 출력 비율 — '백지 아님'
    diversity     : 잉크 마스크 64×64 다운샘플의 평균 쌍별 해밍 거리
"""

import io
from typing import Dict, List

import numpy as np

from .generator import GeneratedSVG


def _render_gray(svg: str, res: int = 128):
    import resvg_py
    from PIL import Image
    png = resvg_py.svg_to_bytes(svg_string=svg, width=res, height=res)
    rgba = Image.open(io.BytesIO(bytes(png))).convert("RGBA")
    bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    return np.asarray(Image.alpha_composite(bg, rgba).convert("L"), dtype=np.float64) / 255.0


class RenderEvaluator:
    def evaluate(self, samples: List[GeneratedSVG]) -> Dict[str, float]:
        if not samples:
            return {}
        n = len(samples)
        renderable = 0
        ink_ratios = []
        masks = []
        for s in samples:
            try:
                img = _render_gray(s.svg_full)
            except Exception:
                continue
            renderable += 1
            ink = img < 0.9
            ink_ratios.append(ink.mean())
            m = ink.reshape(64, 2, 64, 2).any(axis=(1, 3))
            masks.append(m)

        non_degen = sum(1 for r in ink_ratios if 0.005 <= r <= 0.6)
        div = 0.0
        if len(masks) >= 2:
            ds = [np.mean(masks[i] != masks[j])
                  for i in range(len(masks)) for j in range(i + 1, min(i + 6, len(masks)))]
            div = float(np.mean(ds))

        return {
            "n": n,
            "grammar_valid": sum(1 for s in samples if s.is_valid) / n,
            "renderable": renderable / n,
            "mean_ink": float(np.mean(ink_ratios)) if ink_ratios else 0.0,
            "non_degenerate": non_degen / n,
            "diversity": div,
        }
