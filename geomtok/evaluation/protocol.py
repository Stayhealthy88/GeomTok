"""
GeomTok-Eval / 1.1 — 게임-내성 렌더 기반 평가 프로토콜 (PRD §7.5 / G4)
======================================================================
v1.1: (a) 좌표 오차 꼬리 지표 — 분위수 p50/p90/p95/p99 + 인지 가능(>2px)
비율 (PAPER §10 E6). (b) 렌더러 부재(None)와 렌더 파탄(0.0)을 구분 —
렌더러 없는 환경에서 SSIM 0.0 이 '충실도 붕괴'로 오독되는 것을 방지.

토크나이저 품질을 **생성기와 분리**하여 측정하는 공개 표준 프로토콜.
순진한 압축률은 게임 가능(랜덤 베이스라인 1.0)하므로, 본 프로토콜은
원본↔복원의 (1) 속성/좌표 충실도, (2) 요소 수 정확도, (3) 실제 렌더 SSIM,
(4) 토큰 경제(텍스트 BPE 대비)를 함께 본다.

생성기-분리(generator-separated): 측정 대상은 '토크나이저가 원본을 얼마나
충실히 왕복하는가'이지 '생성 모델이 얼마나 잘 그리는가'가 아니다. 따라서
랜덤/사기 토크나이저는 충실도에서 즉시 탄로난다.

내장(builtin) 토크나이저뿐 아니라 임의의 (tokenize_fn, detokenize_fn) 쌍을
평가할 수 있어 제3자·원격 토크나이저 벤치마킹(PRD UC-5)에 쓰인다.

SSIM 은 외부 의존(scikit-image) 없이 numpy 박스필터로 구현한다.
렌더는 cairosvg(우선) 또는 resvg 폴백.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# 텍스트 BPE 베이스라인 어휘 크기 (cl100k_base)
_BPE_VOCAB = 100256


# --------------------------------------------------------------------------- #
# 렌더링
# --------------------------------------------------------------------------- #

_RENDERER_PROBE = {"checked": False, "available": False}


def renderer_available() -> bool:
    """렌더러(cairosvg 또는 resvg) 사용 가능 여부 — 1회 프로브 후 캐시.

    '측정 불가(환경에 렌더러 없음)'와 '충실도 0(내용 파탄)'을 구분하는 단일
    진실 원천. 렌더러가 없으면 SSIM 은 None 으로 보고해야 하며 0.0 으로
    뭉개면 안 된다 (0.0 은 게임-내성 패널티 전용)."""
    if not _RENDERER_PROBE["checked"]:
        probe = ("<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 4 4'>"
                 "<path d='M0 0 L4 4' stroke='black'/></svg>")
        _RENDERER_PROBE["available"] = _render_gray(probe, 8) is not None
        _RENDERER_PROBE["checked"] = True
    return _RENDERER_PROBE["available"]


def _render_gray(svg: str, res: int = 128) -> Optional[np.ndarray]:
    """SVG → res×res 그레이스케일 [0,1] 배열. 실패 시 None.

    흰 배경에 합성 후 휘도화. 렌더러는 cairosvg 우선, resvg 폴백."""
    png = None
    try:
        import cairosvg
        png = cairosvg.svg2png(bytestring=svg.encode("utf-8"),
                               output_width=res, output_height=res,
                               background_color="white")
    except Exception:
        try:
            import resvg_py
            from PIL import Image
            raw = resvg_py.svg_to_bytes(svg_string=svg, width=res, height=res)
            from PIL import Image as _I
            rgba = _I.open(io.BytesIO(bytes(raw))).convert("RGBA")
            bg = _I.new("RGBA", rgba.size, (255, 255, 255, 255))
            return np.asarray(_I.alpha_composite(bg, rgba).convert("L"),
                              dtype=np.float64) / 255.0
        except Exception:
            return None
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(png)).convert("L")
        if img.size != (res, res):
            img = img.resize((res, res))
        return np.asarray(img, dtype=np.float64) / 255.0
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# SSIM (numpy, 의존성 없음)
# --------------------------------------------------------------------------- #

def _box_filter(img: np.ndarray, win: int) -> np.ndarray:
    """win×win 평균 필터 (edge 패딩, 적분영상으로 완전 벡터화)."""
    r = win // 2
    win = 2 * r + 1
    H, W = img.shape
    padded = np.pad(img, r, mode="edge")
    ii = np.zeros((H + 2 * r + 1, W + 2 * r + 1), dtype=np.float64)
    ii[1:, 1:] = np.cumsum(np.cumsum(padded, axis=0), axis=1)
    A = ii[win:win + H, win:win + W]
    B = ii[0:H, win:win + W]
    C = ii[win:win + H, 0:W]
    D = ii[0:H, 0:W]
    return (A - B - C + D) / float(win * win)


def ssim(a: np.ndarray, b: np.ndarray, win: int = 7) -> float:
    """두 그레이스케일 이미지([0,1])의 평균 SSIM (Wang et al. 2004).

    분야 표준 수치를 위해 scikit-image 의 정준 구현(가우시안 창)을 우선 사용하고,
    없으면 numpy 박스필터 폴백으로 동작한다 (OSS 코어는 numpy-only 유지)."""
    if a.shape != b.shape:
        return 0.0
    try:
        from skimage.metrics import structural_similarity as _sk
        return float(_sk(a, b, data_range=1.0))
    except Exception:
        pass
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2
    mu_a = _box_filter(a, win)
    mu_b = _box_filter(b, win)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    sa = _box_filter(a * a, win) - mu_a2
    sb = _box_filter(b * b, win) - mu_b2
    sab = _box_filter(a * b, win) - mu_ab
    num = (2 * mu_ab + C1) * (2 * sab + C2)
    den = (mu_a2 + mu_b2 + C1) * (sa + sb + C2)
    smap = num / np.maximum(den, 1e-12)
    return float(np.clip(smap.mean(), -1.0, 1.0))


def _restyle_stroke(svg: str, width: float) -> str:
    """SVG 의 stroke-width 를 일괄 치환 (SSIM 렌더 시 양측 동일 굵기 보장).

    얇은 1px 미만 스트로크는 안티앨리어싱으로 SSIM 변별력이 떨어지므로, 정답·
    복원 양측을 같은 가시 굵기로 렌더해 양자화 오차만 비교한다."""
    import re
    if 'stroke-width=' in svg:
        return re.sub(r'stroke-width="[^"]*"', f'stroke-width="{width}"', svg)
    # stroke-width 가 없으면 path/그룹에 주입
    return svg.replace('<path ', f'<path stroke-width="{width}" ')


# --------------------------------------------------------------------------- #
# 값/좌표 충실도 (value_fidelity 재사용)
# --------------------------------------------------------------------------- #

from .value_fidelity import _SHAPE_TAGS  # noqa: E402
from ..parser.svg_parser import SVGParser  # noqa: E402
from ..parser.path_parser import CommandType  # noqa: E402

_EVAL_PARSER = SVGParser(normalize_canvas=None)


def _svg_to_points(svg: str) -> List[Tuple[float, float]]:
    """SVG 를 절대좌표 점 시퀀스로 환원 (렌더-구조 독립 정렬용).

    H/V/도형이 모두 PathParser 로 절대 좌표로 환원되므로, 같은 기하를 기술하는
    두 SVG(정답·복원)는 동일 길이·순서의 점열을 만들어 위치별 L2 비교가 가능."""
    try:
        doc = _EVAL_PARSER.parse_string(svg)
    except Exception:
        return []
    from ..parser.arc_flatten import flatten_arcs
    pts: List[Tuple[float, float]] = []
    for elem in doc.elements:
        for cmd in flatten_arcs(elem.commands):
            ap = cmd.abs_params
            ct = cmd.command_type
            if ap is None or ct == CommandType.CLOSE:
                continue
            if ct in (CommandType.MOVE, CommandType.LINE,
                      CommandType.HLINE, CommandType.VLINE):
                pts.append((ap[0], ap[1]))
            elif ct == CommandType.CUBIC:
                pts += [(ap[0], ap[1]), (ap[2], ap[3]), (ap[4], ap[5])]
            elif ct == CommandType.QUADRATIC:
                pts += [(ap[0], ap[1]), (ap[2], ap[3])]
            elif ct == CommandType.ARC:
                pts += [(ap[0], ap[1]), (ap[5], ap[6])]
    return pts


def _attr_errors(gt_svg: str, recon_svg: str) -> Tuple[List[float], bool]:
    """정답↔복원 SVG 의 좌표 L2 오차 리스트 + 요소수 일치 여부.

    두 SVG 를 절대좌표 점열로 환원해 위치별 유클리드 거리(px)를 측정한다."""
    op = _svg_to_points(gt_svg)
    rp = _svg_to_points(recon_svg)
    errs = [math.hypot(ox - rx, oy - ry)
            for (ox, oy), (rx, ry) in zip(op, rp)]
    # 길이 불일치(기하 누락/과잉)는 게임-내성을 위해 패널티로 반영한다 — zip 절단이
    # 빈/짧은 복원을 '완벽'으로 둔갑시키지 않도록. 누락 점당 캔버스 대각(≈최대 오차).
    n_unmatched = abs(len(op) - len(rp))
    if n_unmatched:
        errs.extend([300.0 * (2 ** 0.5)] * n_unmatched)
    n_o = sum(gt_svg.count(f"<{t}") for t in _SHAPE_TAGS) + gt_svg.count("<path")
    n_r = sum(recon_svg.count(f"<{t}") for t in _SHAPE_TAGS) + recon_svg.count("<path")
    return errs, (n_o == n_r)


# --------------------------------------------------------------------------- #
# 프로토콜
# --------------------------------------------------------------------------- #

@dataclass
class SceneScore:
    name: str
    attr_mean_err: float
    attr_max_err: float
    count_match: bool
    render_ssim: Optional[float]   # None = 렌더러 부재(측정 불가), 0.0 = 파탄
    n_tokens: int
    n_tokens_bpe: Optional[int]


@dataclass
class GeomTokEvalProtocol:
    """GeomTok-Eval/1.1 하니스.

    Args:
        tokenize_fn   : svg(text) → token_ids
        detokenize_fn : token_ids → svg(text)
        gt_norm_fn    : (옵션) svg → 정규화 정답 SVG (SSIM/attr 비교 기준).
                        None 이면 원본 SVG 를 정답으로 사용.
        render_res    : SSIM 렌더 해상도.
    """
    tokenize_fn: Callable[[str], List[int]]
    detokenize_fn: Callable[[List[int]], str]
    gt_norm_fn: Optional[Callable[[str], str]] = None
    render_res: int = 256
    ssim_stroke: float = 3.0
    bpe_vocab: int = _BPE_VOCAB
    geom_vocab: int = 5561
    perceptible_px: float = 2.0   # 인지 가능 좌표 오차 임계 (E6 tail 지표)
    results: List[SceneScore] = field(default_factory=list)
    _pooled_errors: List[float] = field(default_factory=list, repr=False)

    def eval_scene(self, name: str, svg: str) -> SceneScore:
        token_ids = self.tokenize_fn(svg)
        recon = self.detokenize_fn(token_ids)
        gt = self.gt_norm_fn(svg) if self.gt_norm_fn else svg

        errs, count_match = _attr_errors(gt, recon)
        self._pooled_errors.extend(errs)

        # 렌더 SSIM (정답 정규화본 vs 복원본 — 양자화 오차만 분리).
        # 양측 stroke 굵기를 동일 가시값으로 맞춰 굵기차가 아닌 위치 오차만 본다.
        # 렌더러 부재(환경)는 None, 렌더 실패(내용)는 0.0 패널티 — 게임-내성 유지.
        if renderer_available():
            a = _render_gray(_restyle_stroke(gt, self.ssim_stroke), self.render_res)
            b = _render_gray(_restyle_stroke(recon, self.ssim_stroke), self.render_res)
            s: Optional[float] = ssim(a, b) if (a is not None and b is not None) else 0.0
        else:
            s = None

        # 텍스트 BPE 베이스라인 (원본 SVG 기준)
        try:
            import tiktoken
            n_bpe = len(tiktoken.get_encoding("cl100k_base").encode(svg))
        except Exception:
            n_bpe = None

        sc = SceneScore(
            name=name,
            attr_mean_err=(sum(errs) / len(errs)) if errs else 0.0,
            attr_max_err=max(errs) if errs else 0.0,
            count_match=count_match,
            render_ssim=s,
            n_tokens=len(token_ids),
            n_tokens_bpe=n_bpe,
        )
        self.results.append(sc)
        return sc

    def summary(self) -> Dict:
        rs = self.results
        if not rs:
            return {"protocol": "GeomTok-Eval/1.1", "summary": {"scenes": 0}}
        n_tok = sum(r.n_tokens for r in rs)
        n_bpe = sum((r.n_tokens_bpe or 0) for r in rs)
        # bits/icon: 토큰수 × log2(어휘). 정직한 정보량 비교 (PRD §2.3).
        bits_geom = (n_tok / len(rs)) * math.log2(self.geom_vocab)
        bits_bpe = (n_bpe / len(rs)) * math.log2(self.bpe_vocab) if n_bpe else 0.0
        # 렌더 SSIM: 렌더러 부재 장면(None)은 평균에서 제외 — 0.0 오염 금지.
        ssims = [r.render_ssim for r in rs if r.render_ssim is not None]
        out = {
            "protocol": "GeomTok-Eval/1.1",
            "summary": {
                "scenes": len(rs),
                "attr_mean_err": round(sum(r.attr_mean_err for r in rs) / len(rs), 4),
                "attr_max_err": round(max(r.attr_max_err for r in rs), 4),
                "count_acc": round(sum(r.count_match for r in rs) / len(rs), 4),
                "render_ssim_mean": (round(sum(ssims) / len(ssims), 4)
                                     if ssims else None),
                "render_scenes": len(ssims),
                "tokens": n_tok,
                "tokens_bpe": n_bpe,
                "compression_vs_baseline": round(n_bpe / n_tok, 4) if n_tok and n_bpe else 0.0,
                "bits_per_icon": round(bits_geom, 1),
                "bits_per_icon_bpe": round(bits_bpe, 1),
            },
        }
        # 좌표 오차 꼬리 (E6): 평균은 꼬리를 숨긴다 — 인지 가능(>2px) 비율과
        # 분위수를 함께 보고해야 "bounded-error" 주장이 검증 가능해진다.
        errs = self._pooled_errors
        if errs:
            arr = np.asarray(errs, dtype=np.float64)
            out["summary"].update({
                "coords_measured": int(arr.size),
                "coord_err_p50": round(float(np.percentile(arr, 50)), 4),
                "coord_err_p90": round(float(np.percentile(arr, 90)), 4),
                "coord_err_p95": round(float(np.percentile(arr, 95)), 4),
                "coord_err_p99": round(float(np.percentile(arr, 99)), 4),
                "perceptible_px": self.perceptible_px,
                "perceptible_err_rate": round(
                    float((arr > self.perceptible_px).mean()), 4),
            })
        return out


# --------------------------------------------------------------------------- #
# 편의 러너 (builtin / remote)
# --------------------------------------------------------------------------- #

def run_builtin_eval(svgs: List[str], names: Optional[List[str]] = None,
                     level: str = "L1", render_res: int = 256,
                     lean: bool = False) -> Dict:
    """내장 GeomTok 토크나이저로 장면 집합을 평가 (PRD /v1/eval builtin).

    lean=True 는 보조 마커(연속성·곡률) 없는 lean L1 스트림을 평가한다 —
    마커는 파생 가능(복원 무영향)하므로 충실도는 동일해야 한다 (PAPER §5.1)."""
    from ..api import GeomTokenizer
    gt = GeomTokenizer.default()
    proto = GeomTokEvalProtocol(
        tokenize_fn=lambda s: gt.tokenize(s, level=level, lean=lean)["token_ids"],
        detokenize_fn=lambda ids: gt.detokenize(
            ids, tokenizer_version=gt.tokenizer_version,
            vocab_id=gt.vocab_id, level=level)["svg"],
        gt_norm_fn=lambda s: gt.normalized_svg(s),
        render_res=render_res,
    )
    for i, svg in enumerate(svgs):
        nm = names[i] if names else f"scene_{i}"
        try:
            proto.eval_scene(nm, svg)
        except Exception:
            continue
    return proto.summary()


def run_remote_eval(svgs: List[str], tokenize_url: str, detokenize_url: str,
                    names: Optional[List[str]] = None,
                    render_res: int = 256, timeout: float = 30.0) -> Dict:
    """제3자(원격) 토크나이저를 동일 하니스로 평가 (PRD UC-5).

    원격 계약:
        POST tokenize_url   {svg} -> {token_ids}
        POST detokenize_url {token_ids} -> {svg}
    생성기-분리 프로토콜이므로 마케팅 주장이 아닌 게임-내성 점수로 비교된다."""
    import urllib.request
    import json as _json

    def _post(url, payload):
        req = urllib.request.Request(
            url, data=_json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return _json.loads(r.read().decode("utf-8"))

    proto = GeomTokEvalProtocol(
        tokenize_fn=lambda s: _post(tokenize_url, {"svg": s})["token_ids"],
        detokenize_fn=lambda ids: _post(detokenize_url, {"token_ids": ids})["svg"],
        gt_norm_fn=None,   # 원격은 정규화 기준이 없으므로 원본 대비
        render_res=render_res,
    )
    for i, svg in enumerate(svgs):
        nm = names[i] if names else f"scene_{i}"
        try:
            proto.eval_scene(nm, svg)
        except Exception:
            continue
    return proto.summary()
