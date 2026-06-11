"""
SVG transform 평탄화 (v0.6 / E2)
==================================
transform 속성을 3x3 affine 행렬로 파싱하고, 그룹 상속을 누적 적용해
좌표를 평탄화한다. 토크나이저는 평탄화된 절대 좌표만 본다 — 실세계
SVG(Figma/Illustrator 내보내기)의 1차 차단 요인 제거.

ARC 제한: 회전·비등방 스케일 하의 타원호 정확 변환은 호→큐빅 변환이
필요하다. 토크나이저가 호 회전·플래그를 이미 버리므로(v0.5.1과 동일
손실 수준), 반지름은 sqrt(|det|) 등방 근사로 스케일한다.
"""

import math
import re
from typing import List, Optional, Tuple

import numpy as np

_CMD_RE = re.compile(r"(matrix|translate|scale|rotate|skewX|skewY)\s*\(([^)]*)\)")
_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def identity() -> np.ndarray:
    return np.eye(3)


def parse_transform(text: Optional[str]) -> np.ndarray:
    """transform 속성 문자열 → 3x3 행렬. 좌→우 순서로 곱한다 (SVG 규격)."""
    M = identity()
    if not text:
        return M
    for name, args in _CMD_RE.findall(text):
        v = [float(n) for n in _NUM_RE.findall(args)]
        T = identity()
        if name == "matrix" and len(v) >= 6:
            a, b, c, d, e, f = v[:6]
            T = np.array([[a, c, e], [b, d, f], [0, 0, 1]], dtype=float)
        elif name == "translate":
            tx = v[0] if v else 0.0
            ty = v[1] if len(v) > 1 else 0.0
            T[0, 2], T[1, 2] = tx, ty
        elif name == "scale":
            sx = v[0] if v else 1.0
            sy = v[1] if len(v) > 1 else sx
            T[0, 0], T[1, 1] = sx, sy
        elif name == "rotate":
            a = math.radians(v[0]) if v else 0.0
            ca, sa = math.cos(a), math.sin(a)
            R = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]], dtype=float)
            if len(v) >= 3:
                cx, cy = v[1], v[2]
                Tp, Tm = identity(), identity()
                Tp[0, 2], Tp[1, 2] = cx, cy
                Tm[0, 2], Tm[1, 2] = -cx, -cy
                R = Tp @ R @ Tm
            T = R
        elif name == "skewX":
            T[0, 1] = math.tan(math.radians(v[0])) if v else 0.0
        elif name == "skewY":
            T[1, 0] = math.tan(math.radians(v[0])) if v else 0.0
        M = M @ T
    return M


def viewbox_matrix(viewbox: Optional[Tuple[float, float, float, float]],
                   canvas_size: float) -> np.ndarray:
    """viewBox → [0, canvas)² 등방 스케일 정규화 행렬 (종횡비 보존, 중앙 정렬)."""
    M = identity()
    if not viewbox:
        return M
    vx, vy, vw, vh = viewbox
    if vw <= 0 or vh <= 0:
        return M
    s = canvas_size / max(vw, vh)
    ox = (canvas_size - vw * s) / 2.0
    oy = (canvas_size - vh * s) / 2.0
    M[0, 0] = M[1, 1] = s
    M[0, 2] = ox - vx * s
    M[1, 2] = oy - vy * s
    return M


def is_identity(M: np.ndarray, tol: float = 1e-9) -> bool:
    return bool(np.allclose(M, np.eye(3), atol=tol))


def _pt(M: np.ndarray, x: float, y: float) -> Tuple[float, float]:
    p = M @ np.array([x, y, 1.0])
    return float(p[0]), float(p[1])


def apply_to_commands(commands: List, M: np.ndarray) -> List:
    """절대좌표가 해석된 PathCommand 리스트에 행렬을 in-place 적용."""
    if is_identity(M):
        return commands
    # ARC 반지름용 등방 스케일 근사
    det = abs(M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0])
    r_scale = math.sqrt(det) if det > 0 else 1.0

    for cmd in commands:
        p = cmd.abs_params
        ct = cmd.command_type.value
        if p:
            if ct in ("M", "L", "H", "V"):
                p[0], p[1] = _pt(M, p[0], p[1])
            elif ct in ("C", "S"):
                for i in (0, 2, 4):
                    p[i], p[i + 1] = _pt(M, p[i], p[i + 1])
            elif ct in ("Q", "T"):
                for i in (0, 2):
                    p[i], p[i + 1] = _pt(M, p[i], p[i + 1])
            elif ct == "A" and len(p) >= 7:
                p[0] *= r_scale
                p[1] *= r_scale
                p[5], p[6] = _pt(M, p[5], p[6])
        if cmd.start_point:
            cmd.start_point = _pt(M, *cmd.start_point)
        if cmd.end_point:
            cmd.end_point = _pt(M, *cmd.end_point)
    return commands
