"""
Arc → Cubic Bézier 평탄화 (정규화 단계)
=========================================
SVG 타원 호(A/a)를 큐빅 베지에 시퀀스로 변환한다. 호 반지름(rx,ry)은 캔버스를
초과할 수 있어(near-flat 호는 반지름이 수백 px) 위치 좌표로 토큰화하면 클램핑
손실이 난다. 호를 베지에로 평탄화하면 모든 좌표가 곡선의 실제 위치 범위 안에
들어와 좌표 토큰화가 무손실 도메인에 머문다.

W3C SVG Implementation Notes(F.6) 의 endpoint→center 변환 + 90° 분할
베지에 근사를 따른다.
"""

from __future__ import annotations

import math
from typing import List

from .path_parser import PathCommand, CommandType


def _arc_to_cubics(sx, sy, rx, ry, phi_deg, large_arc, sweep, ex, ey):
    """호를 큐빅 베지에 제어점 리스트로 변환.

    Returns: [(c1x,c1y,c2x,c2y,ex,ey), ...] — 각 큐빅의 절대 제어점·끝점.
    반지름 0 또는 시작=끝이면 빈 리스트(직선 처리는 호출측)."""
    if rx == 0 or ry == 0:
        return []
    if sx == ex and sy == ey:
        return []

    rx, ry = abs(rx), abs(ry)
    phi = math.radians(phi_deg % 360.0)
    cos_p, sin_p = math.cos(phi), math.sin(phi)

    # 1) 끝점 → 중심 파라미터화
    dx2 = (sx - ex) / 2.0
    dy2 = (sy - ey) / 2.0
    x1p = cos_p * dx2 + sin_p * dy2
    y1p = -sin_p * dx2 + cos_p * dy2

    # 반지름 보정 (스펙 F.6.6)
    lam = (x1p * x1p) / (rx * rx) + (y1p * y1p) / (ry * ry)
    if lam > 1:
        s = math.sqrt(lam)
        rx *= s
        ry *= s

    num = rx * rx * ry * ry - rx * rx * y1p * y1p - ry * ry * x1p * x1p
    den = rx * rx * y1p * y1p + ry * ry * x1p * x1p
    co = math.sqrt(max(0.0, num / den)) if den != 0 else 0.0
    if large_arc == sweep:
        co = -co
    cxp = co * (rx * y1p / ry)
    cyp = co * (-ry * x1p / rx)

    cx = cos_p * cxp - sin_p * cyp + (sx + ex) / 2.0
    cy = sin_p * cxp + cos_p * cyp + (sy + ey) / 2.0

    def angle(ux, uy, vx, vy):
        dot = ux * vx + uy * vy
        nrm = math.hypot(ux, uy) * math.hypot(vx, vy)
        a = math.acos(max(-1.0, min(1.0, dot / nrm))) if nrm else 0.0
        if ux * vy - uy * vx < 0:
            a = -a
        return a

    theta1 = angle(1, 0, (x1p - cxp) / rx, (y1p - cyp) / ry)
    dtheta = angle((x1p - cxp) / rx, (y1p - cyp) / ry,
                   (-x1p - cxp) / rx, (-y1p - cyp) / ry)
    if not sweep and dtheta > 0:
        dtheta -= 2 * math.pi
    elif sweep and dtheta < 0:
        dtheta += 2 * math.pi

    # 2) ≤90° 세그먼트로 분할 후 각 세그먼트를 큐빅 근사
    n_segs = max(1, int(math.ceil(abs(dtheta) / (math.pi / 2.0))))
    delta = dtheta / n_segs
    t = (4.0 / 3.0) * math.tan(delta / 4.0)

    cubics = []
    th = theta1
    px = sx
    py = sy
    for _ in range(n_segs):
        th2 = th + delta
        cos1, sin1 = math.cos(th), math.sin(th)
        cos2, sin2 = math.cos(th2), math.sin(th2)

        # 세그먼트 끝점
        ex_i = (cos_p * rx * cos2 - sin_p * ry * sin2) + cx
        ey_i = (sin_p * rx * cos2 + cos_p * ry * sin2) + cy

        # 제어점 (단위원 미분 × t)
        d1x = -rx * sin1
        d1y = ry * cos1
        d2x = -rx * sin2
        d2y = ry * cos2
        c1x = px + t * (cos_p * d1x - sin_p * d1y)
        c1y = py + t * (sin_p * d1x + cos_p * d1y)
        c2x = ex_i - t * (cos_p * d2x - sin_p * d2y)
        c2y = ey_i - t * (sin_p * d2x + cos_p * d2y)

        cubics.append((c1x, c1y, c2x, c2y, ex_i, ey_i))
        th = th2
        px, py = ex_i, ey_i

    return cubics


def flatten_arcs(commands: List[PathCommand]) -> List[PathCommand]:
    """명령 시퀀스의 모든 ARC 를 CUBIC(또는 직선 LINE)으로 치환.

    절대좌표(resolve_to_absolute) 완료된 명령을 입력으로 가정한다."""
    out: List[PathCommand] = []
    for cmd in commands:
        if cmd.command_type != CommandType.ARC or cmd.abs_params is None:
            out.append(cmd)
            continue
        ap = cmd.abs_params
        sx, sy = cmd.start_point if cmd.start_point else (0.0, 0.0)
        rx, ry, rot = ap[0], ap[1], ap[2]
        large, sweep = int(ap[3]), int(ap[4])
        ex, ey = ap[5], ap[6]
        cubics = _arc_to_cubics(sx, sy, rx, ry, rot, large, sweep, ex, ey)
        if not cubics:
            # 반지름 0/퇴화 → 직선
            out.append(PathCommand(
                CommandType.LINE, False, [ex, ey], abs_params=[ex, ey],
                start_point=(sx, sy), end_point=(ex, ey)))
            continue
        prev = (sx, sy)
        for (c1x, c1y, c2x, c2y, ix, iy) in cubics:
            out.append(PathCommand(
                CommandType.CUBIC, False,
                [c1x, c1y, c2x, c2y, ix, iy],
                abs_params=[c1x, c1y, c2x, c2y, ix, iy],
                start_point=prev, end_point=(ix, iy)))
            prev = (ix, iy)
    return out
