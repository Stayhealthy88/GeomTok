"""
GeomTok FSA 문법 — torch-free 코어 (PRD G3: 유효성 보장 디코딩)
================================================================
GPL 토큰 문법을 유한상태오토마타(FSA)로 추적하여 (1) 다음 허용 토큰 마스크를
산출하고 (2) 토큰 스트림이 well-formed 인지 검증한다. CAD-Tokenizer
(arXiv:2509.21150)의 FSA 제약이 무효율 80%→8%로 줄인 선례를 따른다.

이 모듈은 **순수 파이썬(+numpy 옵션)** 으로 구현되어 OSS 코어(install_requires
=numpy)에서 그대로 동작한다. torch 는 의존하지 않는다. 학습 경로의
`training/grammar.py` 는 이 코어를 감싸 로짓 마스킹용 텐서를 만든다.

문법 (Detokenizer 해석 규약과 일치):
    BOS      → CMD | SHAPE | SPATIAL | SEP | EOS
    CMD(k)   → coord × n(k)  [→ CONT? → CURV?]  → BOUNDARY
    SHAPE(s) → coord(중심/원점) → coord_L6 × m(s)(스칼라) → BOUNDARY
    SPATIAL  → 인자형(60-65,70)은 coord 1개, 마커형(66-69)은 0개 → BOUNDARY
    BOUNDARY → CMD | SHAPE | SPATIAL | SEP | EOS
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

# 명령별 coord 인자 수 (MOVE,LINE,HLINE,VLINE=1; CUBIC=3; QUAD=2; ARC=2; CLOSE=0)
_CMD_COORDS = {10: 1, 11: 1, 12: 1, 13: 1, 14: 3, 15: 2, 16: 2, 17: 0}
# 도형: 중심 coord 1개 이후 스칼라 좌표 수
_SHAPE_SCALARS = {20: 1, 21: 2, 22: 2, 23: 4}
# 공간 관계: 인자형은 coord 1개, 마커형은 0개
_SPATIAL_ARG = {60: 1, 61: 1, 62: 1, 63: 1, 64: 1, 65: 1,
                66: 0, 67: 0, 68: 0, 69: 0, 70: 1}

_BOS, _EOS, _SEP = 1, 2, 3
_CONT_LO, _CONT_HI = 30, 34          # 연속성 [30,34)
_CURV_LO, _CURV_HI = 40, 56          # 곡률 [40,56)


@dataclass
class ValidationResult:
    valid: bool
    position: int = -1          # 첫 위반 위치 (없으면 -1)
    reason: str = ""
    n_tokens: int = 0


class GrammarFSA:
    """GPL 토큰 문법 FSA. 마스킹·검증의 단일 진실 원천 (torch-free)."""

    def __init__(self, vocab):
        self.vocab = vocab
        self.V = vocab.vocab_size
        # 좌표 토큰 ID 집합 (해석은 vocab 권위)
        self._coord_ids: Set[int] = {
            tid for tid in range(self.V) if vocab.id_to_coord(tid) is not None
        }
        self._cont_ids = set(range(_CONT_LO, _CONT_HI))
        self._curv_ids = set(range(_CURV_LO, _CURV_HI))
        self._boundary_ids = (
            set(_CMD_COORDS) | set(_SHAPE_SCALARS) | set(_SPATIAL_ARG)
            | {_EOS, _SEP}
        )

    # ---- 상태 전이 (마스크·검증 공용) ---- #

    @staticmethod
    def _advance(t: int, pending: int, state: str,
                 in_path_cmd: bool) -> Tuple[int, str, bool]:
        """토큰 t 를 소비하여 (pending, state, in_path_cmd) 갱신."""
        if pending > 0:
            pending -= 1
            state = "post" if pending == 0 else "args"
            return pending, state, in_path_cmd
        if t in _CMD_COORDS:
            pending = _CMD_COORDS[t]
            return pending, ("args" if pending else "post"), True
        if t in _SHAPE_SCALARS:
            return 1 + _SHAPE_SCALARS[t], "args", False
        if t in _SPATIAL_ARG:
            pending = _SPATIAL_ARG[t]
            return pending, ("args" if pending else "boundary"), False
        if _CONT_LO <= t < _CONT_HI:
            return 0, "curv", in_path_cmd
        if _CURV_LO <= t < _CURV_HI:
            return 0, "boundary", in_path_cmd
        return 0, "boundary", False     # SEP/EOS/기타

    def _allowed_ids(self, pending: int, state: str,
                     in_path_cmd: bool) -> Set[int]:
        """현재 상태에서 허용되는 토큰 ID 집합."""
        if pending > 0:
            return self._coord_ids
        if state == "post":
            # 경로 명령 뒤에는 연속성·곡률 토큰이 각각 독립적으로(둘 다 옵션)
            # 올 수 있다 — 토크나이저는 cont 없이 curv 만 붙이기도 한다.
            ids = set(self._boundary_ids)
            if in_path_cmd:
                ids |= self._cont_ids | self._curv_ids
            return ids
        if state == "curv":   # 연속성 토큰 직후 — 곡률 또는 경계
            return self._curv_ids | self._boundary_ids
        return set(self._boundary_ids)   # boundary

    # ---- 공개 API ---- #

    def allowed_ids(self, prefix: List[int]) -> Set[int]:
        """접두사 다음에 허용되는 토큰 ID 집합. prefix 는 BOS 로 시작 가정."""
        pending, state, in_path_cmd = 0, "boundary", False
        for t in prefix[1:]:
            pending, state, in_path_cmd = self._advance(t, pending, state, in_path_cmd)
        return self._allowed_ids(pending, state, in_path_cmd)

    def allowed_mask(self, prefix: List[int]):
        """접두사 다음 허용 토큰의 boolean numpy 마스크 (길이 V)."""
        import numpy as np
        mask = np.zeros(self.V, dtype=bool)
        for tid in self.allowed_ids(prefix):
            if 0 <= tid < self.V:
                mask[tid] = True
        return mask

    def validate(self, token_ids: List[int],
                 require_bos_eos: bool = True) -> ValidationResult:
        """토큰 스트림이 문법에 맞는지 단일 패스 O(n) 검증.

        Args:
            token_ids: 검사할 토큰 ID 시퀀스.
            require_bos_eos: BOS 로 시작하고 EOS 로 끝나야 하는지 여부.
        Returns:
            ValidationResult(valid, position, reason, n_tokens)
        """
        n = len(token_ids)
        if n == 0:
            return ValidationResult(False, 0, "empty stream", 0)

        start = 0
        if require_bos_eos:
            if token_ids[0] != _BOS:
                return ValidationResult(False, 0, "missing BOS", n)
            start = 1

        pending, state, in_path_cmd = 0, "boundary", False
        for i in range(start, n):
            t = token_ids[i]
            if t == _BOS:
                return ValidationResult(False, i, "unexpected BOS", n)
            if t == _EOS:
                # EOS 는 boundary/post/curv 의 비-인자 위치에서만 허용
                if pending > 0:
                    return ValidationResult(False, i, "EOS while args pending", n)
                # EOS 이후 토큰이 더 있으면 위반
                if i != n - 1:
                    return ValidationResult(False, i, "tokens after EOS", n)
                return ValidationResult(True, -1, "", n)
            allowed = self._allowed_ids(pending, state, in_path_cmd)
            if t not in allowed:
                return ValidationResult(
                    False, i, f"token {t} not allowed in state "
                              f"(pending={pending},state={state})", n)
            pending, state, in_path_cmd = self._advance(t, pending, state, in_path_cmd)

        if pending > 0:
            return ValidationResult(False, n - 1, "stream ends mid-argument", n)
        if require_bos_eos:
            return ValidationResult(False, n - 1, "missing EOS", n)
        return ValidationResult(True, -1, "", n)

    def is_valid(self, token_ids: List[int],
                 require_bos_eos: bool = True) -> bool:
        return self.validate(token_ids, require_bos_eos=require_bos_eos).valid

    def repair(self, token_ids: List[int]) -> List[int]:
        """모델이 뱉은 토큰열을 문법에 맞게 잘라/마감하여 항상 디코드 가능한
        스트림으로 만든다. 인자 미완(pending>0) 위치에서 절단 후 EOS 부착.

        '모델이 쓰레기를 뱉는' 실패 모드를 제거하는 안전망 (PRD G3)."""
        out: List[int] = [_BOS]
        pending, state, in_path_cmd = 0, "boundary", False
        body = token_ids[1:] if (token_ids and token_ids[0] == _BOS) else token_ids
        for t in body:
            if t == _EOS:
                break
            if t == _BOS:
                continue
            allowed = self._allowed_ids(pending, state, in_path_cmd)
            if t not in allowed:
                # 비허용 토큰은 건너뛴다 (안전 절단)
                continue
            out.append(t)
            pending, state, in_path_cmd = self._advance(t, pending, state, in_path_cmd)
        # 인자 미완이면 마지막 불완전 그룹 제거
        while pending > 0 and len(out) > 1:
            removed = out.pop()
            pending, state, in_path_cmd = 0, "boundary", False
            for t in out[1:]:
                pending, state, in_path_cmd = self._advance(t, pending, state, in_path_cmd)
        out.append(_EOS)
        return out
