"""
GPL 문법 제약 디코딩 (v0.6 / E4)
==================================
GPL 토큰 문법을 FSA로 추적하고 단계마다 허용 토큰 마스크를 산출한다.
CAD-Tokenizer(arXiv:2509.21150)의 FSA 제약이 무효율 80%→8%를 보인 선례.

문법 (Detokenizer 해석 규약과 일치):
    BOS      → CMD | SHAPE | EOS
    CMD(k)   → coord × n(k)  [→ CONT? → CURV?]  → BOUNDARY
    SHAPE(s) → coord(중심/원점) → coord_L6 × m(s)(스칼라) → BOUNDARY
    SPATIAL  → 인자형(60-65,70)은 coord 1개, 마커형(66-69)은 0개 → BOUNDARY
    BOUNDARY → CMD | SHAPE | SPATIAL | SEP | EOS
"""

from typing import List

import torch

_CMD_COORDS = {10: 1, 11: 1, 12: 1, 13: 1, 14: 3, 15: 2, 16: 2, 17: 0}
_SHAPE_SCALARS = {20: 1, 21: 2, 22: 2, 23: 4}   # 중심 coord 이후 스칼라 수
_SPATIAL_ARG = {60: 1, 61: 1, 62: 1, 63: 1, 64: 1, 65: 1, 66: 0, 67: 0, 68: 0, 69: 0, 70: 1}


class GrammarMask:
    """토큰 시퀀스 접두사에 대해 다음 허용 토큰 마스크 생성."""

    def __init__(self, vocab):
        V = vocab.vocab_size
        coord = torch.zeros(V, dtype=torch.bool)
        for tid in range(V):
            if vocab.id_to_coord(tid):
                coord[tid] = True
        self.coord = coord
        self.cont = torch.zeros(V, dtype=torch.bool); self.cont[30:34] = True
        self.curv = torch.zeros(V, dtype=torch.bool); self.curv[40:56] = True
        self.boundary = torch.zeros(V, dtype=torch.bool)
        for tid in list(_CMD_COORDS) + list(_SHAPE_SCALARS) + list(_SPATIAL_ARG):
            self.boundary[tid] = True
        self.boundary[2] = True   # EOS
        self.boundary[3] = True   # SEP

    def allowed(self, prefix: List[int]) -> torch.Tensor:
        pending = 0          # 남은 coord 인자 수
        in_path_cmd = False  # 직전 그룹이 경로 명령(CONT/CURV 허용)
        state = "boundary"

        for t in prefix[1:]:  # BOS 제외
            if pending > 0:
                pending -= 1
                state = "post" if pending == 0 else "args"
                continue
            if t in _CMD_COORDS:
                pending = _CMD_COORDS[t]
                in_path_cmd = True
                state = "args" if pending else "post"
            elif t in _SHAPE_SCALARS:
                pending = 1 + _SHAPE_SCALARS[t]
                in_path_cmd = False
                state = "args"
            elif t in _SPATIAL_ARG:
                pending = _SPATIAL_ARG[t]
                in_path_cmd = False
                state = "args" if pending else "boundary"
            elif 30 <= t < 34:
                state = "curv"
            elif 40 <= t < 56:
                state = "boundary"
            else:  # SEP 등
                state = "boundary"
                in_path_cmd = False

        if pending > 0:
            return self.coord
        if state == "post":
            m = self.boundary.clone()
            if in_path_cmd:
                m |= self.cont
            return m
        if state == "curv":
            return self.curv | self.boundary
        return self.boundary


@torch.no_grad()
def constrained_generate(model, vocab, prompt: torch.Tensor, max_len: int = 64,
                         temperature: float = 0.8, top_k: int = 50,
                         eos_id: int = 2) -> torch.Tensor:
    """FSA 마스크를 로짓에 적용하는 자가회귀 샘플링."""
    gm = GrammarMask(vocab)
    model.eval()
    generated = prompt.clone()
    for _ in range(max_len):
        if generated.size(1) >= model.config.max_seq_len:
            break
        logits = model.forward(generated)[:, -1, :] / temperature
        mask = gm.allowed(generated[0].tolist())
        logits[:, ~mask] = float("-inf")
        if top_k > 0:
            topk_vals, _ = torch.topk(logits, min(top_k, int(mask.sum())))
            logits[logits < topk_vals[:, -1:]] = float("-inf")
        probs = torch.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1)
        generated = torch.cat([generated, nxt], dim=1)
        if nxt.item() == eos_id:
            break
    return generated
