"""
GPL 문법 제약 디코딩 — torch 어댑터 (v1.0)
===========================================
유효성 보장 디코딩의 문법 로직은 이제 torch-free 코어
`geomtok.tokenizer.grammar.GrammarFSA` 에 있다 (PRD G3: OSS 코어 보장).
이 모듈은 그 코어를 감싸 학습/생성 경로용 **로짓 마스킹 텐서**를 제공한다.

CAD-Tokenizer(arXiv:2509.21150)의 FSA 제약이 무효율 80%→8%를 보인 선례.
"""

from typing import List

import torch

from ..tokenizer.grammar import GrammarFSA


class GrammarMask:
    """토큰 시퀀스 접두사에 대해 다음 허용 토큰 마스크(torch.bool) 생성.

    문법 판정은 GrammarFSA(코어)에 위임하고, 여기서는 torch 텐서로 변환만 한다.
    """

    def __init__(self, vocab):
        self.fsa = GrammarFSA(vocab)
        self.V = vocab.vocab_size
        # 자주 쓰는 집합을 텐서로 캐시 (하위호환 속성)
        self.coord = self._to_tensor(self.fsa._coord_ids)
        self.cont = self._to_tensor(self.fsa._cont_ids)
        self.curv = self._to_tensor(self.fsa._curv_ids)
        self.boundary = self._to_tensor(self.fsa._boundary_ids)

    def _to_tensor(self, ids) -> torch.Tensor:
        m = torch.zeros(self.V, dtype=torch.bool)
        idx = [i for i in ids if 0 <= i < self.V]
        if idx:
            m[idx] = True
        return m

    def allowed(self, prefix: List[int]) -> torch.Tensor:
        """접두사 다음 허용 토큰의 torch.bool 마스크."""
        return self._to_tensor(self.fsa.allowed_ids(prefix))


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
