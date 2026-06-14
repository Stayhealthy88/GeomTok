"""
BPE-on-L1 학습 머지 토크나이저 (L2)
=====================================
PRD §7.3 의 L2 = "학습된 비제약 머지". 연구(RESEARCH_SUMMARY §0)에서 수작업
도형 검출(구 composite L2)은 실데이터 발화율 0% 로 폐기되었고, 대신 **L1 토큰
스트림 위에 BPE 머지를 학습**하면 264→151 tok/icon(−43%) 로 수작업 매크로(+5%)를
압도했다. 핵심 발견: "기하 프리미티브 토큰화가 학습 압축의 더 나은 기질(substrate)".

이 모듈은 그 L2 를 구현한다:
  - learn_merges() : L1 스트림 코퍼스에서 인접쌍 BPE 머지를 학습.
  - MergeCodec     : 학습된 머지로 L1↔L2 인코드/디코드 (결정적·무손실).

머지 토큰 ID 는 vocab_size 이상에서 발급되어 기존 L1 어휘와 충돌하지 않는다.
특수 토큰(BOS/EOS/SEP)은 머지 경계로 보호되어 요소·스트림 구조가 보존된다.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

# 머지로 건너뛰지 않는 보호 토큰 (요소/스트림 경계)
_PROTECTED = {1, 2, 3}   # BOS, EOS, SEP


def _count_pairs(streams: List[List[int]], protected: set) -> Counter:
    """모든 스트림에서 인접 토큰쌍 빈도 집계 (보호 토큰 포함 쌍 제외)."""
    pairs: Counter = Counter()
    for s in streams:
        for a, b in zip(s, s[1:]):
            if a in protected or b in protected:
                continue
            pairs[(a, b)] += 1
    return pairs


def _apply_merge(stream: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
    """스트림에서 pair 의 모든 비중첩 출현을 new_id 로 치환."""
    a, b = pair
    out: List[int] = []
    i = 0
    n = len(stream)
    while i < n:
        if i < n - 1 and stream[i] == a and stream[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(stream[i])
            i += 1
    return out


def _stream_pairs(s: List[int], protected: set):
    for a, b in zip(s, s[1:]):
        if a in protected or b in protected:
            continue
        yield (a, b)


def learn_merges(streams: List[List[int]], num_merges: int,
                 vocab_size: int, min_freq: int = 2,
                 protected: Optional[set] = None) -> List[List[int]]:
    """L1 스트림 코퍼스에서 BPE 머지를 학습 (증분 갱신, 표준 고속 BPE).

    매 스텝마다 전 코퍼스를 재스캔하지 않고, 머지가 일어난 스트림만 차분 갱신해
    인접쌍 빈도를 유지한다 — O(코퍼스) 재스캔을 제거.

    Args:
        streams: L1 토큰 ID 시퀀스 리스트 (BOS/EOS 포함 가능).
        num_merges: 학습할 최대 머지 수.
        vocab_size: 기존 L1 어휘 크기 → 머지 ID 시작점.
        min_freq: 머지를 채택할 최소 쌍 빈도.
        protected: 머지 경계로 보호할 토큰 ID 집합.
    Returns:
        merges: [[a, b, new_id], ...] — 순서가 곧 머지 우선순위.
    """
    protected = protected or _PROTECTED
    seqs = [list(s) for s in streams]

    pair_counts: Counter = Counter()
    pair_where: Dict[Tuple[int, int], set] = defaultdict(set)
    for i, s in enumerate(seqs):
        for p in _stream_pairs(s, protected):
            pair_counts[p] += 1
            pair_where[p].add(i)

    merges: List[List[int]] = []
    next_id = vocab_size

    for _ in range(num_merges):
        if not pair_counts:
            break
        # 최빈쌍 — 동률은 (a,b) 역순으로 깨 결정성 확보
        best_pair = max(pair_counts.items(),
                        key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))[0]
        freq = pair_counts[best_pair]
        if freq < min_freq:
            break
        merges.append([best_pair[0], best_pair[1], next_id])

        # best_pair 를 포함한 스트림만 차분 갱신
        for i in list(pair_where[best_pair]):
            s = seqs[i]
            old_pairs = Counter(_stream_pairs(s, protected))
            if best_pair not in old_pairs:
                pair_where[best_pair].discard(i)   # 스테일 항목 정리
                continue
            ns = _apply_merge(s, best_pair, next_id)
            seqs[i] = ns
            new_pairs = Counter(_stream_pairs(ns, protected))
            # 차분: 사라진 쌍 감소, 생긴 쌍 증가
            for p, c in old_pairs.items():
                pair_counts[p] -= c
                if pair_counts[p] <= 0:
                    del pair_counts[p]
                    pair_where.pop(p, None)
            for p, c in new_pairs.items():
                pair_counts[p] += c
                pair_where[p].add(i)

        pair_counts.pop(best_pair, None)
        pair_where.pop(best_pair, None)
        next_id += 1

    return merges


class MergeCodec:
    """학습된 머지로 L1↔L2 인코드/디코드 (결정적·무손실)."""

    def __init__(self, merges: List[List[int]], vocab_size: int):
        self.merges = [list(m) for m in merges]
        self.vocab_size = vocab_size
        self.merge_base_id = vocab_size
        # 인코드용 우선순위 랭크
        self._rank: Dict[Tuple[int, int], int] = {}
        # 디코드용 확장 테이블 new_id → (a, b)
        self._expand: Dict[int, Tuple[int, int]] = {}
        for rank, (a, b, nid) in enumerate(self.merges):
            self._rank[(a, b)] = rank
            self._expand[nid] = (a, b)

    @classmethod
    def from_manifest(cls, manifest) -> "MergeCodec":
        if not manifest.has_merges:
            raise ValueError("manifest has no merges")
        return cls(manifest.merges, manifest.vocab_size)

    # --- 인코드 (L1 → L2) --- #

    def encode(self, l1_ids: List[int]) -> List[int]:
        """L1 토큰열에 머지를 우선순위 순으로 반복 적용해 L2 로 압축."""
        if not self._rank:
            return list(l1_ids)
        seq = list(l1_ids)
        while True:
            # 현재 시퀀스에서 가장 낮은 랭크(=높은 우선순위) 쌍 탐색
            best_rank = None
            best_pos = -1
            for i in range(len(seq) - 1):
                # 보호 토큰(BOS/EOS/SEP)을 가로지르는 머지는 금지 — 수기/외부
                # 머지 테이블이 와도 요소·스트림 경계를 보존 (learn_merges 가드 미러)
                if seq[i] in _PROTECTED or seq[i + 1] in _PROTECTED:
                    continue
                r = self._rank.get((seq[i], seq[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank = r
                    best_pos = i
            if best_rank is None:
                break
            a, b = seq[best_pos], seq[best_pos + 1]
            new_id = self.merge_base_id + best_rank
            seq[best_pos:best_pos + 2] = [new_id]
        return seq

    # --- 디코드 (L2 → L1) --- #

    def decode(self, l2_ids: List[int]) -> List[int]:
        """머지 토큰을 L1 프리미티브로 완전 전개 (무손실)."""
        out: List[int] = []
        for tid in l2_ids:
            if tid in self._expand:
                self._expand_into(tid, out)
            else:
                out.append(tid)
        return out

    def _expand_into(self, tid: int, out: List[int]):
        # 반복적 스택 전개 (깊은 재귀 회피)
        stack = [tid]
        while stack:
            cur = stack.pop()
            pair = self._expand.get(cur)
            if pair is None:
                out.append(cur)
            else:
                # b 를 먼저 push → a 가 먼저 pop 되어 순서 보존
                stack.append(pair[1])
                stack.append(pair[0])

    @property
    def n_merges(self) -> int:
        return len(self.merges)
