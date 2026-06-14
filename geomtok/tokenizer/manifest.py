"""
Vocab 매니페스트 · 결정성 백킹 (v1.0 / G2)
============================================
GeomTok 의 결정성·재현성 보장을 담당하는 **불변(immutable)** 매니페스트.

PRD G2: `tokenizer_version` + `vocab_id` 를 고정하면 비트-동일 인코드/디코드가
보장되어야 한다. 이 모듈은 그 계약을 코드로 고정한다:

  - TOKENIZER_VERSION : 토크나이저 알고리즘 시맨틱 버전 (인코딩 규약 핀)
  - VocabManifest     : id→심볼, 레벨 레이아웃, 좌표 그리드, (옵션) 머지 테이블
                        을 담은 직렬화 가능한 불변 객체. 오프라인 인코드/디코드
                        의 단일 진실 원천(single source of truth).

매니페스트는 `vocab_id` 로 식별되며, 한 번 발행되면 절대 바뀌지 않는다.
어휘를 바꾸려면 새 `vocab_id` 를 발행한다 (예: geom-5561-v2).

설계 원칙:
    동일 (TOKENIZER_VERSION, vocab_id, config) → 동일 비트.
    인코딩은 순수 산술(균일 격자 양자화)이므로 상태/난수가 없다.
    매니페스트는 디코드 테이블의 권위 있는 사본이며, 서버·로컬·오프라인이
    모두 같은 매니페스트를 읽어 비트-동일 결과를 낸다.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

from .vocabulary import (
    GPLVocabulary, SpecialToken, CommandToken, CompositeToken,
    SpatialToken, ContinuityToken, CURVATURE_TOKEN_BASE, N_CURVATURE_BINS,
    COORD_TOKEN_BASE,
)

# 토크나이저 알고리즘 버전 — 인코딩 규약을 바꾸는 변경에서만 올린다.
TOKENIZER_VERSION = "geomtok-1.0.0"

# v1.0 정규 어휘. max_coord_level=6 → vocab_size = 100 + Σ_{l=0..6} 4^l = 5561.
DEFAULT_VOCAB_ID = "geom-5561-v1"
DEFAULT_CANVAS_SIZE = 300.0
DEFAULT_MAX_COORD_LEVEL = 6


def _symbol_for_id(vocab: GPLVocabulary, tid: int) -> Optional[str]:
    """토큰 ID → 사람이 읽는 안정적 심볼 문자열 (매니페스트 표기용).

    반환 None 은 '예약 ID(미발행)' 를 의미한다.
    """
    info = vocab.decode_token_id(tid)
    t = info["type"]
    if t == "special":
        return info["value"]
    if t == "command":
        return info["value"]
    if t == "composite":
        return info["value"]
    if t == "spatial":
        return info["value"]
    if t == "continuity":
        return info["value"]
    if t == "curvature":
        return f"CURV@{info['bin']}"
    if t == "coord":
        return f"C{info['level']}:{info['qx']},{info['qy']}"
    return None  # unknown / reserved


@dataclass(frozen=True)
class VocabManifest:
    """불변 vocab 매니페스트 — 오프라인 인코드/디코드·결정성 백킹.

    필드:
        vocab_id          : 어휘 식별자 (불변)
        tokenizer_version : 인코딩 규약 버전
        canvas_size       : 정규화 캔버스 크기 (px)
        max_coord_level   : 최심 쿼드트리 레벨 (좌표 그리드 해상도 2^L)
        vocab_size        : 총 토큰 수
        coord_token_base  : 좌표 토큰 시작 ID
        level_layout      : 레벨별 (start_id, grid, n_tokens) 레이아웃
        special / command / continuity / curvature / composite / spatial :
            id → 심볼 매핑 (디코드 테이블의 비좌표 영역)
        merges            : (옵션) L2 학습 머지 테이블. None 이면 L1 전용.
                            [[a_id, b_id, new_id], ...] 순서가 곧 머지 우선순위.
        merge_base_id     : 머지 토큰 ID 시작점 (vocab_size 이상)
    """
    vocab_id: str
    tokenizer_version: str
    canvas_size: float
    max_coord_level: int
    vocab_size: int
    coord_token_base: int
    level_layout: List[Dict] = field(default_factory=list)
    special: Dict[int, str] = field(default_factory=dict)
    command: Dict[int, str] = field(default_factory=dict)
    continuity: Dict[int, str] = field(default_factory=dict)
    curvature: Dict[int, str] = field(default_factory=dict)
    composite: Dict[int, str] = field(default_factory=dict)
    spatial: Dict[int, str] = field(default_factory=dict)
    merges: Optional[List[List[int]]] = None
    merge_base_id: Optional[int] = None

    # --- 구축 ---

    @classmethod
    def build(cls, vocab_id: str = DEFAULT_VOCAB_ID,
              canvas_size: float = DEFAULT_CANVAS_SIZE,
              max_coord_level: int = DEFAULT_MAX_COORD_LEVEL,
              merges: Optional[List[List[int]]] = None) -> "VocabManifest":
        """정규 GPL 어휘로부터 매니페스트를 결정적으로 구축."""
        vocab = GPLVocabulary(max_coord_level=max_coord_level)

        # 레벨 레이아웃 (좌표 그리드)
        layout = []
        cur = COORD_TOKEN_BASE
        for level in range(max_coord_level + 1):
            grid = 2 ** level
            n = grid * grid
            layout.append({"level": level, "start_id": cur, "grid": grid, "n_tokens": n})
            cur += n

        def _range_syms(lo: int, hi: int) -> Dict[int, str]:
            out = {}
            for tid in range(lo, hi):
                s = _symbol_for_id(vocab, tid)
                if s is not None:
                    out[tid] = s
            return out

        return cls(
            vocab_id=vocab_id,
            tokenizer_version=TOKENIZER_VERSION,
            canvas_size=float(canvas_size),
            max_coord_level=int(max_coord_level),
            vocab_size=vocab.vocab_size,
            coord_token_base=COORD_TOKEN_BASE,
            level_layout=layout,
            special=_range_syms(0, 5),
            command=_range_syms(10, 18),
            continuity=_range_syms(30, 34),
            curvature=_range_syms(CURVATURE_TOKEN_BASE,
                                  CURVATURE_TOKEN_BASE + N_CURVATURE_BINS),
            composite=_range_syms(20, 24),
            spatial=_range_syms(60, 71),
            merges=merges,
            merge_base_id=(vocab.vocab_size if merges else None),
        )

    # --- 직렬화 ---

    def to_dict(self) -> Dict:
        d = asdict(self)
        # JSON 은 int 키를 문자열로 강제하므로, 명시적으로 문자열화하여 왕복 안정성 확보
        for k in ("special", "command", "continuity", "curvature", "composite", "spatial"):
            d[k] = {str(i): s for i, s in getattr(self, k).items()}
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, d: Dict) -> "VocabManifest":
        def _intkeys(m):
            return {int(k): v for k, v in (m or {}).items()}
        return cls(
            vocab_id=d["vocab_id"],
            tokenizer_version=d["tokenizer_version"],
            canvas_size=float(d["canvas_size"]),
            max_coord_level=int(d["max_coord_level"]),
            vocab_size=int(d["vocab_size"]),
            coord_token_base=int(d["coord_token_base"]),
            level_layout=d.get("level_layout", []),
            special=_intkeys(d.get("special")),
            command=_intkeys(d.get("command")),
            continuity=_intkeys(d.get("continuity")),
            curvature=_intkeys(d.get("curvature")),
            composite=_intkeys(d.get("composite")),
            spatial=_intkeys(d.get("spatial")),
            merges=d.get("merges"),
            merge_base_id=d.get("merge_base_id"),
        )

    @classmethod
    def from_json(cls, s: str) -> "VocabManifest":
        return cls.from_dict(json.loads(s))

    # --- 무결성 ---

    def content_hash(self) -> str:
        """매니페스트 내용의 결정적 SHA-256 (앞 16자). 감사·핀 검증용.

        동일 내용 → 동일 해시. 온프레미스/호스팅 비트-동일 보증을 계약적으로
        검증할 때 사용 (PRD 미해결질문 8)."""
        canonical = json.dumps(self.to_dict(), ensure_ascii=False,
                               sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    @property
    def has_merges(self) -> bool:
        return bool(self.merges)


# 패키지 동봉 매니페스트 경로 (L2 머지 포함, train_merges.py 산출물)
import os as _os
_BUNDLED_DIR = _os.path.join(_os.path.dirname(__file__), "..", "data")


def bundled_manifest_path(vocab_id: str = DEFAULT_VOCAB_ID) -> str:
    return _os.path.normpath(_os.path.join(_BUNDLED_DIR, f"manifest_{vocab_id}.json"))


def load_bundled_manifest(vocab_id: str = DEFAULT_VOCAB_ID) -> Optional[VocabManifest]:
    """동봉된 매니페스트(L2 머지 포함)를 로드. 없으면 None.

    이것이 결정성·오프라인의 단일 진실 원천이다 — 서버·SDK·오프라인이 모두
    같은 파일을 읽어 비트-동일 결과를 낸다 (PRD G2)."""
    path = bundled_manifest_path(vocab_id)
    if _os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as fh:
            return VocabManifest.from_json(fh.read())
    return None
