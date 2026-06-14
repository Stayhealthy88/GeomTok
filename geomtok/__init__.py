"""
GeomTok — geometry-native tokenization for vector graphics (formerly GPL Tokenizer)
=============================================
SVG의 기하학적 구조를 보존하는 토큰화 시스템.

Architecture:
    SVG Text → SVGParser → PathCommands → GeometricAnalyzer → AnnotatedCommands
    → PrimitiveTokenizer → GPL Tokens → Detokenizer → SVG Text
    → GPLEmbedding → GPLTransformer → Generated GPL Tokens → SVG

Modules:
    parser/      : SVG 파싱 및 path 명령어 분해
    analyzer/    : 베지에 곡률, G1/G2 연속성, 공간 관계 분석
    tokenizer/   : Level 1-3 토큰화, ARCS, 어휘 관리
    embedding/   : GPLEmbedding + HMN 초기화 (v0.4)
    training/    : 학습 파이프라인 — 데이터셋, Transformer, 생성, 평가 (v0.5)
    utils/       : 수학 유틸리티
"""

__version__ = "1.0.0"

# 고수준 SDK (PRD §7.4 — pip install 후 <15분 quickstart)
#   from geomtok import tokenize, detokenize, GeomTokenizer
from .api import GeomTokenizer, detect_unsupported, MAX_SVG_BYTES
from .tokenizer.manifest import (
    VocabManifest, TOKENIZER_VERSION, DEFAULT_VOCAB_ID, load_bundled_manifest,
)
from .errors import (
    GeomTokError, ParseError, ParseUnsupportedElement,
    PayloadTooLarge, VocabMismatch, VersionRequired,
)

# 프로세스 단일 기본 토크나이저 (지연 초기화) — 모듈 함수형 인터페이스 백킹
_DEFAULT = None


def _default() -> "GeomTokenizer":
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = GeomTokenizer.default()
    return _DEFAULT


def tokenize(svg: str, level: str = "L1", **kw):
    """SVG → 기하 토큰 (기본 어휘). 로컬·매니지드 동일 결과.

    >>> import geomtok
    >>> out = geomtok.tokenize("<svg viewBox='0 0 24 24'><path d='M4 4 L20 20'/></svg>")
    >>> out["n_tokens"], out["tokenizer_version"]
    """
    return _default().tokenize(svg, level=level, **kw)


def detokenize(token_ids, tokenizer_version: str = TOKENIZER_VERSION,
               vocab_id: str = DEFAULT_VOCAB_ID, level: str = "L1", **kw):
    """기하 토큰 → 유효 SVG (라운드트립). tokenizer_version 핀으로 결정적."""
    return _default().detokenize(token_ids, tokenizer_version=tokenizer_version,
                                 vocab_id=vocab_id, level=level, **kw)


__all__ = [
    "__version__", "tokenize", "detokenize", "GeomTokenizer",
    "VocabManifest", "TOKENIZER_VERSION", "DEFAULT_VOCAB_ID",
    "load_bundled_manifest", "detect_unsupported", "MAX_SVG_BYTES",
    "GeomTokError", "ParseError", "ParseUnsupportedElement",
    "PayloadTooLarge", "VocabMismatch", "VersionRequired",
]
