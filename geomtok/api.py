"""
GeomTok 고수준 파사드 (SDK · 서버 · Eval 공용 코어)
====================================================
PRD §7 의 단건 인터페이스를 구현하는 단일 진입점. 로컬 SDK 와 매니지드 API 가
**동일 코드 · 동일 매니페스트**를 공유하여 비트-동일 결과를 보장한다 (PRD §9).

파이프라인:
    svg(text) → SVGParser(viewBox 정규화 · transform 평탄화)
              → 요소별 L1 토큰 (SEP 로 경계 보존)
              → [옵션] L2 학습 머지(BPE-on-L1)
              → token_ids

    token_ids → [L2 역머지] → Detokenizer(FSA 규약) → 유효 SVG

결정성: 인코딩은 순수 산술(균일 L6 격자). 같은
(tokenizer_version, vocab_id, config) → 같은 비트.
"""

from __future__ import annotations

import re as _re
from typing import Dict, List, Optional, Tuple

from .errors import (
    ParseError, ParseUnsupportedElement, PayloadTooLarge,
    VocabMismatch, VersionRequired,
)
from .parser.svg_parser import SVGParser
from .parser.path_parser import CommandType
from .parser.arc_flatten import flatten_arcs
from .tokenizer.primitive_tokenizer import PrimitiveTokenizer
from .tokenizer.detokenizer import Detokenizer
from .tokenizer.vocabulary import SpecialToken
from .tokenizer.manifest import (
    VocabManifest, TOKENIZER_VERSION, DEFAULT_VOCAB_ID,
    DEFAULT_CANVAS_SIZE, DEFAULT_MAX_COORD_LEVEL,
)

# 입력 한도 (PRD §7.1 / §8 — DoS 방어)
MAX_SVG_BYTES = 256 * 1024          # 256KB (단건)

# v1.0 도메인 밖 — 명시 거부 (PRD NG2/NG5). 키=태그 조각, 값=사유.
_UNSUPPORTED_MARKERS = {
    "<filter": "filter effects",
    "fegaussianblur": "filter effects",
    "<image": "raster image embed",
    "<text": "text element",
    "<tspan": "text element",
    "<lineargradient": "gradient fill",
    "<radialgradient": "gradient fill",
    "<pattern": "pattern fill",
    "<use": "symbol/use reference",
    "<foreignobject": "foreignObject",
    "<animate": "SMIL animation",
    "<animatetransform": "SMIL animation",
    "<set ": "SMIL animation",
}


def detect_unsupported(svg_text: str) -> Optional[str]:
    """v1.0 도메인 밖 요소를 탐지. 발견 시 사유 문자열, 없으면 None.

    path 중심 모노크롬 아이콘 도메인(PRD §3.2)에 한정하기 위해, 렌더 의도가
    있으나 토큰화 불가한 요소(필터·그라디언트·래스터·텍스트·use·애니메이션)를
    명시 거부한다. defs/clipPath/mask 같은 컨테이너는 거부하지 않고 무시한다.

    주석/desc/title/metadata 안의 언급(예: `<!-- <text> placeholder -->`)을
    실제 요소로 오인하지 않도록, 매칭 전 비렌더 텍스트를 제거한다."""
    low = _strip_nonrender_text(svg_text).lower()
    for marker, reason in _UNSUPPORTED_MARKERS.items():
        if marker in low:
            return reason
    return None


_NONRENDER_RE = _re.compile(
    r"<!--.*?-->|<(desc|title|metadata)\b[^>]*>.*?</\1>",
    _re.DOTALL | _re.IGNORECASE)


def _strip_nonrender_text(svg_text: str) -> str:
    """주석·desc·title·metadata 블록 제거 (오거부 방지)."""
    return _NONRENDER_RE.sub("", svg_text)


class GeomTokenizer:
    """GeomTok 코어 토크나이저 파사드.

    사용법:
        gt = GeomTokenizer()                 # 정규 v1.0 어휘
        out = gt.tokenize(svg)               # PRD §7.2 형식 dict
        back = gt.detokenize(out["token_ids"])
    """

    def __init__(self,
                 canvas_size: float = DEFAULT_CANVAS_SIZE,
                 max_coord_level: int = DEFAULT_MAX_COORD_LEVEL,
                 vocab_id: str = DEFAULT_VOCAB_ID,
                 manifest: Optional[VocabManifest] = None):
        if manifest is not None:
            canvas_size = manifest.canvas_size
            max_coord_level = manifest.max_coord_level
            vocab_id = manifest.vocab_id
        self.canvas_size = float(canvas_size)
        self.max_coord_level = int(max_coord_level)
        self.vocab_id = vocab_id
        self.tokenizer_version = TOKENIZER_VERSION

        # 결정성: uniform L6 격자(난수/적응 없음)
        self._tok = PrimitiveTokenizer(
            canvas_size=self.canvas_size, max_coord_level=self.max_coord_level,
            use_adaptive_arcs=False, uniform_coords=True,
        )
        self.vocab = self._tok.vocab
        self.arcs = self._tok.arcs
        self._detok = Detokenizer(self.vocab, self.arcs)
        self._parser = SVGParser(normalize_canvas=self.canvas_size)
        # FSA 문법 검증기 — 좌표-id 집합 재구축 비용을 1회로 (호출당 X)
        from .tokenizer.grammar import GrammarFSA
        self._fsa = GrammarFSA(self.vocab)

        self.manifest = manifest or VocabManifest.build(
            vocab_id=vocab_id, canvas_size=self.canvas_size,
            max_coord_level=self.max_coord_level,
        )
        # L2 머지 코덱 (있을 때만)
        self._merge_codec = None
        if self.manifest.has_merges:
            from .tokenizer.merge_tokenizer import MergeCodec
            self._merge_codec = MergeCodec.from_manifest(self.manifest)

    @classmethod
    def default(cls, vocab_id: str = DEFAULT_VOCAB_ID) -> "GeomTokenizer":
        """동봉 매니페스트(L2 머지 포함)가 있으면 로드, 없으면 L1 전용.

        서버·SDK 의 표준 생성 경로 — 비트-동일 결과의 단일 진실 원천."""
        from .tokenizer.manifest import load_bundled_manifest
        m = load_bundled_manifest(vocab_id)
        return cls(manifest=m) if m is not None else cls(vocab_id=vocab_id)

    @property
    def has_l2(self) -> bool:
        return self._merge_codec is not None

    # ------------------------------------------------------------------ #
    # 인코드
    # ------------------------------------------------------------------ #

    def tokenize(self, svg: str, level: str = "L1",
                 config: Optional[Dict] = None,
                 return_fields: Optional[List[str]] = None,
                 check_size: bool = True,
                 lean: bool = False) -> Dict:
        """SVG 1건 → 기하 토큰. PRD §7.2 응답 형식의 dict 반환.

        Args:
            svg: SVG XML 문자열.
            level: "L1"(기본) | "L2"(학습 머지) | "L3"(공간; 실험).
            config: {canvas_size, max_coord_level} 등 — 현재는 에코·검증용.
            return_fields: 응답에 담을 키 화이트리스트 (None=전체).
            check_size: 페이로드 한도 검사 여부.
            lean: True 면 연속성·곡률 보조 마커 없는 lean L1 스트림.
                마커는 파생 가능(복원 무영향)하면서 시퀀스를 ~24% 늘리고
                held-out NLL 을 ~12% 악화시킨다 (PAPER §5.1) — 모델
                학습·생성용 스트림엔 lean 권장. 기본 False 는 v1.0
                비트-동일성 계약 보존. detokenize 는 양쪽 모두 동일 SVG.
        Raises:
            PayloadTooLarge, ParseUnsupportedElement, ParseError.
        """
        if check_size and len(svg.encode("utf-8")) > MAX_SVG_BYTES:
            raise PayloadTooLarge(
                f"svg exceeds {MAX_SVG_BYTES} bytes",
                limit=MAX_SVG_BYTES)

        reason = detect_unsupported(svg)
        if reason is not None:
            raise ParseUnsupportedElement(f"<{reason}> not tokenizable",
                                          reason=reason)

        try:
            doc = self._parse(svg)
        except ParseUnsupportedElement:
            raise
        except Exception as e:  # noqa: BLE001 — 파서 내부 예외를 표준화
            raise ParseError(f"failed to parse SVG: {e}") from e

        # 요소별 토큰화 → SEP 로 경계 보존 → BOS/EOS 브래킷
        warnings: List[str] = []
        body: List[int] = []
        n_commands = 0
        transforms_flattened = 0
        n_elems = 0
        n_clamped = 0
        for elem in doc.elements:
            cmds = list(elem.commands)
            if not cmds:
                continue
            if elem.transform:
                transforms_flattened += 1
            # viewBox 밖 좌표는 클램핑됨 — 정직한 신호로 카운트
            for cmd in cmds:
                for (px, py) in self._cmd_points(cmd):
                    # arcs.quantize 가 [0, canvas) 로 clamp 하므로 동일 경계로 집계
                    if px < 0 or px > self.canvas_size \
                            or py < 0 or py > self.canvas_size:
                        n_clamped += 1
            res = self._tok.tokenize(cmds, emit_markers=not lean)
            # primitive tokenizer 가 부여한 BOS/EOS 제거 → 가운데 토큰만
            inner = res.token_ids
            if inner and inner[0] == int(SpecialToken.BOS):
                inner = inner[1:]
            if inner and inner[-1] == int(SpecialToken.EOS):
                inner = inner[:-1]
            if n_elems > 0:
                body.append(int(SpecialToken.SEP))
            body.extend(inner)
            n_commands += res.n_commands
            n_elems += 1

        l1_ids = [int(SpecialToken.BOS)] + body + [int(SpecialToken.EOS)]

        if n_clamped:
            warnings.append(
                f"{n_clamped} coordinate(s) outside viewBox were clamped to canvas")

        # 레벨 적용
        applied_level = "L1"
        token_ids = l1_ids
        if level.upper() == "L2":
            if self._merge_codec is not None:
                token_ids = self._merge_codec.encode(l1_ids)
                applied_level = "L2"
                if lean:
                    warnings.append(
                        "L2 merges were learned on full (marker) L1 streams; "
                        "lean+L2 compression may be suboptimal")
            else:
                warnings.append("L2 merges unavailable; falling back to L1")
        elif level.upper() == "L3":
            warnings.append("L3 spatial is experimental; emitting L1 substrate")
        elif level.upper() != "L1":
            warnings.append(f"unknown level '{level}'; emitting L1")

        # BPE 압축비 (동일 어휘 SVG-학습 BPE 기준은 Eval 에서 계산; 여기선
        # 참조 cl100k 대비 비를 best-effort 로 채운다)
        comp = self._compression_vs_bpe(svg, len(token_ids))

        payload = {
            "tokenizer_version": self.tokenizer_version,
            "vocab_id": self.vocab_id,
            "level": applied_level,
            "lean": lean,
            "token_ids": token_ids,
            "tokens": [self._symbol(t) for t in token_ids],
            "n_commands": n_commands,
            "n_tokens": len(token_ids),
            "n_elements": n_elems,
            "compression_ratio": comp,
            "normalization": {
                "viewBox_applied": doc.viewbox is not None,
                "transforms_flattened": transforms_flattened,
            },
            "warnings": warnings,
        }
        if return_fields:
            keep = set(return_fields) | {
                "tokenizer_version", "vocab_id", "level", "lean",
                "n_tokens", "warnings"}
            payload = {k: v for k, v in payload.items() if k in keep}
        return payload

    # ------------------------------------------------------------------ #
    # 디코드
    # ------------------------------------------------------------------ #

    def detokenize(self, token_ids: List[int],
                   tokenizer_version: Optional[str] = None,
                   vocab_id: Optional[str] = None,
                   level: str = "L1",
                   measure: bool = True) -> Dict:
        """토큰 ID → 유효 SVG (라운드트립). PRD §7.2 응답 형식 dict.

        Args:
            token_ids: 인코드 시 받은 토큰 ID.
            tokenizer_version: **필수** (디코드 테이블 핀). 없으면 VersionRequired.
            vocab_id: 인코드와 일치 필수. 불일치 시 VocabMismatch.
            level: "L2" 면 머지 역적용 후 디코드.
        """
        if tokenizer_version is None:
            raise VersionRequired("tokenizer_version is required for detokenize")
        if tokenizer_version != self.tokenizer_version:
            raise VocabMismatch(
                f"tokenizer_version {tokenizer_version} != {self.tokenizer_version}",
                expected=self.tokenizer_version, got=tokenizer_version)
        if vocab_id is not None and vocab_id != self.vocab_id:
            raise VocabMismatch(
                f"vocab_id {vocab_id} != {self.vocab_id}",
                expected=self.vocab_id, got=vocab_id)

        # 토큰 ID 범위 검증 — 잘못된 입력이 raw 예외로 500 나지 않도록 표준화
        upper = (self.manifest.merge_base_id or self.vocab.vocab_size) \
            + (len(self.manifest.merges) if self.manifest.has_merges else 0)
        bad = next((t for t in token_ids
                    if not isinstance(t, int) or t < 0 or t >= upper), None)
        if bad is not None:
            raise VocabMismatch(f"token id {bad} out of range [0,{upper})",
                                max_id=upper)

        # L2 머지 역적용 — level 인자에 의존하지 않고 merge-range id 자동 감지.
        # tokenize() 응답은 디코드 시 level="L2"를 되돌려 넘기라고 안내하지 않으므로,
        # 기본 level="L1" 호출에 L2 id가 와도 무음 손실 없이 정확히 복원한다.
        ids = list(token_ids)
        mb = self.manifest.merge_base_id
        if self._merge_codec is not None and mb is not None and \
                any(isinstance(t, int) and t >= mb for t in ids):
            ids = self._merge_codec.decode(ids)

        # FSA 문법 검증 — "항상 well-formed" 보증을 실제로 실행 (PRD G3).
        # 위반 시 repair 로 디코드 가능한 스트림으로 안전 절단.
        valid = self._fsa.is_valid(ids)
        repaired = False
        if not valid:
            ids = self._fsa.repair(ids)
            repaired = True

        svg = self._detok.to_svg_document(
            ids, width=self.canvas_size, height=self.canvas_size)
        out = {
            "tokenizer_version": self.tokenizer_version,
            "vocab_id": self.vocab_id,
            "svg": svg,
            "n_tokens": len(token_ids),
            "valid": valid,
            "repaired": repaired,
        }
        if measure:
            out["fidelity"] = self._expected_fidelity(ids)
        return out

    # 단위 정사각 셀 내 균일 분포 점→중심 평균 L2 거리 (해석상수).
    _MEAN_DIST_UNIT_CELL = 0.382597858

    def _expected_fidelity(self, ids: List[int]) -> Dict:
        """디코드 시 originals 가 없으므로, 토큰에 실제 등장한 좌표 레벨에서의
        해석적 양자화 오차(설계 보장치)를 보고한다. 가장 얕은 레벨이 곧 가장
        큰 셀 → 가장 보수적인 상한. 균등분포 가정 평균/모서리 최대."""
        levels = []
        for tid in ids:
            info = self.vocab.decode_token_id(tid)
            if info.get("type") == "coord":
                levels.append(info["level"])
        if not levels:
            return {"coord_mean_px": 0.0, "coord_max_px": 0.0,
                    "canvas": self.canvas_size}
        coarsest = min(levels)
        cell = self.canvas_size / (2 ** coarsest)
        return {
            "coord_mean_px": round(self._MEAN_DIST_UNIT_CELL * cell, 4),
            "coord_max_px": round((cell / 2.0) * (2 ** 0.5), 4),
            "canvas": self.canvas_size,
        }

    # ------------------------------------------------------------------ #
    # 헬퍼
    # ------------------------------------------------------------------ #

    def _parse(self, svg: str):
        """SVG 파싱 + 호 평탄화(정규화). 모든 다운스트림이 같은 명령 스트림을
        보도록 단일 경로로 통일한다."""
        doc = self._parser.parse_string(svg)
        for elem in doc.elements:
            elem.commands = flatten_arcs(elem.commands)
        return doc

    def encoded_points(self, svg: str) -> List[Tuple[float, float]]:
        """이 SVG 가 인코딩하는 원본 (x,y) 좌표를 토큰 순서대로 반환.

        라운드트립 좌표 충실도 측정용 — 원본↔복원 정렬 비교의 '원본' 쪽."""
        doc = self._parse(svg)
        pts: List[Tuple[float, float]] = []
        for elem in doc.elements:
            for cmd in elem.commands:
                pts.extend(self._cmd_points(cmd))
        return pts

    @staticmethod
    def _cmd_points(cmd) -> List[Tuple[float, float]]:
        ap = cmd.abs_params
        ct = cmd.command_type
        if ap is None or ct == CommandType.CLOSE:
            return []
        if ct in (CommandType.MOVE, CommandType.LINE,
                  CommandType.HLINE, CommandType.VLINE):
            return [(ap[0], ap[1])]
        if ct == CommandType.CUBIC:
            return [(ap[0], ap[1]), (ap[2], ap[3]), (ap[4], ap[5])]
        if ct == CommandType.QUADRATIC:
            return [(ap[0], ap[1]), (ap[2], ap[3])]
        if ct == CommandType.ARC:
            return [(ap[0], ap[1]), (ap[5], ap[6])]
        return []

    def normalized_svg(self, svg: str,
                       stroke_width: float = 1.0) -> str:
        """파싱·정규화된(viewBox·transform 평탄화) 기하를 양자화 없이 그대로
        디코드 출력과 동일 스타일로 렌더한 SVG 를 반환.

        Eval 의 SSIM 비교에서 '정답(ground-truth)' 측 — 토큰화 양자화 오차만을
        분리 측정하기 위해, 렌더 스타일(stroke/fill)을 복원본과 일치시킨다."""
        doc = self._parse(svg)
        paths = []
        for elem in doc.elements:
            d = self._commands_to_path_d(elem.commands)
            if d:
                paths.append(
                    f'<path d="{d}" fill="none" stroke="black" '
                    f'stroke-width="{stroke_width}"/>')
        c = int(self.canvas_size) if float(self.canvas_size).is_integer() else self.canvas_size
        inner = "\n  ".join(paths)
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{c}" height="{c}" '
                f'viewBox="0 0 {c} {c}">\n  {inner}\n</svg>')

    @staticmethod
    def _commands_to_path_d(commands) -> str:
        """절대좌표 명령 시퀀스를 양자화 없이 path d 문자열로 직렬화."""
        def f(v):
            return f"{v:.3f}".rstrip("0").rstrip(".")
        parts = []
        for cmd in commands:
            ct = cmd.command_type
            ap = cmd.abs_params
            if ct == CommandType.MOVE:
                parts.append(f"M {f(ap[0])} {f(ap[1])}")
            elif ct == CommandType.LINE:
                parts.append(f"L {f(ap[0])} {f(ap[1])}")
            elif ct == CommandType.HLINE:
                parts.append(f"L {f(ap[0])} {f(ap[1])}")
            elif ct == CommandType.VLINE:
                parts.append(f"L {f(ap[0])} {f(ap[1])}")
            elif ct == CommandType.CUBIC:
                parts.append(f"C {f(ap[0])} {f(ap[1])} {f(ap[2])} {f(ap[3])} {f(ap[4])} {f(ap[5])}")
            elif ct == CommandType.QUADRATIC:
                parts.append(f"Q {f(ap[0])} {f(ap[1])} {f(ap[2])} {f(ap[3])}")
            elif ct == CommandType.ARC:
                parts.append(f"A {f(ap[0])} {f(ap[1])} {f(ap[2])} "
                             f"{int(ap[3])} {int(ap[4])} {f(ap[5])} {f(ap[6])}")
            elif ct == CommandType.CLOSE:
                parts.append("Z")
        return " ".join(parts)

    def measure_roundtrip(self, svg: str) -> Dict:
        """단일 SVG 의 라운드트립 좌표 충실도를 측정 (렌더 없이).

        Returns: {mean_error, max_error, n_points, within_bound, ...}
        """
        pts = self.encoded_points(svg)
        return self.arcs.roundtrip_fidelity(pts)

    def _symbol(self, tid: int) -> str:
        if self._merge_codec is not None and tid >= (self.manifest.merge_base_id or 1 << 30):
            return f"MERGE@{tid}"
        info = self.vocab.decode_token_id(tid)
        t = info["type"]
        if t in ("special", "command", "composite", "spatial", "continuity"):
            return info["value"]
        if t == "curvature":
            return f"CURV@{info['bin']}"
        if t == "coord":
            return f"C{info['level']}:{info['qx']},{info['qy']}"
        return f"?{tid}"

    def _compression_vs_bpe(self, svg: str, n_tokens: int) -> float:
        """결정적·의존성 없는 압축비: 원본 SVG 문자 수 / 토큰 수 (= 토큰당
        대체한 소스 문자 수). cl100k 비교(논문 §6에서 strawman 으로 철회)와
        tiktoken 유무에 따른 비결정성(3.2 vs 0.0)을 제거한다. 도메인-BPE 대비
        정직한 압축비(3.54×)는 코퍼스 단위로 /v1/eval 에서 보고한다."""
        if not n_tokens:
            return 0.0
        return round(len(svg) / n_tokens, 4)
