"""
GeomTok 표준 에러 코드 (PRD §7.6)
=================================
모든 코어·API 경로가 동일한 에러 코드를 쓰도록 단일 정의한다. 서버는 이
예외를 잡아 PRD 형식의 JSON {code, message} 으로 직렬화한다.
"""

from __future__ import annotations


class GeomTokError(Exception):
    """모든 GeomTok 도메인 에러의 기반. `code` 는 PRD 표준 코드."""
    code = "GEOMTOK_ERROR"
    http_status = 400

    def __init__(self, message: str = "", **detail):
        super().__init__(message or self.code)
        self.message = message or self.code
        self.detail = detail

    def to_payload(self) -> dict:
        p = {"code": self.code, "message": self.message}
        if self.detail:
            p["detail"] = self.detail
        return p


class ParseError(GeomTokError):
    code = "PARSE_ERROR"


class ParseUnsupportedElement(GeomTokError):
    code = "PARSE_UNSUPPORTED_ELEMENT"


class PayloadTooLarge(GeomTokError):
    code = "PAYLOAD_TOO_LARGE"
    http_status = 413


class VocabMismatch(GeomTokError):
    code = "VOCAB_MISMATCH"


class VersionRequired(GeomTokError):
    code = "VERSION_REQUIRED"
