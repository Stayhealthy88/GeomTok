"""GeomTok 매니지드 API 서버 (FastAPI). OSS 코어와 동일 코드·동일 vocab 공유."""
from .app import create_app, app

__all__ = ["create_app", "app"]
