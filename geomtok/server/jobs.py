"""
GeomTok 비동기 잡 백엔드 (PRD §7.1 / §8)
==========================================
대규모 배치(10k–1M)를 위한 **진짜 비동기** 처리. 제출은 즉시 반환하고
백그라운드 워커가 처리하므로 클라이언트는 `running` 상태와 진행률 증가를
실제로 관찰한다(동기 stub 아님).

설계 원칙 — OSS 코어는 `pip install`(stdlib + fastapi)만으로 동작해야 하므로
Redis/Celery/S3 같은 무거운 의존 없이 **표준 라이브러리만** 사용한다. 단,
교체 가능한 인터페이스(JobStore·BlobStore)를 두어 프로덕션에서 Redis/DB·
S3/GCS 백엔드를 라우트 수정 없이 드롭인할 수 있다.

구성:
    Job          상태/진행률/결과/메타를 담는 레코드
    JobStore     잡 영속화 인터페이스 (InMemoryJobStore = 단일노드 기본)
    BlobStore    대규모 입출력 NDJSON 인터페이스 (LocalBlobStore = file:// 기본)
    JobManager   ThreadPoolExecutor 워커 + 상태 전이 + 취소 + 웹훅
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple

# 잡 상태 (PRD §8)
QUEUED, RUNNING, SUCCEEDED, FAILED, PARTIAL, CANCELLED = (
    "queued", "running", "succeeded", "failed", "partial", "cancelled")
_TERMINAL = {SUCCEEDED, FAILED, PARTIAL, CANCELLED}


@dataclass
class Job:
    job_id: str
    op: str
    status: str = QUEUED
    total: int = 0
    done: int = 0
    n_ok: int = 0
    n_failed: int = 0
    results: List[dict] = field(default_factory=list)   # inline 결과(소규모)
    results_uri: Optional[str] = None                    # blob 결과(대규모)
    error: Optional[dict] = None
    webhook_url: Optional[str] = None
    created_at: float = 0.0
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    tokenizer_version: str = ""
    vocab_id: str = ""

    def public(self, include_results: bool = True) -> dict:
        """PRD §8 상태 응답 형태."""
        d = {
            "job_id": self.job_id, "op": self.op, "status": self.status,
            "progress": {"done": self.done, "total": self.total},
            "partial": self.n_failed > 0 and self.status in _TERMINAL,
            "stats": {"n_ok": self.n_ok, "n_failed": self.n_failed},
            "tokenizer_version": self.tokenizer_version, "vocab_id": self.vocab_id,
            "created_at": self.created_at, "started_at": self.started_at,
            "finished_at": self.finished_at,
        }
        if self.error is not None:
            d["error"] = self.error
        if self.results_uri is not None:
            d["results_url"] = self.results_uri
        if include_results and self.results_uri is None:
            d["results"] = self.results
        return d


# --------------------------------------------------------------------------- #
# 영속화 인터페이스 (교체 가능)
# --------------------------------------------------------------------------- #

class JobStore:
    """잡 영속화 인터페이스. 프로덕션은 Redis/DB 구현으로 교체."""
    def put(self, job: Job) -> None: raise NotImplementedError
    def get(self, job_id: str) -> Optional[Job]: raise NotImplementedError
    def update(self, job: Job) -> None: raise NotImplementedError
    def list_ids(self) -> List[str]: raise NotImplementedError


class InMemoryJobStore(JobStore):
    """스레드 안전 인메모리 잡 스토어 (단일노드 기본)."""
    def __init__(self):
        self._d: Dict[str, Job] = {}
        self._lock = threading.RLock()

    def put(self, job: Job) -> None:
        with self._lock:
            self._d[job.job_id] = job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._d.get(job_id)

    def update(self, job: Job) -> None:
        with self._lock:
            self._d[job.job_id] = job

    def list_ids(self) -> List[str]:
        with self._lock:
            return list(self._d.keys())


class BlobStore:
    """대규모 입출력 NDJSON 인터페이스. 프로덕션은 S3/GCS 구현으로 교체."""
    def read_items(self, uri: str) -> Iterator[dict]: raise NotImplementedError
    def write_ndjson(self, uri: str, rows: Iterable[dict]) -> None: raise NotImplementedError


class LocalBlobStore(BlobStore):
    """file:// 로컬 파일시스템 NDJSON (기본). 한 줄당 JSON 객체."""
    @staticmethod
    def _path(uri: str) -> str:
        return uri[len("file://"):] if uri.startswith("file://") else uri

    def read_items(self, uri: str) -> Iterator[dict]:
        with open(self._path(uri), "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def write_ndjson(self, uri: str, rows: Iterable[dict]) -> None:
        with open(self._path(uri), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")


# --------------------------------------------------------------------------- #
# 잡 매니저
# --------------------------------------------------------------------------- #

# 처리기: (op, item_dict) → (result_dict, ok). app.py 가 GeomTokenizer 로 바인딩.
ProcessFn = Callable[[str, dict], Tuple[dict, bool]]


class JobManager:
    """ThreadPoolExecutor 기반 진짜 비동기 잡 처리.

    submit() 은 즉시 QUEUED 잡을 반환하고 워커 스레드가 RUNNING 으로 전이해
    아이템마다 progress 를 갱신한다. 인라인 결과는 잡에 보관하고(소규모),
    output_uri 가 주어지면 BlobStore 로 스트리밍 기록한다(대규모).
    """

    INLINE_RESULT_CAP = 10_000   # 이 수를 넘으면 결과를 blob 으로만 보관

    def __init__(self, process: ProcessFn,
                 tokenizer_version: str, vocab_id: str,
                 store: Optional[JobStore] = None,
                 blob: Optional[BlobStore] = None,
                 max_workers: int = 2):
        self._process = process
        self._ver = tokenizer_version
        self._vid = vocab_id
        self.store = store or InMemoryJobStore()
        self.blob = blob or LocalBlobStore()
        self._pool = ThreadPoolExecutor(max_workers=max_workers,
                                        thread_name_prefix="geomtok-job")
        self._cancel: Dict[str, threading.Event] = {}
        self._lock = threading.RLock()

    # ---- 제출 ---- #

    def submit(self, op: str, *, items: Optional[List[dict]] = None,
               input_uri: Optional[str] = None, level: str = "L1",
               lean: bool = False,
               on_error: str = "skip", webhook_url: Optional[str] = None,
               output_uri: Optional[str] = None) -> Job:
        """잡 제출 — 즉시 QUEUED 반환. items(인라인) 또는 input_uri(blob) 택1.

        level/lean 은 잡 레벨 기본값이며 아이템 딕셔너리에 병합된다 —
        아이템(NDJSON 라인)이 자체 level/lean 키를 가지면 그 값이 이긴다."""
        job_id = "job_" + uuid.uuid4().hex[:16]
        total = len(items) if items is not None else 0   # blob 입력은 미상→0
        job = Job(job_id=job_id, op=op, status=QUEUED, total=total,
                  webhook_url=webhook_url, created_at=time.time(),
                  results_uri=output_uri,
                  tokenizer_version=self._ver, vocab_id=self._vid)
        self.store.put(job)
        with self._lock:
            self._cancel[job_id] = threading.Event()
        self._pool.submit(self._execute, job_id, items, input_uri,
                          {"level": level, "lean": lean}, on_error)
        return job

    # ---- 실행 (워커 스레드) ---- #

    def _iter_items(self, items, input_uri) -> Iterator[dict]:
        if items is not None:
            yield from items
        elif input_uri:
            yield from self.blob.read_items(input_uri)

    def _execute(self, job_id, items, input_uri, defaults, on_error):
        job = self.store.get(job_id)
        if job is None:
            return
        cancel = self._cancel.get(job_id)
        job.status = RUNNING
        job.started_at = time.time()
        self.store.update(job)

        out_rows: List[dict] = []
        stream_to_blob = job.results_uri is not None
        blob_buffer: List[dict] = []
        try:
            for item in self._iter_items(items, input_uri):
                if cancel is not None and cancel.is_set():
                    job.status = CANCELLED
                    break
                job.total = max(job.total, job.done + 1)   # blob 입력 시 점증
                try:
                    # 잡 레벨 기본값 아래에 아이템 키를 병합 — 아이템이 이긴다.
                    # (v1.0 은 level 을 죽은 파라미터로 무시했다 — v1.1 수정)
                    result, ok = self._process(job.op, {**defaults, **item})
                except Exception as e:  # noqa: BLE001 — 아이템 격리
                    if on_error == "fail_fast":
                        raise
                    result, ok = {"id": item.get("id"), "ok": False,
                                  "error": {"code": "ITEM_ERROR", "message": str(e)}}, False
                job.done += 1
                job.n_ok += int(ok)
                job.n_failed += int(not ok)
                if stream_to_blob:
                    blob_buffer.append(result)
                elif len(job.results) < self.INLINE_RESULT_CAP:
                    job.results.append(result)
                else:
                    stream_to_blob = True            # 캡 초과 → blob 전환
                    job.results_uri = job.results_uri or f"file:///tmp/{job_id}_results.ndjson"
                    blob_buffer = list(job.results) + [result]
                    job.results = []
                if job.done % 50 == 0:               # 주기적 진행률 체크포인트
                    self.store.update(job)

            if stream_to_blob and job.results_uri:
                self.blob.write_ndjson(job.results_uri, blob_buffer)

            if job.status != CANCELLED:
                job.status = SUCCEEDED if job.n_failed == 0 else PARTIAL
        except Exception as e:  # noqa: BLE001 — 잡 전체 실패(fail_fast 등)
            job.status = FAILED
            job.error = {"code": "JOB_FAILED", "message": str(e)}
        finally:
            job.finished_at = time.time()
            self.store.update(job)
            with self._lock:
                self._cancel.pop(job_id, None)
            self._fire_webhook(job)

    # ---- 조회·취소 ---- #

    def get(self, job_id: str) -> Optional[Job]:
        return self.store.get(job_id)

    def cancel(self, job_id: str) -> bool:
        """협조적 취소 — 다음 아이템 경계에서 중단. 이미 종료면 False."""
        job = self.store.get(job_id)
        if job is None or job.status in _TERMINAL:
            return False
        ev = self._cancel.get(job_id)
        if ev is not None:
            ev.set()
        return True

    # ---- 웹훅 (best-effort) ---- #

    def _fire_webhook(self, job: Job) -> None:
        if not job.webhook_url:
            return
        try:
            import urllib.request
            data = json.dumps(job.public(include_results=False)).encode("utf-8")
            req = urllib.request.Request(
                job.webhook_url, data=data,
                headers={"Content-Type": "application/json"}, method="POST")
            urllib.request.urlopen(req, timeout=5).close()
        except Exception:  # noqa: BLE001 — 웹훅 실패는 잡 상태에 영향 없음
            pass

    def shutdown(self):
        self._pool.shutdown(wait=False)
