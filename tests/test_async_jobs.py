"""
비동기 잡 백엔드 테스트 (PRD §7.1 / §8)
========================================
진짜 비동기를 결정적으로 증명한다 — 느린 처리기를 주입해 워커 스레드가
백그라운드에서 도는 동안 submit 이 즉시 반환하고, 클라이언트가 running·진행률
증가를 관찰하며, 협조적 취소가 동작함을 확인한다. blob 입출력·웹훅·에러격리도.
"""

import os
import sys
import time
import json
import tempfile
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geomtok.server.jobs import (
    JobManager, InMemoryJobStore, LocalBlobStore,
    QUEUED, RUNNING, SUCCEEDED, PARTIAL, FAILED, CANCELLED,
)

passed = 0
failed = 0


def check(cond, msg):
    global passed, failed
    if cond:
        print(f"  [PASS] {msg}"); passed += 1
    else:
        print(f"  [FAIL] {msg}"); failed += 1


def slow_process(delay=0.03, fail_ids=()):
    def _p(op, item):
        time.sleep(delay)
        if item.get("id") in fail_ids:
            raise ValueError("boom")
        return {"id": item.get("id"), "ok": True, "n": item.get("id")}, True
    return _p


print("=" * 60)
print("비동기 잡 백엔드 테스트")
print("=" * 60)

print("\n[1] 진짜 비동기 — submit 즉시 반환, running 관찰")
mgr = JobManager(slow_process(0.03), "geomtok-1.0.0", "geom-5561-v1", max_workers=2)
items = [{"id": i, "svg": "x"} for i in range(10)]   # 10×30ms = 300ms
t0 = time.time()
job = mgr.submit("tokenize", items=items)
submit_ms = (time.time() - t0) * 1000
check(submit_ms < 50, f"submit 즉시 반환 ({submit_ms:.0f}ms < 50ms, 300ms 작업)")
check(job.status in (QUEUED, RUNNING), f"제출 직후 상태 queued/running ({job.status})")

seen = set()
progresses = []
for _ in range(300):
    j = mgr.get(job.job_id)
    seen.add(j.status)
    progresses.append(j.done)
    if j.status in (SUCCEEDED, PARTIAL, FAILED, CANCELLED):
        break
    time.sleep(0.01)
check(RUNNING in seen, f"running 상태 관찰됨 ({seen})")
check(progresses[-1] == 10 and any(0 < p < 10 for p in progresses),
      f"진행률 점증 관찰 (done 궤적 끝={progresses[-1]})")
check(mgr.get(job.job_id).status == SUCCEEDED, "최종 succeeded")
check(mgr.get(job.job_id).n_ok == 10, "10건 모두 성공")

print("\n[2] 협조적 취소 — 다음 아이템 경계에서 중단")
mgr2 = JobManager(slow_process(0.04), "v", "vid", max_workers=1)
job2 = mgr2.submit("tokenize", items=[{"id": i} for i in range(50)])  # 2s
time.sleep(0.1)                       # 몇 건 처리되게
ok = mgr2.cancel(job2.job_id)
check(ok, "실행 중 cancel 요청 수락")
for _ in range(300):
    if mgr2.get(job2.job_id).status in (CANCELLED, SUCCEEDED, PARTIAL):
        break
    time.sleep(0.01)
fin = mgr2.get(job2.job_id)
check(fin.status == CANCELLED, f"취소됨 ({fin.status})")
check(fin.done < 50, f"전량 처리 전 중단 (done={fin.done} < 50)")
check(mgr2.cancel(job2.job_id) is False, "종료된 잡 재취소 → False")

print("\n[3] 에러 격리 (on_error=skip) + partial")
mgr3 = JobManager(slow_process(0.005, fail_ids={2, 5}), "v", "vid")
job3 = mgr3.submit("tokenize", items=[{"id": i} for i in range(8)], on_error="skip")
for _ in range(300):
    if mgr3.get(job3.job_id).status in (SUCCEEDED, PARTIAL, FAILED):
        break
    time.sleep(0.01)
j3 = mgr3.get(job3.job_id)
check(j3.status == PARTIAL, f"일부 실패 → partial ({j3.status})")
check(j3.n_ok == 6 and j3.n_failed == 2, f"n_ok=6 n_failed=2 (실제 {j3.n_ok}/{j3.n_failed})")
check(j3.public()["partial"] is True, "public().partial == True")

print("\n[4] blob 입출력 (file:// NDJSON, 대규모 경로)")
ind = tempfile.mktemp(suffix=".ndjson"); outd = tempfile.mktemp(suffix=".ndjson")
with open(ind, "w") as f:
    for i in range(15):
        f.write(json.dumps({"id": i}) + "\n")
mgr4 = JobManager(slow_process(0.002), "v", "vid")
job4 = mgr4.submit("tokenize", input_uri="file://" + ind, output_uri="file://" + outd)
for _ in range(300):
    if mgr4.get(job4.job_id).status in (SUCCEEDED, PARTIAL, FAILED):
        break
    time.sleep(0.01)
j4 = mgr4.get(job4.job_id)
check(j4.status == SUCCEEDED and j4.done == 15, f"blob 입력 15건 처리 (done={j4.done})")
out_lines = [json.loads(l) for l in open(outd) if l.strip()]
check(len(out_lines) == 15, f"결과 NDJSON 15줄 기록 ({len(out_lines)})")
check("results_url" in j4.public(), "응답에 results_url (인라인 결과 대신 blob)")

print("\n[5] 웹훅 best-effort — 종료 시 POST")
received = {}
ev = threading.Event()
from http.server import BaseHTTPRequestHandler, HTTPServer

class _H(BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        received["body"] = json.loads(self.rfile.read(n))
        self.send_response(200); self.end_headers()
        ev.set()
    def log_message(self, *a): pass

srv = HTTPServer(("127.0.0.1", 0), _H)
port = srv.server_address[1]
threading.Thread(target=srv.handle_request, daemon=True).start()
mgr5 = JobManager(slow_process(0.002), "v", "vid")
mgr5.submit("tokenize", items=[{"id": 1}],
            webhook_url=f"http://127.0.0.1:{port}/hook")
got = ev.wait(timeout=5)
srv.server_close()
check(got and received.get("body", {}).get("status") == SUCCEEDED,
      f"웹훅 수신 + status=succeeded (got={got})")

print("\n" + "=" * 60)
print(f"  {passed} passed, {failed} failed")
print("=" * 60)
sys.exit(0 if failed == 0 else 1)
