"""pytest 설정 — 패키지 루트를 import 경로에 추가하고, 스크립트형 테스트는
직접 수집에서 제외(test_all_scripts.py가 서브프로세스로 실행)."""
import glob
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

# 스크립트형 테스트(자체 sys.exit)는 pytest가 직접 import하면 INTERNALERROR.
# test_all_scripts.py만 수집하고 나머지 test_*.py는 무시한다.
collect_ignore = [
    p for p in glob.glob(os.path.join(_ROOT, "tests", "test_*.py"))
    if os.path.basename(p) != "test_all_scripts.py"
]
