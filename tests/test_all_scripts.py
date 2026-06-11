"""pytest 진입점 — 스크립트형 테스트 파일들을 서브프로세스로 실행하고
종료 코드 0(전부 통과)을 단언한다. CI/`pytest`에서 한 번에 수집·실행 가능.

각 테스트 파일은 자체 pass/fail 카운터로 동작하고 실패 시 exit code 1을 반환하므로,
여기서는 그 종료 코드만 검사한다. (개별 파일을 직접 실행해도 동일하게 동작.)
"""
import os
import subprocess
import sys

import pytest

_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_DIR)
_SCRIPTS = sorted(
    f for f in os.listdir(_DIR)
    if f.startswith("test_") and f.endswith(".py") and f != "test_all_scripts.py"
)


@pytest.mark.parametrize("script", _SCRIPTS)
def test_script_passes(script):
    env = dict(os.environ, PYTHONPATH=_ROOT)
    r = subprocess.run([sys.executable, os.path.join(_DIR, script)],
                       cwd=_ROOT, env=env, capture_output=True, text=True)
    assert r.returncode == 0, f"{script} failed:\n{r.stdout[-2000:]}\n{r.stderr[-1000:]}"
