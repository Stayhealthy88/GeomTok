"""pytest 설정 — 패키지 루트를 import 경로에 추가."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
