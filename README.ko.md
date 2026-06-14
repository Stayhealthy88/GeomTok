<p align="center">
  <img src="assets/hero_banner.svg" alt="GeomTok — 벡터 그래픽을 위한 기하 네이티브 토큰화" width="100%"/>
</p>

<p align="center">
  <a href="LICENSE"><img alt="License: Apache-2.0" src="https://img.shields.io/badge/License-Apache_2.0-8b6cff.svg"></a>
  <img alt="Python 3.9+" src="https://img.shields.io/badge/python-3.9%2B-37e6d4.svg">
  <img alt="Tests" src="https://img.shields.io/badge/tests-251_passing-54e08a.svg">
  <img alt="Core deps" src="https://img.shields.io/badge/core-numpy_only-ff5d9e.svg">
  <a href="PAPER.md"><img alt="Paper" src="https://img.shields.io/badge/paper-PAPER.md-ffc857.svg"></a>
  <a href="README.md"><img alt="English" src="https://img.shields.io/badge/lang-English-aeb6d6.svg"></a>
</p>

---

## 왜 GeomTok인가

언어 모델은 벡터 그래픽(SVG)을 산문처럼 토큰화합니다 — `150.5` 같은 좌표를 `1`, `50`, `.`, `5`로 잘게 쪼개버려, *공간 속 한 점*이라는 의미가 사라집니다. **GeomTok**은 SVG를 기하 프리미티브 토큰(명령·도형·어휘 추가 0의 고정소수점 좌표 코덱)으로 매핑하는 기하 네이티브 토크나이저로, 모델이 기하를 기하로 읽게 합니다.

**torch 없이 `numpy`만으로 동작하는 OSS 코어** + 매니지드 FastAPI 서비스 + 생성기가 아닌 **토크나이저 자체를 평가하는** 렌더 기반 프로토콜 **GeomTok-Eval**로 구성됩니다.

<p align="center">
  <img src="assets/fig1_pipeline.svg" alt="GeomTok 파이프라인" width="90%"/>
</p>

## 핵심 발견: 압축 ≠ 모델가능성

동일한 소형 트랜스포머에서 토크나이저만 바꾸는 통제 실험 결과, 기하 프리미티브 토큰이 도메인 학습 텍스트 BPE보다 held-out 모델링에서 **26% 우수**하고, 반직관적으로 **더 압축된 학습-머지 변형보다도 우수**합니다. 더 많이 압축한다고 다운스트림 모델링이 좋아지는 게 아닙니다.

<p align="center">
  <img src="assets/fig2_compression_modelability.svg" alt="압축 vs 모델가능성" width="78%"/>
</p>

| 토크나이저 (동일 2.5M 모델, 실세계 아이콘) | tokens/icon | held-out NLL bits/icon ↓ |
|---|---|---|
| 도메인 char-BPE | 159 | 775 ± 6 |
| GeomTok-L1 + 학습 BPE *(최고 압축)* | 59 | 666 ± 2 |
| **GeomTok-L1** | 104 | **576 ± 4** |

이 격차는 모델 용량(0.9M → 9.3M)에 따라 **유지·확대**되고, 2번째 코퍼스에서 **재현**됩니다. 전체 연구: **[PAPER.md](PAPER.md)**.

## 왕복 충실도는 진짜다 — 무손실이 아니라 정직하다

GeomTok의 좌표 코덱은 경계-한정 오차(평균 **1.78px**, 최대 3.37px / 300px 캔버스), 렌더 **SSIM 0.929** — 3.54× 압축의 정직한 비용입니다. 시도했던 곡률 적응 격자는 *직선을 뒤틀어* 폐기하고 균일 격자를 채택했습니다:

<p align="center">
  <img src="assets/fig3_render_panel.svg" alt="원본 vs 균일격자 vs 적응 쿼드트리 왕복" width="62%"/>
</p>

## 빠른 시작

```bash
pip install geomtok                 # 코어 (numpy만)
pip install 'geomtok[eval,server]'  # + 렌더 평가 + 매니지드 API
```

```python
import geomtok

out = geomtok.tokenize("<svg viewBox='0 0 24 24'><path d='M4 4 L20 4 L20 20 Z'/></svg>")
print(out["n_tokens"], out["tokenizer_version"], out["vocab_id"])

svg = geomtok.detokenize(out["token_ids"],
                         tokenizer_version=out["tokenizer_version"],
                         vocab_id=out["vocab_id"])["svg"]   # FSA 유효 SVG 보장
```

매니지드 API 로컬 실행:

```bash
geomtok-serve --port 8000
```

| 라우트 | 기능 |
|---|---|
| `POST /v1/tokenize` · `/v1/detokenize` | 단건 SVG ↔ 토큰; 결정적·FSA 검증 |
| `POST /v1/eval` | GeomTok-Eval 렌더 기반 프로토콜 (내장 또는 원격 토크나이저) |
| `POST /v1/batch` · `/v1/stream` | 동기 배치(≤1000, ≤32MB) · NDJSON 스트리밍 |
| `POST /v1/batch/jobs` · `GET`/`DELETE /v1/batch/jobs/{id}` | **진짜 비동기** 잡 — 워커·실시간 진행률·취소·웹훅 |
| `GET /v1/vocab/{id}` · `/v1/healthz` | 불변 매니페스트 · 헬스체크 |

엔드포인트별 적합성·프로덕션 여부: **[API_STATUS.md](API_STATUS.md)**.

## v1.0 제공 범위

| 능력 | 상태 |
|---|---|
| 파서 · transform 평탄화 · viewBox 정규화 · 호 평탄화 | ✅ OSS 코어 |
| L1 기하 토큰 + **어휘 추가 0 스칼라 고정소수점 좌표 코덱** (0.04px) | ✅ OSS 코어 |
| **L2 = 학습 BPE-on-L1 머지** (수작업 매크로 폐기 — 실데이터 0% 발화) | ✅ OSS 코어 |
| FSA 문법 제약 디코딩 (torch-free) — **유효 SVG 보장** | ✅ OSS 코어 |
| 불변 vocab 매니페스트 — 비트-동일·오프라인 인코드/디코드 | ✅ OSS 코어 |
| GeomTok-Eval/1.0 — 렌더 SSIM·값수준 좌표오차·파싱률·토큰 경제 | ✅ OSS 코어 |
| 매니지드 API (FastAPI): 진짜 비동기 잡 + NDJSON 스트림 포함 9 라우트 | ✅ `[server]` |

> **생성은 v1.0의 명시적 비목표입니다.** 현 모델은 토이 규모(2.5M·CPU·단색 path 아이콘). 토크나이저+평가가 프로덕션 산출물이고, 생성은 Phase 2(펀딩 게이트)입니다.

## 무엇이 독보적인가

2024–2026 SVG 토큰화 문헌(HiVG·OmniSVG·LLM4SVG·StrokeNUWA·InternSVG·CNM·GeoBPE)에 대조 검증:

- **SVG에서 held-out NLL 기반 통제 토크나이저-스왑** — 동일 백본에 토크나이저만 분리한 선행 SVG 연구 없음.
- **GeomTok-Eval: 생성기 분리 토크나이저 프로토콜** — 기존 SVG 벤치마크(VGBench·SVGenius·VectorGym·LOO)는 *생성기*를 평가.
- **학습 비제약 머지가 HiVG식 구조-제약 머지를 이김** (동일 예산 172 vs 236 tokens/icon) — 미발표 직접 비교.
- **어휘 추가 0 스칼라 고정소수점 코덱** — HiVG는 좌표 토큰 2,384개·OmniSVG ~40k 추가; GeomTok은 0개.

"압축 ≠ 모델가능성"은 텍스트(PathPiece, EMNLP'24)·래스터 이미지(arXiv:2412.16326, NeurIPS'25)에 선례가 있으나, GeomTok은 **벡터그래픽스 최초**이며 반대 방향(압축이 신뢰성 있게 도움 안 됨)을 보입니다.

## 로드맵

- [x] **v1.0** — torch-free OSS 코어·GeomTok-Eval·매니지드 FastAPI. *실세계 아이콘 검증: 파싱+왕복 100%, FSA 유효, 결정적.*
- [ ] **Phase 2** — 대규모 생성 (현재 토이 규모; v1.0 비목표)
- [ ] **Phase 3** — Figma / Canva 플러그인

## 라이선스

**Apache-2.0** — 코어·평가 프로토콜·vocab 매니페스트 전부 오픈(크리플링 없음). 수익은 알고리즘 비공개가 아니라 매니지드 호스팅·SLA·지원에서. [LICENSE](LICENSE)·[NOTICE](NOTICE) 참조.

<p align="center"><sub>GeomTok — 모델에게 기하를 기하로 보는 법을 가르친다.</sub></p>
