# GeomTok v1.0 — Product Requirements Document (PRD)

> 문서 버전: 1.0 (Draft for Build) · 작성일: 2026-06-12 · 라이선스: Apache-2.0 · 상태: 빌드 승인 대기

---

## 1. 개요 (TL;DR)

GeomTok은 SVG를 **기하 프리미티브 단위로 토큰화**하는 오픈소스 토크나이저와, 토크나이저 품질을 **렌더 기반으로 게임-내성(game-resistant) 있게** 평가하는 GeomTok-Eval 툴킷이다. 최종 사용자용 앱이 아니라 디자인 툴·SaaS 벤더가 자사 제품에 "벡터 AI"를 붙일 때 바닥에 까는 **인프라(API/SDK)**다. 핵심 가치는 검증된 실측치에 있다 — 실제 아이콘 2,682개에 대해 **100% 파싱 + 라운드트립**, 동일 5,561 어휘 기준 텍스트 토크나이저 대비 **3.54배 토큰 압축**, 300px 캔버스에서 좌표 오차 **평균 1.78px / 최대 3.37px**, 렌더 **SSIM 0.929**. 가장 중요한 발견은 **기하 프리미티브 토큰이 다운스트림에 최적 기질(substrate)**이라는 점이다: 기하 토큰은 84% 렌더 가능한 SVG를 생성하는 반면 텍스트 토크나이저는 0%다. v1.0은 **토크나이저 + 평가만** 프로덕션으로 출하한다. 생성(generation)은 현재 2.5M 파라미터 토이 스케일이므로 **명시적 비목표**이며 대규모 학습 투자가 선행되어야 하는 후속 단계다. 사업은 완전 오픈소스(Apache-2.0)이고, 수익은 매니지드 호스팅 + 지원/컨설팅에서 나온다.

### 핵심 결정 요약

| 항목 | 결정 |
|---|---|
| 제품 형태 | 개발자용 토크나이저 SDK + GeomTok-Eval + 매니지드 API (인프라, 엔드유저 앱 아님) |
| 라이선스 | Apache-2.0 (코어 전체 오픈, 크리플링 없음) |
| 비즈니스 모델 | OSS 무료 → 매니지드 호스팅 + Enterprise(SSO/SLA/VPC) + 지원/컨설팅 수익 |
| 1차 타깃 | Figma/Canva급 디자인툴, 아이콘 라이브러리, 노코드 벤더 |
| 2차 타깃 | SVG 모델을 만드는 AI/ML 팀 |
| v1.0 범위 | 토크나이저(파싱·정규화·인코드/디코드) + GeomTok-Eval + 매니지드 API |
| v1.0 비목표 | **생성 모델 제품화** (토이 스케일, 후속 Phase 2) |
| 검증 수치 | 라운드트립 100% (2,682 아이콘), 3.54x 압축, 1.78/3.37px, SSIM 0.929, 84% vs 0% 렌더 |
| 결정성 보장 | `tokenizer_version` + `vocab_id` 고정 시 비트-동일 재현 |
| 유효성 보장 | FSA 문법 제약 디코딩으로 항상 well-formed SVG |
| 차별화 축 | (1) 학습된 비제약 머지가 구조-제약 머지(HiVG)보다 우수, (2) 규칙 기반 매크로는 실데이터에서 0% 발화, (3) 생성기-분리 평가 프로토콜은 공개 등가물이 없음 |

---

## 2. 배경·문제 (왜 지금)

### 2.1 트리거 — 왜 지금인가

디자인툴/SaaS 리더십이 "생성형 벡터" 기능을 다음 릴리스에 약속하고, 엔지니어링 리드의 첫 스파이크(LLM에게 SVG를 직접 뱉게 하기)가 대부분 깨진 출력을 반환하는 순간이 시장 진입 트리거다. 이 시점에 PM은 (a) 신뢰할 수 있는 기질과 (b) 그것이 작동함을 증명할 방법을 멀티-쿼터 빌드에 인력을 투입하기 전에 필요로 한다.

### 2.2 문제 — 오늘 무엇이 깨지는가

1. **유효성 붕괴.** 범용 LLM에 SVG를 뱉게 하면 상당 비율이 구조적으로 무효한 출력을 낸다. 텍스트 토크나이저 베이스라인은 **0% 렌더 가능 SVG**를 생성한다 — 즉 현 상태로는 기능 출시 자체가 불가능하다.
2. **측정 불가능성.** 표현(representation) 선택이 좋은지 나쁜지 판단할 falsifiable한 방법이 없다. 순진한 지표는 게임 가능하다(랜덤 베이스라인이 1.0을 받을 수 있음). 엔지니어링 주장이 반증 불가능해진다.
3. **실데이터에서 조용히 실패.** 실세계 아이콘 세트는 순진한 파서를 깨뜨린다(중첩 transform, viewBox 변형, 다양한 path 인코딩). "데모에서는 됨"이 고객 파일에서 조용히 실패한다.
4. **압축 신화.** 압축 중심 벤더가 "토큰이 작을수록 모델이 좋다"를 과대 판매한다. 팀에는 이를 반박할 도구가 없다.
5. **언디퍼런시에이티드 중노동.** in-house로 견고한 SVG 토크나이저를 짓는 것은 multi-month 늪이다 — path 문법 엣지케이스, transform 수학, viewBox 정규화, 라운드트립 충실도 테스트 모두.

### 2.3 연구 근거 (검증된 실측, 부풀리지 않음)

| 발견 | 수치 | 함의 |
|---|---|---|
| 라운드트립 | 실제 아이콘 **2,682개에서 100%** 파싱+라운드트립 | 고객 파일이 조용히 깨지지 않음 |
| 압축 | 동일 5,561 어휘 텍스트 토크나이저 대비 **3.54x** | 시퀀스 단축 → 모델/서빙 비용 감소 |
| 좌표 충실도 | 300px 캔버스 **평균 1.78px / 최대 3.37px** | 시각적 드리프트 설계 기준 제공 |
| 렌더 충실도 | **SSIM 0.929** | 픽셀 단위 재구성 품질 |
| **다운스트림 기질** | 기하 프리미티브 토큰 **576 bits/icon** vs 텍스트 BPE **775**, **~28σ** | 기하 토큰이 최적 기질 |
| **모델 가능성** | 기하 토큰 **84% 렌더 가능** vs 텍스트 토크나이저 **0%** | 출시 가능 기능 vs 데모의 차이 |
| 압축 ≠ 모델 가능성 | 가장 많이 압축된 변형이 **오히려 더 나쁘게** 모델링/생성 | 압축 신화의 반증 |
| 좌표 코덱 | 스칼라 고정소수점 **0.04px 오차, 어휘 추가 0** | 양자화 오차/어휘 폭증 회피 |
| 유효성 | FSA 문법 제약 디코딩 → 유효 SVG **보장** | "모델이 쓰레기 뱉음" 실패 모드 제거 |

> 정직성 원칙: 위 수치 외 일반화 금지. "first", "pre-structured advantage" 같은 이미 반박된 우월 표현 금지. 부정 결과(압축 ≠ 모델 가능성)는 숨기지 않고 신뢰 후크로 전면 사용.

---

## 3. 목표·비목표

### 3.1 목표 (측정 가능)

| # | 목표 | 측정 기준 |
|---|---|---|
| G1 | 프로덕션급 토크나이저 출하 | 파트너 코퍼스 ≥3개에서 100% 파싱+라운드트립, 평균 좌표오차 ≤1.78px / SSIM ≥0.929 유지 |
| G2 | 결정적·재현 가능한 인코딩 | `tokenizer_version`+`vocab_id` 고정 시 비트-동일 출력; 불변 vocab 매니페스트로 오프라인 인코드/디코드 가능 |
| G3 | 유효성 보장 디코딩 | 모델 생성 토큰 스트림에 대해 FSA 문법 제약으로 well-formed SVG 100% |
| G4 | GeomTok-Eval을 공개 표준으로 | 게임-내성 렌더 기반 프로토콜 공개 + 공개 리더보드; 제3자 ≥2건이 점수 보고 |
| G5 | 개발자 경험(DX) | `pip install`부터 첫 tokenize+detokenize 성공까지 **<15분** |
| G6 | OSS 채택 | 1년차 PyPI+npm 누적 25,000 설치, GitHub 3,000 stars |
| G7 | 매니지드 시드 수익 | 1년차 말 **$8–15K MRR** (학습 신호로서, 1년차 명제 아님) |

### 3.2 비목표 (명시적 Non-Goals)

| # | 비목표 | 근거 |
|---|---|---|
| NG1 | **생성 모델 제품화** | 현 모델은 2.5M 파라미터, CPU, 단색 path 아이콘만. 토이 스케일. 대규모 데이터/학습 투자가 선행되어야 함. v1.0에서는 절대 "생성 가능"으로 마케팅하지 않음. Phase 2로 명시 로드맵화 |
| NG2 | 멀티컬러/그라디언트/필터/래스터 임베드 전체 지원 | v1.0은 path 중심 모노크롬 아이콘 도메인에 집중. `<filter>` 등은 `PARSE_UNSUPPORTED_ELEMENT`로 명시 거부 |
| NG3 | 엔드유저 GUI / 디자인 캔버스 | 인프라(API/SDK)이지 앱이 아님 |
| NG4 | Figma/Canva 플러그인 직접 출하 | Phase 3. v1.0은 SDK/API만 |
| NG5 | 임의 SVG/SVG-애니메이션/SMIL 전체 명세 커버리지 | 정적 벡터 기하에 한정 |
| NG6 | 자체 호스팅 코어 크리플링 | OSS는 완전 기능. 수익은 운영 고통(SLA·스케일·드리프트)에서만 |

---

## 4. 타깃 사용자·페르소나

| 페르소나 | 역할 | 핵심 JTBD | 오늘의 고통 | GeomTok이 주는 것 |
|---|---|---|---|---|
| **Priya** — Vector-AI PM (디자인툴 SaaS) | 'AI 기능' 로드맵 오너, VP Product 보고, 보드가 생성형 벡터 푸시 | 매번 렌더 가능한 SVG를 내는 AI 기능 출시; build-vs-buy 디리스크; 숫자로 로드맵 방어; 무효 벡터로 인한 지원 티켓 회피 | LLM이 무효 SVG 다발; 게임 가능한 내부 벤치마크로 주장 반증 불가; 실아이콘이 파서 깨뜨림; 압축 과대판매 반박 불가 | 2,682 아이콘 100% 라운드트립; 84% vs 0% 렌더 증명; 게임-내성 평가로 falsifiable 결정; 정직한 단계 스토리(지금=토크나이저+평가) |
| **Marcus** — 벡터-AI 임베딩 엔지니어링 리드 | 벡터 파이프라인 오너, in-house 빌드를 피하려는 당사자 | 파서 없이 ingest→tokenize→(model)→valid-SVG 파이프라인; 캔버스 닿기 전 유효성 보장; 메시한 코퍼스를 canonical화; 코덱·어휘를 데이터로 선택/측정 | 견고한 토크나이저 in-house가 multi-month 늪; 손수 양자화의 가시적 오차·어휘 폭증; 자유 디코딩이 렌더 크래시; 커스텀 토크나이저가 dev 코퍼스에 과적합 | drop-in SDK(Apache-2.0): parse·flatten·normalize·tokenize·detokenize 100% 라운드트립; 0.04px 코덱(어휘 0 추가); FSA 보장 디코드; 3.54x 압축 (압축=품질 함정 없이) |
| **Dana** — 아이콘 라이브러리/디자인-에셋 플랫폼 Eng-PM | 수만 SVG 카탈로그 + 툴링 오너; 위생·검색·디덥·AI 저작 | 거대 이질 카탈로그 정규화; 기하 인식 기능(근접중복 탐지, 스타일 일관성, 벡터 검색/클러스터링); on-grid/on-style 저작 보조; 매니지드 인프라로 운영 | 인코딩·중복 transform·viewBox 불일치로 시각적 동일 아이콘이 바이트상 상이 → 디덥/검색 불안정; 공유 기하 표현 부재 → 픽셀(비싸고 손실) 또는 메타데이터(부정확)로 유사도; bespoke 파이프라인을 작은 팀이 유지 | canonical 기하 표현으로 디덥/검색/클러스터링 기반; 배치/스트림/잡 API로 대규모 백필; 매니지드 운영으로 자체 파이프라인 제거 |

> 2차: SVG 모델을 짓는 AI/ML 팀 — GeomTok-Eval을 벤치마크로, 기하 토큰을 학습 기질로 사용.

---

## 5. 핵심 유스케이스

### UC-1 — Build-vs-Buy 디리스크 스파이크 (Priya)
**상황:** 리더십이 생성형 벡터 기능을 다음 릴리스에 약속. **흐름:** Marcus가 `pip install geomtok`으로 자사 아이콘 1,000개를 `/v1/eval`에 통과 → 자사 텍스트 토크나이저 베이스라인이 0% 렌더 가능임을 발견, 기하 L1이 84%임을 확인 → Priya가 이 private GeomTok-Eval 리포트로 멀티-쿼터 빌드 결정을 방어. **성공:** 인력 투입 전에 falsifiable 근거 확보.

### UC-2 — 파이프라인 부트스트랩 (Marcus)
**상황:** 백로그 티켓이 "SVG 토크나이제이션+유효성 레이어 in-house 1–2 쿼터" 추정. **흐름:** Marcus가 Apache-2.0 SDK로 parse→flatten→normalize→tokenize→detokenize를 1일 만에 PoC; 자사 코퍼스에서 라운드트립 100% 확인; FSA 디코드로 무효 출력 실패 모드 제거. **성공:** multi-month 빌드를 파일럿으로 대체.

### UC-3 — 카탈로그 정규화 + 기하 검색 (Dana)
**상황:** 수만 개 이질 SVG 카탈로그. **흐름:** `/v1/batch/jobs`로 전체 코퍼스를 canonical 토큰으로 백필(S3 in/out) → 토큰 표현으로 근접중복 탐지·클러스터링·스타일 일관성 검사. **성공:** 시각적 동일 아이콘이 동일 canonical 표현 → 디덥/검색 신뢰성 확보.

### UC-4 — 고처리량 인덱싱 (Dana/Marcus)
**상황:** CI에서 신규 에셋 매 커밋 토큰화. **흐름:** `/v1/stream`(NDJSON)으로 파이프라인 ingest, 백프레셔 인지, 라인당 결과 스트림. **성공:** 바운디드 메모리로 지속 인덱싱.

### UC-5 — 표현 의사결정 / 벤치마킹 (2차 AI/ML 팀)
**상황:** 자체 SVG 모델용 토크나이저 선택. **흐름:** `/v1/eval`에 `{remote: {tokenize_url, detokenize_url}}`로 제3자 토크나이저를 동일 하니스로 평가 → 576<666<775 bits/icon, 0%/84% 렌더 비교. **성공:** 마케팅 주장이 아닌 게임-내성 점수로 선택.

### UC-6 — 결정성 핀 고정 / 오프라인 (Marcus)
**상황:** 재현 가능한 학습 파이프라인 필요. **흐름:** `GET /v1/vocab/{vocab_id}`로 불변 매니페스트 받아 오프라인 인코드/디코드, `tokenizer_version` 고정. **성공:** 비트-동일 재현.

---

## 6. 제품 범위 — OSS 코어 vs 매니지드 서비스

| 능력 | OSS 코어 (Apache-2.0, 무료) | 매니지드 서비스 (유료) |
|---|---|---|
| 파서 / transform 평탄화 / viewBox 정규화 | ✅ 완전 | ✅ (동일 코드) |
| 토크나이즈 / 디토크나이즈 (L1/L2/L3) | ✅ 완전 | ✅ |
| 스칼라 고정소수점 좌표 코덱 (0.04px) | ✅ | ✅ |
| FSA 문법 제약 디코딩 (유효성 보장) | ✅ | ✅ |
| GeomTok-Eval 프로토콜 + 하니스 | ✅ 완전 + 공개 리더보드 | ✅ + private 리포트 |
| Python SDK + JS 바인딩 | ✅ | ✅ |
| vocab 매니페스트 (불변, 오프라인) | ✅ 다운로드 | ✅ 호스팅·버전 관리 |
| 단건 `/v1/tokenize` `/v1/detokenize` | 로컬 실행 | ✅ 호스팅 엔드포인트 |
| 고처리량 배치 (`/v1/batch`, vectorized CPU) | 직접 구축 필요 | ✅ 코얼레싱·캐싱·50x급 처리량 |
| 비동기 대규모 잡 (`/v1/batch/jobs`, 10k–1M) | 직접 구축 필요 | ✅ 큐·버킷 in/out·웹훅 |
| NDJSON 스트리밍 (`/v1/stream`) | 직접 구축 필요 | ✅ 백프레셔·바운디드 메모리 |
| 라운드트립 100% **SLA 보장** | ❌ (best-effort) | ✅ 계약 SLA |
| 코퍼스 드리프트 모니터링 / 정규화 컨설팅 | ❌ | ✅ |
| SSO / VPC / 온프레미스 / 데이터 격리 | ❌ | ✅ Enterprise |
| 지원·SLA·버전 핀 운영 | 커뮤니티 | ✅ 계약 |

> 경계 원칙: **알고리즘을 팔지 않는다. 알고리즘 주변의 운영 고통을 판다** — 진화하는 고객 코퍼스에 대한 SLA 보장 라운드트립, 고QPS/배치 처리량, 정규화+드리프트 모니터링, 버전 핀+문법 검증 서비스, 컨설팅. OSS는 진짜로 완전하게(크리플링 없이) 유지 → 신뢰 극대화, 수익은 스케일+보증+전문성에서.

---

## 7. 기능 요구사항

### 7.1 API 엔드포인트

| 메서드 | 경로 | 목적 | 한도/비고 |
|---|---|---|---|
| POST | `/v1/tokenize` | SVG 1건 → 기하 토큰 ID (기본 L1; L2 합성/L3 공간 옵션). 코어. `tokenizer_version`+config로 결정적 | svg ≤256KB; `return`으로 페이로드 상세도 선택 |
| POST | `/v1/detokenize` | 토큰 ID → 유효 SVG (라운드트립). 좌표 충실 재구성 | `tokenizer_version` **필수**(디코드 테이블); `vocab_id`는 인코드와 일치 필수 |
| POST | `/v1/eval` | GeomTok-Eval 렌더 기반 프로토콜 실행 (내장 또는 caller 토크나이저). 게임-내성 | `tokenizer`={builtin} 또는 {remote:{tokenize_url,detokenize_url}} |
| POST | `/v1/batch` | 동기 배치 tokenize/detokenize (vectorized CPU) | items ≤1000, 페이로드 ≤32MB, `on_error`∈{skip,fail_fast} |
| POST | `/v1/batch/jobs` | 비동기 대규모 잡 (10k–1M), 매니페스트/버킷 ref | 웹훅+폴; `GET /v1/batch/jobs/{id}` 상태 |
| POST | `/v1/stream` | NDJSON 스트리밍 tokenize, 백프레셔 인지, 순서 보존 | `application/x-ndjson`, chunked |
| GET | `/v1/vocab/{vocab_id}` | 불변 vocab 매니페스트 (id→심볼, 레벨 레이아웃, 좌표 그리드) | 오프라인 인코드/디코드·결정성 백킹 |

### 7.2 예시 페이로드

**`POST /v1/tokenize` 요청**
```json
{ "svg": "<svg viewBox='0 0 24 24'><path d='M4 4 L20 4 L20 20 Z'/><circle cx='12' cy='12' r='6'/></svg>",
  "level": "L1", "config": {"canvas_size": 300, "max_coord_level": 6},
  "return": ["token_ids", "tokens", "metadata"] }
```
**응답**
```json
{ "tokenizer_version": "geomtok-1.0.0", "vocab_id": "geom-5561-v1", "level": "L1",
  "token_ids": [2,17,140,140,17,820,140,3],
  "tokens": ["<BOS>","MOVE_TO","X@140","Y@140","..."],
  "n_commands": 4, "n_tokens": 41, "compression_ratio": 3.54,
  "normalization": {"viewBox_applied": true, "transforms_flattened": 1}, "warnings": [] }
```

**`POST /v1/detokenize` 응답**
```json
{ "svg": "<svg viewBox=\"0 0 300 300\"><path d=\"M140 140 ...Z\"/><circle cx=\"150\" cy=\"150\" r=\"75\"/></svg>",
  "n_tokens": 41, "valid": true,
  "fidelity": {"coord_mean_px": 1.78, "coord_max_px": 3.37, "canvas": 300} }
```

**`POST /v1/eval` 응답(요약)**
```json
{ "protocol": "GeomTok-Eval/1.0",
  "summary": {"scenes": 2, "attr_mean_err": 1.78, "attr_max_err": 3.37, "count_acc": 1.0,
              "render_ssim_mean": 0.929, "tokens": 86, "tokens_bpe": 311, "compression_vs_baseline": 3.54} }
```

**`POST /v1/batch` 응답(부분 실패)**
```json
{ "op": "tokenize", "results": [
    {"id":"ic_001","ok":true,"token_ids":[2,17,3],"n_tokens":41,"compression_ratio":3.54},
    {"id":"ic_002","ok":false,"error":{"code":"PARSE_UNSUPPORTED_ELEMENT","message":"<filter> not tokenizable"}} ],
  "stats": {"n_ok": 1, "n_failed": 1, "server_ms": 38} }
```

### 7.3 토큰 레벨

| 레벨 | 의미 | 기본 | 용도 |
|---|---|---|---|
| L1 | 기하 프리미티브 (MOVE_TO, LINE_TO, 좌표 등) | ✅ 기본 | 다운스트림 최적 기질 (84% 렌더, 576 bits/icon) |
| L2 | 합성 (학습된 비제약 머지) | 옵션 | 시퀀스 추가 단축; HiVG 구조-제약 머지보다 우수 |
| L3 | 공간 (spatial) | 옵션 | 레이아웃 인식 다운스트림 |

> 압축 ≠ 모델 가능성: 가장 압축된 변형이 모델링에 더 나쁨. **기본은 L1**으로 두고 압축은 명시 옵트인. 문서에서 이 트레이드오프를 정직하게 노출.

### 7.4 SDK 요구사항
- **Python**: `pip install geomtok`. `tokenize() / detokenize() / eval()`이 로컬·매니지드 모두 동일 인터페이스. 첫 성공까지 <15분 quickstart.
- **JS 바인딩**: npm 패키지. 동일 결정성 보장.
- 카피-페이스트 라운드트립 데모 + 인터랙티브 `showcase.html`을 랜딩 아티팩트로.

### 7.5 GeomTok-Eval 요구사항
- 메트릭: `attr_err`(평균/최대), `count_acc`, `render_ssim`, `token_economy`.
- **생성기-분리(generator-separated)**: 토크나이저 품질을 생성기 품질과 분리 측정 → 게임 불가(랜덤 베이스라인 1.0 불가).
- 렌더 기반: 프록시가 아닌 실제 렌더 SSIM.
- 내장 베이스라인: `cl100k_bpe` 등 텍스트 BPE 대조.
- 공개 리더보드: 텍스트-BPE vs 기하 vs HiVG/GeoBPE류, 576<666<775 bits/icon, 0%/84%/35% 렌더 성공 공개.

### 7.6 버전·결정성 요구사항
- 모든 응답에 `tokenizer_version` + `vocab_id` 에코.
- 동일 `tokenizer_version`+config → **비트-동일** 출력 (서버 디폴트는 버전별로 핀).
- vocab 매니페스트는 **불변**; 오프라인 인코드/디코드 가능; 디토크나이즈 시 `tokenizer_version` 필수, `vocab_id` 인코드 일치 필수.
- 에러 코드 표준화: `PARSE_ERROR`, `PARSE_UNSUPPORTED_ELEMENT`, `PAYLOAD_TOO_LARGE`, `VOCAB_MISMATCH`, `VERSION_REQUIRED`.

---

## 8. 비기능 요구사항

| 영역 | 요구사항 |
|---|---|
| 지연 (단건) | `/v1/tokenize` p50 sub-ms CPU 연산; API 왕복 p95 ≤50ms (동일 리전) |
| 처리량 (배치) | `/v1/batch` vectorized CPU 코얼레싱; OSS 라이브러리 대비 매니지드가 유의미 우위(배칭·캐싱·호스티드 vocab)가 무료→유료 전환 wedge |
| 대규모 잡 | `/v1/batch/jobs` 10k–1M 아이템; 진행률·부분 성공(`partial`)·`errors_url`·웹훅 |
| 스트리밍 | `/v1/stream` 바운디드 메모리(양단)·백프레셔·순서 보존 |
| SLA (Enterprise) | 라운드트립 100% 계약 보장(고객 코퍼스), 가동률 SLA, 응답 시간 SLA |
| 멀티테넌시 | 테넌트별 격리; 버킷 in/out은 테넌트 버킷 사용; 잡/스트림 테넌트 스코프 |
| 보안/IP | 입력 SVG는 처리 후 비보존 옵션; Enterprise는 VPC/온프레미스/데이터 격리; 페이로드 한도(256KB/32MB)로 DoS 방어; `<filter>` 등 미지원 요소 명시 거부로 입력 표면 축소 |
| 결정성 | 버전·vocab 핀으로 재현 보장 (비기능적 신뢰 기둥) |
| 가용성 | 매니지드 서비스 목표 99.9% (Enterprise SLA로 격상) |
| 관측성 | quickstart 텔레메트리(time-to-first-token), 토큰 처리량 카운터, 잡 진행률 |

---

## 9. 아키텍처 (텍스트 다이어그램)

```
                         ┌─────────────────────────────────────────────┐
   OSS 코어 (Apache-2.0)  │  geomtok (Python) / @geomtok (JS)            │
   ─ 로컬 실행 가능 ──────▶│  parse → transform-flatten → viewBox-norm   │
                         │  → tokenize(L1/L2/L3) → [coord codec 0.04px] │
                         │  → detokenize [FSA 문법 제약 → valid SVG]    │
                         │  GeomTok-Eval 하니스 (render SSIM/attr/econ) │
                         └──────────────────┬──────────────────────────┘
                                            │ 동일 코드, 동일 vocab 매니페스트
                                            ▼
   매니지드 서비스         ┌─────────────────────────────────────────────┐
   (호스팅·SLA·스케일)     │  API Gateway / Auth(테넌트) / Rate-limit      │
                         │   ├ /v1/tokenize, /v1/detokenize  (단건)     │
                         │   ├ /v1/eval        (생성기-분리 프로토콜)    │
                         │   ├ /v1/batch       (vectorized CPU 코얼레싱) │
                         │   ├ /v1/stream      (NDJSON 백프레셔)         │
                         │   ├ /v1/batch/jobs ─┐                         │
                         │   └ /v1/vocab/{id}  │                         │
                         │                     ▼                         │
                         │   Queue → Worker Pool(CPU) → 결과 라이터       │
                         │                     │           │             │
                         │   테넌트 S3/R2 ◀────┘     Webhook 통지        │
                         │   불변 Vocab Registry (결정성 백킹)           │
                         │   Cache (정규화·토큰 결과)                    │
                         └─────────────────────────────────────────────┘
```

핵심: OSS와 매니지드가 **동일 토크나이저 코드 + 동일 불변 vocab**을 공유 → 로컬과 호스팅 결과가 비트-동일. 매니지드는 그 위에 스케일/큐/캐시/SLA/테넌시만 추가.

---

## 10. 지표·성공기준 (1년차 목표)

| 지표 | 목표 |
|---|---|
| PyPI+npm 누적 설치 | 25,000 (월 5,000 run-rate by M12) |
| GitHub stars | 3,000 (외부 기여자 ≥40, 커뮤니티 eval 케이스 ≥10) |
| 매니지드 무료 가입 | 600 (월간 활성 호출 ≥120) |
| 디자인 파트너 로고 | 3–5 (서면 파일럿/LOI, public 케이스 스터디 ≥1) |
| 처리 토큰 누적 | 50억 (월 1억+ run-rate by M12) |
| 호스티드 MRR | $8–15K (유료 2–3 + 컨설팅 1–2; 학습 신호) |
| GeomTok-Eval 인용/채택 | 학술 인용 8–12, 제3자 모델 논문/레포 점수 보고 ≥2 |
| 라운드트립 SLA 유지 | 파트너 코퍼스 ≥3개 100%; 신규 데이터에서 ≤1.78px / SSIM ≥0.929 유지 |
| Time-to-first-token | `pip install`→첫 성공 <15분 |

---

## 11. 가격·사업모델

원칙: **per-artifact(per-1K SVG) 가격** — Unstructured.io의 페이지당 가격이 최적 템플릿. GeomTok의 "페이지"는 아이콘/SVG. CPU 한계비용이 near-zero이므로 순수 per-call 메터링은 비합리(sub-ms → 분수 센트). 월 최소(monthly minimum)로 마진 방어, 타이어키커 필터.

| 티어 | 월 최소 | 포함/가격 | 대상 | 핵심 가치 |
|---|---|---|---|---|
| **OSS (Self-host)** | $0 | Apache-2.0 전체 코어, GeomTok-Eval, vocab 매니페스트 | 모두 | 신뢰·채택. 크리플링 없음 |
| **Free (Managed)** | $0 | 15,000 SVG/월 무료, 단건 API | 평가·소규모 | self-serve 온보딩 |
| **Pro** | $50/월 floor | 볼륨 슬라이더, ~$1 / 1,000 SVG, 스케일 시 declining(예: $0.90→$0.46/1K), 배치·스트림 | Marcus/Dana 스케일 | 호스티드 처리량·캐싱 |
| **Scale** | $500/월 floor | 비동기 잡(10k–1M), 우선 처리, declining rate, 잡 SLA | 대규모 백필 | 코퍼스 인덱싱 |
| **Enterprise** | sales-led | SSO, 라운드트립 100% **계약 SLA**, VPC/온프레미스, 데이터 격리, 버전 핀 운영, 지원 | IP 민감 디자인툴 | 자체 호스팅 가능하므로 **가치는 보증·격리·운영** |
| **Consulting/Support 리테이너** | 별도 | 코퍼스 온보딩·정규화, 드리프트 모니터링, (후속) 생성 공동 R&D | 파트너 | 생성 성숙 전 브릿지 |

수익 논리: 알고리즘이 아니라 **운영 고통**을 판매. Apache-2.0이 self-host를 기술적으로 허용하므로 유료 가치는 스케일+SLA+VPC+전문성에 명시 정렬.

---

## 12. GTM·출시계획

**GTM 모션 (병렬):**
1. **OSS-led / PLG:** `pip install geomtok` quickstart, 카피-페이스트 라운드트립 데모, `showcase.html` 랜딩. 스케일/SLA가 아플 때만 매니지드로 전환.
2. **Benchmark-as-marketing:** GeomTok-Eval을 렌더 기반·생성기-분리·게임-내성 표준으로. 공개 리더보드(576<666<775 bits/icon, 0%/84%/35% 렌더). 벤더가 돌려보면 자사 텍스트 토크나이저가 0% 렌더임을 스스로 발견 → 수요 생성 엔진.
3. **Design-partner 등대 계정:** 3–5 벤더와 'vector AI readiness' 파일럿. 그들의 실제 코퍼스로 100% 라운드트립 증명 + private GeomTok-Eval 리포트. 파일럿→레퍼런스 로고·케이스 스터디.
4. **Research-credibility:** 논문을 ACL/CVPR/NeurIPS 트랙 + arXiv, GeomTok/GeomTok-Eval 네이밍 락. 부정-결과 무결성(압축≠모델 가능성)을 신뢰 후크로.
5. **DevRel 쿡북:** "토크나이제이션 안 짓고 디자인툴에 벡터 AI 붙이기", FSA 보장 디코딩, 스칼라 코덱 튜토리얼. HN·arXiv·ML 뉴스레터·디자인-엔지니어링 커뮤니티.
6. **매니지드+컨설팅 업셀:** 벤더가 self-host 안 할 boring-but-critical 레이어 수익화.

**출시 단계:**

| 단계 | 기간(목표) | 산출물 | 게이트 |
|---|---|---|---|
| Private Alpha | M0–M2 | OSS 코어 + SDK + GeomTok-Eval, 2,682 아이콘 라운드트립 100% 재현 | 내부 벤치 통과 |
| Design-Partner Beta | M2–M5 | 매니지드 API(단건+배치), private 리포트, 파트너 1–2 코퍼스 100% | 파트너 코퍼스 라운드트립 100% |
| Public Launch (v1.0) | M5–M6 | 공개 OSS + 리더보드 + arXiv + 매니지드 GA(Free/Pro/Scale) | <15분 TTF, SLA 운영 준비 |
| Scale-out | M6–M12 | Enterprise(VPC/온프레미스), 스트림/잡 GA, 케이스 스터디 | $8–15K MRR, 25K 설치 |

---

## 13. 리스크·완화

| 리스크 | 설명 | 완화 |
|---|---|---|
| **Self-host 중력** | 결정적 라이브러리가 Apache-2.0 → "pip install 가능한데 왜 호스팅 비용?" 매니지드 수익이 0 근처 정체 가능 | 알고리즘이 아니라 운영 고통 판매: 진화 코퍼스 SLA 라운드트립, 고QPS/배치, 정규화+드리프트 모니터링, 버전 핀+FSA 검증 서비스, 컨설팅. OSS는 완전 유지(신뢰), 스케일+보증+전문성 수익화 |
| **생성 미성숙** | 2.5M params·CPU·단색 path만. 바이어가 "벡터 AI=생성"으로 기대 후 이탈 | 급진적 정직: 토크나이저+Eval은 지금 프로덕션, 생성은 명시 로드맵 후속(대규모 학습 필요). 기질 가치 먼저(84% vs 0%, bits/icon 26% 우위). 생성은 파트너와 공동-펀딩 R&D로 제안(출하 약속 아님) |
| **경쟁/선점** | HiVG(arXiv:2604.05072), GeoBPE(arXiv:2511.11758), OmniSVG류; 일부 헤드라인 주장은 문헌이 선점 | 방어 가능 3축: (i) 학습된 비제약 머지 > HiVG 구조-제약, (ii) 손수 매크로는 실데이터 0% 발화(규칙 기반 실패 모드), (iii) 생성기-분리 평가 공개 등가물 없음. "first"/"pre-structured advantage" 등 반박된 superlative 금지. Eval을 카테고리 정의 자산으로 → 경쟁자가 우리 벤치 위에서 측정됨 |
| **팀 밴드위스** | OSS 인프라 + 매니지드 + 학술 출판 + 디자인 파트너를 소규모 팀이 동시 | 범위 엄격(v1.0=토크나이저+Eval), 생성 비목표화, PLG로 세일즈 부담 최소화, 파트너 3–5로 한정 |
| **도메인 일반화** | 모노크롬 path 아이콘 밖(멀티컬러·필터·복잡 일러스트)에서 충실도 저하 | v1.0 도메인 명시 한정; 미지원 요소 명시 거부; 신규 코퍼스 SLA는 검증 후 부여 |

---

## 14. 로드맵

| Phase | 내용 | 게이트(다음 단계 진입 조건) |
|---|---|---|
| **Phase 1 — 토크나이저 + 평가 (v1.0, 현재)** | 파서·정규화·L1/L2/L3 토큰·0.04px 코덱·FSA 디코드·GeomTok-Eval·SDK·매니지드 API. 프로덕션-레디 | ✅ 파트너 코퍼스 ≥3개 라운드트립 100%, ≤1.78px/SSIM 0.929 유지; 공개 리더보드 가동; 25K 설치·3–5 파트너 |
| **Phase 2 — 생성 (후속)** | 토이 스케일(2.5M·CPU·단색)에서 대규모-데이터 학습으로. 파트너 공동-펀딩 R&D | 진입: 대규모 학습 펀딩 확보 + 파트너 데이터 확보. 졸업: 기하 토큰 위 생성이 파트너 도메인에서 출하 가능 품질(렌더 가능률·SSIM 목표 충족), 비-토이 데이터 학습 검증 |
| **Phase 3 — Figma/Canva 통합** | 디자인툴 플러그인/임베드, on-grid/on-style 저작 보조 | 진입: Phase 2 생성이 프로덕션 품질 + 디자인 파트너 통합 계약. 졸업: 1개 이상 디자인툴에 라이브 기능 |

---

## 15. 미해결 질문

1. **L2/L3 디폴트 정책** — 모델 가능성 데이터를 더 모은 뒤 L2 합성을 언제 기본 옵션으로 노출할지? 현재는 L1 기본 유지가 안전.
2. **매니지드 처리량 wedge 정량화** — OSS 라이브러리 대비 매니지드 배치의 실제 배수(목표 ~50x급)를 어디까지 측정·보장할지? 무료→유료 전환률을 좌우.
3. **per-1K SVG 가격 캘리브레이션** — Pro $1/1K, Scale declining 곡선의 정확한 브레이크포인트는 파트너 사용량 데이터 확보 후 확정.
4. **신규 코퍼스 SLA 부여 기준** — 새 고객 코퍼스에 100% 라운드트립 SLA를 부여하기 위한 사전 검증 자동화 수준(셀프-서브 검증 vs 컨설팅 게이팅)?
5. **도메인 확장 우선순위** — 멀티컬러/그라디언트/복잡 일러스트 중 어느 것을 먼저 확장할지, 그리고 그 확장이 좌표 충실도/어휘에 미치는 영향.
6. **생성 공동-펀딩 구조** — Phase 2를 파트너 R&D 공동 펀딩으로 갈 때 IP/라이선스(Apache-2.0 코어와 분리?) 구조.
7. **Eval 거버넌스** — 공개 리더보드의 제출/검증 거버넌스(악의적 제출 방지, 재현성 요구)를 누가 운영하는가.
8. **온프레미스 결정성 보증** — Enterprise 온프레미스 배포에서 호스팅과 비트-동일 결과를 어떻게 계약적으로 보증·감사할지.