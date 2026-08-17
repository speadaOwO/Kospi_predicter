# 📈 KOSPI 예측 모델 개발 프로젝트 계획서

> **프로젝트 목표:** 다양한 금융 수치 지표와 텍스트(뉴스) 감성 분석을 결합하여 PyTorch Transformer 기반의 KOSPI 단기/중기 예측 모델을 개발합니다.

---

## 1. 프로젝트 개요 (Overview)

| 항목 | 내용 |
| :--- | :--- |
| **프로젝트명** | KOSPI 인공지능 예측 모델 개발 |
| **핵심 기술** | Python, PyTorch, Transformer, Pandas |
| **데이터 소스** | 금융 수치 데이터 (pykrx, yfinance) + 뉴스 감성 분석 (딥서치, 싱크풀 등) |
| **형상 관리** | GitHub 기반 버전 관리 및 협업 |

---

## 2. 데이터 수집 및 파이프라인 (Data Pipeline)

모델 학습을 위해 거시경제 지표, 기술적 지표, 시장 수급 및 뉴스 감성 데이터를 종합적으로 수집하고 `pandas`를 이용해 가공합니다.

### 📊 수집 데이터 상세 분류

1. **주가 및 기본 수치**
   * 코스피 종가 및 거래량 (타겟 변수)

2. **기술적 지표 & 변동성/환율**
   * **지표:** MA5, MA20, RSI, MACD
   * **변동성 및 환율:** VIX, USD/KRW

3. **글로벌 증시 및 금리**
   * S&P 500, NASDAQ, SOX (반도체 지수)
   * 미국 10년물 국채 금리

4. **시장 수급 및 파생상품**
   * 외국인 순매수 동향
   * 코스피 200 야간선물, 옵션 지표

5. **텍스트 및 감성 분석 (News & Sentiment)**
   * **구현 난이도 및 정확도 , 데이터 누수 문제 우려로 시행하지 않음**
---

## 3. 기술 스택 및 아키텍처 (Tech Stack & Model)

* **Data Processing:** `pandas`, `pykrx`, `yfinance`
* **Deep Learning Framework:** `PyTorch` (파이토치)
* **Model Architecture:** **Transformer**
  * *선정 이유:* 시계열 데이터의 장기 의존성(Long-term dependency)을 학습하고, 다양한 이질적 지표들 간의 상관관계를 Self-Attention 메커니즘을 통해 효과적으로 포착하기 위함입니다.

---

## 4. 개발 단계 및 로드맵 (Milestones)

- [ ] **Phase 1: 데이터 수집 및 전처리 (EDA)**
  * `pykrx`, `yfinance`를 활용한 데이터 크롤링 자동화
  * 결측치 처리, 이상치 제거 및 스케일링(Scaling)
- [ ] **Phase 2: 감성 분석 파이프라인 구축**
  * 감정 분석을 대체하는 지수들을 사용하기
- [ ] **Phase 3: 모델 설계 및 학습 (Modeling)**
  * PyTorch 기반 Transformer 모델 구조 구현
  * Train / Validation / Test 데이터셋 분할 및 하이퍼파라미터 튜닝
- [ ] **Phase 4: 평가 및 최적화**
  * 예측 오차 평가 (MSE, MAE 등) 및 과적합(Overfitting) 방지 대책 적용
- [ ] **Phase 5: 문서화 및 배포**
  * GitHub 레포지토리 구축 및 README 작성
"""
