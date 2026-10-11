# KOSPI Predictor 📈

**A Deep Learning Approach to Next-Day KOSPI Direction and Return Prediction**

CNN, LSTM, Transformer 및 CNN-Transformer 모델을 활용하여 다음 거래일의 KOSPI 방향과 수익률을 예측하는 시계열 분석 프로젝트입니다.

본 프로젝트는 단순히 높은 정확도를 기록하는 것보다, 모델 구조와 입력 시계열 길이가 예측 성능에 미치는 영향을 실험적으로 비교하는 것을 목표로 합니다.

> **Status:** Research & Experimentation

## 1. 프로젝트 개요

주식시장은 다양한 국내외 시장 요인과 경제 상황의 영향을 받습니다. 그러나 금융 시계열은 노이즈가 많고 시장의 특성이 시간에 따라 변하기 때문에, 과거 데이터만으로 미래를 정확하게 예측하는 것은 어렵습니다.

본 프로젝트에서는 과거 시장 데이터를 입력으로 사용하여 다음 거래일의 KOSPI 수익률과 상승·하락 방향을 예측합니다.

### 주요 연구 질문

- CNN, LSTM, Transformer 및 CNN-Transformer는 예측 성능에서 어떤 차이를 보이는가?
- 입력 구간 길이를 늘리면 예측 성능이 개선되는가?
- 서로 다른 난수 시드에서도 모델의 성능이 안정적으로 유지되는가?
- 정확도 외에 ROC-AUC, Balanced Accuracy 등의 지표를 활용하면 모델의 한계를 어떻게 파악할 수 있는가?

## 2. 데이터

주요 입력 데이터는 주가지수, 시장 변동성 및 환율 등의 수치형 시계열 데이터입니다. 뉴스 감성 분석은 사용하지 않습니다.

### 데이터 구성

| 데이터 유형 | 변수 예시 |
|---|---|
| 국내 주식시장 | KOSPI |
| 미국 주식시장 | S&P 500, NASDAQ, Dow Jones |
| 반도체 시장 | Philadelphia Semiconductor Index (SOX) |
| 일본 주식시장 | Nikkei 225 |
| 시장 변동성 | VIX, VKOSPI |
| 환율 | USD/KRW |
| 금리 | 미국·한국·일본 기준금리, 미국 국채금리 |
| 거시경제 | 물가상승률 등 |

위 목록은 프로젝트의 데이터 구성 및 확장 대상입니다. 모든 변수가 현재 학습 데이터에 포함되어 있는 것은 아니며, 실제 사용 여부는 전처리 코드와 데이터 파일을 기준으로 확인해야 합니다.

### 데이터 처리

- 날짜를 기준으로 시계열 데이터를 정렬합니다.
- 입력 특성과 예측 대상 변수를 구성합니다.
- 일정 길이의 과거 데이터를 하나의 입력 시퀀스로 변환합니다.
- 학습·검증·테스트 데이터를 시간 순서에 따라 분리합니다.
- 정규화에 필요한 통계량은 학습 데이터에서 계산합니다.

거시경제 지표는 실제 발표 시점을 고려해야 합니다. 발표 이전에 해당 지표의 값을 모델 입력에 포함하면 미래 정보 누수(Data Leakage)가 발생할 수 있으므로, 실제 이용 가능 시점에 맞춘 처리가 필요합니다.

## 3. 모델 아키텍처

서로 다른 시계열 모델을 구현하고 동일한 예측 문제에서 성능을 비교합니다.

| 모델 | 핵심 아이디어 |
|---|---|
| CNN | 합성곱을 통해 국소적인 시계열 패턴 추출 |
| LSTM | 순환 구조를 활용한 시계열 의존성 학습 |
| Transformer | Self-Attention을 활용한 시점 간 관계 학습 |
| CNN-Transformer | CNN의 국소 패턴 추출과 Transformer의 관계 학습 결합 |

### CNN-Transformer

현재 주요 실험 모델은 CNN과 Transformer를 결합한 하이브리드 구조입니다.

주요 구성 요소는 다음과 같습니다.

1. **Feature Gating:** 입력 특성별 가중치를 조정합니다.
2. **Input Projection & Layer Normalization:** 입력 특성을 모델 차원으로 변환하고 정규화합니다.
3. **Positional Embedding:** 시퀀스 내 시간적 위치 정보를 반영합니다.
4. **Temporal CNN:** 인접 시점 사이의 국소적인 패턴을 추출합니다.
5. **Transformer Encoder:** Self-Attention을 통해 시점 간 관계를 학습합니다.
6. **Attention Pooling & Fusion:** 시퀀스 정보를 통합합니다.
7. **Dual Output Heads:** 수익률 회귀와 방향 분류를 동시에 수행합니다.

### 다중 과제 학습 (Multi-Task Learning)

모델은 두 가지 예측 과제를 동시에 학습합니다.

- **Regression:** 다음 거래일의 수익률 예측
- **Classification:** 다음 거래일의 상승·하락 방향 분류

회귀에는 Huber Loss, 분류에는 Binary Cross-Entropy Loss를 사용합니다. 이를 통해 수익률 예측과 방향 분류 정보를 함께 학습하도록 설계했습니다.

## 4. 실험 설정

CNN-Transformer의 주요 실험에서 사용한 설정은 다음과 같습니다.

| 항목 | 설정 |
|---|---|
| Model dimension | 64 |
| Attention heads | 4 |
| Transformer layers | 2 |
| Feed-forward dimension | 128 |
| Optimizer | AdamW |
| Initial learning rate | 3 × 10⁻⁴ |
| Weight decay | 1 × 10⁻³ |
| Regression loss | Huber Loss |
| Classification loss | BCEWithLogitsLoss |
| Regression loss weight | 0.35 |
| Classification threshold | 0.50 |
| Early stopping | 사용 |

실험에 따라 입력 구간 길이와 난수 시드를 변경하여 성능을 비교합니다.

## 5. 실험 결과

### 5.1. 모델별 성능 비교: Window 60

테스트 표본은 497개이며, 항상 상승을 예측하는 기준선의 정확도는 60.36%입니다.

| Model | Seed | Accuracy | ROC-AUC |
|---|---:|---:|---:|
| Transformer | 100 | 54.53% | 0.634 |
| Transformer | 200 | 56.54% | 0.677 |
| Transformer | 300 | 57.55% | 0.667 |
| LSTM | 100 | 53.72% | 0.604 |
| LSTM | 200 | 58.15% | 0.641 |
| CNN | 100 | 60.36% | 미측정 |
| CNN | 200 | 57.75% | 미측정 |
| CNN | 300 | 60.36% | 미측정 |
| CNN-Transformer | 100 | 64.79% | 0.640 |
| CNN-Transformer | 200 | 60.76% | 0.610 |
| CNN-Transformer | 300 | 51.71% | 0.647 |

Window 60에서는 CNN-Transformer가 시드에 따라 큰 성능 편차를 보였습니다. 따라서 특정 시드의 결과만으로 모델의 우수성을 판단하기 어렵습니다.

### 5.2. 모델별 성능 비교: Window 120

테스트 표본은 437개이며, 항상 상승 예측 기준선은 60.18%입니다.

| Model | Seed | Accuracy | Balanced Accuracy | ROC-AUC |
|---|---:|---:|---:|---:|
| Transformer | 100 | 64.07% | 미측정 | 0.662 |
| Transformer | 300 | 66.36% | 미측정 | 0.658 |
| LSTM | 100 | 54.46% | 55.85% | 0.608 |
| LSTM | 300 | 59.50% | 53.42% | 0.578 |
| CNN | 100 | 60.18% | 미측정 | 미측정 |
| CNN | 300 | 60.18% | 미측정 | 미측정 |
| CNN-Transformer | 100 | 67.28% | 61.92% | 0.679 |
| CNN-Transformer | 300 | 65.68% | 64.19% | 0.685 |

Window 120의 CNN-Transformer는 두 시드 모두 65% 이상의 정확도를 기록했습니다. 평균 정확도는 66.48%, 평균 ROC-AUC는 약 0.682입니다.

### 5.3. 입력 구간 길이 비교

| Window | Model | Test samples | Accuracy | Balanced Accuracy | ROC-AUC |
|---:|---|---:|---:|---:|---:|
| 60 | CNN-Transformer | 497 | 59.09%* | 58.65%* | 0.632* |
| 120 | CNN-Transformer | 437 | 66.48%* | 63.06%* | 0.682* |
| 240 | CNN-Transformer | 317 | 71.92% | 62.79% | 0.699 |
| 480 | CNN-Transformer | 77 | 64.94% | 55.60% | 0.668 |

\* 여러 시드 결과의 단순 평균입니다. Window 240과 480은 각각 단일 실험 결과입니다.

Window 240에서 정확도 71.92%, ROC-AUC 0.6989를 기록했습니다. 반면 Window 480에서는 정확도가 64.94%로 낮아졌으며, Balanced Accuracy도 55.60%에 그쳤습니다.

이 결과는 입력 구간을 길게 설정하는 것만으로 예측 성능이 지속적으로 향상되지는 않을 가능성을 보여줍니다.

단, 입력 구간별 테스트 표본 수와 기간이 서로 다르므로 이 표만으로 최적의 입력 길이를 확정할 수는 없습니다.

### 5.4. Window 240의 예측 특성

Window 240 모델의 테스트 결과는 다음과 같습니다.

| Metric | Result |
|---|---:|
| Accuracy | 71.92% |
| Balanced Accuracy | 62.79% |
| Precision (Up) | 71.22% |
| Recall (Up) | 94.61% |
| ROC-AUC | 0.6989 |
| MCC | 0.3479 |

혼동 행렬:

| 실제 / 예측 | 하락 | 상승 |
|---|---:|---:|
| 하락 | 35 | 78 |
| 상승 | 11 | 193 |

상승 재현율은 94.61%였지만 하락 재현율은 30.97%였습니다. 이는 모델이 상승 방향을 선호하는 편향을 보였음을 의미합니다. 따라서 71.92%의 정확도만으로 균형 잡힌 방향 예측 능력을 주장하기는 어렵습니다.

### 5.5. Window 480의 학습 결과

Window 480 실험은 다음과 같은 결과를 기록했습니다.

- 최적 에폭: 5
- 최적 검증 손실: 0.82731
- 학습 시간: 약 19.4분
- 테스트 정확도: 64.94%
- ROC-AUC: 0.6681
- 테스트 표본: 77개

학습 손실은 감소했지만 검증 성능은 초기 에폭 이후 개선되지 않았습니다. 또한 하락 재현율이 13.33%로 낮게 나타났습니다. 테스트 표본이 적고 단일 시드 결과이므로 추가 검증이 필요합니다.

## 6. 현재까지의 결론

현재 실험에서는 CNN-Transformer가 유망한 후보로 나타났으며, Window 240에서 가장 높은 단일 실험 정확도와 ROC-AUC를 기록했습니다.

그러나 높은 정확도가 모든 방향에서 균형 잡힌 예측 능력을 의미하지는 않습니다. 시드별 편차, 상승·하락 클래스 불균형, 테스트 표본 수의 차이 등이 결과 해석에 영향을 줍니다.

따라서 현 단계의 결과는 확정적인 예측 성능의 입증이 아니라, 후속 실험의 방향을 설정하기 위한 탐색적 결과로 해석해야 합니다.

## 7. 향후 연구 계획

- 모든 입력 구간에서 동일한 테스트 날짜를 사용하여 공정하게 비교
- 각 설정에서 여러 난수 시드로 반복 실험
- Balanced Accuracy, ROC-AUC, MCC를 활용한 종합 평가
- 하락 방향 예측 성능과 클래스 편향 개선
- 거시경제 변수의 실제 발표 시점을 반영한 데이터 처리
- 거래 비용과 슬리피지를 고려한 별도의 전략 검증
- 장기 구간과 여러 시장 국면에서의 일반화 성능 확인

## 8. 기술 스택

- **Language:** Python
- **Deep Learning:** PyTorch
- **Data Processing:** NumPy, Pandas
- **Market Data:** yfinance 및 기타 데이터 제공원
- **Evaluation:** scikit-learn
- **Visualization:** Matplotlib

## 9. 프로젝트 구조

실제 저장소 구조에 맞게 조정해야 하는 예시입니다.

```text
Kospi_predicter/
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── download.py
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
├── README.md
├── requirements.txt
└── .gitignore
```

학습 스크립트와 데이터 파일의 실제 경로는 현재 저장소 구조를 기준으로 확인해야 합니다. 대용량 원본 데이터, 모델 체크포인트, API 키 등은 필요에 따라 Git 추적에서 제외합니다.

## 10. 재현성 및 주의 사항

- 테스트 데이터는 모델 선택이나 임계값 조정에 사용하지 않는 것을 원칙으로 합니다.
- 스케일러는 학습 데이터에만 적합시킵니다.
- 거시경제 데이터는 발표 시점 이전의 정보가 유입되지 않도록 처리합니다.
- 서로 다른 Window 결과는 테스트 날짜와 표본 구성을 확인한 뒤 비교합니다.
- 시드별 평균과 변동성을 함께 보고합니다.

본 프로젝트는 교육 및 연구 목적으로 진행되는 실험입니다. 모델의 과거 예측 성능이 미래 수익을 보장하지 않으며, 투자 판단을 위한 금융 자문을 제공하지 않습니다.

---

*This project explores deep learning architectures for time-series forecasting and evaluates their limitations under different experimental settings.*
