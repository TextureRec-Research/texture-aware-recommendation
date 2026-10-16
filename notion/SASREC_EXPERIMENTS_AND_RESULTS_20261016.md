> 대응 코드: `../code/exp16/`, `../code/exp24/` · 최신 결과: `../results/RESULTS_17_full_table_exp16_recipe.md`, `../results/evaluation.json`

# SASRec·촉감 추천 실험 목록, 코드 흐름과 결과

- 정리 기준일: 2026-10-16 (Asia/Seoul)
- 목적: `sasrec/` 패키지 안의 각 코드가 어떤 실험을 수행하는지, 무엇을 비교했고 결과가 무엇이었는지를 한 문서에서 확인한다.
- 이 문서는 교수님 열람용이다. 패키지에는 원본 데이터 약 87GB와 체크포인트 약 3.5GB가 없으므로 실행은 원본 실험 폴더에서만 가능하다.

## 1. 결과 출처와 재검증 상태

### 기존 Notion 자료에서 가져온 결과

1. `/home/user/onsesang/material_span_grounding/notion/30_QA_BACKUP_TABLES.pdf`의 **B2. 구매 이력 기반 추천에 촉감 결합**
   - R0, R1, R1b, R2, R7의 test 결과를 가져왔다.
   - 이 표는 test 시점 이력 3개 이상 사용자 29,991명을 대상으로 한 Exp24 post-hoc 결과다.
2. `/home/user/onsesang/material_span_grounding/experiments/16_strong_recommender_tactile/notion/STRONG_RECOMMENDER_TACTILE_RESULTS.md`
   - 전체 카탈로그, 공식 split, SASRec 선택 과정과 alpha=0 결과의 배경을 확인했다.

### Notion에 없어 2026-10-16에 다시 실행한 결과

1. `17_train_full_exp16_recipe.py evaluate`
   - 재학습하지 않고 저장된 16개 체크포인트를 validation/test에서 다시 평가했다.
   - 원본에 쓰지 않도록 `/tmp`에 동일 디렉터리 구조를 만든 뒤 체크포인트와 데이터 캐시를 읽기 전용으로 연결했다.
   - 새로 생성된 `evaluation.json`은 패키지의 `../results/evaluation.json`과 **byte 단위로 완전히 일치**했다.
2. `14_tactile_space_diag.py`
   - 마지막 상품/이력 평균 × cosine/L2의 네 촉감 단독 retrieval 조합을 다시 실행했다.
3. `15_cold_zero_reeval.py`
   - 약한 Exp24 레시피의 R0·R3·R4·R5·R6 체크포인트를 원본/cold-zero 조건에서 다시 평가했다.
4. R6 `router_weights()` 확인
   - test cohort 앞 256명에서 router 평균을 약한 레시피와 강한 레시피 각각 다시 계산했다.
5. 강한 레시피 R3 residual gate 대 shuffle
   - 저장된 사용자별 rank로 10,000회 paired bootstrap을 다시 계산했다.

재실행은 평가만 수행했다. 저장된 체크포인트가 이미 있으므로 동일 모델을 처음부터 재학습하지 않았다. 기존 결과 문서와 수치가 일치했으며 원본 폴더에는 파일을 생성하거나 수정하지 않았다.

## 2. 공통 데이터와 평가 조건

| 항목 | 값 |
|---|---:|
| 데이터 | Amazon Reviews'23 `Amazon_Fashion` 공식 0core/last-out split |
| 전체 카탈로그 | 825,869 parent item |
| test 평가 사용자 | 이력 3개 이상 29,991명 |
| cold target | 21,447명, 약 71.5% |
| warm target | 8,544명 |
| primary metric | NDCG@10 |
| 추가 지표 | HR@10, MRR@10, Recall@100/500/3000 |
| seen-item 처리 | 평가 시점 이전에 본 상품 제외 |
| cold 정의 | train 양성 등장 횟수 `train_count == 0` |

각 사용자에게 정답 상품은 하나다. 따라서 Recall@K는 정답이 상위 K개 안에 포함되는 사용자 비율이다. cold target 비율이 매우 높고 ID 기반 SASRec은 학습하지 않은 item을 추천하기 어려워 절대 NDCG가 낮다.

## 3. 코드 파일별 실험 역할

### Exp16: 원래 강한 SASRec 기준선

| 파일 | 수행하는 역할 | 연결되는 실험 |
|---|---|---|
| `code/exp16/models.py` | Transformer block과 SASRec item/position embedding을 정의한다. | Exp16 원래 SASRec 기준선 |
| `code/exp16/recommender.py` | 공식 split을 읽고 학습 batch, negative sampling, validation early stopping, 전체 카탈로그 평가를 수행한다. | Exp16 SASRec 학습·평가 |
| `code/exp16/common.py` | 데이터 경로, JSON 기록, seed 등 공통 보조 기능을 제공한다. | Exp16 전체 |
| `code/exp16/configs/backbones.json` | SASRec 차원·layer·head·dropout·batch와 learning-rate grid를 고정한다. | Exp16 hyperparameter 선택 |
| `code/exp16/configs/protocol.json` | split, 평가 지표, 후보 K 규칙 등 실험 프로토콜을 기록한다. | Exp16 재현 프로토콜 |

### Exp24: 촉감 결합 및 진단

| 파일 | 수행하는 역할 | 연결되는 실험 |
|---|---|---|
| `code/exp24/common24.py` | Exp24 경로, cohort, sequence matrix, 지표와 paired bootstrap을 제공한다. | 모든 Exp24 실험 |
| `code/exp24/models24.py` | R0 SASRec, R3 세 fusion, R4 multitask, R6 mixture/router, R2 gate, R7 reranker를 정의한다. | R0·R2·R3·R4·R6·R7 |
| `code/exp24/02_evaluate.py` | validation/test 이력과 seen item을 만들고 `query × catalog_matrix` 전체 카탈로그 rank를 계산한다. | 공통 평가, R0 재현 |
| `code/exp24/03_train.py` | 약한 Exp24 공통 학습 루프와 R0/R3/R4/R5/R6 모델 생성을 담당한다. | 초기 R3–R6 비교 |
| `code/exp24/04_candidates.py` | SASRec top-500, 촉감 top-500, 표준화 촉감 top-500과 reranking 특성을 생성한다. | R1·R2·R7 후보 생성, 촉감 후보 recall |
| `code/exp24/05_rerank.py` | R1/R1b alpha 선택, R2 gate 학습, R7 후보 합집합과 reranker 학습을 수행한다. epoch 0 no-op도 validation 후보에 포함한다. | R1·R1b·R2·R7 |
| `code/exp24/06_analysis.py` | R0 대비 paired bootstrap 및 이력 길이·촉감 일관성 사용자군 분석을 수행한다. | R1·R2·R7 유의성 분석 |
| `code/exp24/10_evaluate_trained.py` | 약한 루프로 학습된 모든 R3/R4/R5/R6 체크포인트를 평가하고 family별 대표를 validation으로 선택한다. | 초기 R3–R6 전체 비교 |
| `code/exp24/12_trained_significance.py` | 같은 약한 학습 루프의 R0와 R3–R6을 비교하고 R3 actual-vs-shuffle 대조를 계산한다. | 초기 유의성 및 shuffle control |
| `code/exp24/14_tactile_space_diag.py` | 촉감 공간만으로 전체 카탈로그를 검색하고 cold/warm Recall을 계산한다. | 촉감 단독 retrieval, R7 후보 병목 |
| `code/exp24/15_cold_zero_reeval.py` | cold ID embedding을 0으로 만드는 조건과 R3 fused embedding까지 0으로 만드는 조건을 재평가한다. | cold item 실패 원인 진단 |
| `code/exp24/16_train_exp16_recipe.py` | Exp16의 Adam, lr, clip, negative exclusion, early stopping을 복원해 R0/R5를 학습한다. | 강한 기준선 복원 |
| `code/exp24/17_train_full_exp16_recipe.py` | 강한 Exp16 레시피로 R0·R3·R4·R5·R6을 학습하고 3-seed 및 bootstrap 표를 생성한다. | 최종 비교 실험 |
| `code/exp24/run_training_queue.sh` | 약한 공통 루프의 R0·R3·R4·R5·R6 실행 인자를 순서대로 기록한다. | 초기 학습 queue |
| `code/exp24/run_candidates_all.sh` | R7에 필요한 train/validation/test 후보 캐시를 만든다. | R7 전처리 |
| `code/exp24/run_posthoc.sh` | 후보 생성 후 R1/R2/R7 학습과 통계 분석을 연속 실행한다. | post-hoc 전체 흐름 |

## 4. 실험 목록과 결과

### 4.1 R0 — 촉감이 없는 SASRec 기준선

R0는 모든 촉감 variant의 비교 기준이다. 두 종류의 R0 수치를 구분해야 한다.

| R0 종류 | Test NDCG@10 | 의미 |
|---|---:|---|
| Exp16 저장 체크포인트 | 0.00349417 | 기존 Notion B2의 R0/R1/R7 기준선 |
| 약한 Exp24 루프, seed 20260917 | 0.00179340 | 초기 R3–R6과 같은 약한 학습 루프의 기준선 |
| 강한 Exp16 레시피, seed 20260904 | 0.00338117 | 최신 R3/R4/R5/R6 비교의 같은-seed 기준선 |
| 강한 Exp16 레시피, 3-seed | **0.00334725 ± 0.00005383** | seed 20260904/05/06 평균 ± 표본 sd |

약한 루프의 낮은 R0 때문에 초기 실험에서 일부 variant가 좋아 보였다. 최종 결론은 강한 레시피의 같은-seed R0를 기준으로 해석해야 한다.

### 4.2 R1/R1b — 고정 비율 tactile reranking

- R1: 촉감 정보가 유효한 SASRec 후보에만 percentile tactile score를 고정 alpha로 보간한다.
- R1b: 모든 후보를 같은 척도로 맞춘 뒤 고정 alpha를 적용한다.
- validation이 두 방법 모두 `alpha=0`을 선택했다.

| 방법 | Test NDCG@10 | Δ vs R0 | 결과 출처 |
|---|---:|---:|---|
| R0 | 0.00349417 | 기준 | 기존 Notion B2 |
| R1 | 0.00349417 | 0 | 기존 Notion B2 |
| R1b | 0.00349417 | 0 | 기존 Notion B2 |

결론: validation이 촉감 항을 끈 결과이므로 test ranking은 R0와 완전히 동일하다. 이는 작은 비유의 개선이 아니라 선택 단계에서 tactile contribution이 0이 된 경우다.

### 4.3 R2 — 사용자·상품별 학습 gate

R2는 고정 alpha 대신 후보마다 gate를 예측한다. validation-selected 설정의 gate 평균은 `0.00250357`, 표준편차는 약 `9.94e-7`로 거의 상수였다.

| 방법 | Test NDCG@10 | Δ vs R0 | 95% CI | 판정 |
|---|---:|---:|---|---|
| R2 learned gate | 0.00341802 | −0.00007615 | [−0.00017460, +0.00002230] | 유의차 없음 |

결론: adaptive gate가 사용자·상품에 따라 의미 있게 달라지지 않았고 R0를 개선하지 못했다. 기존 Notion B2의 반올림 값 `0.00342`와 일치한다.

### 4.4 R7 — SASRec+촉감 후보와 학습 reranker

R7은 SASRec 500개 후보와 표준화 촉감 500개 후보의 합집합을 만든 뒤 learning-to-rank head를 학습한다.

| 항목 | Validation | Test |
|---|---:|---:|
| SASRec 후보 Recall@500 | 0.038083 | 0.035177 |
| 합집합 후보 Recall | 0.040052 | 0.036878 |
| 선택된 reranker epoch | **0 (untrained no-op)** | validation 선택을 그대로 적용 |
| R7 NDCG@10 | 0.00300162 | 0.00349417 |
| R7 − R0 | 0 | 0 |

학습 가능한 train 사용자는 target이 후보 안에 있던 2,031명, 전체의 6.8%뿐이었다. 촉감 후보를 합치면 candidate recall은 조금 증가하지만 validation이 학습 전 상태를 선택해 최종 ranking은 R0와 동일했다. 이 결과는 기존 Notion B2의 R7 `0.00349`, Δ `0`과 일치한다.

### 4.5 R3 — item embedding 단계의 tactile feature fusion

R3는 item ID embedding에 14차원 촉감 벡터의 projection을 결합한다.

- residual gate: `LayerNorm(ID + gate × tactile)`
- concat projection: `[ID; tactile]`을 projection
- FiLM: tactile로 ID embedding의 scale과 bias를 생성
- shuffle: residual gate와 구조·parameter 수는 같고 촉감 테이블만 결정적으로 섞은 대조군

#### 최종 강한 레시피 결과

| R3 변형 | Test NDCG@10 | Δ vs 같은-seed R0 | 95% CI | 판정 |
|---|---:|---:|---|---|
| residual gate | 0.00109932 | −0.00228186 | [−0.00288654, −0.00168874] | 유의 하락 |
| concat projection | 0.00046656 | −0.00291461 | [−0.00351011, −0.00234148] | 유의 하락 |
| FiLM | 0.00085745 | −0.00252372 | [−0.00311952, −0.00193196] | 유의 하락 |
| shuffle | 0.00079314 | −0.00258803 | [−0.00318398, −0.00200438] | 유의 하락 |

강한 레시피에서 residual gate − shuffle은 `+0.00030617`, 95% CI `[+0.00012955, +0.00048575]`로 residual gate가 유의하게 높았다. 그러나 둘 다 R0보다 크게 낮다. 따라서 “실제 촉감과 shuffle이 항상 같다”가 아니라, **약한 레시피에서는 둘의 차이가 유의하지 않았고 강한 레시피에서는 residual이 더 높지만 둘 다 실패했다**가 정확하다.

R3는 seed 20260904 하나만 실행했고 residual/concat/shuffle의 best epoch가 최대 epoch 50이었다. 완전히 수렴하지 않았을 가능성이 있다.

### 4.6 R4 — 촉감 예측 보조 손실

R4는 SASRec 추천 손실에 현재 sequence representation으로 정답 상품의 촉감 벡터를 예측하는 BCE 보조 손실을 더한다.

| λ | Test NDCG@10 | Δ vs R0 | 95% CI | 판정 |
|---:|---:|---:|---|---|
| 0.05 | 0.00338502 | +0.00000385 | [−0.00000873, +0.00001541] | 유의차 없음 |
| 0.20 | 0.00346298 | +0.00008181 | [−0.00004575, +0.00021422] | 유의차 없음 |

결론: 강한 기준선에서는 두 λ 모두 R0와 유의한 차이가 없다. 초기 약한 루프에서는 일부 λ가 하락했지만 validation-selected λ=0.01은 test에서 R0와 차이가 없었다.

### 4.7 R5 — tactile hard-negative sampling

R5는 target과 같은 category×popularity 그룹에서 촉감 거리가 먼 상품을 negative로 섞는다. SHUFFLED 대조군은 동일한 mining 절차를 유지하고 촉감 테이블만 섞는다.

| seed | R0 | R5 실제 | Δ 실제−R0 | R5 shuffled | Δ 실제−shuffled |
|---:|---:|---:|---:|---:|---:|
| 20260904 | 0.00338117 | 0.00259433 | −0.00078684, 유의 | 0.00341243 | −0.00081810, 유의 |
| 20260905 | 0.00328519 | 0.00318413 | −0.00010105, n.s. | 0.00329820 | −0.00011407, n.s. |
| 20260906 | 0.00337540 | 0.00351539 | +0.00013999, 유의 | 0.00333375 | +0.00018164, 유의 |

| 3-seed 요약 | 평균 ± sd |
|---|---:|
| R0 | 0.00334725 ± 0.00005383 |
| R5 실제 | 0.00309795 ± 0.00046654 |
| R5 shuffled | 0.00334813 ± 0.00005845 |

초기 약한 R0에서는 R5 hard0.5가 `0.00271676`, R0가 `0.00179340`이라 상승처럼 보였다. 그러나 실제와 shuffled의 차이는 `−0.00000462`, 95% CI `[−0.00041189, +0.00038502]`로 없었다. 강한 3-seed에서는 효과 방향까지 바뀌었다. 따라서 “촉감 hard negative가 개선한다” 또는 “항상 해롭다” 모두 지지되지 않는다. 확인된 사실은 실제 촉감 R5의 seed 분산이 매우 크다는 점이다.

### 4.8 R6 — general expert와 tactile-sequence expert의 mixture

R6는 SASRec general query와 촉감 sequence를 GRU로 인코딩한 query를 router로 혼합한다. 후보 item 쪽은 ID embedding을 사용한다.

| 학습 레시피 | R0 | R6 | Δ vs R0 | 판정 |
|---|---:|---:|---:|---|
| 초기 약한 루프 | 0.00179340 | 0.00192620 | +0.00013281 | 유의차 없음 |
| 최종 강한 레시피 | 0.00338117 | 0.00255813 | −0.00082304, CI [−0.00122197, −0.00042310] | 유의 하락 |

test 사용자 앞 256명의 router 평균을 재계산한 결과:

| 체크포인트 | general expert | tactile expert |
|---|---:|---:|
| 초기 약한 레시피, seed 20260917 | 0.00001970 | **0.99998033** |
| 최종 강한 레시피, seed 20260904 | 0.00008505 | **0.99991494** |

두 레시피 모두 router가 사실상 tactile expert 하나만 사용한다. 설계 의도인 사용자별 expert 혼합이 이루어졌다고 보기 어렵다. 또한 이 구조는 “촉감만 쓰는 recommender”가 아니다. query는 거의 촉감 expert지만 후보 쪽은 여전히 ID item embedding이다.

### 4.9 촉감 공간 단독 전체 카탈로그 retrieval

학습하지 않은 14차원 Last2 공간에서 과거 이력으로 query를 만들고 전체 825,869개 상품을 직접 검색했다.

| query·거리 | 그룹 | Recall@100 | Recall@500 | Recall@3000 |
|---|---|---:|---:|---:|
| 마지막 상품·cosine | 전체 | 0.001867 | 0.003568 | 0.010036 |
|  | cold | 0.001912 | **0.003497** | 0.009792 |
|  | warm | 0.001756 | 0.003745 | 0.010651 |
| 마지막 상품·L2 | 전체 | 0.001801 | 0.003634 | 0.010070 |
|  | cold | 0.001865 | **0.003497** | 0.009885 |
|  | warm | 0.001639 | 0.003979 | 0.010534 |
| 이력 평균·cosine | 전체 | 0.000267 | 0.001100 | 0.005835 |
|  | cold | 0.000233 | 0.001026 | 0.005036 |
| 이력 평균·L2 | 전체 | 0.000233 | 0.001034 | 0.005768 |
|  | cold | 0.000187 | 0.000979 | 0.004989 |
| popularity | 전체 | 0.019573 | 0.037745 | 0.083158 |
|  | cold | 0 | 0 | 0 |

촉감 단독 retrieval은 cold item을 원리적으로 찾을 수 있지만 최고 cold Recall@500이 `0.003497`, 즉 약 0.35%다. “cold를 전혀 못 찾는다”는 것보다는 낫지만 실제 후보 검색기로는 매우 낮다. 이력 평균보다 마지막 상품 하나를 query로 쓰는 편이 약 3배 높았다.

### 4.10 cold-zero 재평가

초기 약한 루프에서 train에 없던 item ID embedding을 추론 전에 0으로 만들었다.

| 항목 | 결과 |
|---|---|
| R0 원본 → cold-zero | 0.00179340 → 0.00177195 |
| Exp16 R0와의 격차 중 닫힌 비율 | −1.26%; 오히려 소폭 악화 |
| cold Recall@500 | 모든 모델·조건에서 0 |
| R5 실제−shuffled, cold-zero | +0.00003681, CI [−0.00037432, +0.00044198], n.s. |
| R3 residual−R0, cold-zero | −0.00046964, CI [−0.00092958, +0.00001869], n.s. |
| R6 원본/zero | 둘 다 0.00192620 |

결론: 약한 R0의 성능 격차를 cold ID 행 처리 하나로 설명할 수 없다. R3의 fused vector에는 ID 행을 0으로 해도 촉감 항이 남으며, fused vector 전체를 0으로 하는 조건에서도 R0보다 낮았다.

## 5. 전체 결론

1. 강한 동일 레시피에서 R3 네 변형은 모두 R0보다 유의하게 낮다.
2. R4 보조 손실은 R0와 유의한 차이가 없다.
3. R5는 약한 기준선에서 좋아 보였지만 실제-vs-shuffle 차이가 없었고, 강한 3-seed에서는 효과 방향이 바뀌었다.
4. R6는 router가 두 레시피 모두 tactile expert에 99.99% 이상 몰렸다. 약한 기준선에서는 R0와 차이가 없었지만 강한 기준선에서는 유의하게 하락했다.
5. R7은 촉감 후보를 추가해 Recall을 조금 올렸지만 validation이 untrained no-op을 선택해 최종 순위가 R0와 같았다.
6. 촉감 단독 retrieval은 cold Recall@500 약 0.35%로, cold item을 일부 찾지만 실용적인 회수율은 아니다.
7. test target 약 71.5%가 cold item이라는 데이터 조건이 모든 ID 기반 모델의 절대 성능을 크게 제한한다.

따라서 현재 실험에서 촉감 feature가 일반 next-item 추천을 안정적으로 개선한다는 근거는 없다. 촉감 신호가 전혀 없다는 뜻은 아니지만, 현재 14차원 공간·후보 검색·학습 구조에서는 그 신호를 추천 이득으로 연결하지 못했다.

## 6. 해석 시 주의점

- R3·R4·R6 강한 레시피 결과는 seed 하나다.
- R3 residual/concat/shuffle은 최대 50 epoch가 best여서 미수렴 가능성이 있다.
- 사용자 단위 bootstrap은 사용자 표본 변동만 반영하고 학습 seed 변동은 반영하지 않는다.
- test split은 이전 실험에서 이미 관찰됐으므로 사후 탐색 결과다.
- next-item target은 촉감 만족도 label이 아니다.
- 전체 카탈로그 0core 결과를 k-core SASRec 논문 수치와 직접 비교하면 안 된다.
- R6의 “content-only” 해석은 부정확하다. query 쪽은 거의 tactile expert지만 후보 쪽은 ID embedding이다.
