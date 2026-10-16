# 열람용 스냅샷(2026-10-16). 실행은 아래 원본 위치에서만 가능

SASRec 및 촉감 feature SASRec 관련 코드와 최신 결과를 모은 사본이다. 코드의 변수명, 함수, import, 경로 문자열, 하이퍼파라미터와 실행 로직은 바꾸지 않았으며, 코드 파일에는 열람을 돕는 한국어 `#` 주석만 추가했다.

패키지 내부 코드는 원본 디렉터리 구조와 데이터·체크포인트를 포함하지 않으므로 실행용이 아니다. 실제 실행 위치는 다음과 같다.

```text
/home/user/onsesang/material_span_grounding/experiments/24_adaptive_tactile_recommendation
```

원본에서 실행한 명령은 다음과 같다.

```bash
cd /home/user/onsesang/material_span_grounding/experiments/24_adaptive_tactile_recommendation
/home/user/onsesang/miniconda3/envs/texture/bin/python scripts/17_train_full_exp16_recipe.py train
/home/user/onsesang/miniconda3/envs/texture/bin/python scripts/17_train_full_exp16_recipe.py evaluate
```

## 권장 열람 순서

1. `results/RESULTS_17_full_table_exp16_recipe.md` — 최신 Exp16 레시피 재학습, feature 변형 비교, 3-seed 결과와 한계.
2. `results/evaluation.json` — 보고서 표의 원본 수치, 사용자별 paired bootstrap, seed별 결과.
3. `code/exp24/models24.py` — 기본 SASRec(R0), feature SASRec(R3), multitask(R4), mixture(R6), gate/reranker 정의.
4. `code/exp24/17_train_full_exp16_recipe.py` — 최신 비교표를 생성한 학습·평가 driver.
5. `code/exp24/16_train_exp16_recipe.py` — Exp16 학습 레시피를 복원한 R0/R5 재학습 코드.
6. `code/exp24/03_train.py`, `10_evaluate_trained.py`, `12_trained_significance.py`, `15_cold_zero_reeval.py` — 약한 공통 루프의 R3–R6 학습·평가와 cold-zero 재평가.
7. `code/exp24/04_candidates.py`, `05_rerank.py`, `06_analysis.py` — R7 후보 생성, reranker, R0 대비 유의성·사용자군 분석.
8. `code/exp24/14_tactile_space_diag.py` — 마지막 상품/이력 평균 × cosine/L2의 촉감 단독 전체 카탈로그 retrieval.
9. `code/exp24/run_*.sh` — 약한 R3–R6 및 R7 파이프라인에 실제 사용한 실행 순서와 인자.
10. `code/exp24/02_evaluate.py`, `common24.py` — 전체 카탈로그 평가와 공통 데이터·지표 함수.
11. `code/exp16/models.py`, `recommender.py`, `common.py`, `configs/` — 원래 SASRec 구현과 학습 레시피·프로토콜.
12. `notion/SASREC_EXPERIMENTS_AND_RESULTS_20261016.md` — 코드 파일별 역할, R0–R7·촉감 단독 retrieval·cold-zero 실험 설명과 검증된 결과를 통합한 교수님 열람용 문서.

## 요약 실험과 코드 대응

| 요약 항목 | 포함된 핵심 코드 | 범위 |
|---|---|---|
| R3 feature fusion 및 shuffle 대조 | `models24.py`, `03_train.py`, `17_train_full_exp16_recipe.py` | residual gate, concat projection, FiLM, shuffle과 강한 레시피 평가 |
| R4 촉감 예측 보조 손실 | `models24.py`, `03_train.py`, `17_train_full_exp16_recipe.py` | multitask loss와 λ 조건 학습·평가 |
| R5 촉감 hard negative | `03_train.py`, `16_train_exp16_recipe.py`, `17_train_full_exp16_recipe.py` | 약한/강한 기준선, 실제/섞음 촉감, 3-seed 비교 |
| R6 촉감 expert | `models24.py`, `03_train.py`, `10_evaluate_trained.py`, `17_train_full_exp16_recipe.py` | general/tactile expert와 router, 약한·강한 레시피 평가 |
| R7 후보 추가 + reranker | `04_candidates.py`, `05_rerank.py`, `06_analysis.py` | SASRec+촉감 후보 합집합, epoch 0 no-op을 포함한 validation 선택, test 평가 |
| 촉감 공간 단독 retrieval | `14_tactile_space_diag.py` | 마지막 상품/이력 평균 × cosine/L2, cold/warm Recall@100/500/3000 |

R6의 `router_weights()` 구현은 `models24.py`에 있다. 다만 기존 문서에 기록된 test 사용자 256명의 router 평균 `[general 0.0000197, tactile 0.99998]`를 계산한 일회성 명령은 별도 원본 스크립트로 저장돼 있지 않아 복사할 파일이 없다. 이 패키지에는 없는 실행 로직을 새로 만들어 넣지 않았다.

`notion/SASREC_EXPERIMENTS_AND_RESULTS_20261016.md`는 기존 Notion의 R0/R1/R1b/R2/R7 결과와 2026-10-16에 원본을 쓰지 않는 방식으로 재실행한 R3–R6·촉감 단독 retrieval·cold-zero 평가를 함께 정리한다. 재실행한 최신 `evaluation.json`은 패키지에 포함된 파일과 byte 단위로 일치했다.

## 핵심 결과

평가는 Amazon Reviews'23 `Amazon_Fashion` 공식 0core/last-out split, 전체 카탈로그 825,869개, test 시점 이력 3개 이상 사용자 29,991명을 대상으로 했다.

| 모델 | Test NDCG@10 | 같은 seed R0 대비 | 판정 |
|---|---:|---:|---|
| R0 SASRec, seed 20260904 | 0.00338117 | 기준 | — |
| R3 residual gate | 0.00109932 | −0.00228186 | 유의 하락 |
| R3 concat projection | 0.00046656 | −0.00291461 | 유의 하락 |
| R3 FiLM | 0.00085745 | −0.00252372 | 유의 하락 |
| R3 shuffled feature control | 0.00079314 | −0.00258803 | 유의 하락 |

R0의 3-seed test NDCG@10은 `0.00334725 ± 0.00005383`(평균 ± 표본 표준편차)이다. R4 multitask 두 조건은 R0와 유의한 차이가 없었고, R5 tactile hard negative는 seed에 따라 효과 방향이 바뀌었으며, R6 mixture는 seed 20260904에서 유의하게 하락했다.

## 실험 한계

- R3 feature 모델 네 조건은 seed `20260904` 하나만 실행했다.
- R3 residual gate, concat projection, shuffle은 최대 50 epoch에서 최고 validation 성능이 나와 완전히 수렴하지 않았을 가능성이 있다.
- 실제 촉감 feature를 섞은 shuffle 대조군도 함께 하락했으므로 R3 하락 전체를 촉감 정보 자체의 영향으로만 해석할 수 없다. 구조와 최적화 영향이 함께 들어 있다.
- test target 약 71.5%가 train에서 양성으로 등장하지 않은 cold item이다. 이들에 대한 SASRec 추천 지표는 0이므로 절대 NDCG가 낮다.
- 전체 카탈로그와 0core split을 사용했으므로 k-core 필터를 적용한 일반적인 SASRec 논문 수치와 직접 비교하면 안 된다.
- test split은 Exp16에서 이미 관찰된 뒤의 사후 탐색 결과다. 체크포인트 선택에는 validation만 사용했다.
- 사용자 단위 bootstrap은 사용자 표본 변동만 반영하며 학습 seed 변동을 반영하지 않는다.

## 포함하지 않은 항목

- 원본 데이터 약 87GB
- 학습 체크포인트 약 3.5GB
- `cache/prepared.npz`, `cache/trained_ranks.npz` 및 생성된 rank·로그
- Exp16 체크포인트와 `selection_lock.json`
- 최신 3-seed 결과 이전 문서인 `REPORT.md`, `GPT_SASREC_EXPLAINER_20261010.md`, `STRONG_RECOMMENDER_TACTILE_RESULTS.md`
- 2026-09-27의 `30_QA_BACKUP_TABLES.pdf` 및 `B2_implicit_rec.png`: 기존 R0/R1/R2/R7 결과만 포함하고 최신 R3·3-seed 결과가 없어 제외했다.

따라서 패키지의 코드를 실행하려면 원본 위치의 데이터, 캐시, 체크포인트가 필요하다.

`run_training_queue.sh`의 R0–R6 학습 블록은 이 패키지의 `03_train.py`로 닫힌다. 파일 끝의 별도 tactile-source ablation 준비 블록은 이번 요약 실험 범위가 아니며 원본의 `08_catalog_tactile.py`, Exp21 모델 코드와 체크포인트를 요구한다. 실행 로직을 바꾸지 않기 위해 해당 호출은 사본에 그대로 남겼다.

## 원본 경로 대응표

| 패키지 파일 | 실제 원본 파일 |
|---|---|
| `results/RESULTS_17_full_table_exp16_recipe.md` | `/home/user/onsesang/material_span_grounding/experiments/24_adaptive_tactile_recommendation/RESULTS_17_full_table_exp16_recipe.md` |
| `results/evaluation.json` | `/home/user/onsesang/material_span_grounding/experiments/24_adaptive_tactile_recommendation/artifacts/exp16recipe_full/evaluation.json` |
| `code/exp24/models24.py` | `/home/user/onsesang/material_span_grounding/experiments/24_adaptive_tactile_recommendation/scripts/models24.py` |
| `code/exp24/17_train_full_exp16_recipe.py` | `/home/user/onsesang/material_span_grounding/experiments/24_adaptive_tactile_recommendation/scripts/17_train_full_exp16_recipe.py` |
| `code/exp24/16_train_exp16_recipe.py` | `/home/user/onsesang/material_span_grounding/experiments/24_adaptive_tactile_recommendation/scripts/16_train_exp16_recipe.py` |
| `code/exp24/03_train.py` | `/home/user/onsesang/material_span_grounding/experiments/24_adaptive_tactile_recommendation/scripts/03_train.py` |
| `code/exp24/02_evaluate.py` | `/home/user/onsesang/material_span_grounding/experiments/24_adaptive_tactile_recommendation/scripts/02_evaluate.py` |
| `code/exp24/common24.py` | `/home/user/onsesang/material_span_grounding/experiments/24_adaptive_tactile_recommendation/scripts/common24.py` |
| `code/exp24/04_candidates.py` | `/home/user/onsesang/material_span_grounding/experiments/24_adaptive_tactile_recommendation/scripts/04_candidates.py` |
| `code/exp24/05_rerank.py` | `/home/user/onsesang/material_span_grounding/experiments/24_adaptive_tactile_recommendation/scripts/05_rerank.py` |
| `code/exp24/06_analysis.py` | `/home/user/onsesang/material_span_grounding/experiments/24_adaptive_tactile_recommendation/scripts/06_analysis.py` |
| `code/exp24/10_evaluate_trained.py` | `/home/user/onsesang/material_span_grounding/experiments/24_adaptive_tactile_recommendation/scripts/10_evaluate_trained.py` |
| `code/exp24/12_trained_significance.py` | `/home/user/onsesang/material_span_grounding/experiments/24_adaptive_tactile_recommendation/scripts/12_trained_significance.py` |
| `code/exp24/14_tactile_space_diag.py` | `/home/user/onsesang/material_span_grounding/experiments/24_adaptive_tactile_recommendation/scripts/14_tactile_space_diag.py` |
| `code/exp24/15_cold_zero_reeval.py` | `/home/user/onsesang/material_span_grounding/experiments/24_adaptive_tactile_recommendation/scripts/15_cold_zero_reeval.py` |
| `code/exp24/run_training_queue.sh` | `/home/user/onsesang/material_span_grounding/experiments/24_adaptive_tactile_recommendation/scripts/run_training_queue.sh` |
| `code/exp24/run_candidates_all.sh` | `/home/user/onsesang/material_span_grounding/experiments/24_adaptive_tactile_recommendation/scripts/run_candidates_all.sh` |
| `code/exp24/run_posthoc.sh` | `/home/user/onsesang/material_span_grounding/experiments/24_adaptive_tactile_recommendation/scripts/run_posthoc.sh` |
| `code/exp16/models.py` | `/home/user/onsesang/material_span_grounding/experiments/16_strong_recommender_tactile/models.py` |
| `code/exp16/recommender.py` | `/home/user/onsesang/material_span_grounding/experiments/16_strong_recommender_tactile/recommender.py` |
| `code/exp16/common.py` | `/home/user/onsesang/material_span_grounding/experiments/16_strong_recommender_tactile/common.py` |
| `code/exp16/configs/backbones.json` | `/home/user/onsesang/material_span_grounding/experiments/16_strong_recommender_tactile/configs/backbones.json` |
| `code/exp16/configs/protocol.json` | `/home/user/onsesang/material_span_grounding/experiments/16_strong_recommender_tactile/configs/protocol.json` |

`common.py`와 `configs/protocol.json`은 요청 목록의 코드가 직접 import하거나 읽는 추가 의존 파일이라 함께 포함했다. JSON은 주석을 지원하지 않으므로 원본 경로를 이 표에 기록했다.

## 스냅샷 출처

- Git 저장소 루트: `/home/user/onsesang`
- Git HEAD: `203424ba5989c56538a4094cea7285b4e2dd893e`
- 스냅샷 기준일: 2026-10-16, Asia/Seoul

## 서버에서 확인한 환경

- 실험 환경 Python: `3.10.20` (`/home/user/onsesang/miniconda3/envs/texture/bin/python`)
- 시스템 기본 Python: `3.12.2`
- PyTorch: `2.5.1+cu121`
- PyTorch CUDA build: `12.1`
- GPU: `NVIDIA A100 80GB PCIe`
- NVIDIA driver: `560.35.03`
- GPU memory: `81920 MiB`

위 정보는 2026-10-16에 서버에서 직접 확인한 값만 기록했다.
