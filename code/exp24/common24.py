#!/usr/bin/env python3
# ============================================================
# [열람용 사본] 2026-10-16 시점 스냅샷. 이 파일은 실행용이 아님.
# 실제로 돌아가는 원본 경로:
#   /home/user/onsesang/material_span_grounding/experiments/24_adaptive_tactile_recommendation/scripts/common24.py
# 원본은 이후 실험으로 바뀔 수 있음.
# 역할: Exp24 공통 경로·설정과 시퀀스 구성·평가 지표·bootstrap 함수를 제공한다.
# ============================================================
"""Shared data, features and metrics for experiment 24.

Split semantics (inherited from exp16b/exp17b, not redefined here):
  Amazon Reviews'23 Amazon_Fashion, official 0core/last_out.  Per user the last
  event is the test target, the second-to-last the validation target, the rest
  train.  Training events: 157,490 over 76,755 users; catalog 825,869 parents.

Evaluation cohorts (identical objects to exp17b):
  validation = users with train history >= 3        (14,731)
  test       = users with history >= 3 at test time (29,991)

Leakage rules enforced here:
  * a user's tactile profile is built ONLY from items the user interacted with
    strictly before the evaluation point;
  * candidate tactile features come ONLY from the Last2 image predictor, never
    from the target item's reviews, rating or review-derived labels;
  * previously seen items are removed from every ranking, for every model;
  * ties break by ascending iid, the exp17b rule.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# 원본 실험 디렉터리 구조를 기준으로 데이터와 산출물 경로를 계산한다.
EXP = Path(__file__).resolve().parents[1]
PROJECT = EXP.parents[1]
E16 = PROJECT / "experiments/16_strong_recommender_tactile"
E17 = PROJECT / "experiments/17_history3_recommender_refresh"

DATA = E16 / "data"
ARTIFACTS = EXP / "artifacts"
RESULTS = EXP / "results"
PLOTS = EXP / "plots"
LOGS = EXP / "logs"
CACHE = EXP / "cache"
CHECKPOINTS = EXP / "checkpoints"
for _d in (ARTIFACTS, RESULTS, PLOTS, LOGS, CACHE, CHECKPOINTS):
    _d.mkdir(parents=True, exist_ok=True)

# 모델·평가에서 공통으로 쓰는 seed와 SASRec 하이퍼파라미터다.
SEED = 20260917
MAXLEN = 50
DIM = 64
LAYERS = 2
HEADS = 2
DROPOUT = 0.2
BATCH = 256
NEGATIVES = 64
TOP_K = 10
CANDIDATE_K_GRID = [100, 300, 500, 1000, 3000]

TACTILE_CLASSES = [
    "soft", "firm", "smooth", "rough", "non_elastic", "elastic", "thin",
    "thick", "flexible", "stiff", "warm", "cool", "spongy", "crisp",
]
# The 8 classes exp17b treated as reliable, kept for the ablation.
RELIABLE_CLASSES = ["smooth", "rough", "thin", "thick", "flexible", "stiff", "warm", "cool"]

PREPARED = CACHE / "prepared.npz"


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=float))


def read_json(path: Path):
    return json.loads(Path(path).read_text())


# --------------------------------------------------------------------- loading
# 준비된 시퀀스·촉감 feature 배열을 읽어 이름별 배열로 반환한다.
def load_prepared() -> dict:
    blob = np.load(PREPARED, allow_pickle=False)
    return {k: blob[k] for k in blob.files}


# split별 과거 이력을 오른쪽 정렬·0 padding된 SASRec 입력으로 변환한다.
def sequence_matrix(sequences_flat, offsets, users, maxlen=MAXLEN, extra=None):
    """Right-aligned, zero-padded item-id matrix for the given users.

    Item ids stored in the flat array are already shifted by one (0 = padding),
    matching exp16b's SASRec convention.  ``extra`` optionally appends one more
    id per user (used to fold the validation event into the test-time history).
    """
    out = np.zeros((len(users), maxlen), dtype=np.int64)
    for row, user in enumerate(users):
        start, end = offsets[user], offsets[user + 1]
        history = sequences_flat[start:end]
        if extra is not None and extra[user] > 0:
            history = np.concatenate([history, [extra[user]]])
        history = history[-maxlen:]
        if history.size:
            out[row, maxlen - history.size:] = history
    return out


# ------------------------------------------------------------- tactile profile
# 평가 시점 이전 이력에서 사용자의 평균 촉감 profile과 지원도를 계산한다.
def user_tactile_profile(sequences_flat, offsets, users, tactile, available,
                         extra=None, maxlen=None):
    """Mean tactile vector over the user's *past* items only.

    Returns (profile, support, consistency) where
      support     = how many past items had a usable tactile vector,
      consistency = 1 - mean per-dimension std over those items, in [0, 1];
                    high means the user's past items agree on their feel.
    """
    dim = tactile.shape[1]
    profile = np.zeros((len(users), dim), dtype=np.float32)
    support = np.zeros(len(users), dtype=np.int32)
    consistency = np.zeros(len(users), dtype=np.float32)
    for row, user in enumerate(users):
        start, end = offsets[user], offsets[user + 1]
        history = sequences_flat[start:end]
        if extra is not None and extra[user] > 0:
            history = np.concatenate([history, [extra[user]]])
        if maxlen:
            history = history[-maxlen:]
        items = history[history > 0] - 1              # undo the +1 padding shift
        items = items[available[items]]
        support[row] = items.size
        if items.size:
            block = tactile[items]
            profile[row] = block.mean(axis=0)
            consistency[row] = float(1.0 - block.std(axis=0).mean() * 2.0) if items.size > 1 else 1.0
    return profile, support, np.clip(consistency, 0.0, 1.0)


# ------------------------------------------------------------------- metrics
# 점수가 큰 상품을 우선하고 동점은 작은 column index를 우선해 rank를 구한다.
def rank_of_target(scores: np.ndarray, target_column: int) -> int:
    """Rank of the target among scores, ties broken by ascending index."""
    value = scores[target_column]
    return int(1 + (scores > value).sum() + (scores[:target_column] == value).sum())


# 정답이 하나인 next-item 평가의 NDCG·HR·Recall·MRR을 rank에서 계산한다.
def metrics_from_ranks(ranks: np.ndarray, k: int = TOP_K) -> dict:
    ranks = np.asarray(ranks, dtype=np.float64)
    hit = ranks <= k
    return {
        f"NDCG@{k}": float(np.where(hit, 1.0 / np.log2(ranks + 1.0), 0.0).mean()),
        f"HR@{k}": float(hit.mean()),
        f"Recall@{k}": float(hit.mean()),   # one relevant item per user
        f"MRR@{k}": float(np.where(hit, 1.0 / ranks, 0.0).mean()),
        "users": int(ranks.size),
    }


# 동일 사용자의 두 방법 차이를 사용자 단위로 복원추출해 신뢰구간을 구한다.
def paired_bootstrap_by_user(a: np.ndarray, b: np.ndarray, replicates=1000, seed=SEED):
    """Paired bootstrap over users on a per-user metric vector."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = a.size
    draws = np.empty(replicates)
    for r in range(replicates):
        idx = rng.integers(0, n, size=n)
        draws[r] = a[idx].mean() - b[idx].mean()
    return {
        "delta": float(a.mean() - b.mean()),
        "ci95_low": float(np.percentile(draws, 2.5)),
        "ci95_high": float(np.percentile(draws, 97.5)),
        "replicates": replicates,
        "improved": int((a > b).sum()),
        "worsened": int((a < b).sum()),
        "unchanged": int((a == b).sum()),
    }


def per_user_ndcg(ranks: np.ndarray, k: int = TOP_K) -> np.ndarray:
    ranks = np.asarray(ranks, dtype=np.float64)
    return np.where(ranks <= k, 1.0 / np.log2(ranks + 1.0), 0.0)
