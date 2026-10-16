#!/usr/bin/env python3
# ============================================================
# [열람용 사본] 2026-10-16 시점 스냅샷. 이 파일은 실행용이 아님.
# 실제로 돌아가는 원본 경로:
#   /home/user/onsesang/material_span_grounding/experiments/24_adaptive_tactile_recommendation/scripts/14_tactile_space_diag.py
# 원본은 이후 실험으로 바뀔 수 있음.
# 역할: 14차원 촉감 공간만으로 전체 카탈로그 retrieval을 수행하고 cold/warm Recall을 진단한다.
# ============================================================
"""Is the Last2 tactile space usable as a shared retrieval space for SASRec?

Read-only diagnostic, run before any training.  Nothing is written to disk; the
report goes to stdout as JSON (RESULTS_14_tactile_space_diag.md transcribes it).

  B  cold/warm split of the cached per-user test ranks (R0 exp16 checkpoint,
     R0 retrained, R3 x4) -- cache/ranks__R0_reproduction__test.npz and
     cache/trained_ranks.npz, written by 02_evaluate.py and 10_evaluate_trained.py
  C  cold/warm split of the cached tactile ANN candidate sets (04_candidates.py)
  D  tactile-only full-catalog retrieval: query {last item, history mean} x
     distance {cosine, L2}, Recall@{100,500,3000} on all / cold / warm targets,
     next to popularity; plus how discriminative the 14-d vectors are.

Cohort, histories, seen filter and cold definition are 04_candidates.evaluation_point
and train_count == 0, i.e. exactly what 13_retrieval_diagnostic.py uses.
Ties break by ascending iid, which is ascending parent_asin (asserted below).

Run: <texture python> scripts/14_tactile_space_diag.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch

import common24 as C
from importlib import import_module

candidates_module = import_module("04_candidates")

EXPECTED_TEST_USERS = 29_991
KS = (100, 500, 3000)
RANDOM_BASELINE_RECALL_500 = 0.001     # 13_retrieval_diagnostic.py, random-init SASRec
SAMPLE_ITEMS = 1000
BATCH = 256


# 사용자군별 Recall@K와 사용자 수를 묶어 반환한다.
def recall_block(ranks, groups, ks=KS):
    return {name: {"n_users": int(mask.sum()),
                   **{f"Recall@{k}": float((ranks[mask] <= k).mean()) for k in ks}}
            for name, mask in groups.items()}


# 사용자군별 NDCG 평균을 계산한다.
def ndcg_block(ranks, groups):
    return {name: float(C.per_user_ndcg(ranks[mask]).mean()) for name, mask in groups.items()}


# 촉감 쿼리와 전체 상품 벡터의 cosine 또는 L2 순위에서 target rank를 구한다.
def full_catalog_ranks(queries, valid_query, tactile, available, histories, targets,
                       metric, device):
    """Rank of each target over all items by tactile similarity to the query.

    Items without a tactile vector score -inf (as in 04_candidates.py); seen items
    score -inf; a user without a usable query gets rank n_items + 1 (a miss).
    float64 throughout so that identical vectors produce exactly equal scores and
    the ascending-iid tie-break applies.
    """
    n_items = tactile.shape[0]
    items = torch.as_tensor(tactile, dtype=torch.float64, device=device)
    usable = torch.as_tensor(available, device=device)
    if metric == "cosine":
        items_n = torch.nn.functional.normalize(items, dim=1)
    else:
        item_sq = (items * items).sum(1)
    index = torch.arange(n_items, device=device)
    ranks = np.full(len(queries), n_items + 1, dtype=np.int64)
    # 본 상품과 촉감 벡터가 없는 상품을 제외하고 iid 오름차순으로 동점을 해소한다.
    for begin in range(0, len(queries), BATCH):
        rows = np.arange(begin, min(begin + BATCH, len(queries)))
        q = torch.as_tensor(queries[rows], dtype=torch.float64, device=device)
        if metric == "cosine":
            scores = torch.nn.functional.normalize(q, dim=1) @ items_n.T
        else:
            # -||q - x||^2 up to the per-user constant ||q||^2, which does not change order
            scores = 2.0 * (q @ items.T) - item_sq[None, :]
        scores = scores.masked_fill(~usable[None, :], -torch.inf)
        for k, row in enumerate(rows):
            history = histories[row]
            if history.size:
                scores[k, torch.as_tensor(history - 1, device=device)] = -torch.inf
        target_index = torch.as_tensor(targets[rows] - 1, dtype=torch.long, device=device)
        target_score = scores.gather(1, target_index[:, None]).squeeze(1)
        greater = (scores > target_score[:, None]).sum(1)
        equal_before = ((scores == target_score[:, None])
                        & (index[None, :] < target_index[:, None])).sum(1)
        block = (1 + greater + equal_before).cpu().numpy()
        # a target without a tactile vector scores -inf and cannot be retrieved:
        # count it as a miss rather than as a tie among the -inf items
        block[~torch.isfinite(target_score).cpu().numpy()] = n_items + 1
        ok = valid_query[rows]
        ranks[rows[ok]] = block[ok]
    return ranks


# 같은 seen-filter와 동점 규칙을 적용한 popularity 기준 순위를 계산한다.
def popularity_ranks(train_count, histories, targets):
    """Popularity rank with the same seen filter and ascending-iid tie-break."""
    n_items = train_count.size
    order = np.lexsort((np.arange(n_items), -train_count))
    pop_rank = np.empty(n_items, dtype=np.int64)
    pop_rank[order] = np.arange(1, n_items + 1)
    out = np.empty(len(targets), dtype=np.int64)
    for row, (history, target) in enumerate(zip(histories, targets)):
        t = pop_rank[target - 1]
        out[row] = t - int((pop_rank[history - 1] < t).sum())
    return out


# 기존 rank·후보 진단과 네 가지 촉감 단독 retrieval 조합을 한 JSON 보고서로 출력한다.
def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = C.load_prepared()
    n_items = int(data["n_items"][0])
    train_count = data["train_count"]
    tactile = data["tactile"]
    available = data["tactile_available"]

    catalog = pd.read_parquet(C.DATA / "catalog.parquet", columns=["iid", "parent_asin"])
    assert (catalog["iid"].to_numpy() == np.arange(n_items)).all()
    assert catalog["parent_asin"].is_monotonic_increasing, "iid order != parent_asin order"

    users, matrix, targets, histories = candidates_module.evaluation_point(data, "test")
    if len(users) != EXPECTED_TEST_USERS:
        print(json.dumps({"STOP": f"test cohort has {len(users)} users, expected "
                                  f"{EXPECTED_TEST_USERS}"}))
        return 1
    offsets = data["offsets"]
    train_lengths = np.diff(offsets)[users]
    # train 양성 이력이 전혀 없는 target을 cold item으로 정의한다.
    cold = train_count[targets - 1] == 0
    groups = {"all": np.ones(len(users), dtype=bool), "cold": cold, "warm": ~cold}

    report = {
        "device": str(device),
        "cohort": {
            "test_users": int(len(users)),
            "cold_targets": int(cold.sum()), "warm_targets": int((~cold).sum()),
            "history_length_at_test_min": int(min(h.size for h in histories)),
            "train_only_history_min": int(train_lengths.min()),
            "users_with_train_only_history_lt_3": int((train_lengths < 3).sum()),
            "targets_without_tactile_vector": int((~available[targets - 1]).sum()),
        },
    }

    # B: R0 및 R3 네 변형의 저장 rank를 cold/warm으로 다시 분해한다.
    # ------------------------------------------------------------------ B
    cache = np.load(C.CACHE / "trained_ranks.npz")
    reproduction = np.load(C.CACHE / "ranks__R0_reproduction__test.npz")
    assert (reproduction["users"] == users).all() and (reproduction["targets"] == targets).all()
    rank_sets = {"R0_exp16_checkpoint": reproduction["ranks"]}
    for tag in ("R0_seed20260917", "R3_residual_gate_seed20260917",
                "R3_concat_proj_seed20260917", "R3_film_seed20260917",
                "R3_shuffle_seed20260917"):
        rank_sets[tag] = cache[f"{tag}__test"]
    stored = pd.read_csv(C.RESULTS / "trained_methods_all_variants.csv").set_index("tag")
    report["B_cold_warm_trained"] = {}
    for tag, ranks in rank_sets.items():
        assert ranks.shape == targets.shape
        entry = {"NDCG@10": ndcg_block(ranks, groups),
                 "recall": recall_block(ranks, groups, ks=(500,))}
        if tag in stored.index:     # cache ordering check against the stored CSV
            entry["matches_stored_test_NDCG@10"] = bool(
                abs(entry["NDCG@10"]["all"] - stored.loc[tag, "test_NDCG@10"]) < 1e-12)
        report["B_cold_warm_trained"][tag] = entry

    # C: R7 후보 생성 단계 자체의 cold/warm recall을 측정한다.
    # ------------------------------------------------------------------ C
    block = np.load(C.CACHE / "candidates__test.npz")
    assert (block["users"] == users).all() and (block["targets"] == targets).all()
    report["C_tactile_ann_candidates"] = {}
    for name in ("tactile_candidates", "tactile_std_candidates", "sasrec_candidates"):
        cand = block[name]
        hit_rank = np.where((cand == targets[:, None]).any(1),
                            (cand == targets[:, None]).argmax(1) + 1, n_items + 1)
        report["C_tactile_ann_candidates"][name] = recall_block(
            hit_rank, groups, ks=(100, 300, 500))
    report["C_tactile_ann_candidates"]["users_without_tactile_profile"] = int(
        (block["support"] == 0).sum())

    # D1-D4: 마지막 상품/이력 평균 쿼리와 cosine/L2의 네 조합을 평가한다.
    # ------------------------------------------------------------------ D1-D4
    last_item = np.array([h[-1] - 1 for h in histories])
    q_last = tactile[last_item]
    q_last_valid = available[last_item]
    q_mean = np.zeros_like(q_last)
    q_mean_valid = np.zeros(len(users), dtype=bool)
    for row, history in enumerate(histories):
        items = history - 1
        items = items[available[items]]
        if items.size:
            q_mean[row] = tactile[items].mean(0)
            q_mean_valid[row] = True
    report["D_query_coverage"] = {"q1_last_item_has_tactile": int(q_last_valid.sum()),
                                  "q2_any_history_item_has_tactile": int(q_mean_valid.sum())}

    report["D_tactile_retrieval"] = {}
    for qname, queries, valid in (("q1_last_item", q_last, q_last_valid),
                                  ("q2_history_mean", q_mean, q_mean_valid)):
        for metric in ("cosine", "L2"):
            ranks = full_catalog_ranks(queries, valid, tactile, available, histories,
                                       targets, metric, device)
            report["D_tactile_retrieval"][f"{qname}__{metric}"] = recall_block(ranks, groups)
            print(f"done {qname} {metric}", file=sys.stderr, flush=True)
    report["D_tactile_retrieval"]["popularity"] = recall_block(
        popularity_ranks(train_count, histories, targets), groups)

    # D5: 촉감 벡터의 고유성과 코사인 0.99 이상 이웃 수로 변별력을 진단한다.
    # ------------------------------------------------------------------ D5
    usable_vectors = tactile[available]
    rounded = np.round(usable_vectors.astype(np.float64), 2)
    unique_rounded = np.unique(rounded, axis=0).shape[0]
    rng = np.random.default_rng(C.SEED)
    usable_ids = np.flatnonzero(available)
    sample = rng.choice(usable_ids, size=SAMPLE_ITEMS, replace=False)
    normed = torch.nn.functional.normalize(
        torch.as_tensor(tactile[usable_ids], dtype=torch.float64, device=device), dim=1)
    position = {int(i): j for j, i in enumerate(usable_ids)}
    near = []
    for begin in range(0, SAMPLE_ITEMS, BATCH):
        idx = torch.as_tensor([position[int(i)] for i in sample[begin:begin + BATCH]],
                              device=device)
        sims = normed[idx] @ normed.T
        near.extend(((sims >= 0.99).sum(1) - 1).cpu().tolist())   # minus the item itself
    near = np.array(near)
    report["D5_discriminability"] = {
        "catalog_items": n_items,
        "items_without_tactile_vector": int((~available).sum()),
        "unique_vectors_rounded_2dp": int(unique_rounded),
        "items_with_tactile_vector": int(available.sum()),
        "sample_items": SAMPLE_ITEMS, "sample_seed": C.SEED,
        "other_items_cosine_ge_0.99": {
            "median": float(np.median(near)), "mean": float(near.mean()),
            "min": int(near.min()), "max": int(near.max()),
            "fraction_of_usable_catalog_median": float(np.median(near) / available.sum()),
        },
    }

    # D6: cold Recall@500 최고 조합을 무작위 기준과 비교한다.
    # ------------------------------------------------------------------ D6
    combos = {k: v for k, v in report["D_tactile_retrieval"].items() if k != "popularity"}
    best = max(combos, key=lambda k: combos[k]["cold"]["Recall@500"])
    value = combos[best]["cold"]["Recall@500"]
    report["D6_cold_best"] = {"combination": best, "Recall@500": value,
                              "random_baseline": RANDOM_BASELINE_RECALL_500,
                              "ratio_to_random": value / RANDOM_BASELINE_RECALL_500}

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
