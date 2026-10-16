#!/usr/bin/env python3
# ============================================================
# [열람용 사본] 2026-10-16 시점 스냅샷. 이 파일은 실행용이 아님.
# 실제로 돌아가는 원본 경로:
#   /home/user/onsesang/material_span_grounding/experiments/24_adaptive_tactile_recommendation/scripts/15_cold_zero_reeval.py
# 원본은 이후 실험으로 바뀔 수 있음.
# 역할: 약한 레시피 체크포인트의 cold ID 행 제거 조건을 재평가하고 R0 대비 차이를 검정한다.
# ============================================================
"""Does zeroing cold-item embeddings at inference close the Exp24 R0 gap?

Re-evaluates the saved Exp24 checkpoints -- no training -- under:
  (a)  original      : exactly 10_evaluate_trained.py; asserted against
                       cache/trained_ranks.npz (NDCG@10 within 1e-8)
  (b)  cold-zero     : model.item.weight[1:][cold] = 0 in place before scoring, as
                       exp16b's Scorer does (16_strong_recommender_tactile/
                       recommender.py:78).  In place, so it reaches the query side
                       too whenever a history contains a cold item.  For R3 the
                       hook still adds the tactile term on top of the zeroed ID row.
  (b') R3 only       : (b) plus item_embedding() returns 0 for cold ids, so the
                       fused vector itself is 0 on both the query and catalog side.
cold = train_count == 0 (13_retrieval_diagnostic.py:74).  Ties break by ascending
iid inside 02_evaluate.rank_targets.

Nothing is written to disk; the JSON report goes to stdout and is transcribed into
RESULTS_15_cold_zero_reeval.md.

Run: <texture python> scripts/15_cold_zero_reeval.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

import common24 as C
from importlib import import_module

evaluator = import_module("02_evaluate")
trained = import_module("10_evaluate_trained")

TAGS = [
    "R0_seed20260917",
    "R3_residual_gate_seed20260917", "R3_concat_proj_seed20260917",
    "R3_film_seed20260917", "R3_shuffle_seed20260917",
    "R4_lambda0.01_seed20260917", "R4_lambda0.05_seed20260917",
    "R4_lambda0.1_seed20260917", "R4_lambda0.2_seed20260917",
    "R4_lambda0.5_seed20260917",
    "R5_hard0.5_seed20260917", "R5_hard0.5_SHUFFLED_seed20260917",
    "R6_seed20260917",
]
BASELINE = "R0_seed20260917"
EXP16_R0_TEST_NDCG = 0.00349417
EXPECTED_USERS = {"validation": 14_731, "test": 29_991}
BATCH = 512                     # 10_evaluate_trained.py default
REPLICATES = 2000


# 전체·cold·warm 사용자군별 NDCG, HR, MRR, Recall을 계산한다.
def metrics(ranks, groups):
    out = {}
    for name, mask in groups.items():
        r = ranks[mask].astype(np.float64)
        hit = r <= 10
        out[name] = {
            "users": int(mask.sum()),
            "NDCG@10": float(np.where(hit, 1.0 / np.log2(r + 1.0), 0.0).mean()),
            "HR@10": float(hit.mean()),
            "MRR@10": float(np.where(hit, 1.0 / r, 0.0).mean()),
            **{f"Recall@{k}": float((r <= k).mean()) for k in (100, 500, 3000)},
        }
    return out


# 추론 전에 train에서 보지 못한 상품의 ID embedding 행을 0으로 만든다.
def zero_cold_rows(model, cold_rows):
    with torch.no_grad():
        model.item.weight[1:][cold_rows] = 0


# R3의 query와 catalog 양쪽에서 cold fused embedding 전체를 0으로 강제한다.
def zero_cold_fused(model, cold_shifted):
    """Instance-level override: self.item_embedding resolves here first, so both
    forward() (query side) and TactileSASRec.catalog_matrix() see it."""
    original = model.item_embedding

    def item_embedding(ids):
        return original(ids).masked_fill(cold_shifted[ids].unsqueeze(-1), 0)

    model.item_embedding = item_embedding


# 동일 사용자의 NDCG 차이에 대한 paired bootstrap과 유의성 판정을 반환한다.
def bootstrap(per_user, a, b):
    stat = C.paired_bootstrap_by_user(per_user[a], per_user[b], replicates=REPLICATES,
                                      seed=C.SEED)
    stat["significant"] = bool(stat["ci95_low"] > 0 or stat["ci95_high"] < 0)
    return stat


# 원본·cold-zero·R3 full-zero 조건을 평가하고 모델별 R0 대비 차이를 출력한다.
def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = C.load_prepared()
    n_items = int(data["n_items"][0])
    # train 양성 횟수가 0인 카탈로그 상품을 cold item으로 정의한다.
    cold = data["train_count"] == 0
    cold_rows = torch.as_tensor(cold, device=device)
    cold_shifted = torch.zeros(n_items + 1, dtype=torch.bool, device=device)
    cold_shifted[1:] = cold_rows
    cache = np.load(C.CACHE / "trained_ranks.npz")

    blocks = {}
    for split in ("validation", "test"):
        users, matrix, targets = evaluator.history_and_targets(data, split)
        assert len(users) == EXPECTED_USERS[split], (split, len(users))
        seen = evaluator.seen_items(data, split, users)
        target_cold = cold[targets - 1]
        blocks[split] = (matrix, targets, seen,
                         {"all": np.ones(len(users), bool), "cold": target_cold,
                          "warm": ~target_cold})

    # 체크포인트마다 원본 조건과 cold-zero 조건을 같은 cohort에서 다시 계산한다.
    report = {"device": str(device), "cold_catalog_items": int(cold.sum()),
              "metrics": {}, "assert_a_matches_cache": {}}
    per_user = {}           # (condition, split, tag) -> per-user NDCG@10
    for tag in TAGS:
        saved = torch.load(C.CHECKPOINTS / f"{tag}.pt", map_location="cpu", weights_only=False)
        conditions = ["a", "b"] + (["b_prime"] if saved["variant"] == "R3" else [])
        for condition in conditions:
            model = trained.rebuild(saved, data, n_items, device)
            if condition in ("b", "b_prime"):
                zero_cold_rows(model, cold_rows)
            if condition == "b_prime":
                zero_cold_fused(model, cold_shifted)
            for split, (matrix, targets, seen, groups) in blocks.items():
                ranks, _ = evaluator.rank_targets(model, matrix, targets, seen, device,
                                                  batch=BATCH)
                result = metrics(ranks, groups)
                report["metrics"].setdefault(tag, {}).setdefault(condition, {})[split] = result
                per_user[(condition, split, tag)] = C.per_user_ndcg(ranks)
                if condition == "a":
                    cached = C.per_user_ndcg(cache[f"{tag}__{split}"]).mean()
                    diff = abs(result["all"]["NDCG@10"] - cached)
                    report["assert_a_matches_cache"][f"{tag}__{split}"] = diff
                    assert diff < 1e-8, f"{tag} {split}: (a) differs from cache by {diff}"
                print(f"{tag} {condition} {split} NDCG@10 {result['all']['NDCG@10']:.8f}",
                      file=sys.stderr, flush=True)
            del model
            torch.cuda.empty_cache()

    # cold 행 제거가 약한 R0와 Exp16 R0 사이 격차를 얼마나 설명하는지 계산한다.
    # ---- R0 gap closed by (b)
    a = report["metrics"][BASELINE]["a"]["test"]["all"]["NDCG@10"]
    b = report["metrics"][BASELINE]["b"]["test"]["all"]["NDCG@10"]
    report["R0_gap_closed"] = {"NDCG_a": a, "NDCG_b": b, "exp16_checkpoint": EXP16_R0_TEST_NDCG,
                               "fraction": (b - a) / (EXP16_R0_TEST_NDCG - a)}

    # 모든 variant를 같은 조건의 R0와 사용자 단위 paired bootstrap으로 비교한다.
    # ---- paired bootstrap, every comparison under (a) and (b) with the same draws
    comparisons = [(tag, BASELINE) for tag in TAGS if tag != BASELINE]
    comparisons.append(("R5_hard0.5_seed20260917", "R5_hard0.5_SHUFFLED_seed20260917"))
    report["bootstrap"] = {}
    for condition in ("a", "b"):
        for split in ("validation", "test"):
            for mine, base in comparisons:
                key = f"{condition}__{split}__{mine}__vs__{base}"
                report["bootstrap"][key] = bootstrap(
                    {t: per_user[(condition, split, t)] for t in (mine, base)}, mine, base)
    # R3 (b') against R0 (b): both have cold vectors fully zeroed
    for split in ("validation", "test"):
        for tag in TAGS:
            if tag.startswith("R3_"):
                report["bootstrap"][f"b_prime__{split}__{tag}__vs__{BASELINE}(b)"] = bootstrap(
                    {tag: per_user[("b_prime", split, tag)],
                     BASELINE: per_user[("b", split, BASELINE)]}, tag, BASELINE)

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
