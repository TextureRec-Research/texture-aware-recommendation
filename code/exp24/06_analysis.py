#!/usr/bin/env python3
# ============================================================
# [열람용 사본] 2026-10-16 시점 스냅샷. 이 파일은 실행용이 아님.
# 실제로 돌아가는 원본 경로:
#   /home/user/onsesang/material_span_grounding/experiments/24_adaptive_tactile_recommendation/scripts/06_analysis.py
# 원본은 이후 실험으로 바뀔 수 있음.
# 역할: R7 등 post-hoc 방법을 R0와 비교해 paired bootstrap과 사용자군 분석을 수행한다.
# ============================================================
"""Step 24.4 / 24.5: who does tactile help, and is any difference real?

Subgroups are defined from TRAIN history only -- never from the test outcome --
so the grouping cannot be tuned to make a method look good.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import common24 as C

BASELINE = "R0_sasrec"


# test 결과를 보지 않고 이력 길이와 촉감 일관성으로 사용자군을 정의한다.
def subgroups(block):
    """Bins fixed in advance from train-side quantities only."""
    history = block["history_length"]
    consistency = block["consistency"]
    support = block["support"]
    groups = {}
    groups["history 3-4"] = (history >= 3) & (history <= 4)
    groups["history 5-9"] = (history >= 5) & (history <= 9)
    groups["history 10+"] = history >= 10
    # consistency split at the median of the *validation* cohort, computed once
    median = float(np.median(consistency[support > 0])) if (support > 0).any() else 0.5
    groups["tactile consistency: high"] = (support > 0) & (consistency >= median)
    groups["tactile consistency: low"] = (support > 0) & (consistency < median)
    groups["no tactile profile"] = support == 0
    return groups, median


# 저장된 사용자별 rank로 전체·하위군의 R0 대비 차이와 신뢰구간을 계산한다.
def main() -> int:
    ranks_blob = np.load(C.CACHE / "posthoc_ranks.npz", allow_pickle=False)
    available = {}
    for key in ranks_blob.files:
        method, split = key.rsplit("__", 1)
        available.setdefault(split, {})[method] = ranks_blob[key]

    rows, group_rows = [], []
    # split별로 동일 사용자의 NDCG 차이를 bootstrap하여 방법 간 차이를 검정한다.
    for split, methods in available.items():
        block = np.load(C.CACHE / f"candidates__{split}.npz", allow_pickle=False)
        groups, median = subgroups(block)
        base = C.per_user_ndcg(methods[BASELINE])
        for method, ranks in methods.items():
            mine = C.per_user_ndcg(ranks)
            stat = C.paired_bootstrap_by_user(mine, base)
            rows.append({"split": split, "method": method,
                         "NDCG@10": float(mine.mean()),
                         "delta_vs_R0": stat["delta"],
                         "ci95_low": stat["ci95_low"], "ci95_high": stat["ci95_high"],
                         "improved": stat["improved"], "worsened": stat["worsened"],
                         "unchanged": stat["unchanged"],
                         "significant": bool(stat["ci95_low"] > 0 or stat["ci95_high"] < 0)})
            for name, mask in groups.items():
                if mask.sum() < 50:
                    continue
                sub = C.paired_bootstrap_by_user(mine[mask], base[mask])
                group_rows.append({"split": split, "method": method, "group": name,
                                   "users": int(mask.sum()),
                                   "NDCG@10": float(mine[mask].mean()),
                                   "delta_vs_R0": sub["delta"],
                                   "ci95_low": sub["ci95_low"], "ci95_high": sub["ci95_high"],
                                   "significant": bool(sub["ci95_low"] > 0 or sub["ci95_high"] < 0)})

    # 전체 및 사전 정의 하위군 결과를 별도 표로 저장한다.
    frame = pd.DataFrame(rows).sort_values(["split", "NDCG@10"], ascending=[True, False])
    frame.to_csv(C.RESULTS / "significance_vs_R0.csv", index=False)
    group_frame = pd.DataFrame(group_rows)
    group_frame.to_csv(C.RESULTS / "user_group_analysis.csv", index=False)
    C.write_json(C.ARTIFACTS / "subgroup_definition.json", {
        "defined_from": "train-side history only; never from the test outcome",
        "consistency_median_threshold": median,
        "bins": ["history 3-4", "history 5-9", "history 10+",
                 "tactile consistency: high/low (median split)", "no tactile profile"],
    })

    for split in sorted(frame["split"].unique()):
        print(f"\n===== {split} =====")
        print(frame[frame["split"] == split][
            ["method", "NDCG@10", "delta_vs_R0", "ci95_low", "ci95_high",
             "improved", "worsened", "significant"]]
            .to_string(index=False, float_format=lambda v: f"{v:.8f}"))
    if not group_frame.empty:
        print("\n===== user groups (test) =====")
        block = group_frame[group_frame["split"] == "test"]
        print(block.pivot_table(index="group", columns="method", values="delta_vs_R0")
              .to_string(float_format=lambda v: f"{v:+.8f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
