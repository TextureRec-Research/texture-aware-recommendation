#!/usr/bin/env python3
# ============================================================
# [열람용 사본] 2026-10-16 시점 스냅샷. 이 파일은 실행용이 아님.
# 실제로 돌아가는 원본 경로:
#   /home/user/onsesang/material_span_grounding/experiments/24_adaptive_tactile_recommendation/scripts/12_trained_significance.py
# 원본은 이후 실험으로 바뀔 수 있음.
# 역할: 같은 약한 학습 루프의 R0 대비 R3·R4·R5·R6 유의성과 R3 shuffle 대조를 계산한다.
# ============================================================
"""Paired significance for the trained backbones, and the corrected exp20 control.

R0 here is the backbone trained by THIS loop with the same seed, so R3/R4/R5/R6 are
compared against a baseline that saw the same data, schedule and initialisation --
not against the exp16b checkpoint, which was trained separately.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import common24 as C

BASELINE_TAG = "R0_seed20260917"


# 저장된 사용자별 rank에서 paired bootstrap 신뢰구간과 aligned-vs-shuffle 대조를 만든다.
def main() -> int:
    blob = np.load(C.CACHE / "trained_ranks.npz", allow_pickle=False)
    frame = pd.read_csv(C.RESULTS / "trained_methods_all_variants.csv")
    # 동일 사용자에 대한 NDCG 차이를 같은 루프의 R0와 paired bootstrap한다.
    rows = []
    for split in ("validation", "test"):
        base_key = f"{BASELINE_TAG}__{split}"
        if base_key not in blob.files:
            print(f"missing {base_key}")
            continue
        base = C.per_user_ndcg(blob[base_key])
        for tag in frame["tag"]:
            key = f"{tag}__{split}"
            if key not in blob.files:
                continue
            mine = C.per_user_ndcg(blob[key])
            stat = C.paired_bootstrap_by_user(mine, base)
            variant = frame[frame["tag"] == tag]["variant"].iloc[0]
            rows.append({"split": split, "tag": tag, "variant": variant,
                         "NDCG@10": float(mine.mean()),
                         "delta_vs_R0_same_loop": stat["delta"],
                         "ci95_low": stat["ci95_low"], "ci95_high": stat["ci95_high"],
                         "improved": stat["improved"], "worsened": stat["worsened"],
                         "significant": bool(stat["ci95_low"] > 0 or stat["ci95_high"] < 0)})
    out = pd.DataFrame(rows)
    out.to_csv(C.RESULTS / "trained_significance_vs_R0.csv", index=False)

    # 실제 촉감과 결정적으로 섞은 촉감을 동일 구조에서 직접 비교한다.
    # corrected exp20-style control: aligned tactile table vs a permuted one, same capacity
    aligned_tag = "R3_residual_gate_seed20260917"
    shuffle_tag = "R3_shuffle_seed20260917"
    control = {"note": ("R3 residual-gate fusion with the true tactile table vs the same "
                        "architecture whose tactile table is deterministically permuted. "
                        "Identical parameter count, so a delta near zero means the tactile "
                        "branch contributed capacity rather than tactile information. This "
                        "replaces an earlier version that mistakenly used R0 -- a model with "
                        "no tactile branch at all -- as the aligned arm."),
               "aligned_tag": aligned_tag, "shuffle_tag": shuffle_tag}
    for split in ("validation", "test"):
        a_key, s_key = f"{aligned_tag}__{split}", f"{shuffle_tag}__{split}"
        if a_key in blob.files and s_key in blob.files:
            a = C.per_user_ndcg(blob[a_key])
            s = C.per_user_ndcg(blob[s_key])
            stat = C.paired_bootstrap_by_user(a, s)
            control[f"{split}_delta_NDCG@10"] = stat["delta"]
            control[f"{split}_ci95_low"] = stat["ci95_low"]
            control[f"{split}_ci95_high"] = stat["ci95_high"]
            control[f"{split}_significant"] = bool(stat["ci95_low"] > 0 or stat["ci95_high"] < 0)
    C.write_json(C.RESULTS / "aligned_vs_shuffle.json", control)

    print("===== trained variants vs R0 trained by the same loop =====")
    for split in ("validation", "test"):
        block = out[out["split"] == split].sort_values("NDCG@10", ascending=False)
        print(f"\n--- {split} ---")
        print(block[["tag", "variant", "NDCG@10", "delta_vs_R0_same_loop",
                     "ci95_low", "ci95_high", "improved", "worsened", "significant"]]
              .to_string(index=False, float_format=lambda v: f"{v:.8f}"))
    print("\n===== corrected aligned - shuffle control =====")
    for k, v in control.items():
        if k != "note":
            print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
