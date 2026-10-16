#!/usr/bin/env python3
# ============================================================
# [열람용 사본] 2026-10-16 시점 스냅샷. 이 파일은 실행용이 아님.
# 실제로 돌아가는 원본 경로:
#   /home/user/onsesang/material_span_grounding/experiments/24_adaptive_tactile_recommendation/scripts/10_evaluate_trained.py
# 원본은 이후 실험으로 바뀔 수 있음.
# 역할: 약한 Exp24 레시피로 학습한 R0·R3·R4·R5·R6 체크포인트를 평가하고 validation으로 대표 변형을 고른다.
# ============================================================
"""Evaluate every trained Exp24 checkpoint on validation and test.

Within each family (R3 fusion, R4 lambda, R5 hard-negative ratio) the variant is
chosen by **validation** NDCG@10 before test is read; the per-variant test numbers
are still written out so the selection is auditable rather than merely asserted.

The R3 `shuffle` control is the exp20-style test that separates "tactile
information" from "extra parameters": it has identical capacity to
`residual_gate` but its tactile table is deterministically permuted.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch

import common24 as C
import models24 as M
from importlib import import_module

evaluator = import_module("02_evaluate")
trainer = import_module("03_train")


# 체크포인트 메타데이터와 state_dict에서 동일한 모델 variant를 복원한다.
def rebuild(saved, data, n_items, device):
    variant = saved["variant"]
    model = trainer.build_model(variant, data, n_items, device,
                                saved.get("lambda_tactile", 0.1),
                                saved.get("fusion", "residual_gate"),
                                int(saved.get("seed", C.SEED)))
    model.load_state_dict(saved["state_dict"])
    return model.to(device).eval()


# 모든 체크포인트의 validation/test rank를 계산한 뒤 family별 대표를 validation으로 선택한다.
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=512)
    args = ap.parse_args()

    device = torch.device(args.device)
    data = C.load_prepared()
    n_items = int(data["n_items"][0])

    # 각 split의 평가 cohort, target, seen item을 한 번만 구성해 모든 모델에 재사용한다.
    blocks = {}
    for split in ("validation", "test"):
        users, matrix, targets = evaluator.history_and_targets(data, split)
        blocks[split] = (users, matrix, targets, evaluator.seen_items(data, split, users))

    # R3/R4/R5/R6의 모든 저장 체크포인트를 동일한 전체 카탈로그 평가로 비교한다.
    rows, per_user = [], {}
    for path in sorted(C.CHECKPOINTS.glob("*.pt")):
        saved = torch.load(path, map_location="cpu", weights_only=False)
        tag = saved.get("tag", path.stem)
        print(f"[{tag}] variant={saved['variant']} fusion={saved.get('fusion')} "
              f"lambda={saved.get('lambda_tactile')} hard={saved.get('hard_ratio')} "
              f"best_epoch={saved.get('best_epoch')}", flush=True)
        model = rebuild(saved, data, n_items, device)
        row = {"tag": tag, "variant": saved["variant"], "fusion": saved.get("fusion", ""),
               "lambda_tactile": saved.get("lambda_tactile", np.nan),
               "hard_ratio": saved.get("hard_ratio", np.nan),
               "lr": saved.get("lr", np.nan), "seed": saved.get("seed", C.SEED),
               "best_epoch": saved.get("best_epoch", -1)}
        for split, (users, matrix, targets, seen) in blocks.items():
            start = time.time()
            ranks, _ = evaluator.rank_targets(model, matrix, targets, seen, device,
                                              batch=args.batch)
            metrics = C.metrics_from_ranks(ranks)
            for key, value in metrics.items():
                if key != "users":
                    row[f"{split}_{key}"] = value
            per_user[f"{tag}__{split}"] = ranks
            print(f"  {split}: NDCG@10 {metrics['NDCG@10']:.8f} "
                  f"({time.time() - start:.0f}s)", flush=True)
        rows.append(row)
        del model
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    if not rows:
        print("no checkpoints found")
        return 0

    frame = pd.DataFrame(rows)
    frame.to_csv(C.RESULTS / "trained_methods_all_variants.csv", index=False)
    np.savez_compressed(C.CACHE / "trained_ranks.npz", **per_user)

    # test를 선택에 사용하지 않고 validation NDCG만으로 family별 대표를 고른다.
    # validation-selected representative per family
    selection, selected_rows = {}, []
    for variant, group in frame.groupby("variant"):
        best = group.sort_values("validation_NDCG@10", ascending=False).iloc[0]
        selection[variant] = {
            "tag": best["tag"], "fusion": best["fusion"],
            "lambda_tactile": None if pd.isna(best["lambda_tactile"]) else float(best["lambda_tactile"]),
            "hard_ratio": None if pd.isna(best["hard_ratio"]) else float(best["hard_ratio"]),
            "validation_NDCG@10": float(best["validation_NDCG@10"]),
            "selected_on": "validation NDCG@10, before test was read",
        }
        selected_rows.append(best)
    C.write_json(C.ARTIFACTS / "trained_variant_selection.json", selection)
    pd.DataFrame(selected_rows).to_csv(C.RESULTS / "trained_methods.csv", index=False)

    print("\n===== all variants =====")
    print(frame[["tag", "variant", "validation_NDCG@10", "test_NDCG@10"]]
          .sort_values("validation_NDCG@10", ascending=False)
          .to_string(index=False, float_format=lambda v: f"{v:.8f}"))

    # R3 실제 촉감과 shuffle을 같은 용량에서 비교해 촉감 정보 자체의 효과를 분리한다.
    # the exp20-style control, if both members are present
    # R0 also stores fusion="residual_gate" (the argparse default), so filtering on
    # fusion alone picked R0 -- a model with NO tactile branch -- as the "aligned"
    # arm and made the exp20 control meaningless.  Require variant R3 as well.
    shuffle = frame[(frame["variant"] == "R3") & (frame["fusion"] == "shuffle")]
    aligned = frame[(frame["variant"] == "R3") & (frame["fusion"] == "residual_gate")]
    if len(shuffle) and len(aligned):
        for split in ("validation", "test"):
            delta = float(aligned[f"{split}_NDCG@10"].iloc[0]) - float(shuffle[f"{split}_NDCG@10"].iloc[0])
            print(f"\naligned - shuffle ({split}) NDCG@10: {delta:+.10f}")
        C.write_json(C.RESULTS / "aligned_vs_shuffle.json", {
            "note": ("Identical capacity, tactile table deterministically permuted in the "
                     "shuffle arm. A delta near zero means the tactile branch contributed "
                     "parameters, not tactile information -- the exp20 finding, retested."),
            **{f"{split}_delta_NDCG@10":
               float(aligned[f"{split}_NDCG@10"].iloc[0]) - float(shuffle[f"{split}_NDCG@10"].iloc[0])
               for split in ("validation", "test")},
            "aligned_tag": aligned["tag"].iloc[0], "shuffle_tag": shuffle["tag"].iloc[0],
        })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
