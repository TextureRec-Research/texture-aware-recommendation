#!/usr/bin/env python3
# ============================================================
# [열람용 사본] 2026-10-16 시점 스냅샷. 이 파일은 실행용이 아님.
# 실제로 돌아가는 원본 경로:
#   /home/user/onsesang/material_span_grounding/experiments/24_adaptive_tactile_recommendation/scripts/05_rerank.py
# 원본은 이후 실험으로 바뀔 수 있음.
# 역할: 캐시된 후보에서 R1·R2·R7을 학습하고 validation으로 설정과 epoch를 선택한다.
# ============================================================
"""R1, R2 and R7: post-hoc methods over the cached candidates.

R1   fixed global alpha, reproducing exp17b's percentile interpolation exactly
     (tactile applied only to candidates that have a tactile vector AND a
     user-category profile -- the asymmetry the audit identified).
R1b  the same fixed alpha but applied to EVERY candidate on one common scale.
     This is a diagnostic: it separates "tactile does not help" from
     "the exp17b combination rule re-ordered items by tactile availability".
R2   learned per-(user,item) gate g(u,i) in place of the global alpha.
R7   union of SASRec and tactile candidates, scored by a learned reranker.

Every learned component is fitted ONLY on the train evaluation point (target =
the user's last train event).  Hyper-parameters are chosen on validation.  Test is
scored once, after the configuration is written to artifacts/.
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
from torch.nn import functional as F

import common24 as C
import models24 as M

ALPHA_GRID = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


# 한 split의 후보·점수·사용자 특성 캐시를 메모리로 읽는다.
def load_split(split):
    blob = np.load(C.CACHE / f"candidates__{split}.npz", allow_pickle=False)
    return {k: blob[k] for k in blob.files}


# 행별 점수를 동점 평균 순위의 0~1 percentile로 변환한다.
def percentile(values, mask=None):
    """Average-tie percentile within each row, in [0, 1]."""
    order = np.argsort(values, axis=1, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float32)
    rows = np.arange(values.shape[0])[:, None]
    ranks[rows, order] = np.arange(values.shape[1], dtype=np.float32)[None, :]
    return ranks / max(values.shape[1] - 1, 1)


# 과거 행동과 이미지 촉감 정보만으로 후보별 reranking 특성을 만든다.
def build_features(block, data, candidates, base_scores, source_flags=None):
    """(U, K, F) feature tensor.  Only past behaviour, image tactile, popularity."""
    tactile = data["tactile"]
    available = data["tactile_available"]
    category_id = data["category_id"].astype(np.int64)
    train_count = data["train_count"]

    items = candidates - 1
    profile = block["profile"]
    norms = np.linalg.norm(profile, axis=1, keepdims=True)
    unit_profile = profile / np.maximum(norms, 1e-8)
    item_vectors = tactile[items]                                     # (U, K, 14)
    item_norm = np.linalg.norm(item_vectors, axis=2, keepdims=True)
    cosine = (item_vectors * unit_profile[:, None, :]).sum(axis=2) / np.maximum(item_norm[..., 0], 1e-8)
    cosine[norms[:, 0] == 0] = 0.0

    base_percentile = percentile(base_scores)
    tactile_percentile = percentile(cosine)
    availability = available[items].astype(np.float32)
    popularity = np.log1p(train_count[items])
    popularity = popularity / max(popularity.max(), 1e-8)
    category_match = block["category_hist"][np.arange(len(items))[:, None], category_id[items]]
    history_length = np.log1p(block["history_length"].astype(np.float32))[:, None]
    history_length = np.repeat(history_length / max(history_length.max(), 1e-8), items.shape[1], axis=1)
    consistency = np.repeat(block["consistency"][:, None], items.shape[1], axis=1)
    support = np.log1p(block["support"].astype(np.float32))[:, None]
    support = np.repeat(support / max(support.max(), 1e-8), items.shape[1], axis=1)
    # how confident the image predictor is: distance of each probability from 0.5
    confidence = np.abs(item_vectors - 0.5).mean(axis=2) * 2.0

    stack = [base_percentile, cosine, tactile_percentile, availability, popularity,
             category_match, history_length, consistency, support, confidence]
    if source_flags is not None:
        stack.extend(source_flags)
    return np.stack(stack, axis=2).astype(np.float32), cosine, base_percentile, tactile_percentile


# 후보 집합 안에서 target 순위를 계산하고 후보에 없으면 miss로 처리한다.
def ranks_from_scores(scores, candidates, targets):
    """Rank of the target within the candidate list; a miss ranks past the end.

    Slots holding candidate id 0 are padding, not items, and are excluded so they
    can never displace the target.
    """
    scores = np.where(candidates > 0, scores, -np.inf)
    hit = candidates == targets[:, None]
    found = hit.any(axis=1)
    # Pick the target's score by index, NOT via (scores * hit).sum().  With -inf
    # in the padded slots that product evaluates -inf * 0 = NaN, which poisons
    # target_score, makes every later comparison False, and hands the row a
    # spurious rank of 1 -- it inflated validation NDCG@10 about fivefold before
    # this was caught.  Indexing never touches the sentinel arithmetically.
    row_index = np.arange(candidates.shape[0])
    target_score = np.where(found, scores[row_index, hit.argmax(axis=1)], -np.inf)
    greater = (scores > target_score[:, None]).sum(axis=1)
    equal_lower_id = ((scores == target_score[:, None]) &
                      (candidates < targets[:, None])).sum(axis=1)
    ranks = 1 + greater + equal_lower_id
    ranks[~found] = candidates.shape[1] + 1
    return ranks


# R1은 촉감 정보가 유효한 후보에만 고정 alpha 보간을 적용한다.
def r1_scores(base_percentile, tactile_percentile, alpha, valid):
    """exp17b rule: interpolate only where the tactile term is valid."""
    return np.where(valid, base_percentile + alpha * (tactile_percentile - base_percentile),
                    base_percentile)


# R1b는 모든 후보에 같은 척도의 고정 alpha 보간을 적용한다.
def r1b_scores(base_percentile, tactile_percentile, alpha):
    """Same alpha, but every candidate moves on one common scale."""
    return (1.0 - alpha) * base_percentile + alpha * tactile_percentile


# train target으로 head를 학습하되 epoch 0을 포함해 validation NDCG가 가장 높은 상태를 고른다.
def train_head(head, features, candidates, targets, epochs, lr, device, seed,
               combine=None, base_percentile=None, tactile_percentile=None,
               negatives=127, batch=512, label="", validation=None):
    """Listwise softmax over one positive plus sampled negatives from the candidates.

    The returned head is the epoch with the best *validation* NDCG@10, including
    epoch 0 (the untrained no-op, which equals the R0 ordering).  So a learned
    method can only be reported as beating the baseline if it actually does so on
    data it was not fitted on.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    optimiser = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    hit = candidates == targets[:, None]
    usable = np.nonzero(hit.any(axis=1))[0]
    if usable.size == 0:
        raise RuntimeError("no training user has the target inside its candidate set")
    positive_column = hit[usable].argmax(axis=1)
    print(f"  [{label}] fitting on {usable.size} users "
          f"({usable.size / len(candidates):.1%} of the split)", flush=True)

    def validation_ndcg():
        if validation is None:
            return None
        scores, _ = apply_head(head, validation["features"], device, combine=combine,
                               base_percentile=validation.get("bp"),
                               tactile_percentile=validation.get("tp"))
        ranks = ranks_from_scores(scores, validation["candidates"], validation["targets"])
        return C.metrics_from_ranks(ranks)["NDCG@10"]

    # 학습하지 않은 no-op 상태도 후보로 두어 학습이 실제로 개선될 때만 채택한다.
    best = {"ndcg": validation_ndcg(), "epoch": 0,
            "state": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}}
    if best["ndcg"] is not None:
        print(f"  [{label}] epoch 0 (untrained no-op) val NDCG@10 {best['ndcg']:.8f}", flush=True)
    for epoch in range(epochs):
        order = rng.permutation(usable.size)
        total, steps = 0.0, 0
        for begin in range(0, order.size, batch):
            take = order[begin:begin + batch]
            rows = usable[take]
            positives = positive_column[take]
            sampled = rng.integers(0, candidates.shape[1], size=(rows.size, negatives))
            columns = np.concatenate([positives[:, None], sampled], axis=1)
            picked = torch.as_tensor(
                features[rows[:, None], columns], dtype=torch.float32, device=device)
            if combine is None:
                logits = head(picked)
            else:
                gate = head(picked)
                bp = torch.as_tensor(base_percentile[rows[:, None], columns],
                                     dtype=torch.float32, device=device)
                tp = torch.as_tensor(tactile_percentile[rows[:, None], columns],
                                     dtype=torch.float32, device=device)
                logits = (bp + gate * (tp - bp)) * 20.0
            loss = F.cross_entropy(logits, torch.zeros(rows.size, dtype=torch.long, device=device))
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += float(loss.item())
            steps += 1
        current = validation_ndcg()
        message = f"  [{label}] epoch {epoch + 1}/{epochs} loss {total / max(steps,1):.5f}"
        if current is not None:
            message += f" val NDCG@10 {current:.8f}"
            if best["ndcg"] is None or current > best["ndcg"]:
                best = {"ndcg": current, "epoch": epoch + 1,
                        "state": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}}
                message += "  *best*"
        print(message, flush=True)
    if best["state"] is not None:
        head.load_state_dict(best["state"])
        print(f"  [{label}] selected epoch {best['epoch']} "
              f"(val NDCG@10 {best['ndcg']:.8f})", flush=True)
    return head


@torch.inference_mode()
# 학습된 gate 또는 reranker를 큰 후보 텐서에 chunk 단위로 적용한다.
def apply_head(head, features, device, chunk=2048, combine=None,
               base_percentile=None, tactile_percentile=None):
    out = np.zeros(features.shape[:2], dtype=np.float32)
    gates = np.zeros(features.shape[:2], dtype=np.float32) if combine else None
    for begin in range(0, features.shape[0], chunk):
        rows = slice(begin, min(begin + chunk, features.shape[0]))
        block = torch.as_tensor(features[rows], dtype=torch.float32, device=device)
        value = head(block)
        if combine is None:
            out[rows] = value.cpu().numpy()
        else:
            gate = value.cpu().numpy()
            gates[rows] = gate
            bp, tp = base_percentile[rows], tactile_percentile[rows]
            out[rows] = bp + gate * (tp - bp)
    return out, gates


# R1/R2/R7의 validation 선택, test 1회 평가, rank 캐시 저장을 순서대로 수행한다.
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=C.SEED)
    args = ap.parse_args()

    device = torch.device(args.device)
    data = C.load_prepared()
    splits = {s: load_split(s) for s in ("train", "validation", "test")}
    results, per_user_ranks = [], {}

    # R1/R1b: validation에서 고정 alpha를 선택한다.
    # ---------------------------------------------------------------- R1 / R1b
    prepared = {}
    for split, block in splits.items():
        candidates = block["sasrec_candidates"]
        features, cosine, bp, tp = build_features(block, data, candidates,
                                                  block["sasrec_scores"])
        valid = (data["tactile_available"][candidates - 1]
                 & (block["support"][:, None] > 0))
        prepared[split] = {"candidates": candidates, "features": features,
                           "bp": bp, "tp": tp, "valid": valid,
                           "targets": block["targets"]}

    alpha_rows = []
    for alpha in ALPHA_GRID:
        block = prepared["validation"]
        for name, fn in (("R1_exp17_rule", lambda a: r1_scores(block["bp"], block["tp"], a, block["valid"])),
                         ("R1b_common_scale", lambda a: r1b_scores(block["bp"], block["tp"], a))):
            ranks = ranks_from_scores(fn(alpha), block["candidates"], block["targets"])
            metrics = C.metrics_from_ranks(ranks)
            alpha_rows.append({"method": name, "alpha": alpha, **metrics})
    alpha_frame = pd.DataFrame(alpha_rows)
    alpha_frame.to_csv(C.RESULTS / "alpha_search_validation.csv", index=False)
    selected = {}
    for name in ("R1_exp17_rule", "R1b_common_scale"):
        subset = alpha_frame[alpha_frame["method"] == name]
        best = subset.sort_values(["NDCG@10", "alpha"], ascending=[False, True]).iloc[0]
        selected[name] = float(best["alpha"])
        print(f"{name}: validation-selected alpha = {best['alpha']} "
              f"(NDCG@10 {best['NDCG@10']:.8f})", flush=True)
    C.write_json(C.ARTIFACTS / "alpha_selection.json", selected)

    # R2: 초기 gate bias와 학습률을 sweep하고 validation 성능으로 하나를 고른다.
    # ------------------------------------------------------------------- R2
    # The gate's starting point matters: initialise it near zero and it may never
    # leave (a global alpha of ~0 in disguise); initialise it near 0.5 and it
    # starts far from the baseline.  Which is right is a hyper-parameter, so it is
    # swept and chosen on VALIDATION, and the gate's dispersion is recorded for
    # every setting so a degenerate "adaptive" gate cannot be reported as adaptive.
    train_block, train_prepared = splits["train"], prepared["train"]
    gate_sweep = []
    best_gate = None
    for init_bias in (-6.0, -2.0, 0.0):
        for lr in (args.lr, args.lr * 10):
            candidate_gate = M.TactileGate(train_prepared["features"].shape[2],
                                           init_bias=init_bias).to(device)
            train_head(candidate_gate, train_prepared["features"],
                       train_prepared["candidates"], train_prepared["targets"],
                       args.epochs, lr, device, args.seed,
                       combine="gate", base_percentile=train_prepared["bp"],
                       tactile_percentile=train_prepared["tp"],
                       label=f"R2 gate init{init_bias} lr{lr}",
                       validation=prepared["validation"])
            scores, gates = apply_head(candidate_gate, prepared["validation"]["features"],
                                       device, combine="gate",
                                       base_percentile=prepared["validation"]["bp"],
                                       tactile_percentile=prepared["validation"]["tp"])
            ranks = ranks_from_scores(scores, prepared["validation"]["candidates"],
                                      prepared["validation"]["targets"])
            ndcg = C.metrics_from_ranks(ranks)["NDCG@10"]
            row = {"init_bias": init_bias, "lr": lr, "validation_NDCG@10": ndcg,
                   "gate_mean": float(gates.mean()), "gate_std": float(gates.std()),
                   "gate_p05": float(np.percentile(gates, 5)),
                   "gate_p95": float(np.percentile(gates, 95)),
                   "across_user_std": float(gates.mean(axis=1).std())}
            gate_sweep.append(row)
            print(f"  [R2 sweep] init {init_bias} lr {lr}: val NDCG {ndcg:.8f} "
                  f"gate mean {row['gate_mean']:.5f} std {row['gate_std']:.2e}", flush=True)
            if best_gate is None or ndcg > best_gate[0]:
                best_gate = (ndcg, candidate_gate, row)
    gate = best_gate[1]
    pd.DataFrame(gate_sweep).to_csv(C.RESULTS / "r2_gate_sweep.csv", index=False)
    C.write_json(C.ARTIFACTS / "r2_gate_selection.json", {
        "selected_on": "validation NDCG@10",
        "selected": best_gate[2],
        "sweep": gate_sweep,
        "note": ("gate_std near zero means the learned gate is a global constant in "
                 "disguise, i.e. it did NOT become user- or item-specific."),
    })
    print(f"  [R2] selected {best_gate[2]}", flush=True)

    # R7: SASRec 후보와 표준화 촉감 후보를 합친 뒤 학습형 reranker를 적합한다.
    # ------------------------------------------------------------------- R7
    union_prepared = {}
    for split, block in splits.items():
        sasrec = block["sasrec_candidates"]
        # A24-1: validation prefers the standardised tactile ANN
        # (Recall@500 0.002240 vs 0.001358).  Selected on validation only.
        tactile = block["tactile_std_candidates"]
        merged, in_s, in_t = [], [], []
        width = sasrec.shape[1] + tactile.shape[1]
        for row in range(len(sasrec)):
            union, first = np.unique(np.concatenate([sasrec[row], tactile[row]]),
                                     return_index=True)
            # Pad with 0, an id no catalog item uses.  An earlier version repeated
            # union[0] here, which let one arbitrary item be counted several times
            # when computing the target's rank.  Padding is needed for 23 % of rows
            # (mean 0.3 slots), so the bias was small, but it is removed rather
            # than argued away: ranks_from_scores gives every 0 slot -inf.
            padded = np.zeros(width, dtype=np.int64)
            padded[:union.size] = union
            merged.append(padded)
            in_s.append(np.isin(padded, sasrec[row]).astype(np.float32))
            in_t.append(np.isin(padded, tactile[row]).astype(np.float32))
        merged = np.stack(merged)
        in_s, in_t = np.stack(in_s), np.stack(in_t)
        # backbone score for union members: known for SASRec members, else min
        score_lookup = np.full(merged.shape, -np.inf, dtype=np.float32)
        for row in range(len(sasrec)):
            mapping = dict(zip(sasrec[row].tolist(), block["sasrec_scores"][row].tolist()))
            floor = float(block["sasrec_scores"][row].min())
            score_lookup[row] = [mapping.get(int(c), floor) for c in merged[row]]
        features, cosine, bp, tp = build_features(block, data, merged, score_lookup,
                                                  source_flags=[in_s, in_t])
        union_prepared[split] = {"candidates": merged, "features": features,
                                 "bp": bp, "tp": tp, "targets": block["targets"]}
        if split != "train":
            recall = float((merged == block["targets"][:, None]).any(axis=1).mean())
            sasrec_recall = float((sasrec == block["targets"][:, None]).any(axis=1).mean())
            print(f"[{split}] candidate recall  sasrec {sasrec_recall:.6f} "
                  f"-> union {recall:.6f}", flush=True)

    # base_index 0 is base_percentile in build_features' stack order
    reranker = M.Reranker(union_prepared["train"]["features"].shape[2], base_index=0).to(device)
    train_head(reranker, union_prepared["train"]["features"],
               union_prepared["train"]["candidates"], union_prepared["train"]["targets"],
               args.epochs, args.lr, device, args.seed, label="R7 reranker",
               validation=union_prepared["validation"])

    # 선택이 끝난 뒤 validation과 test를 평가하고 사용자별 rank를 저장한다.
    # ------------------------------------------------------------- evaluation
    gate_stats = {}
    for split in ("validation", "test"):
        block, union = prepared[split], union_prepared[split]
        entries = {
            "R0_sasrec": block["bp"],
            "R1_fixed_alpha": r1_scores(block["bp"], block["tp"],
                                        selected["R1_exp17_rule"], block["valid"]),
            "R1b_fixed_alpha_common_scale": r1b_scores(block["bp"], block["tp"],
                                                       selected["R1b_common_scale"]),
        }
        scores, gates = apply_head(gate, block["features"], device, combine="gate",
                                   base_percentile=block["bp"], tactile_percentile=block["tp"])
        entries["R2_learned_gate"] = scores
        gate_stats[split] = {
            "mean": float(gates.mean()), "std": float(gates.std()),
            "p05": float(np.percentile(gates, 5)), "p50": float(np.percentile(gates, 50)),
            "p95": float(np.percentile(gates, 95)),
            "per_user_std_mean": float(gates.std(axis=1).mean()),
            "across_user_std_of_mean": float(gates.mean(axis=1).std()),
        }
        np.savez_compressed(C.CACHE / f"gate_values__{split}.npz", gates=gates,
                            history_length=splits[split]["history_length"],
                            consistency=splits[split]["consistency"])

        for name, score in entries.items():
            ranks = ranks_from_scores(score, block["candidates"], block["targets"])
            per_user_ranks[(name, split)] = ranks
            results.append({"method": name, "split": split, "candidates": "sasrec_top500",
                            **C.metrics_from_ranks(ranks)})

        union_score, _ = apply_head(reranker, union["features"], device)
        ranks = ranks_from_scores(union_score, union["candidates"], union["targets"])
        per_user_ranks[("R7_hybrid_candidates_reranker", split)] = ranks
        results.append({"method": "R7_hybrid_candidates_reranker", "split": split,
                        "candidates": "union_sasrec500_tactile500",
                        **C.metrics_from_ranks(ranks)})

    frame = pd.DataFrame(results)
    frame.to_csv(C.RESULTS / "posthoc_methods.csv", index=False)
    C.write_json(C.RESULTS / "gate_distribution.json", gate_stats)
    np.savez_compressed(C.CACHE / "posthoc_ranks.npz",
                        **{f"{m}__{s}": r for (m, s), r in per_user_ranks.items()})

    print("\n===== post-hoc methods =====")
    print(frame.pivot_table(index="method", columns="split",
                            values=["NDCG@10", "HR@10", "MRR@10"])
          .to_string(float_format=lambda v: f"{v:.8f}"))
    print("\ngate distribution:", gate_stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
