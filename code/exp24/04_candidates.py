#!/usr/bin/env python3
# ============================================================
# [열람용 사본] 2026-10-16 시점 스냅샷. 이 파일은 실행용이 아님.
# 실제로 돌아가는 원본 경로:
#   /home/user/onsesang/material_span_grounding/experiments/24_adaptive_tactile_recommendation/scripts/04_candidates.py
# 원본은 이후 실험으로 바뀔 수 있음.
# 역할: R7용 SASRec·촉감 후보와 누수 없는 reranking 입력 특성을 split별로 생성한다.
# ============================================================
"""Generate and cache candidates + reranking features for R1, R2 and R7.

Three evaluation points, all leakage-safe:
  train      : target = the user's LAST TRAIN event, history = train events before it.
               This is the only place a learned reranker/gate may be fitted.
  validation : target = the validation event, history = train events.
  test       : target = the test event, history = train + validation events.

Candidate sets per user:
  sasrec  : top-K by the frozen exp16b SASRec backbone
  tactile : top-K by cosine similarity between the user's tactile profile
            (built from PAST items only) and each catalog item's Last2 vector
  union   : the two combined -- this is R7's candidate generator

Features are built only from: the backbone score, the image-derived tactile
vectors, train-period popularity, category, and the user's own past behaviour.
The target item's reviews, rating and review-derived labels are never touched.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

import common24 as C
from importlib import import_module

evaluator = import_module("02_evaluate")

TACTILE_K = 500
SASREC_K = 500


# train·validation·test마다 과거 이력만 사용해 사용자, target, seen 이력을 구성한다.
def evaluation_point(data, split):
    """users, history matrix, target, and the set of already-seen items."""
    offsets, sequences = data["offsets"], data["sequences"]
    if split == "train":
        lengths = np.diff(offsets)
        users = np.nonzero(lengths >= 2)[0]
        targets = np.array([sequences[offsets[u + 1] - 1] for u in users], dtype=np.int64)
        histories = [sequences[offsets[u]:offsets[u + 1] - 1] for u in users]
    else:
        users = data[f"{split}_users"]
        targets = data[f"{split}_target"][users]
        extra = data["validation_target"] if split == "test" else None
        histories = []
        for u in users:
            history = sequences[offsets[u]:offsets[u + 1]]
            if extra is not None and extra[u] > 0:
                history = np.concatenate([history, [extra[u]]])
            histories.append(history)
    matrix = np.zeros((len(users), C.MAXLEN), dtype=np.int64)
    for row, history in enumerate(histories):
        cut = history[-C.MAXLEN:]
        if cut.size:
            matrix[row, C.MAXLEN - cut.size:] = cut
    return users, matrix, targets, histories


# 사용자의 과거 상품에서 촉감 평균·일관성·카테고리 분포를 계산한다.
def tactile_profiles(histories, tactile, available, category_id, n_categories):
    """Mean tactile vector, support, consistency and category histogram from the
    user's past items only."""
    dim = tactile.shape[1]
    profile = np.zeros((len(histories), dim), dtype=np.float32)
    support = np.zeros(len(histories), dtype=np.int32)
    consistency = np.zeros(len(histories), dtype=np.float32)
    category_hist = np.zeros((len(histories), n_categories), dtype=np.float32)
    for row, history in enumerate(histories):
        items = history[history > 0] - 1
        if items.size:
            counts = np.bincount(category_id[items], minlength=n_categories)
            category_hist[row] = counts / max(counts.sum(), 1)
        usable = items[available[items]]
        support[row] = usable.size
        if usable.size:
            block = tactile[usable]
            profile[row] = block.mean(axis=0)
            consistency[row] = 1.0 - float(block.std(axis=0).mean() * 2.0) if usable.size > 1 else 1.0
    return profile, support, np.clip(consistency, 0.0, 1.0), category_hist


# SASRec top-K와 촉감 top-K를 생성하고 R7이 읽는 후보 캐시를 저장한다.
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "validation", "test"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--checkpoint",
                    default=str(C.E16 / "artifacts/checkpoints/sasrec_lr0.0003.pt"))
    ap.add_argument("--sasrec-k", type=int, default=SASREC_K)
    ap.add_argument("--tactile-k", type=int, default=TACTILE_K)
    args = ap.parse_args()

    data = C.load_prepared()
    n_items = int(data["n_items"][0])
    device = torch.device(args.device)
    tactile = data["tactile"]
    available = data["tactile_available"]
    category_id = data["category_id"].astype(np.int64)
    n_categories = int(category_id.max()) + 1
    train_count = data["train_count"]

    users, matrix, targets, histories = evaluation_point(data, args.split)
    print(f"[{args.split}] {len(users)} users", flush=True)

    profile, support, consistency, category_hist = tactile_profiles(
        histories, tactile, available, category_id, n_categories)

    # 고정된 Exp16 SASRec으로 전체 카탈로그 기준 점수와 후보를 만든다.
    model = evaluator.load_exp16_sasrec(args.checkpoint, n_items, device)
    catalog = model.catalog_matrix().to(device)

    tactile_t = torch.as_tensor(tactile, device=device)
    tactile_norm = torch.nn.functional.normalize(tactile_t, dim=1)
    # 촉감 차원별 표준화 버전도 계산해 validation에서 선택한 R7 후보 공간을 제공한다.
    # A24-1: Last2 emits 14 independent sigmoids, so every item vector sits in
    # [0,1]^14 with a large shared positive mean and raw cosine is near 1 for
    # almost any pair.  Standardising each dimension by its catalog mean/std makes
    # the similarity measure differences instead of magnitude.  Statistics come
    # from image-derived vectors only -- no user, interaction or review data.
    usable = torch.as_tensor(available, device=device)
    centre = tactile_t[usable].mean(dim=0, keepdim=True)
    spread = tactile_t[usable].std(dim=0, keepdim=True).clamp_min(1e-6)
    tactile_std = torch.nn.functional.normalize((tactile_t - centre) / spread, dim=1)
    availability = torch.as_tensor(available, device=device)

    k_s, k_t = args.sasrec_k, args.tactile_k
    out_sasrec = np.zeros((len(users), k_s), dtype=np.int64)
    out_sasrec_score = np.zeros((len(users), k_s), dtype=np.float32)
    out_tactile = np.zeros((len(users), k_t), dtype=np.int64)
    out_tactile_sim = np.zeros((len(users), k_t), dtype=np.float32)
    out_tactile_std = np.zeros((len(users), k_t), dtype=np.int64)
    out_tactile_std_sim = np.zeros((len(users), k_t), dtype=np.float32)
    full_rank = np.zeros(len(users), dtype=np.int64)

    start = time.time()
    # 각 배치에서 이미 본 상품을 제외한 SASRec·촉감 후보를 각각 추출한다.
    for begin in range(0, len(users), args.batch):
        rows = slice(begin, min(begin + args.batch, len(users)))
        seq = torch.as_tensor(matrix[rows], dtype=torch.long, device=device)
        with torch.no_grad():
            query = model.query(seq)
            scores = (query @ catalog.T).clone()
        for k, row in enumerate(range(rows.start, rows.stop)):
            history = histories[row]
            if history.size:
                scores[k, history - 1] = -torch.inf
        # full-catalog rank of the target, for the R0 reference on this split
        target_index = torch.as_tensor(targets[rows] - 1, dtype=torch.long, device=device)
        target_score = scores.gather(1, target_index[:, None]).squeeze(1)
        index = torch.arange(n_items, device=device)
        full_rank[rows] = (1 + (scores > target_score[:, None]).sum(1)
                           + ((scores == target_score[:, None])
                              & (index[None, :] < target_index[:, None])).sum(1)).cpu().numpy()
        top = scores.topk(k_s, dim=1)
        out_sasrec[rows] = top.indices.cpu().numpy() + 1
        out_sasrec_score[rows] = top.values.cpu().numpy()

        # tactile ANN: cosine between the user's past-only profile and item vectors
        with torch.no_grad():
            query_profile = torch.nn.functional.normalize(
                torch.as_tensor(profile[rows], device=device), dim=1)
            similarity = query_profile @ tactile_norm.T
            similarity = similarity.masked_fill(~availability[None, :], -torch.inf).clone()
            no_profile = torch.as_tensor(support[rows] == 0, device=device)
            for k, row in enumerate(range(rows.start, rows.stop)):
                history = histories[row]
                if history.size:
                    similarity[k, history - 1] = -torch.inf
            top_t = similarity.topk(k_t, dim=1)
        out_tactile[rows] = top_t.indices.cpu().numpy() + 1
        values = top_t.values.cpu().numpy()
        values[no_profile.cpu().numpy()] = 0.0
        out_tactile_sim[rows] = values

        # A24-1 variant: standardised tactile space
        with torch.no_grad():
            raw_profile = torch.as_tensor(profile[rows], device=device)
            standard_profile = torch.nn.functional.normalize(
                (raw_profile - centre) / spread, dim=1)
            similarity_std = standard_profile @ tactile_std.T
            similarity_std = similarity_std.masked_fill(~availability[None, :], -torch.inf).clone()
            for k, row in enumerate(range(rows.start, rows.stop)):
                history = histories[row]
                if history.size:
                    similarity_std[k, history - 1] = -torch.inf
            top_std = similarity_std.topk(k_t, dim=1)
        out_tactile_std[rows] = top_std.indices.cpu().numpy() + 1
        values_std = top_std.values.cpu().numpy()
        values_std[no_profile.cpu().numpy()] = 0.0
        out_tactile_std_sim[rows] = values_std
        if begin % (args.batch * 20) == 0:
            print(f"  {rows.stop}/{len(users)} "
                  f"{rows.stop / max(time.time() - start, 1e-9):.1f} users/s", flush=True)

    # 세 split이 동일 포맷으로 후속 reranker에 입력되도록 후보와 특성을 함께 저장한다.
    np.savez_compressed(
        C.CACHE / f"candidates__{args.split}.npz",
        users=users, targets=targets, history_matrix=matrix,
        sasrec_candidates=out_sasrec, sasrec_scores=out_sasrec_score,
        tactile_candidates=out_tactile, tactile_similarity=out_tactile_sim,
        tactile_std_candidates=out_tactile_std, tactile_std_similarity=out_tactile_std_sim,
        full_rank=full_rank,
        profile=profile, support=support, consistency=consistency,
        category_hist=category_hist,
        history_length=np.array([h.size for h in histories], dtype=np.int32),
    )

    # 후보 단계의 recall을 기록해 reranker 이전의 회수 병목을 측정한다.
    # candidate recall, the quantity the audit identified as the real bottleneck
    recall = {}
    for name, block in (("sasrec", out_sasrec), ("tactile", out_tactile),
                        ("tactile_std", out_tactile_std)):
        for k in (100, 300, 500):
            if k <= block.shape[1]:
                recall[f"{name}@{k}"] = float(
                    (block[:, :k] == targets[:, None]).any(axis=1).mean())
    for name, block in (("tactile", out_tactile), ("tactile_std", out_tactile_std)):
        union_hit = ((out_sasrec == targets[:, None]).any(axis=1)
                     | (block == targets[:, None]).any(axis=1))
        recall[f"union_{name}@{k_s}+{k_t}"] = float(union_hit.mean())
    recall["full_catalog_recall@10"] = float((full_rank <= 10).mean())
    C.write_json(C.RESULTS / f"candidate_recall__{args.split}.json", recall)
    print(f"[{args.split}] candidate recall: {recall}", flush=True)
    print(f"[{args.split}] done in {time.time() - start:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
