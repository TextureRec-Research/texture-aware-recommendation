#!/usr/bin/env python3
# ============================================================
# [열람용 사본] 2026-10-16 시점 스냅샷. 이 파일은 실행용이 아님.
# 실제로 돌아가는 원본 경로:
#   /home/user/onsesang/material_span_grounding/experiments/24_adaptive_tactile_recommendation/scripts/03_train.py
# 원본은 이후 실험으로 바뀔 수 있음.
# 역할: Exp24 모델 variant를 만들고 negative sampling·학습·검증을 수행한다.
# ============================================================
"""Train one Exp24 recommendation variant.

Every variant shares the exp16b/exp17b training recipe so that differences are
attributable to the architecture, not the optimiser:
  dim 64, 2 layers, 2 heads, maxlen 50, dropout 0.2, batch 256,
  64 sampled negatives, AdamW, max 50 epochs, validate every 5, patience 5,
  selection on validation NDCG@10 over the history>=3 cohort.

R5 (tactile-aware hard negatives) changes only the negative sampler.  Hard
negatives are drawn using the *image* tactile predictor, never the target item's
review labels.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

import common24 as C
import models24 as M
from importlib import import_module

evaluator = import_module("02_evaluate")


# train split에서 다음 상품 예측 쌍을 만들 수 있는 사용자만 선택한다.
def build_training_pairs(data):
    """Next-item pairs from the train split only, right-aligned per user."""
    offsets, sequences = data["offsets"], data["sequences"]
    users = np.nonzero(np.diff(offsets) >= 2)[0]          # need at least one (input, target)
    return users


def make_batches(data, users, rng, batch=C.BATCH):
    order = rng.permutation(len(users))
    for begin in range(0, len(order), batch):
        yield users[order[begin:begin + batch]]


# 각 사용자 이력을 한 칸 이동한 입력(seq)과 다음 상품 정답(pos)으로 나눈다.
def sequences_for_training(data, users, maxlen=C.MAXLEN):
    """Input = history[:-1], target = history[1:], right aligned."""
    offsets, sequences = data["offsets"], data["sequences"]
    seq = np.zeros((len(users), maxlen), dtype=np.int64)
    pos = np.zeros((len(users), maxlen), dtype=np.int64)
    for row, user in enumerate(users):
        history = sequences[offsets[user]:offsets[user + 1]][-(maxlen + 1):]
        inputs, targets = history[:-1], history[1:]
        seq[row, maxlen - inputs.size:] = inputs
        pos[row, maxlen - targets.size:] = targets
    return seq, pos


class NegativeSampler:
    """Random negatives, optionally mixed with tactile-hard negatives.

    A tactile-hard negative is an item that is plausible for the same taste
    neighbourhood -- similar popularity band and same category -- but whose
    predicted tactile vector is far from the positive's.  It forces the model to
    use tactile evidence to separate items that collaborative signal alone would
    confuse.
    """

    def __init__(self, data, hard_ratio=0.0, seed=C.SEED, device="cpu", pool=256):
        self.n_items = int(data["n_items"][0])
        self.hard_ratio = hard_ratio
        self.rng = np.random.default_rng(seed)
        self.pool = pool
        self.device = device
        if hard_ratio > 0:
            self.tactile = torch.as_tensor(data["tactile"], device=device)
            self.category = torch.as_tensor(data["category_id"].astype(np.int64), device=device)
            counts = data["train_count"]
            # popularity band by log-count decile, computed on train counts only
            band = np.zeros(self.n_items, dtype=np.int64)
            positive = counts > 0
            if positive.any():
                band[positive] = np.digitize(
                    np.log1p(counts[positive]),
                    np.quantile(np.log1p(counts[positive]), np.linspace(0.1, 0.9, 9)))
            self.band = torch.as_tensor(band, device=device)
            # hard negative 후보군은 카테고리와 train 인기도 구간의 교차로 정의한다.
            self.by_group = {}
            groups = (self.category * 16 + self.band).cpu().numpy()
            order = np.argsort(groups, kind="mergesort")
            sorted_groups = groups[order]
            edges = np.flatnonzero(np.diff(sorted_groups)) + 1
            for chunk in np.split(order, edges):
                if chunk.size > 1:
                    self.by_group[int(groups[chunk[0]])] = chunk

    def __call__(self, positives: np.ndarray) -> np.ndarray:
        """positives: (B, L) shifted ids, 0 = padding.  Returns same shape."""
        # 기본은 전체 카탈로그 균등 음성이며, 일부를 촉감 거리가 먼 후보로 교체한다.
        negatives = self.rng.integers(1, self.n_items + 1, size=positives.shape)
        if self.hard_ratio <= 0:
            return negatives
        valid = np.flatnonzero(positives.reshape(-1) > 0)
        if valid.size == 0:
            return negatives
        take = self.rng.random(valid.size) < self.hard_ratio
        chosen = valid[take]
        if chosen.size == 0:
            return negatives
        flat_pos = positives.reshape(-1)[chosen] - 1
        group = (self.category[flat_pos] * 16 + self.band[flat_pos]).cpu().numpy()
        anchor = self.tactile[flat_pos]
        flat_neg = negatives.reshape(-1)
        for offset, (index, key) in enumerate(zip(chosen, group)):
            pool_items = self.by_group.get(int(key))
            if pool_items is None or pool_items.size < 2:
                continue
            sample = pool_items[self.rng.integers(0, pool_items.size,
                                                  size=min(self.pool, pool_items.size))]
            candidate = self.tactile[torch.as_tensor(sample, device=self.device)]
            distance = (candidate - anchor[offset]).abs().mean(dim=1)
            flat_neg[index] = int(sample[int(distance.argmax())]) + 1
        return flat_neg.reshape(positives.shape)


# R0/R3/R4/R5/R6 이름을 실제 SASRec 계열 클래스와 설정에 연결한다.
def build_model(variant, data, n_items, device, lambda_tactile, fusion, seed):
    torch.manual_seed(seed)
    tactile = torch.as_tensor(data["tactile"])
    if variant in ("R0", "R1", "R2", "R5", "R7"):
        model = M.SASRec(n_items, dim=C.DIM, layers=C.LAYERS, heads=C.HEADS,
                         maxlen=C.MAXLEN, dropout=C.DROPOUT)
    elif variant == "R3":
        if fusion == "shuffle":
            generator = torch.Generator().manual_seed(seed)
            tactile = tactile[torch.randperm(tactile.shape[0], generator=generator)]
        model = M.TactileSASRec(n_items, tactile, fusion=fusion, dim=C.DIM,
                                layers=C.LAYERS, heads=C.HEADS, maxlen=C.MAXLEN,
                                dropout=C.DROPOUT)
    elif variant == "R4":
        model = M.MultiTaskSASRec(n_items, tactile, dim=C.DIM, layers=C.LAYERS,
                                  heads=C.HEADS, maxlen=C.MAXLEN, dropout=C.DROPOUT,
                                  lambda_tactile=lambda_tactile)
    elif variant == "R6":
        model = M.MixtureSASRec(n_items, tactile, dim=C.DIM, layers=C.LAYERS,
                                heads=C.HEADS, maxlen=C.MAXLEN, dropout=C.DROPOUT)
    else:
        raise ValueError(variant)
    return model.to(device)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["R0", "R3", "R4", "R5", "R6"])
    ap.add_argument("--fusion", default="residual_gate",
                    choices=["residual_gate", "concat_proj", "film", "shuffle"])
    ap.add_argument("--lambda-tactile", type=float, default=0.1)
    ap.add_argument("--hard-ratio", type=float, default=0.0)
    ap.add_argument("--shuffle-tactile", action="store_true",
                    help=("R5 control: keep the hard-negative SAMPLING PROCEDURE "
                          "identical but permute the tactile table deterministically. "
                          "If the gain survives this, it came from mining harder "
                          "negatives, not from tactile information."))
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--validate-every", type=int, default=5)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    tag = args.tag or f"{args.variant}_lr{args.lr}_seed{args.seed}"
    device = torch.device(args.device)
    data = C.load_prepared()
    n_items = int(data["n_items"][0])
    # 모델 초기화와 데이터 순서가 같은 seed에서 재현되도록 RNG를 고정한다.
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    users = build_training_pairs(data)
    print(f"[{tag}] training users {len(users)}  items {n_items}  device {device}", flush=True)

    model = build_model(args.variant, data, n_items, device, args.lambda_tactile,
                        args.fusion, args.seed)
    sampler_data = data
    if args.shuffle_tactile:
        generator = np.random.default_rng(args.seed)
        permuted = dict(data)
        order = generator.permutation(data["tactile"].shape[0])
        permuted["tactile"] = data["tactile"][order]
        sampler_data = permuted
        print(f"[{tag}] R5 CONTROL: tactile table permuted for negative mining", flush=True)
    sampler = NegativeSampler(sampler_data, hard_ratio=args.hard_ratio, seed=args.seed,
                              device=str(device))
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    validation_users, validation_matrix, validation_targets = \
        evaluator.history_and_targets(data, "validation")
    validation_seen = evaluator.seen_items(data, "validation", validation_users)

    best = {"ndcg": -1.0, "epoch": -1, "state": None}
    history = []
    patience = 0
    start = time.time()
    # 다음 상품 예측 손실로 학습하고 주기적으로 validation NDCG@10을 확인한다.
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, batches = 0.0, 0
        for batch_users in make_batches(data, users, rng):
            seq, pos = sequences_for_training(data, batch_users)
            neg = sampler(pos)
            seq_t = torch.as_tensor(seq, device=device)
            pos_t = torch.as_tensor(pos, device=device)
            neg_t = torch.as_tensor(neg, device=device)
            optimiser.zero_grad()
            loss = model.loss(seq_t, pos_t, neg_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            total += float(loss.item())
            batches += 1
        if epoch % args.validate_every:
            print(f"[{tag}] epoch {epoch} loss {total / max(batches,1):.5f}", flush=True)
            continue
        model.eval()
        ranks, _ = evaluator.rank_targets(model, validation_matrix, validation_targets,
                                          validation_seen, device, batch=512)
        metrics = C.metrics_from_ranks(ranks)
        history.append({"epoch": epoch, "loss": total / max(batches, 1), **metrics})
        print(f"[{tag}] epoch {epoch} loss {total / max(batches,1):.5f} "
              f"val NDCG@10 {metrics['NDCG@10']:.8f}", flush=True)
        # validation NDCG@10 개선 여부로 체크포인트와 early stopping을 결정한다.
        if metrics["NDCG@10"] > best["ndcg"]:
            best = {"ndcg": metrics["NDCG@10"], "epoch": epoch,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                print(f"[{tag}] early stop at epoch {epoch}", flush=True)
                break

    torch.save({"state_dict": best["state"], "variant": args.variant, "tag": tag,
                "shuffle_tactile": bool(args.shuffle_tactile),
                "fusion": args.fusion, "lambda_tactile": args.lambda_tactile,
                "hard_ratio": args.hard_ratio, "lr": args.lr, "seed": args.seed,
                "best_epoch": best["epoch"], "validation_ndcg": best["ndcg"]},
               C.CHECKPOINTS / f"{tag}.pt")
    C.write_json(C.LOGS / f"train__{tag}.json", {
        "tag": tag, "variant": args.variant, "fusion": args.fusion,
        "lambda_tactile": args.lambda_tactile, "hard_ratio": args.hard_ratio,
        "lr": args.lr, "seed": args.seed, "best_epoch": best["epoch"],
        "best_validation_ndcg": best["ndcg"], "elapsed_seconds": time.time() - start,
        "history": history,
    })
    print(f"[{tag}] done: best epoch {best['epoch']} val NDCG@10 {best['ndcg']:.8f} "
          f"in {time.time() - start:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
