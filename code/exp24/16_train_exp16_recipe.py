#!/usr/bin/env python3
# ============================================================
# [열람용 사본] 2026-10-16 시점 스냅샷. 이 파일은 실행용이 아님.
# 실제로 돌아가는 원본 경로:
#   /home/user/onsesang/material_span_grounding/experiments/24_adaptive_tactile_recommendation/scripts/16_train_exp16_recipe.py
# 원본은 이후 실험으로 바뀔 수 있음.
# 역할: Exp16 학습 레시피로 R0·R5·R5-shuffled를 재학습하고 비교한다.
# ============================================================
"""Retrain R0 / R5 / R5-shuffled under the exp16b training recipe.

Why: the Exp24 R0 (test NDCG@10 0.00179) is half the exp16b checkpoint (0.00349),
and RESULTS_15 ruled out inference-time cold zeroing.  This script keeps Exp24's
data, cohorts and final evaluation (02_evaluate.rank_targets) and changes only the
training recipe to match 16_strong_recommender_tactile/recommender.py:

  Adam, no weight decay (recommender.py:149) | lr 3e-4 | grad clip 5 (:185)
  1 uniform negative per position over the full catalog, rejecting every item in
    the user's train history (:167-177)
  early stopping on NDCG@10 over all 281,395 official validation targets, via
    exp16b's own evaluate() (:187-198), every 5 epochs, patience 5, max 50
  cold rows (train_count == 0) reset to 0 in place at every validation (:78)
  seed 20260904; torch threads 8; TF32 off (:8-9)

R5 keeps 03_train.NegativeSampler's hard-negative mining (same grouping, same
hard pick) but its uniform part is replaced by the exclusion sampler above.

Modes
  train    --variant R0|R5|R5_shuffled   -> artifacts/exp16recipe/<tag>.pt + log
  evaluate                                -> history>=3 metrics, step-0 check,
                                             paired bootstrap; JSON to stdout and
                                             artifacts/exp16recipe/evaluation.json

exp16b's event() is deliberately not called: it appends to exp16b's artifacts and
posts to Discord when DISCORD_WEBHOOK_URL is set.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import torch

import common24 as C
import models24 as M
from importlib import import_module

sys.path.insert(1, str(C.E16))
import recommender as R          # exp16b: RecData, evaluate      # noqa: E402

evaluator = import_module("02_evaluate")
trainer = import_module("03_train")

OUT = C.EXP / "artifacts" / "exp16recipe"
# Exp16과 맞춘 seed·학습률·gradient clipping·early stopping 설정이다.
SEED16 = 20260904
LR = 3e-4
CLIP = 5.0
MAX_EPOCHS = 50
VALIDATE_EVERY = 5
PATIENCE = 5
HARD_RATIO = 0.5
EXP16_CHECKPOINTS = C.E16 / "artifacts" / "checkpoints"
EXP16_R0_TEST_NDCG = 0.00349417
EXP24_R0_TEST_NDCG = 0.0017933950160958480
TAGS = {"R0": "R0_exp16recipe", "R5": "R5_hard0.5_exp16recipe",
        "R5_shuffled": "R5_hard0.5_SHUFFLED_exp16recipe"}
BOOTSTRAP_REPLICATES = 10_000


# --------------------------------------------------------------- recipe checks
# Exp24 준비 데이터와 Exp16 데이터·하이퍼파라미터가 같은지 학습 전에 검사한다.
def assert_recipe(data, d16):
    config = json.loads((C.E16 / "configs" / "backbones.json").read_text())
    expected = {"dim": C.DIM, "sasrec_layers": C.LAYERS, "sasrec_heads": C.HEADS,
                "sasrec_maxlen": C.MAXLEN, "sasrec_dropout": C.DROPOUT,
                "sasrec_batch_size": C.BATCH, "patience": PATIENCE,
                "validation_every_epochs": VALIDATE_EVERY, "maximum_epochs": MAX_EPOCHS,
                "seed": SEED16}
    for key, value in expected.items():
        assert config[key] == value, (key, config[key], value)
    assert LR in config["learning_rates"]
    # same training users, same chronological sequences
    offsets, sequences = data["offsets"], data["sequences"]
    users24 = trainer.build_training_pairs(data)
    users16 = np.array([u for u, h in d16.hist.items() if len(h) >= 2])
    assert np.array_equal(users24, users16), "training users differ"
    for u, h in d16.hist.items():
        assert np.array_equal(sequences[offsets[u]:offsets[u + 1]] - 1, h), u
    assert int(data["n_items"][0]) == d16.nitems
    assert np.array_equal(data["train_count"] == 0, d16.cold)
    return users16


def assert_validation_reproduces(d16, device):
    """exp16b's evaluate() on its own checkpoints must give the stored numbers."""
    n_items = d16.nitems
    for lr in ("0.0003", "0.001"):
        stored = json.loads((EXP16_CHECKPOINTS / f"sasrec_lr{lr}.complete.json").read_text())
        model = evaluator.load_exp16_sasrec(EXP16_CHECKPOINTS / f"sasrec_lr{lr}.pt", n_items, device)
        result, _ = R.evaluate("sasrec", model, d16, "validation", device, 256)
        got = result["all_official_targets"]["ndcg_at_10"]
        want = stored["validation"]["all_official_targets"]["ndcg_at_10"]
        assert abs(got - want) < 1e-9, (lr, got, want)
        print(f"[check] exp16b sasrec lr{lr} validation NDCG@10 {got:.10f} == stored", flush=True)
        del model


# ------------------------------------------------------------------- sampling
# 사용자 train 이력의 모든 양성 상품을 균등 음성에서 제외한다.
def exclusion_negatives(rng, full, n, n_items):
    """recommender.py:173-176, verbatim logic: uniform over [0, n_items), redraw
    while the item is anywhere in the user's train history."""
    v = rng.integers(n_items, size=(n, 1))
    for k in range(n):
        while v[k, 0] in full:
            v[k, 0] = rng.integers(n_items)
    return v[:, 0]


class ExclusionHardSampler(trainer.NegativeSampler):
    """03_train.NegativeSampler with its uniform draw replaced.

    __init__ (category x popularity-band grouping, tactile table) is inherited.
    replace_hard() is NegativeSampler.__call__ from the line after its uniform
    draw (03_train.py:100-122), unchanged: the same fraction of positions get a
    same-group, tactile-farthest negative from a 256-item pool.  Hard picks are
    not filtered against the user's history -- only the uniform part is.
    """

    def replace_hard(self, positives: np.ndarray, negatives: np.ndarray) -> np.ndarray:
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
        flat_neg = negatives.reshape(-1).copy()
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


# 사용자별 오른쪽 정렬 시퀀스와 다음 상품 양성·음성을 만든다.
def batch_tensors(d16, uids, rng, n_items):
    """recommender.py:167-177 for SASRec (nneg = 1), ids shifted by one."""
    seq = np.zeros((len(uids), C.MAXLEN), dtype=np.int64)
    pos = seq.copy()
    neg = seq.copy()
    for j, u in enumerate(uids):
        full = d16.hist[int(u)]
        h = full[-C.MAXLEN - 1:]
        n = len(h) - 1
        seq[j, -n:] = h[:-1] + 1
        pos[j, -n:] = h[1:] + 1
        neg[j, -n:] = exclusion_negatives(rng, full, n, n_items) + 1
    return seq, pos, neg


# ---------------------------------------------------------------------- train
# train에 등장하지 않은 cold item의 ID embedding을 평가 전에 0으로 만든다.
def zero_cold_rows(model, cold_rows):
    with torch.no_grad():
        model.item.weight[1:][cold_rows] = 0


def train(variant, device):
    # Exp16 seed와 수치 설정을 고정해 R0와 hard-negative 변형을 같은 조건에서 학습한다.
    torch.set_num_threads(8)
    torch.manual_seed(SEED16)
    np.random.seed(SEED16)
    torch.backends.cuda.matmul.allow_tf32 = False
    OUT.mkdir(parents=True, exist_ok=True)
    tag = TAGS[variant]
    checkpoint = OUT / f"{tag}.pt"
    assert not checkpoint.exists(), f"{checkpoint} exists; refusing to overwrite"

    data = C.load_prepared()
    d16 = R.RecData()
    n_items = d16.nitems
    users = assert_recipe(data, d16)
    assert_validation_reproduces(d16, device)
    cold_rows = torch.as_tensor(d16.cold, device=device)

    torch.manual_seed(SEED16)
    model = M.SASRec(n_items, dim=C.DIM, layers=C.LAYERS, heads=C.HEADS,
                     maxlen=C.MAXLEN, dropout=C.DROPOUT).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)
    rng = np.random.default_rng(SEED16)

    sampler = None
    if variant in ("R5", "R5_shuffled"):
        sampler_data = data
        if variant == "R5_shuffled":
            order = np.random.default_rng(SEED16).permutation(data["tactile"].shape[0])
            sampler_data = dict(data)
            sampler_data["tactile"] = data["tactile"][order]
        sampler = ExclusionHardSampler(sampler_data, hard_ratio=HARD_RATIO, seed=SEED16,
                                       device=str(device))

    log_path = OUT / f"train__{tag}.jsonl"
    log = open(log_path, "w")

    def record(**row):
        row = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "tag": tag, **row}
        log.write(json.dumps(row) + "\n")
        log.flush()
        print(json.dumps(row), flush=True)

    record(status="started", variant=variant, lr=LR, seed=SEED16, clip=CLIP,
           optimizer="Adam", weight_decay=0.0, hard_ratio=HARD_RATIO if sampler else 0.0,
           training_users=int(len(users)))
    best, bad, best_epoch = -1.0, 0, -1
    begin = time.monotonic()
    # 사용자 단위 배치를 순회하며 next-item 양성/음성 손실을 최적화한다.
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        losses = []
        indices = rng.permutation(len(users))
        for start in range(0, len(indices), C.BATCH):
            uids = users[indices[start:start + C.BATCH]]
            optimiser.zero_grad(set_to_none=True)
            seq, pos, neg = batch_tensors(d16, uids, rng, n_items)
            if sampler is not None:
                neg = sampler.replace_hard(pos, neg)
            tensors = [torch.as_tensor(a, device=device) for a in (seq, pos, neg)]
            loss = model.loss(*tensors)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{tag} nonfinite loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            optimiser.step()
            losses.append(float(loss.detach()))
        record(status="epoch", epoch=epoch, loss=float(np.mean(losses)),
               seconds=round(time.monotonic() - begin, 1))
        # 전체 공식 validation target의 NDCG@10으로 저장과 조기 종료를 결정한다.
        if epoch % VALIDATE_EVERY == 0 or epoch == MAX_EPOCHS:
            zero_cold_rows(model, cold_rows)          # explicit; R.evaluate repeats it
            result, _ = R.evaluate("sasrec", model, d16, "validation", device, 256)
            assert float(model.item.weight[1:][cold_rows].abs().max()) == 0.0
            value = result["all_official_targets"]["ndcg_at_10"]
            record(status="validation", epoch=epoch, **result["all_official_targets"],
                   train_history_ge_3_ndcg_at_10=result["train_history_ge_3"]["ndcg_at_10"])
            if value > best:
                best, bad, best_epoch = value, 0, epoch
                torch.save({"state_dict": {k: v.detach().cpu().clone()
                                           for k, v in model.state_dict().items()},
                            "variant": variant, "tag": tag, "recipe": "exp16b",
                            "lr": LR, "seed": SEED16, "clip": CLIP, "hard_ratio":
                            HARD_RATIO if sampler else 0.0, "epoch": epoch,
                            "validation_all_official": result["all_official_targets"]},
                           checkpoint)
            else:
                bad += 1
            if bad >= PATIENCE:
                break
    record(status="complete", best_epoch=best_epoch, best_validation_ndcg_at_10=best,
           seconds=round(time.monotonic() - begin, 1))
    log.close()


# ------------------------------------------------------------------- evaluate
# 전체·cold·warm 집단별 ranking 지표를 동일한 정답 순위에서 계산한다.
def metrics(ranks, groups):
    out = {}
    for name, mask in groups.items():
        r = ranks[mask].astype(np.float64)
        hit = r <= 10
        out[name] = {"users": int(mask.sum()),
                     "NDCG@10": float(np.where(hit, 1.0 / np.log2(r + 1.0), 0.0).mean()),
                     "HR@10": float(hit.mean()),
                     "MRR@10": float(np.where(hit, 1.0 / r, 0.0).mean()),
                     **{f"Recall@{k}": float((r <= k).mean()) for k in (100, 500, 3000)}}
    return out


def evaluate(device):
    data = C.load_prepared()
    n_items = int(data["n_items"][0])
    cold = data["train_count"] == 0
    blocks = {}
    # target이 train에 등장했는지에 따라 cold/warm 평가 집단을 나눈다.
    for split in ("validation", "test"):
        users, matrix, targets = evaluator.history_and_targets(data, split)
        seen = evaluator.seen_items(data, split, users)
        target_cold = cold[targets - 1]
        blocks[split] = (matrix, targets, seen, {"all": np.ones(len(users), bool),
                                                 "cold": target_cold, "warm": ~target_cold})

    def run(model):
        out, ranks = {}, {}
        for split, (matrix, targets, seen, groups) in blocks.items():
            r, _ = evaluator.rank_targets(model, matrix, targets, seen, device, batch=512)
            out[split] = metrics(r, groups)
            ranks[split] = r
        return out, ranks

    report = {"metrics": {}, "training": {}}
    per_user = {}
    # step 0: exp16b checkpoints, both learning rates
    for lr in ("0.001", "0.0003"):
        name = f"exp16b_sasrec_lr{lr}"
        model = evaluator.load_exp16_sasrec(EXP16_CHECKPOINTS / f"sasrec_lr{lr}.pt", n_items, device)
        report["metrics"][name], ranks = run(model)
        stored = json.loads((EXP16_CHECKPOINTS / f"sasrec_lr{lr}.complete.json").read_text())
        report["training"][name] = {
            "best_epoch": stored["best_epoch"],
            "validation_all_official_ndcg_at_10": stored["validation"]["all_official_targets"]["ndcg_at_10"]}
        for split, r in ranks.items():
            per_user[(name, split)] = C.per_user_ndcg(r)
        del model
    # Exp24 R0 from the cache (RESULTS_15 asserted it equals a fresh evaluation)
    cache = np.load(C.CACHE / "trained_ranks.npz")
    report["metrics"]["exp24_R0_seed20260917"] = {
        split: metrics(cache[f"R0_seed20260917__{split}"], blocks[split][3])
        for split in ("validation", "test")}
    # the three retrained models
    for variant, tag in TAGS.items():
        saved = torch.load(OUT / f"{tag}.pt", map_location="cpu", weights_only=False)
        model = M.SASRec(n_items, dim=C.DIM, layers=C.LAYERS, heads=C.HEADS,
                         maxlen=C.MAXLEN, dropout=C.DROPOUT)
        model.load_state_dict(saved["state_dict"])
        model = model.to(device).eval()
        report["metrics"][tag], ranks = run(model)
        log = [json.loads(line) for line in open(OUT / f"train__{tag}.jsonl")]
        done = [row for row in log if row["status"] == "complete"][-1]
        report["training"][tag] = {
            "best_epoch": done["best_epoch"],
            "validation_all_official_ndcg_at_10": done["best_validation_ndcg_at_10"],
            "last_epoch": max(row["epoch"] for row in log if row["status"] == "epoch"),
            "seconds": done["seconds"],
            "validation_curve": [(row["epoch"], row["ndcg_at_10"]) for row in log
                                 if row["status"] == "validation"]}
        for split, r in ranks.items():
            per_user[(tag, split)] = C.per_user_ndcg(r)
        np.savez_compressed(OUT / f"ranks__{tag}.npz", **{s: r for s, r in ranks.items()})
        del model

    r0 = report["metrics"][TAGS["R0"]]["test"]["all"]["NDCG@10"]
    report["R0_restoration"] = {
        "exp24_R0": EXP24_R0_TEST_NDCG, "exp16recipe_R0": r0, "exp16b_checkpoint": EXP16_R0_TEST_NDCG,
        "fraction": (r0 - EXP24_R0_TEST_NDCG) / (EXP16_R0_TEST_NDCG - EXP24_R0_TEST_NDCG)}

    pairs = [(TAGS["R5"], TAGS["R0"]), (TAGS["R5_shuffled"], TAGS["R0"]),
             (TAGS["R5"], TAGS["R5_shuffled"]), (TAGS["R0"], "exp16b_sasrec_lr0.0003")]
    report["bootstrap"] = {}
    for split in ("validation", "test"):
        for a, b in pairs:
            stat = C.paired_bootstrap_by_user(per_user[(a, split)], per_user[(b, split)],
                                              replicates=BOOTSTRAP_REPLICATES, seed=C.SEED)
            stat["significant"] = bool(stat["ci95_low"] > 0 or stat["ci95_high"] < 0)
            report["bootstrap"][f"{split}__{a}__minus__{b}"] = stat

    (OUT / "evaluation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "evaluate"])
    ap.add_argument("--variant", choices=list(TAGS))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    if args.mode == "train":
        assert args.variant, "--variant is required for train"
        train(args.variant, device)
    else:
        evaluate(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
