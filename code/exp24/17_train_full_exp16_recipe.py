#!/usr/bin/env python3
# ============================================================
# [열람용 사본] 2026-10-16 시점 스냅샷. 이 파일은 실행용이 아님.
# 실제로 돌아가는 원본 경로:
#   /home/user/onsesang/material_span_grounding/experiments/24_adaptive_tactile_recommendation/scripts/17_train_full_exp16_recipe.py
# 원본은 이후 실험으로 바뀔 수 있음.
# 역할: Exp16 레시피로 SASRec·feature 변형 전체와 3-seed 조건을 재학습·평가한다.
# ============================================================
"""Rebuild the Exp24 section-7.2 table under the exp16b training recipe.

Reuses scripts/16_train_exp16_recipe.py (recipe asserts, exclusion negatives,
exp16b batch construction) and models24 / 03_train.build_model unchanged.

Differences from 16_train_exp16_recipe.py
  * early stopping uses validate_all_official(): exp16b's evaluate() logic
    (all 281,395 official validation targets, popularity rank for users without
    train history, seen filter, ascending-iid ties) but scoring through
    model.query() x model.catalog_matrix(), so R3 (fused catalog) and R6 (router
    query) are scored the way they are evaluated.  exp16b's own evaluate() is
    SASRec-only (model(seq)[:, -1] x item.weight).  For plain SASRec the two are
    the same computation; both are asserted to reproduce exp16b's stored
    validation NDCG@10 within 1e-9 before any training.
  * R5 hard negatives also skip items in the user's train history: candidates in
    the 256-item pool are tried from tactile-farthest down and the first item not
    in the history is taken (RESULTS_16 filtered only the uniform negatives).

Modes
  train     trains every missing (variant, seed) in PLAN, one process
  evaluate  history>=3 metrics + paired bootstrap; JSON to stdout and
            artifacts/exp16recipe_full/evaluation.json

Run: <texture python> scripts/17_train_full_exp16_recipe.py train|evaluate
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
import pandas as pd
import torch

import common24 as C
from importlib import import_module

base = import_module("16_train_exp16_recipe")       # also puts exp16b on sys.path
R = base.R
evaluator = base.evaluator
trainer = base.trainer

OUT = C.EXP / "artifacts" / "exp16recipe_full"
SEEDS_EXTRA = (20260905, 20260906)
BOOTSTRAP_REPLICATES = 10_000

# 비교표의 모델명, feature fusion, 보조 손실 가중치, hard-negative 모드를 고정한다.
# (tag stem, build_model variant, fusion, lambda, hard-negative mode)
VARIANTS = {
    "R0":                    ("R0", "residual_gate", 0.1, None),
    "R3_residual_gate":      ("R3", "residual_gate", 0.1, None),
    "R3_concat_proj":        ("R3", "concat_proj", 0.1, None),
    "R3_film":               ("R3", "film", 0.1, None),
    "R3_shuffle":            ("R3", "shuffle", 0.1, None),
    "R4_lambda0.05":         ("R4", "residual_gate", 0.05, None),
    "R4_lambda0.2":          ("R4", "residual_gate", 0.2, None),
    "R5_hard0.5":            ("R5", "residual_gate", 0.1, "real"),
    "R5_hard0.5_SHUFFLED":   ("R5", "residual_gate", 0.1, "shuffled"),
    "R6":                    ("R6", "residual_gate", 0.1, None),
}
PLAN = [(name, base.SEED16) for name in VARIANTS]
PLAN += [(name, seed) for seed in SEEDS_EXTRA
         for name in ("R0", "R5_hard0.5", "R5_hard0.5_SHUFFLED")]


def tag_of(name, seed):
    return f"{name}_seed{seed}"


# ------------------------------------------------------------ early stopping
@torch.no_grad()
def validate_all_official(model, d16, device, batch=256):
    """exp16b recommender.evaluate(split='validation') with generic scoring."""
    model.eval()
    # 전체 공식 validation target을 사용하고 이력이 없는 사용자는 인기도 순위로 처리한다.
    targets = pd.read_parquet(R.DATA / "validation_targets.parquet")
    hist = d16.hist
    ranks = np.empty(len(targets), dtype=np.int64)
    active = []
    for j, row in enumerate(targets.itertuples()):
        h = hist.get(int(row.uid), [])
        if len(h):
            active.append(j)
        else:
            ranks[j] = int(d16.pop_rank[row.iid])
    catalog = model.catalog_matrix().to(device)
    ids = torch.arange(d16.nitems, device=device)
    for start in range(0, len(active), batch):
        idx = active[start:start + batch]
        rows = targets.iloc[idx]
        uids = rows.uid.to_numpy()
        seq = torch.as_tensor(R.padded(hist, uids, model.maxlen), device=device)
        scores = model.query(seq) @ catalog.T
        r_idx, c_idx = [], []
        for j, u in enumerate(uids):
            h = hist[int(u)]
            r_idx.extend([j] * len(h))
            c_idx.extend(h)
        scores[torch.as_tensor(r_idx, device=device), torch.as_tensor(c_idx, device=device)] = -torch.inf
        t = torch.as_tensor(rows.iid.to_numpy(), device=device)
        ts = scores[torch.arange(len(idx), device=device), t]
        assert torch.isfinite(ts).all()
        rr = 1 + (scores > ts[:, None]).sum(-1) + ((scores == ts[:, None]) & (ids[None, :] < t[:, None])).sum(-1)
        ranks[idx] = rr.cpu().numpy()
    out = {"all_official_targets": R.metrics(ranks)}
    out["train_history_ge_3"] = R.metrics(ranks[targets.train_history_length.to_numpy() >= 3])
    return out


def assert_generic_validation(d16, device):
    for lr in ("0.0003", "0.001"):
        stored = json.loads((base.EXP16_CHECKPOINTS / f"sasrec_lr{lr}.complete.json").read_text())
        want = stored["validation"]["all_official_targets"]["ndcg_at_10"]
        model = evaluator.load_exp16_sasrec(base.EXP16_CHECKPOINTS / f"sasrec_lr{lr}.pt", d16.nitems, device)
        got = validate_all_official(model, d16, device)["all_official_targets"]["ndcg_at_10"]
        assert abs(got - want) < 1e-9, (lr, got, want)
        print(f"[check] generic validation, exp16b sasrec lr{lr}: {got:.10f} == stored", flush=True)
        del model


# -------------------------------------------------------- hard negative mining
class HistoryAwareHardSampler(base.ExclusionHardSampler):
    """RESULTS_16's sampler, plus: a hard pick already in the user's train history
    is replaced by the next tactile-farthest pool item that is not."""

    def replace_hard(self, positives, negatives, histories):
        if self.hard_ratio <= 0:
            return negatives
        valid = np.flatnonzero(positives.reshape(-1) > 0)
        if valid.size == 0:
            return negatives
        take = self.rng.random(valid.size) < self.hard_ratio
        chosen = valid[take]
        if chosen.size == 0:
            return negatives
        width = positives.shape[1]
        flat_pos = positives.reshape(-1)[chosen] - 1
        group = (self.category[flat_pos] * 16 + self.band[flat_pos]).cpu().numpy()
        anchor = self.tactile[flat_pos]
        flat_neg = negatives.reshape(-1).copy()
        self.history_hits = getattr(self, "history_hits", 0)
        for offset, (index, key) in enumerate(zip(chosen, group)):
            pool_items = self.by_group.get(int(key))
            if pool_items is None or pool_items.size < 2:
                continue
            sample = pool_items[self.rng.integers(0, pool_items.size,
                                                  size=min(self.pool, pool_items.size))]
            candidate = self.tactile[torch.as_tensor(sample, device=self.device)]
            distance = (candidate - anchor[offset]).abs().mean(dim=1)
            order = torch.argsort(distance, descending=True, stable=True).cpu().numpy()
            seen = histories[index // width]
            # 촉감 거리가 먼 순서대로 보되 사용자 train 이력 상품은 건너뛴다.
            for o in order:
                item = int(sample[o])
                if item not in seen:
                    flat_neg[index] = item + 1
                    break
                self.history_hits += 1
            # if every pool item is in the history, the (history-excluded)
            # uniform negative already in flat_neg is kept
        return flat_neg.reshape(positives.shape)


# ---------------------------------------------------------------------- train
# 하나의 모델 조건과 seed를 Exp16 레시피로 학습하고 최선 validation 상태를 저장한다.
def train_one(name, seed, data, d16, users, device):
    variant, fusion, lam, hard = VARIANTS[name]
    tag = tag_of(name, seed)
    checkpoint = OUT / f"{tag}.pt"
    if checkpoint.exists() and (OUT / f"train__{tag}.complete.json").exists():
        print(f"[{tag}] exists, skipping", flush=True)
        return
    n_items = d16.nitems
    cold_rows = torch.as_tensor(d16.cold, device=device)
    # variant마다 동일한 seed에서 초기화와 배치 순서를 재현한다.
    np.random.seed(seed)
    model = trainer.build_model(variant, data, n_items, device, lam, fusion, seed)
    optimiser = torch.optim.Adam(model.parameters(), lr=base.LR)
    rng = np.random.default_rng(seed)
    sampler = None
    if hard is not None:
        sampler_data = data
        if hard == "shuffled":
            order = np.random.default_rng(seed).permutation(data["tactile"].shape[0])
            sampler_data = dict(data)
            sampler_data["tactile"] = data["tactile"][order]
        sampler = HistoryAwareHardSampler(sampler_data, hard_ratio=base.HARD_RATIO, seed=seed,
                                          device=str(device))
    log = open(OUT / f"train__{tag}.jsonl", "w")

    def record(**row):
        row = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "tag": tag, **row}
        log.write(json.dumps(row) + "\n")
        log.flush()
        print(json.dumps(row), flush=True)

    record(status="started", variant=variant, fusion=fusion, lambda_tactile=lam, hard=hard,
           seed=seed, lr=base.LR, clip=base.CLIP, optimizer="Adam")
    best, bad, best_epoch = -1.0, 0, -1
    begin = time.monotonic()
    # 공통 학습 루프에서 모델 구조와 sampler 조건만 바꿔 비교한다.
    for epoch in range(1, base.MAX_EPOCHS + 1):
        model.train()
        losses = []
        indices = rng.permutation(len(users))
        for start in range(0, len(indices), C.BATCH):
            uids = users[indices[start:start + C.BATCH]]
            optimiser.zero_grad(set_to_none=True)
            seq, pos, neg = base.batch_tensors(d16, uids, rng, n_items)
            if sampler is not None:
                neg = sampler.replace_hard(pos, neg, [set(d16.hist[int(u)].tolist()) for u in uids])
            tensors = [torch.as_tensor(a, device=device) for a in (seq, pos, neg)]
            loss = model.loss(*tensors)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{tag} nonfinite loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), base.CLIP)
            optimiser.step()
            losses.append(float(loss.detach()))
        record(status="epoch", epoch=epoch, loss=float(np.mean(losses)),
               seconds=round(time.monotonic() - begin, 1))
        # cold ID embedding을 중립화한 뒤 전체 공식 validation NDCG@10을 선택 기준으로 쓴다.
        if epoch % base.VALIDATE_EVERY == 0 or epoch == base.MAX_EPOCHS:
            base.zero_cold_rows(model, cold_rows)
            assert float(model.item.weight[1:][cold_rows].abs().max()) == 0.0
            result = validate_all_official(model, d16, device)
            value = result["all_official_targets"]["ndcg_at_10"]
            record(status="validation", epoch=epoch, **result["all_official_targets"],
                   train_history_ge_3_ndcg_at_10=result["train_history_ge_3"]["ndcg_at_10"])
            if value > best:
                best, bad, best_epoch = value, 0, epoch
                torch.save({"state_dict": {k: v.detach().cpu().clone()
                                           for k, v in model.state_dict().items()},
                            "name": name, "variant": variant, "fusion": fusion,
                            "lambda_tactile": lam, "hard": hard, "seed": seed, "tag": tag,
                            "recipe": "exp16b", "lr": base.LR, "epoch": epoch,
                            "validation_all_official": result["all_official_targets"]},
                           checkpoint)
            else:
                bad += 1
            if bad >= base.PATIENCE:
                break
    summary = {"tag": tag, "best_epoch": best_epoch, "best_validation_ndcg_at_10": best,
               "last_epoch": epoch, "seconds": round(time.monotonic() - begin, 1),
               "hard_history_hits": getattr(sampler, "history_hits", None)}
    record(status="complete", **summary)
    log.close()
    (OUT / f"train__{tag}.complete.json").write_text(json.dumps(summary, indent=2))
    del model, optimiser
    torch.cuda.empty_cache()


def train_all(device):
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    OUT.mkdir(parents=True, exist_ok=True)
    data = C.load_prepared()
    d16 = R.RecData()
    users = base.assert_recipe(data, d16)
    base.assert_validation_reproduces(d16, device)        # exp16b evaluate(), 1e-9
    assert_generic_validation(d16, device)                # this script's validator, 1e-9
    for name, seed in PLAN:
        train_one(name, seed, data, d16, users, device)


# ------------------------------------------------------------------- evaluate
# 저장된 설정으로 모델을 다시 만들고 선택된 state_dict를 적재한다.
def rebuild(saved, data, n_items, device):
    model = trainer.build_model(saved["variant"], data, n_items, device,
                                saved["lambda_tactile"], saved["fusion"], saved["seed"])
    model.load_state_dict(saved["state_dict"])
    return model.to(device).eval()


def evaluate(device):
    data = C.load_prepared()
    n_items = int(data["n_items"][0])
    cold = data["train_count"] == 0
    blocks = {}
    # 각 split을 전체·cold target·warm target 집단으로 고정해 평가한다.
    for split in ("validation", "test"):
        users, matrix, targets = evaluator.history_and_targets(data, split)
        seen = evaluator.seen_items(data, split, users)
        tc = cold[targets - 1]
        blocks[split] = (matrix, targets, seen, {"all": np.ones(len(users), bool),
                                                 "cold": tc, "warm": ~tc})
    report = {"metrics": {}, "training": {}, "bootstrap": {}}
    per_user = {}
    for name, seed in PLAN:
        tag = tag_of(name, seed)
        saved = torch.load(OUT / f"{tag}.pt", map_location="cpu", weights_only=False)
        model = rebuild(saved, data, n_items, device)
        report["metrics"][tag] = {}
        ranks_out = {}
        for split, (matrix, targets, seen, groups) in blocks.items():
            ranks, _ = evaluator.rank_targets(model, matrix, targets, seen, device, batch=512)
            report["metrics"][tag][split] = base.metrics(ranks, groups)
            per_user[(tag, split)] = C.per_user_ndcg(ranks)
            ranks_out[split] = ranks
        np.savez_compressed(OUT / f"ranks__{tag}.npz", **ranks_out)
        report["training"][tag] = json.loads((OUT / f"train__{tag}.complete.json").read_text())
        print(f"[{tag}] test NDCG@10 {report['metrics'][tag]['test']['all']['NDCG@10']:.8f}",
              file=sys.stderr, flush=True)
        del model
        torch.cuda.empty_cache()

    # 같은 사용자의 NDCG 차이에 paired bootstrap을 적용한다.
    def boot(a, b, split):
        stat = C.paired_bootstrap_by_user(per_user[(a, split)], per_user[(b, split)],
                                          replicates=BOOTSTRAP_REPLICATES, seed=C.SEED)
        stat["significant"] = bool(stat["ci95_low"] > 0 or stat["ci95_high"] < 0)
        return stat

    seeds = sorted({seed for _, seed in PLAN})
    for split in ("validation", "test"):
        for name, seed in PLAN:
            if name != "R0":
                key = f"{split}__{tag_of(name, seed)}__minus__{tag_of('R0', seed)}"
                report["bootstrap"][key] = boot(tag_of(name, seed), tag_of("R0", seed), split)
        for seed in seeds:
            a, b = tag_of("R5_hard0.5", seed), tag_of("R5_hard0.5_SHUFFLED", seed)
            report["bootstrap"][f"{split}__{a}__minus__{b}"] = boot(a, b, split)

    # R0와 R5 조건은 세 seed의 평균과 표본 표준편차를 함께 기록한다.
    summary = {}
    for name in ("R0", "R5_hard0.5", "R5_hard0.5_SHUFFLED"):
        for split in ("validation", "test"):
            values = [report["metrics"][tag_of(name, s)][split]["all"]["NDCG@10"] for s in seeds]
            summary[f"{name}__{split}"] = {"per_seed": dict(zip(map(str, seeds), values)),
                                           "mean": float(np.mean(values)),
                                           "sd": float(np.std(values, ddof=1))}
    report["three_seed_summary"] = summary
    (OUT / "evaluation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "evaluate"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    if args.mode == "train":
        train_all(device)
    else:
        evaluate(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
