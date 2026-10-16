#!/usr/bin/env bash
# ============================================================
# [열람용 사본] 2026-10-16 시점 스냅샷. 이 파일은 실행용이 아님.
# 실제로 돌아가는 원본 경로:
#   /home/user/onsesang/material_span_grounding/experiments/24_adaptive_tactile_recommendation/scripts/run_training_queue.sh
# 원본은 이후 실험으로 바뀔 수 있음.
# 역할: 약한 Exp24 공통 루프로 R0·R3·R4·R5·R6 학습 작업을 순서대로 실행한다.
# ============================================================
# Trains every Exp24 variant that needs a retrained backbone (R3, R4, R5, R6)
# plus a fresh R0 under the identical loop, so R3-R6 are compared against a
# baseline trained the same way rather than against a checkpoint from exp16b.
#
# Waits for (a) every foreign job and (b) the Exp22 VLM supervisor, so Exp22
# keeps its priority and we never contend with another user again.
set -u
EXP="/home/user/onsesang/material_span_grounding/experiments/24_adaptive_tactile_recommendation"
PY="/home/user/onsesang/miniconda3/envs/texture/bin/python"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OTHER_PIDS="83996 84693 80079"
# 기존 GPU 작업과 Exp22 평가가 끝날 때까지 기다린 뒤 학습을 시작한다.
while :; do
  alive=0
  for pid in $OTHER_PIDS; do kill -0 "$pid" 2>/dev/null && alive=$((alive+1)); done
  if [ "$alive" -eq 0 ]; then break; fi
  echo "$(date -Is) waiting: ${alive} foreign job(s) on the GPU"
  sleep 120
done
echo "$(date -Is) foreign jobs clear"

while pgrep -f "run_vlm_when_free.sh|04_score_vlm.py" >/dev/null 2>&1; do
  echo "$(date -Is) waiting: Exp22 VLM scoring still running"
  sleep 120
done
echo "$(date -Is) Exp22 VLM stage clear; starting Exp24 training"

run() {
  local tag="$1"; shift
  if [ -f "$EXP/checkpoints/${tag}.pt" ]; then
    echo "$(date -Is) ${tag}: checkpoint exists, skipping"; return
  fi
  echo "$(date -Is) ${tag}: training"
  "$PY" -u "$EXP/scripts/03_train.py" --tag "$tag" "$@" \
    > "$EXP/logs/train__${tag}.log" 2>&1 \
    || echo "$(date -Is) ${tag}: FAILED (exit $?)"
}

SEED=20260917
# 같은 seed와 학습 루프에서 R0 기준선과 촉감 variant를 차례로 학습한다.
# R0 under this loop, the baseline R3-R6 are measured against
run "R0_seed${SEED}"            --variant R0 --seed $SEED
# R3: the three fusions the prompt allows, plus the exp20-style shuffle control
for f in residual_gate concat_proj film shuffle; do
  run "R3_${f}_seed${SEED}"     --variant R3 --fusion "$f" --seed $SEED
done
# R5: hard-negative ratio, a small grid only
for r in 0.25 0.5 0.75; do
  run "R5_hard${r}_seed${SEED}" --variant R5 --hard-ratio "$r" --seed $SEED
done
# R4: small lambda grid
for l in 0.01 0.05 0.1 0.2 0.5; do
  run "R4_lambda${l}_seed${SEED}" --variant R4 --lambda-tactile "$l" --seed $SEED
done
# R6: mixture of experts
run "R6_seed${SEED}"            --variant R6 --seed $SEED

echo "$(date -Is) training queue finished"

# 아래 블록은 본 비교 이후 별도의 촉감 source ablation 입력을 만드는 선택 단계다.
# --- tactile-source ablation inputs -----------------------------------------
# Last2 already has a full-catalog cache from exp16b.  FabricVST-A/B do not, so
# they are produced here, after the must-have training is complete.
for m in fabricvst_B fabricvst_A; do
  if [ -f "$EXP/cache/catalog_tactile__${m}.npz" ] && \
     grep -q '"coverage"' "$EXP/logs/catalog_tactile__${m}.json" 2>/dev/null; then
    echo "$(date -Is) catalog tactile ${m}: done, skipping"; continue
  fi
  echo "$(date -Is) catalog tactile ${m}: inferring over the Amazon catalog"
  "$PY" -u "$EXP/scripts/08_catalog_tactile.py" --model "$m" --batch-size 256 \
    > "$EXP/logs/08_catalog_${m}.log" 2>&1 || echo "$(date -Is) ${m}: FAILED"
done
echo "$(date -Is) ablation inputs ready"
