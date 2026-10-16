#!/usr/bin/env bash
# ============================================================
# [열람용 사본] 2026-10-16 시점 스냅샷. 이 파일은 실행용이 아님.
# 실제로 돌아가는 원본 경로:
#   /home/user/onsesang/material_span_grounding/experiments/24_adaptive_tactile_recommendation/scripts/run_candidates_all.sh
# 원본은 이후 실험으로 바뀔 수 있음.
# 역할: R7에 필요한 train·validation·test 후보 캐시를 04_candidates.py로 생성한다.
# ============================================================
set -u
EXP="/home/user/onsesang/material_span_grounding/experiments/24_adaptive_tactile_recommendation"
PY="/home/user/onsesang/miniconda3/envs/texture/bin/python"
# CPU only: the shared A100 still belongs to another user's job.
# 이미 만들어진 split은 건너뛰고 세 평가 지점의 후보를 순서대로 생성한다.
for split in validation train test; do
  if [ -f "$EXP/cache/candidates__${split}.npz" ]; then
    echo "$(date -Is) $split: cached, skipping"; continue
  fi
  echo "$(date -Is) $split: generating"
  "$PY" -u "$EXP/scripts/04_candidates.py" --split "$split" --device cpu --batch 128 \
    > "$EXP/logs/04_candidates_${split}.log" 2>&1 || echo "$(date -Is) $split FAILED"
done
echo "$(date -Is) all candidate splits done"
