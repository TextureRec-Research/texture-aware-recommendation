#!/usr/bin/env bash
# ============================================================
# [열람용 사본] 2026-10-16 시점 스냅샷. 이 파일은 실행용이 아님.
# 실제로 돌아가는 원본 경로:
#   /home/user/onsesang/material_span_grounding/experiments/24_adaptive_tactile_recommendation/scripts/run_posthoc.sh
# 원본은 이후 실험으로 바뀔 수 있음.
# 역할: R7 후보 캐시를 기다린 뒤 reranker 학습과 R0 대비 분석을 연속 실행한다.
# ============================================================
set -u
EXP="/home/user/onsesang/material_span_grounding/experiments/24_adaptive_tactile_recommendation"
PY="/home/user/onsesang/miniconda3/envs/texture/bin/python"
# 세 split의 후보가 모두 준비된 뒤 post-hoc 학습과 통계 분석을 실행한다.
while [ "$(ls $EXP/cache/candidates__*.npz 2>/dev/null | wc -l)" -lt 3 ]; do
  echo "$(date -Is) waiting for candidate splits ($(ls $EXP/cache/candidates__*.npz 2>/dev/null | wc -l)/3)"
  sleep 30
done
echo "$(date -Is) all candidate splits ready; running R1/R1b/R2/R7 on CPU"
"$PY" -u "$EXP/scripts/05_rerank.py" --device cpu > "$EXP/logs/05_rerank.log" 2>&1
status=$?
echo "$(date -Is) 05_rerank exit ${status}"
if [ $status -eq 0 ]; then
  "$PY" -u "$EXP/scripts/06_analysis.py" > "$EXP/logs/06_analysis.log" 2>&1
  echo "$(date -Is) 06_analysis exit $?"
fi
