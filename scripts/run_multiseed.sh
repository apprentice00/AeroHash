#!/usr/bin/env bash
# Multi-seed training, out-of-sample evaluation, and result aggregation.
#
# Train, evaluate, and summarize the GAP model across seeds; optionally run frequency-domain ablations or code-length sweeps.
# Usage: bash scripts/run_multiseed.sh
#       SEEDS="42" bash scripts/run_multiseed.sh
#       FREQ_ABLATION=1 bash scripts/run_multiseed.sh
set -u -o pipefail

if [ -n "${HEADS:-}${HEAD:-}${HEAD_MODE:-}${NUM_QUERIES:-}${QUERY_SA:-}${DEC_LAYERS:-}" ]; then
  echo "[ABORT] Only GAP is supported; remove HEADS/HEAD/HEAD_MODE/NUM_QUERIES/QUERY_SA/DEC_LAYERS."
  exit 2
fi

AEROHASH_BIN="${AEROHASH_BIN:-aerohash}"
config_args=()
if [ -n "${CONFIG:-}" ]; then
  config_args=(--config "${CONFIG}")
fi

CODE_LENGTH="${CODE_LENGTH:-96}"
SEEDS="${SEEDS:-42 43 44}"
OUTROOT="${OUTROOT:-runs_multiseed}"
MANIFEST="${OUTROOT}/manifest.tsv"

mkdir -p "${OUTROOT}"

# GPU preflight: exit immediately if no GPU is available (otherwise every combination fails at model.to(cuda),
# exhausting all combinations and leaving empty checkpoint directories). Set SKIP_GPU_CHECK=1 to skip.
if [ "${SKIP_GPU_CHECK:-0}" != "1" ]; then
  if ! python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "[ABORT] No available GPU detected (torch.cuda.is_available()=False)."
    echo "        Restore GPU availability first (nvidia-smi should list devices), then retry;"
    echo "        or force continuation with SKIP_GPU_CHECK=1 (not recommended; CPU execution will be very slow or fail)."
    exit 2
  fi
fi

# Manifest header (append if the file exists; preserve previous entries)
if [ ! -f "${MANIFEST}" ]; then
  printf 'tag\thead\tseed\ttf\ttfc\tckpt_dir\tstatus\n' > "${MANIFEST}"
fi

run_one() {
  # $1=tag $2=seed $3=tf $4=tfc [$5=code_length]
  local tag="$1" head="gap_mlp" seed="$2" tf="$3" tfc="$4"
  local cl="${5:-${CODE_LENGTH}}"          # Code length for this run (defaults to global CODE_LENGTH)
  local info="ms_${tag}_s${seed}"
  local ckpt_root="${CKPT_ROOT:-checkpoints}"
  local ckpt_name="${info}_$(date '+%Y-%m-%d-%H-%M-%S')_${BASHPID}"
  local ckpt_dir="${ckpt_root}/${ckpt_name}"
  echo "==================================================================="
  echo "[$(date '+%F %T')] RUN tag=${tag} head=${head} seed=${seed} tf=${tf} tfc=${tfc} bits=${cl}"
  echo "==================================================================="

  SEED="${seed}" TF_USE="${tf}" TFC_USE="${tfc}" \
    CODE_LENGTH="${cl}" RUN_INFO="${info}" CKPT_ROOT="${ckpt_root}" CKPT_NAME="${ckpt_name}" \
    "${AEROHASH_BIN}" train "${config_args[@]}" > "${OUTROOT}/${info}.train.log" 2>&1
  local rc=$?

  if [ ${rc} -ne 0 ]; then
    echo "[WARN] Training failed rc=${rc}; see ${OUTROOT}/${info}.train.log"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${tag}" "${head}" "${seed}" "${tf}" "${tfc}" "-" "train_failed" >> "${MANIFEST}"
    return
  fi

  # The training invocation owns this exact output directory.
  if [ -z "${ckpt_dir}" ] || [ ! -f "${ckpt_dir}/model.t" ]; then
    echo "[WARN] Checkpoint model.t not found"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${tag}" "${head}" "${seed}" "${tf}" "${tfc}" "-" "no_ckpt" >> "${MANIFEST}"
    return
  fi

  # Recompute using the standard evaluation protocol (write metrics_weather.json + codes_*.npy to ckpt_dir)
  "${AEROHASH_BIN}" evaluate "${config_args[@]}" --ckpt "${ckpt_dir}/model.t" --code-length "${cl}" --outdir "${ckpt_dir}" \
    > "${OUTROOT}/${info}.eval.log" 2>&1
  local eval_rc=$?
  if [ ${eval_rc} -ne 0 ]; then
    echo "[WARN] Evaluation failed rc=${eval_rc}; see ${OUTROOT}/${info}.eval.log"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${tag}" "${head}" "${seed}" "${tf}" "${tfc}" "${ckpt_dir}" "eval_failed" >> "${MANIFEST}"
    return
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${tag}" "${head}" "${seed}" "${tf}" "${tfc}" "${ckpt_dir}" "ok" >> "${MANIFEST}"
  echo "[$(date '+%F %T')] DONE ${info} -> ${ckpt_dir}"
}

if [ "${BITSWEEP:-0}" = "1" ]; then
  # GAP code-length sensitivity; keep other modules at their default settings.
  # Run once for each code length × seed. Override with BITS; defaults to 16 32 64 128.
  BITS="${BITS:-16 32 64 128}"
  for bits in ${BITS}; do
    for seed in ${SEEDS}; do
      run_one "bit_${bits}" "${seed}" 1 1 "${bits}"
    done
  done
elif [ "${FREQ_ABLATION:-0}" = "1" ]; then
  # Frequency-domain encoder / TFC 2x2 ablation with GAP
  for seed in ${SEEDS}; do
    run_one "freq_time"     "${seed}" 0 0
    run_one "freq_tf"       "${seed}" 1 0
    run_one "freq_tfc"      "${seed}" 0 1
    run_one "freq_both"     "${seed}" 1 1
  done
else
  for seed in ${SEEDS}; do
    run_one "gap_mlp" "${seed}" 1 1
  done
fi

echo "==================================================================="
echo "[$(date '+%F %T')] ALL RUNS FINISHED. Summary:"
"${AEROHASH_BIN}" collect --manifest "${MANIFEST}" --code-length "${CODE_LENGTH}"
