#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/naveen/neural-playground"
PYTHON="/home/naveen/SecondOrbit-150M/.venv/bin/python"
TMP_TS="/tmp/gen_phi3_variant_model.ts"
TMP_MJS="/tmp/gen_phi3_variant_model.mjs"
TMP_MODEL="/tmp/generated_phi3_variant_model.py"

run_variant() {
  local label="$1"
  local layers="$2"
  local hidden="$3"
  local intermediate="$4"
  local heads="$5"
  local kv_heads="$6"
  local head_dim="$7"
  local positions="$8"
  local vocab="$9"
  local seq_len="${10}"

  cat >"${TMP_TS}" <<EOF
import { writeFileSync } from "node:fs";
import { buildPhi3ArchitectureSpec } from "${ROOT}/packages/ir-schema/src/phi3.ts";
import { exportPhi3IrToPyTorch } from "${ROOT}/packages/exporter-pytorch/src/phi3-ir-export.ts";

const spec = buildPhi3ArchitectureSpec({
  numHiddenLayers: ${layers},
  hiddenSize: ${hidden},
  intermediateSize: ${intermediate},
  numAttentionHeads: ${heads},
  numKeyValueHeads: ${kv_heads},
  headDim: ${head_dim},
  maxPositionEmbeddings: ${positions},
  vocabSize: ${vocab},
  tieWordEmbeddings: false,
});
writeFileSync("${TMP_MODEL}", exportPhi3IrToPyTorch(spec), "utf8");
EOF

  npx esbuild "${TMP_TS}" --bundle --platform=node --format=esm --outfile="${TMP_MJS}" >/dev/null
  node "${TMP_MJS}" >/dev/null

  echo "[phi3-parity] variant=${label}"
  "${PYTHON}" "${ROOT}/scripts/check_phi3_parity.py" \
    --model-file "${TMP_MODEL}" \
    --layers "${layers}" \
    --hidden-size "${hidden}" \
    --intermediate-size "${intermediate}" \
    --heads "${heads}" \
    --kv-heads "${kv_heads}" \
    --positions "${positions}" \
    --vocab-size "${vocab}" \
    --seq-len "${seq_len}"
  echo
}

run_variant "phi3-mini" 2 192 512 6 6 32 64 256 8
run_variant "phi3-medium" 2 320 896 10 5 32 64 256 8
run_variant "phi3.5-mini" 2 192 512 6 6 32 128 256 8
