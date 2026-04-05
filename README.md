# Neural Playground

Visual transformer architecture builder with exact-family editing, PyTorch export, full-project export, and local checkpoint pruning.

## How To Use The Website

Typical builder flow:

1. start the frontend

```bash
cd /home/naveen/neural-playground
npm install
npm run dev:web
```

2. open the app in your browser, usually:

```text
http://127.0.0.1:5173
```

3. choose a starting point

- load a built-in template such as:
  - `GPT-2`
  - `LLaMA`
  - `Mistral 7B v0.3`
  - `Phi-3 Mini`
  - `Gemma 4 31B`
- or start from the default graph and add blocks manually

4. edit the architecture

- drag blocks on the canvas
- click a block to edit its parameters in the inspector
- connect blocks by using the canvas connection flow
- use exact-family blocks when you want exact-family export

5. watch validation

- invalid graphs are shown in the validation panel
- export is blocked when the graph is structurally invalid
- common issues like cycles, wrong arity, and invalid output paths are surfaced before export

6. export artifacts

- `Graph JSON` for save/share/load
- `PyTorch Model` for generated `model.py`
- `Full Project` for a runnable starter training project

7. load saved work later

- use `Save` to download a project file
- use `Load` to restore a saved project or raw graph JSON

Typical pruning flow:

1. start the prune service in another terminal

```bash
cd /home/naveen/neural-playground
npm run dev:prune-service
```

2. open the pruning page in the app

3. enter a Hugging Face model id

- example:
  - `microsoft/Phi-3-mini-4k-instruct`
  - `mistralai/Mistral-7B-Instruct-v0.3`

4. inspect metadata

- the app fetches config + weight-index metadata
- it detects the transformer layer prefix and layer count
- it tells you whether broad block pruning is supported

5. choose blocks to keep or drop

6. run local prune

- the backend downloads the model snapshot if needed
- rewrites config and weights
- validates original and pruned model load
- reports memory usage deltas

7. review the output path and validation report

## Current Scope

Supported exact families:

- `GPT-2`
- `LLaMA`
- `Phi-3`
  - `Phi-3 Mini`
  - `Phi-3 Medium`
  - `Phi-3.5 Mini` preset on the same core family path
- `Gemma 4` text
- `Mistral`

Supported product capabilities:

- canvas-based architecture editing
- exact-family template import
- graph save/load as project JSON
- generated `model.py` export
- full starter PyTorch project export
- local checkpoint pruning with validation
- regression and browser test coverage

## Repo Shape

```text
apps/web                  React/Vite frontend
apps/prune-service        local Node service for HF inspection + pruning orchestration
packages/block-schema     block definitions, contracts, defaults
packages/ir-schema        exact family IR and graph<->IR mappers
packages/validator        topology, dimension, and export validation
packages/exporter-pytorch exact-family and generic PyTorch/project exporters
packages/test-suite       V1 regression suite
scripts                   parity harnesses, prune runner, setup helpers
```

## Setup

```bash
cd /home/naveen/neural-playground
npm install
```

If you want pruning support, set up the prune Python env too:

```bash
npm run setup
```

## Run The App

Frontend only:

```bash
npm run dev:web
```

Then open the local Vite URL, usually `http://127.0.0.1:5173`.

Frontend + pruning backend:

1. configure the prune service token if needed

```bash
cp apps/prune-service/base_conf.example.json apps/prune-service/base_conf.json
```

Then edit `apps/prune-service/base_conf.json`:

```json
{
  "host": "127.0.0.1",
  "port": 8787,
  "hfToken": ""
}
```

2. start the prune service

```bash
npm run dev:prune-service
```

3. in another terminal, start the web app

```bash
npm run dev:web
```

Notes:

- gated Hugging Face models require `hfToken`
- the prune service calls the local Python runner in `scripts/prune_checkpoint.py`
- pruning is a local workflow, not a browser-only feature

## Templates

Current built-in templates include:

- `GPT-2`
- `LLaMA`
- `Gemma 4 31B`
- `Mistral 7B v0.3`
- `Phi-3 Mini`
- `Phi-3 Medium`
- `Phi-3.5 Mini`

## Export Workflow

The app can export:

- `Graph JSON`
- `PyTorch Model`
- `Full Project`

Full project export includes generated files such as:

- `src/kurra_ai_cb/model.py`
- `src/kurra_ai_cb/train.py`
- `configs/model.yaml`
- `configs/train.yaml`
- `scripts/train.py`
- helper runtime modules

Export is blocked when the graph is structurally invalid.

## Run An Exported Project

From an exported project directory:

```bash
cd /path/to/exported-project
PYTHONPATH=src /home/naveen/SecondOrbit-150M/.venv/bin/python scripts/train.py
```

With token-id shard data:

```bash
cd /path/to/exported-project
PYTHONPATH=src /home/naveen/SecondOrbit-150M/.venv/bin/python scripts/train.py \
  --train-shards-glob "/path/to/train/*.npy" \
  --val-shards-glob "/path/to/val/*.npy" \
  --output-dir artifacts/run
```

If you need CPU-only execution:

```bash
cd /path/to/exported-project
CUDA_VISIBLE_DEVICES="" PYTHONPATH=src /home/naveen/SecondOrbit-150M/.venv/bin/python scripts/train.py \
  --train-shards-glob "/path/to/train/*.npy" \
  --val-shards-glob "/path/to/val/*.npy" \
  --output-dir artifacts/run
```

## Pruning Workflow

The pruning page can:

- inspect HF model metadata
- detect repeated transformer layer prefixes
- choose kept/dropped block indices
- run local checkpoint rewrite
- validate original vs pruned model load
- report memory usage deltas

Current family detection in the pruning inspector includes:

- `gpt2`
- `llama`
- `mistral`
- `phi3`

The pruning backend is still generic block-pruning infrastructure, not family-specific training code.

## Tests

Repo-wide checks:

```bash
npm run check
npm run build
```

V1 regression suite:

```bash
npm run test:v1
```

Browser/e2e tests:

```bash
npm run test:e2e
```

All tests:

```bash
npm run test:all
```

Playwright starts the web app automatically on `http://127.0.0.1:4173`.

## Parity Harnesses

Family parity scripts live in [`scripts/`](/home/naveen/neural-playground/scripts):

- `check_gpt2_parity.py`
- `check_llama_parity.py`
- `check_phi3_parity.py`
- `check_gemma4_parity.py`
- `check_mistral_parity.py`

These are used to compare generated exact-family exports against native `transformers` implementations on small configs.

## Product Boundaries

This repo is not an arbitrary graph compiler.

Important current constraints:

- export assumes an acyclic graph
- exact-family export expects exact family block sets
- export currently targets decoder-first causal LM flows
- runtime helpers are training-oriented and minimal
- exact family support does not imply full HF inference-runtime parity
  - cache
  - rolling KV buffers
  - advanced generation helpers
  - all model-specific inference APIs

## Status

This is the main product repo.

`SecondOrbit-150M` is a separate reference/training environment used for local validation, parity runs, and exported project execution, not the app itself.
