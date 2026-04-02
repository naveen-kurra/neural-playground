# Neural Playground

Visual neural architecture builder for decoder-first transformer experimentation, learning, and PyTorch project export.

## What It Does

Neural Playground lets users:

- add and connect architecture blocks on a visual canvas
- drag nodes and inspect block parameters
- configure training settings like optimizer, loss, activation, and learning rate
- define custom hook names for optimizer, loss, and activation flows
- validate graph structure and dimension-related issues
- save and load graph projects as JSON
- export generated `model.py`
- export a full starter PyTorch project scaffold

## Current Product Scope

The current app is intentionally focused on a decoder-first path.

Supported blocks:

- `Input`
- `Embedding`
- `TransformerBlock`
- `MLP`
- `LayerNorm`
- `Softmax`
- `Output`

Supported presets:

- `Small GPT`
- `Decoder Block`
- `Text LM Starter`

Supported export path:

- decoder-oriented PyTorch model export
- decoder-oriented starter training project export
- custom hook stub generation for:
  - activation
  - loss
  - optimizer

## Current Limits

This is not an arbitrary neural graph compiler yet.

Current constraints:

- export assumes an acyclic graph
- export assumes exactly one `Input`
- export assumes exactly one `Embedding`
- export assumes exactly one `Output`
- full project export assumes decoder-style training
- full project export requires `Output.headType = LanguageModel`
- multi-branch execution is not compiled as true graph execution
- browser-side training is not part of this version

The validator supports multiple validation modes, and export now uses stricter mode checks before allowing generated artifacts.

## Repo Shape

```text
apps/web                  React app for the visual builder
packages/block-schema     Shared block types, contracts, and block definitions
packages/ir-schema        Exact internal architecture representation for supported families
packages/validator        Graph, topology, and dimension validation
packages/exporter-pytorch Graph-to-PyTorch and project export logic
docs                      Product and implementation notes
```

## Local Setup

```bash
cd /home/naveen/neural-playground
npm install
npm run dev:web
```

Then open the local Vite URL, usually `http://localhost:5173`.

## Quality Checks

```bash
npm run check
npm run build
```

## Save / Load

- `Save` downloads a `.project.json` file
- `Load` restores a previously saved project or raw graph JSON

## Export

Artifacts currently available in the app:

- `Graph JSON`
- `PyTorch Model`
- `Full Project`

Full project export includes generated files like:

- `src/kurra_ai_cb/model.py`
- `configs/model.yaml`
- `configs/train.yaml`
- `scripts/train.py`
- reusable training helpers
- `CUSTOM_HOOKS.md`

## Custom Hook Workflow

If the user selects `Custom` for:

- optimizer
- loss
- activation

the app collects the custom hook name and carries it into exported configs and code stubs. The exported project then tells the user exactly where to implement that custom hook.

## Status

This repo is the product codebase.

`SecondOrbit-150M` is a reference implementation for model/export behavior and reusable training concepts, not the main app itself.
