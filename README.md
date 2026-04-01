# Neural Playground

Visual drag-and-drop playground for building, configuring, validating, and exporting neural network architectures.

## Repo Shape

```text
apps/web                  React app for the visual builder
packages/block-schema     Shared block types and starter block definitions
packages/validator        Graph and shape validation rules
packages/exporter-pytorch Graph-to-PyTorch export logic
docs                      Product and implementation notes
```

## MVP Goal

Ship a browser-based builder where users can:

- drag blocks onto a canvas
- connect blocks into a model graph
- edit block parameters in an inspector
- validate obvious graph/config mistakes
- export the graph as JSON
- export starter PyTorch code

## Build Order

1. Shared graph and block schema
2. Validator package
3. React canvas app
4. PyTorch exporter

## First Blocks

- `Input`
- `Embedding`
- `TransformerBlock`
- `MLP`
- `LayerNorm`
- `Softmax`
- `Output`

## Status

This repo is the product foundation. The existing `SecondOrbit-150M` repo remains a reference implementation for model/export behavior, not the main app.

## Local Setup

```bash
cd /home/naveen/neural-playground
npm install
npm run dev:web
```
