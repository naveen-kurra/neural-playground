# MVP Outline

## Product

`Neural Playground` is a visual architecture builder for learners and R&D users.

## V1 Scope

- graph canvas
- block palette
- node inspector
- training config panel
- graph serialization
- starter PyTorch export

## V1 Non-Goals

- full browser training
- multi-user collaboration
- dataset hosting
- remote job orchestration

## Initial Technical Choices

- `React`
- `TypeScript`
- `React Flow`
- `Zustand`
- package-level shared schema and validator logic

## Immediate Next Tasks

1. Finalize block schema and node metadata.
2. Add validation rules for graph structure and block config.
3. Scaffold `apps/web` with a canvas and starter palette.
4. Add JSON export from the shared graph model.

