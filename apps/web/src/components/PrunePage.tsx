import { useMemo, useState } from "react";

type PrunePageProps = {
  onBackToBuilder: () => void;
};

type ModelFamily = "auto" | "gpt2" | "llama" | "unknown";

export function PrunePage(props: PrunePageProps) {
  const { onBackToBuilder } = props;
  const [modelId, setModelId] = useState("");
  const [modelFamily, setModelFamily] = useState<ModelFamily>("auto");
  const [layerCount, setLayerCount] = useState(12);

  const keptLayerCount = useMemo(() => Math.max(1, Math.floor(layerCount)), [layerCount]);

  return (
    <div className="tool-page">
      <header className="tool-header">
        <div>
          <p className="eyebrow">Neural Playground</p>
          <h1>Pruning Tool</h1>
          <p className="panel-copy">
            Broad checkpoint pruning for Hugging Face-style models. Start with whole-layer removal and keep the original
            model class workflow.
          </p>
        </div>
        <div className="canvas-actions">
          <button type="button" className="ghost-button" onClick={onBackToBuilder}>
            Back To Builder
          </button>
        </div>
      </header>

      <div className="prune-layout">
        <section className="canvas-panel">
          <div className="panel-header">
            <p className="eyebrow">Import</p>
            <h2>Model Source</h2>
          </div>
          <div className="form-stack">
            <label className="field">
              <span>Hugging Face Model ID</span>
              <input
                type="text"
                value={modelId}
                placeholder="meta-llama/Llama-3.2-1B or gpt2"
                onChange={(event) => setModelId(event.target.value)}
              />
            </label>
            <label className="field">
              <span>Model Family</span>
              <select value={modelFamily} onChange={(event) => setModelFamily(event.target.value as ModelFamily)}>
                <option value="auto">Auto Detect</option>
                <option value="gpt2">GPT-2 Style</option>
                <option value="llama">LLaMA Style</option>
                <option value="unknown">Unknown / Inspect Only</option>
              </select>
            </label>
            <p className="muted">
              First version will focus on whole-layer pruning with updated config plus remapped weights, not architecture
              rewrites.
            </p>
          </div>
        </section>

        <section className="canvas-panel">
          <div className="panel-header">
            <p className="eyebrow">Plan</p>
            <h2>Layer Pruning</h2>
          </div>
          <div className="form-stack">
            <label className="field">
              <span>Target Layer Count</span>
              <input
                type="number"
                min="1"
                max="256"
                value={layerCount}
                onChange={(event) => setLayerCount(Number(event.target.value))}
              />
            </label>
          </div>
          <div className="prune-summary">
            <strong>Current Scope</strong>
            <span>Whole transformer block removal only</span>
            <strong>Output Artifacts</strong>
            <span>`config.json`, pruning manifest, checkpoint remap script</span>
            <strong>Target Layers</strong>
            <span>{keptLayerCount}</span>
          </div>
        </section>

        <section className="issues-panel">
          <div className="panel-header">
            <p className="eyebrow">Next</p>
            <h2>Implementation Roadmap</h2>
          </div>
          <div className="issue-card warning">
            <div className="issue-header">
              <strong>Phase 1</strong>
            </div>
            <span>Fetch model config and tensor index, detect layer prefix, and build a whole-layer pruning plan.</span>
          </div>
          <div className="issue-card warning">
            <div className="issue-header">
              <strong>Phase 2</strong>
            </div>
            <span>Generate updated config plus a weight remap script that preserves surviving pretrained layers.</span>
          </div>
          <div className="issue-card warning">
            <div className="issue-header">
              <strong>Later</strong>
            </div>
            <span>Head pruning, FFN shrinking, and more family adapters after the broad layer-pruning workflow is stable.</span>
          </div>
        </section>
      </div>
    </div>
  );
}

