import { useEffect, useMemo, useRef, useState } from "react";
import { fetchHuggingFaceModelViaService, type HuggingFaceFetchResult } from "../app/huggingface";
import { downloadTextFile } from "../app/file-utils";
import { buildPruningManifest, buildUpdatedConfig, buildWeightRemapScript } from "../app/pruning-artifacts";
import { buildLayerRemap, getEffectiveLayerIndices } from "../app/pruning";
import { checkPruneServiceHealth, runLocalPrune, type PruneServiceLogEvent, type RunLocalPruneResult } from "../app/prune-service";

type PrunePageProps = {
  onBackToBuilder: () => void;
};

type ModelFamily = "auto" | "gpt2" | "llama" | "unknown";
type PrunePreset = "none" | "drop-last-4" | "drop-last-8" | "keep-every-other" | "keep-first-half";
type RunLogItem = {
  id: number;
  receivedAt: number;
  event: PruneServiceLogEvent;
};

export function PrunePage(props: PrunePageProps) {
  const { onBackToBuilder } = props;
  const [modelId, setModelId] = useState("");
  const [modelFamily, setModelFamily] = useState<ModelFamily>("auto");
  const [layerCount, setLayerCount] = useState(12);
  const [fetchState, setFetchState] = useState<"idle" | "loading" | "success" | "error">("idle");
  const [fetchError, setFetchError] = useState("");
  const [fetchedModel, setFetchedModel] = useState<HuggingFaceFetchResult | null>(null);
  const [droppedLayerIndices, setDroppedLayerIndices] = useState<number[]>([]);
  const [inputDir, setInputDir] = useState("");
  const [outputDir, setOutputDir] = useState("");
  const [cacheDir, setCacheDir] = useState("");
  const [runState, setRunState] = useState<"idle" | "checking" | "running" | "success" | "error">("idle");
  const [runError, setRunError] = useState("");
  const [runResult, setRunResult] = useState<RunLocalPruneResult | null>(null);
  const [runLogs, setRunLogs] = useState<RunLogItem[]>([]);
  const [selectedPreset, setSelectedPreset] = useState<PrunePreset>("none");
  const runLogSeqRef = useRef(0);
  const logsContainerRef = useRef<HTMLDivElement | null>(null);

  const keptLayerCount = useMemo(() => Math.max(1, Math.floor(layerCount)), [layerCount]);
  const configKeys = useMemo(() => Object.keys(fetchedModel?.config ?? {}).sort(), [fetchedModel]);
  const layerCountHint = fetchedModel?.inspection.layerCountHint ?? null;
  const availableLayerIndices = useMemo(
    () => getEffectiveLayerIndices(fetchedModel?.inspection.detectedLayerIndices ?? [], layerCountHint),
    [fetchedModel, layerCountHint]
  );
  const actualLayerCount = availableLayerIndices.length;
  const keptLayerIndices = useMemo(
    () => availableLayerIndices.filter((index) => !droppedLayerIndices.includes(index)),
    [availableLayerIndices, droppedLayerIndices]
  );
  const layerRemap = useMemo(() => buildLayerRemap(keptLayerIndices), [keptLayerIndices]);
  const pruningManifest = useMemo(
    () =>
      fetchedModel
        ? buildPruningManifest(fetchedModel, keptLayerIndices, droppedLayerIndices, layerRemap)
        : null,
    [droppedLayerIndices, fetchedModel, keptLayerIndices, layerRemap]
  );
  const updatedConfig = useMemo(
    () => (fetchedModel ? buildUpdatedConfig(fetchedModel, layerRemap) : null),
    [fetchedModel, layerRemap]
  );
  const weightRemapScript = useMemo(
    () => (pruningManifest ? buildWeightRemapScript(pruningManifest) : ""),
    [pruningManifest]
  );
  const memoryDeltaMiB = useMemo(() => {
    if (!runResult) {
      return null;
    }
    const original = runResult.validation.load.memory.originalModelMiB;
    const pruned = runResult.validation.load.memory.prunedModelMiB;
    if (original === null || pruned === null) {
      return null;
    }
    return Number((original - pruned).toFixed(2));
  }, [runResult]);
  const memoryDeltaPercent = useMemo(() => {
    if (!runResult || memoryDeltaMiB === null) {
      return null;
    }
    const original = runResult.validation.load.memory.originalModelMiB;
    if (original === null || original <= 0) {
      return null;
    }
    return Number(((memoryDeltaMiB / original) * 100).toFixed(2));
  }, [memoryDeltaMiB, runResult]);
  const hasDetectedPrunableStack = Boolean(fetchedModel?.inspection.detectedLayerPrefix) && actualLayerCount > 0;
  const canDownloadArtifacts = Boolean(pruningManifest && updatedConfig && keptLayerIndices.length > 0 && hasDetectedPrunableStack);
  const canRunLocalPrune = Boolean(
    fetchedModel &&
    keptLayerIndices.length > 0 &&
    outputDir.trim() &&
    hasDetectedPrunableStack &&
    runState !== "checking" &&
    runState !== "running"
  );
  const pruningReadinessMessage = useMemo(() => {
    if (!fetchedModel) {
      return "Fetch a model first to inspect whether a prunable transformer block stack exists.";
    }
    if (!hasDetectedPrunableStack) {
      return "No prunable transformer block stack was detected from the fetched model metadata.";
    }
    if (!fetchedModel.inspection.broadPruningSupported) {
      return "Metadata inspection needs manual review before broad whole-block pruning is trusted.";
    }
    if (!outputDir.trim()) {
      return "Choose an output directory to enable the local prune run.";
    }
    if (keptLayerIndices.length === 0) {
      return "Keep at least one transformer block before pruning.";
    }
    return "Ready to prune the detected transformer block stack.";
  }, [fetchedModel, hasDetectedPrunableStack, outputDir, keptLayerIndices.length]);
  const fetchStatusLabel = useMemo(() => {
    if (fetchState === "loading") {
      return "Fetching metadata";
    }
    if (fetchState === "success") {
      return "Ready";
    }
    if (fetchState === "error") {
      return "Failed";
    }
    return "Not fetched";
  }, [fetchState]);
  const runButtonLabel = runState === "checking" ? "Checking Service..." : runState === "running" ? "Pruning..." : "Run Local Prune";
  const firstLogReceivedAt = runLogs[0]?.receivedAt ?? null;

  useEffect(() => {
    const container = logsContainerRef.current;
    if (!container) {
      return;
    }
    container.scrollTop = container.scrollHeight;
  }, [runLogs.length]);

  async function handleFetchModel() {
    try {
      setFetchState("loading");
      setFetchError("");
      const serviceHealthy = await checkPruneServiceHealth();
      if (!serviceHealthy) {
        throw new Error("Prune service is not running. Start it with `npm run dev:prune-service` from the repo root.");
      }
      const result = await fetchHuggingFaceModelViaService(modelId);
      setFetchedModel(result);
      setFetchState("success");
      setDroppedLayerIndices([]);

      if (result.resolvedFamily === "gpt2" || result.resolvedFamily === "llama") {
        setModelFamily(result.resolvedFamily);
      }

      const nextLayerCount =
        result.inspection.layerCountHint ??
        (result.inspection.detectedLayerIndices.length > 0 ? result.inspection.detectedLayerIndices.length : 12);
      setLayerCount(nextLayerCount);
      setSelectedPreset("none");
      setRunResult(null);
      setRunError("");
      setRunLogs([]);
      runLogSeqRef.current = 0;
    } catch (error) {
      setFetchedModel(null);
      setFetchState("error");
      setFetchError(error instanceof Error ? error.message : "Failed to fetch model metadata.");
    }
  }

  function toggleLayer(layerIndex: number) {
    setDroppedLayerIndices((current) =>
      current.includes(layerIndex) ? current.filter((value) => value !== layerIndex) : [...current, layerIndex].sort((a, b) => a - b)
    );
  }

  function applyPreset(preset: PrunePreset) {
    setSelectedPreset(preset);

    if (availableLayerIndices.length === 0) {
      return;
    }

    let nextDropped: number[] = [];
    if (preset === "drop-last-4") {
      nextDropped = availableLayerIndices.slice(-4);
    } else if (preset === "drop-last-8") {
      nextDropped = availableLayerIndices.slice(-8);
    } else if (preset === "keep-every-other") {
      nextDropped = availableLayerIndices.filter((_, index) => index % 2 === 1);
    } else if (preset === "keep-first-half") {
      const keepCount = Math.max(1, Math.ceil(availableLayerIndices.length / 2));
      nextDropped = availableLayerIndices.slice(keepCount);
    }

    setDroppedLayerIndices(nextDropped);
    setLayerCount(Math.max(1, availableLayerIndices.length - nextDropped.length));
  }

  function clearDroppedLayers() {
    if (availableLayerIndices.length === 0) {
      return;
    }
    setSelectedPreset("none");
    setDroppedLayerIndices([]);
    setLayerCount(Math.max(1, availableLayerIndices.length));
  }

  function dropAllLayers() {
    if (availableLayerIndices.length === 0) {
      return;
    }
    setSelectedPreset("none");
    setDroppedLayerIndices([...availableLayerIndices]);
    setLayerCount(1);
  }

  function clearRunLogs() {
    setRunLogs([]);
    runLogSeqRef.current = 0;
  }

  function formatLogOffset(receivedAt: number): string {
    if (!firstLogReceivedAt) {
      return "0.0s";
    }
    return `${((receivedAt - firstLogReceivedAt) / 1000).toFixed(1)}s`;
  }

  function downloadPruningArtifacts() {
    if (!pruningManifest || !updatedConfig) {
      return;
    }

    downloadTextFile("pruned-config.json", JSON.stringify(updatedConfig, null, 2));
    downloadTextFile("prune_manifest.json", JSON.stringify(pruningManifest, null, 2));
    downloadTextFile("convert_weights.py", weightRemapScript);
  }

  async function handleRunLocalPrune() {
    if (!fetchedModel) {
      setRunState("error");
      setRunError("Fetch a model first so the pruning plan is defined.");
      return;
    }

    try {
      setRunState("checking");
      setRunError("");
      setRunResult(null);
      setRunLogs([]);
      runLogSeqRef.current = 0;

      const serviceHealthy = await checkPruneServiceHealth();
      if (!serviceHealthy) {
        throw new Error("Prune service is not running. Start it with `npm run dev:prune-service` from the repo root.");
      }

      setRunState("running");
      const result = await runLocalPrune({
        inputDir: inputDir.trim() || undefined,
        outputDir,
        modelId: fetchedModel.modelId,
        cacheDir: cacheDir.trim() || undefined,
        droppedLayerIndices,
        layerPrefix: fetchedModel.inspection.detectedLayerPrefix ?? undefined,
        layerCountKey: fetchedModel.inspection.layerCountKey ?? undefined,
        strategy: "drop"
      }, (event) => {
        const id = runLogSeqRef.current++;
        setRunLogs((current) => [...current, { id, receivedAt: Date.now(), event }]);
      });
      setRunResult(result);
      setRunState("success");
    } catch (error) {
      setRunState("error");
      setRunError(error instanceof Error ? error.message : "Local prune run failed.");
    }
  }

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
        <div className="tool-header-actions">
          <button type="button" className="ghost-button" onClick={onBackToBuilder}>
            Back To Builder
          </button>
        </div>
      </header>

      <div className="prune-layout">
        <section className="canvas-panel prune-panel">
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
            <button type="button" className="ghost-button" onClick={handleFetchModel} disabled={fetchState === "loading"}>
              {fetchState === "loading" ? <span className="inline-spinner" aria-hidden="true" /> : null}
              {fetchState === "loading" ? "Fetching metadata..." : "Fetch Model Metadata"}
            </button>
          </div>
          {fetchError ? (
            <div className="prune-result">
              <strong>Metadata Error</strong>
              <span>{fetchError}</span>
            </div>
          ) : null}
          {fetchedModel ? (
            <div className="prune-result prune-kv">
              <div className="prune-result-header">
                <strong>Resolved Model</strong>
                <span className={`prune-badge ${fetchedModel.inspection.broadPruningSupported ? "success" : "warning"}`}>
                  {fetchedModel.inspection.broadPruningSupported ? "Likely supported" : "Needs review"}
                </span>
              </div>
              <strong>Model ID</strong>
              <span>{fetchedModel.modelId}</span>
              <strong>Detected Family</strong>
              <span>{fetchedModel.resolvedFamily}</span>
              <strong>Broad Pruning Support</strong>
              <span>{fetchedModel.inspection.broadPruningSupported ? "Likely supported" : "Needs manual review"}</span>
              <strong>Pruning Readiness</strong>
              <span>{hasDetectedPrunableStack ? "Transformer block stack detected" : "No prunable stack detected"}</span>
              <strong>Weight Index</strong>
              <span>{fetchedModel.weightIndex ? "Found model.safetensors.index.json" : "No safetensors index found"}</span>
              <strong>Layer Count Hint</strong>
              <span>{layerCountHint ?? "Not found in config"}</span>
              <strong>Layer Count Key</strong>
              <span>{fetchedModel.inspection.layerCountKey ?? "Not detected"}</span>
              <strong>Detected Layer Prefix</strong>
              <span>{fetchedModel.inspection.detectedLayerPrefix ?? "Not detected"}</span>
              <strong>Detected Layer Indices</strong>
              <span>
                {fetchedModel.inspection.detectedLayerIndices.length > 0
                  ? `${fetchedModel.inspection.detectedLayerIndices.slice(0, 8).join(", ")}${
                      fetchedModel.inspection.detectedLayerIndices.length > 8 ? " ..." : ""
                    }`
                  : "None"}
              </span>
              <strong>Effective Layer Count</strong>
              <span>{actualLayerCount || "Not detected"}</span>
              <strong>Sample Layer Keys</strong>
              <span>
                {fetchedModel.inspection.sampleLayerKeys.length > 0
                  ? fetchedModel.inspection.sampleLayerKeys.slice(0, 3).join(" | ")
                  : "None"}
              </span>
              <strong>Config Keys</strong>
              <span>{configKeys.slice(0, 10).join(", ")}{configKeys.length > 10 ? " ..." : ""}</span>
            </div>
          ) : null}
        </section>

        <section className="canvas-panel prune-panel">
          <div className="panel-header">
            <p className="eyebrow">Plan</p>
            <h2>Transformer Block Pruning</h2>
          </div>
          <div className="form-stack">
            <label className="field">
              <span>Quick Preset</span>
              <select value={selectedPreset} onChange={(event) => applyPreset(event.target.value as PrunePreset)}>
                <option value="none">No preset</option>
                <option value="drop-last-4">Drop Last 4 Blocks</option>
                <option value="drop-last-8">Drop Last 8 Blocks</option>
                <option value="keep-every-other">Keep Every Other Block</option>
                <option value="keep-first-half">Keep First Half</option>
              </select>
            </label>
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
            <label className="field">
              <span>Local Model Directory</span>
              <input
                type="text"
                value={inputDir}
                placeholder="Optional: /path/to/local/model-dir"
                onChange={(event) => setInputDir(event.target.value)}
              />
            </label>
            <label className="field">
              <span>Cache Directory</span>
              <input
                type="text"
                value={cacheDir}
                placeholder="Optional: ~/.cache/neural-playground/hf"
                onChange={(event) => setCacheDir(event.target.value)}
              />
            </label>
            <label className="field">
              <span>Output Directory</span>
              <input
                type="text"
                value={outputDir}
                placeholder="/path/to/output/pruned-model"
                onChange={(event) => setOutputDir(event.target.value)}
              />
            </label>
          </div>
          {availableLayerIndices.length > 0 ? (
            <>
              <div className="layer-picker-header">
                <span>{availableLayerIndices.length} total · {droppedLayerIndices.length} dropped · {keptLayerIndices.length} kept</span>
                <div className="layer-picker-actions">
                  <button type="button" className="inline-button" onClick={clearDroppedLayers}>Clear All</button>
                  <button type="button" className="inline-button" onClick={dropAllLayers}>Drop All</button>
                </div>
              </div>
              <div className="layer-picker">
                {availableLayerIndices.map((layerIndex) => {
                  const dropped = droppedLayerIndices.includes(layerIndex);
                  return (
                    <button
                      type="button"
                      key={layerIndex}
                      className={`layer-chip ${dropped ? "dropped" : "kept"}`}
                      onClick={() => toggleLayer(layerIndex)}
                    >
                      {dropped ? "Drop" : "Keep"} L{layerIndex}
                    </button>
                  );
                })}
              </div>
            </>
          ) : (
            <div className="prune-empty-state">
              <strong>No layers loaded yet</strong>
              <span>Fetch model metadata to view and toggle layer chips.</span>
            </div>
          )}
          <div className="prune-summary prune-kv">
            <strong>Current Scope</strong>
            <span>Whole transformer block removal only. Embeddings, heads, and non-block modules are preserved.</span>
            <strong>Output Artifacts</strong>
            <span>`config.json`, pruning manifest, checkpoint remap script</span>
            <strong>Target Layers</strong>
            <span>{keptLayerCount}</span>
            <strong>Detected Layers</strong>
            <span>{actualLayerCount || "Not detected"}</span>
            <strong>Dropped Layers</strong>
            <span>{droppedLayerIndices.length > 0 ? droppedLayerIndices.join(", ") : "None"}</span>
            <strong>Kept Layers</strong>
            <span>{keptLayerIndices.length > 0 ? keptLayerIndices.slice(0, 12).join(", ") : "None"}{keptLayerIndices.length > 12 ? " ..." : ""}</span>
            <strong>Fetch Status</strong>
            <span>{fetchStatusLabel}</span>
            <strong>Readiness</strong>
            <span>{pruningReadinessMessage}</span>
          </div>
          <div className="prune-contact-card">
            <span className="prune-contact-title">✦ Interested in running pruning?</span>
            <span>Reach out at <a href="mailto:naveen.research45@gmail.com">naveen.research45@gmail.com</a> — we'd love to set you up.</span>
          </div>
          <div className="canvas-actions prune-actions">
            <button
              type="button"
              className="ghost-button"
              disabled
            >
              Download Pruning Artifacts
            </button>
            <button
              type="button"
              className="danger-button"
              disabled
            >
              Run Local Prune
            </button>
          </div>
          {canRunLocalPrune && droppedLayerIndices.length > 0 ? (
            <p className="prune-run-warning">About to drop {droppedLayerIndices.length} layers from {fetchedModel!.modelId}. This writes files to disk and cannot be undone.</p>
          ) : null}
          {runError ? (
            <div className="prune-result">
              <strong>Local Prune Error</strong>
              <span>{runError}</span>
            </div>
          ) : null}
          {runResult?.validation.ok ? <p className="project-status">Pruned model validated successfully.</p> : null}
          {runResult ? (
            <div className="prune-result prune-kv">
              <strong>Prune Output</strong>
              <span>{runResult.outputDir}</span>
              <strong>Model Source</strong>
              <span>{runResult.downloadedFromHub ? `Downloaded automatically to ${runResult.inputDir}` : runResult.inputDir}</span>
              <strong>Validation</strong>
              <span>{runResult.validation.ok ? "Passed" : "Needs review"}</span>
              <strong>Layer Prefix Used</strong>
              <span>{runResult.layerPrefix ?? "Not detected"}</span>
              <strong>Layer Count Key Used</strong>
              <span>{runResult.layerCountKey ?? "Not updated"}</span>
              <strong>Written Files</strong>
              <span>{runResult.writtenFiles.join(", ")}</span>
              <strong>Copied Support Files</strong>
              <span>{runResult.copiedSupportFiles.length > 0 ? runResult.copiedSupportFiles.join(", ") : "None"}</span>
              <strong>Configured Layer Count</strong>
              <span>
                {runResult.validation.config.configuredLayerCount ?? "Not updated"} / expected {runResult.validation.config.expectedLayerCount}
              </span>
              <strong>Detected Pruned Prefix</strong>
              <span>{runResult.validation.checkpoint.detectedLayerPrefix ?? "Not detected"}</span>
              <strong>Pruned Layer Indices</strong>
              <span>
                {runResult.validation.checkpoint.detectedLayerIndices.length > 0
                  ? `${runResult.validation.checkpoint.detectedLayerIndices.slice(0, 12).join(", ")}${
                      runResult.validation.checkpoint.detectedLayerIndices.length > 12 ? " ..." : ""
                    }`
                  : "None"}
              </span>
              <strong>Original Config Load</strong>
              <span>{runResult.validation.load.originalConfigLoadOk ? "OK" : "Failed"}</span>
              <strong>Pruned Config Load</strong>
              <span>{runResult.validation.load.prunedConfigLoadOk ? "OK" : "Failed"}</span>
              <strong>Original Model Load</strong>
              <span>{runResult.validation.load.originalModelLoadOk ? "OK" : runResult.validation.load.skippedModelLoadReason ? "Skipped" : "Failed"}</span>
              <strong>Pruned Model Load</strong>
              <span>{runResult.validation.load.prunedModelLoadOk ? "OK" : runResult.validation.load.skippedModelLoadReason ? "Skipped" : "Failed"}</span>
              <strong>Validation Device</strong>
              <span>{runResult.validation.load.memory.device}</span>
              <strong>Original Model Memory</strong>
              <span>
                {runResult.validation.load.memory.originalModelMiB !== null
                  ? `${runResult.validation.load.memory.originalModelMiB} MiB`
                  : "Unavailable"}
              </span>
              <strong>Pruned Model Memory</strong>
              <span>
                {runResult.validation.load.memory.prunedModelMiB !== null
                  ? `${runResult.validation.load.memory.prunedModelMiB} MiB`
                  : "Unavailable"}
              </span>
              <strong>Memory Saved</strong>
              <span>
                {memoryDeltaMiB !== null
                  ? `${memoryDeltaMiB} MiB${memoryDeltaPercent !== null ? ` (${memoryDeltaPercent}%)` : ""}`
                  : "Unavailable"}
              </span>
              {runResult.validation.load.skippedModelLoadReason ? (
                <>
                  <strong>Load Skip Reason</strong>
                  <span>{runResult.validation.load.skippedModelLoadReason}</span>
                </>
              ) : null}
              {runResult.validation.load.warnings.length > 0 ? (
                <>
                  <strong>Validation Warnings</strong>
                  <span>{runResult.validation.load.warnings.join(" | ")}</span>
                </>
              ) : null}
            </div>
          ) : null}
        </section>

        <section className="issues-panel prune-panel">
          <div className="panel-header row">
            <div>
              <p className="eyebrow">Progress</p>
              <h2>Pruning Progress</h2>
            </div>
            {runLogs.length > 0 ? (
              <button type="button" className="inline-button" onClick={clearRunLogs}>
                Clear
              </button>
            ) : null}
          </div>
          {runLogs.length === 0 ? (
            <div className="prune-empty-state">
              <strong>No prune run yet</strong>
              <span>Run local prune to stream backend progress logs here.</span>
            </div>
          ) : null}
          <div className="prune-log-list" ref={logsContainerRef}>
            {runLogs.map((entry) => (
              <div key={entry.id} className="prune-log-entry">
                <div className="prune-log-header">
                  <strong>{entry.event.stage}</strong>
                  <span className="prune-log-time">{formatLogOffset(entry.receivedAt)}</span>
                </div>
                <p>{entry.event.message}</p>
              </div>
            ))}
          </div>
          {runLogs.length > 0 ? <p className="muted">Log entries: {runLogs.length}</p> : null}
        </section>

        <section className="issues-panel prune-panel">
          <div className="panel-header">
            <p className="eyebrow">Remap</p>
            <h2>Layer Remap Preview</h2>
          </div>
          {layerRemap.length === 0 ? (
            <div className="prune-empty-state">
              <strong>No remap yet</strong>
              <span>Fetch a model and keep at least one layer to preview remapped indices.</span>
            </div>
          ) : null}
          <div className="prune-remap-list">
            {layerRemap.slice(0, 16).map((entry) => (
              <div key={`${entry.newIndex}-${entry.oldIndex}`} className="prune-remap-entry">
                <div className="prune-log-header">
                  <strong>New Layer {entry.newIndex}</strong>
                </div>
                <p>Copies weights from original layer {entry.oldIndex}.</p>
              </div>
            ))}
          </div>
          {layerRemap.length > 16 ? <p className="muted">Showing first 16 remap entries of {layerRemap.length}.</p> : null}
        </section>
      </div>
    </div>
  );
}
