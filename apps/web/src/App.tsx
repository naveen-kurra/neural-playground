import { useCallback, useEffect, useMemo, useRef, useState, type ChangeEvent, type PointerEvent as ReactPointerEvent } from "react";
import { getBlockDefinition, type BlockEdge, type BlockField, type BlockNode, type ModelGraph } from "@neural-playground/block-schema";
import {
  exportGPT2IrProjectFiles,
  exportGPT2IrToPyTorch,
  exportHybridIrProjectFiles,
  exportHybridIrToPyTorch,
  exportLlamaIrProjectFiles,
  exportLlamaIrToPyTorch,
  exportModelGraphToPyTorch,
  exportProjectFiles
} from "@neural-playground/exporter-pytorch";
import {
  buildGPT2ArchitectureSpec,
  buildLlamaArchitectureSpec,
  mapModelGraphToGPT2Ir,
  mapModelGraphToHybridIr,
  mapModelGraphToLlamaIr,
  projectGPT2IrToModelGraph,
  projectLlamaIrToModelGraph
} from "@neural-playground/ir-schema";
import JSZip from "jszip";
import {
  inferSequenceDimensions,
  validateConfigCompatibility,
  validateEdge,
  validateGraph,
  validateTopology,
  type ValidationIssue
} from "@neural-playground/validator";
import { createNode, defaultTrainingConfig } from "./app/defaults";
import { normalizeGraphForGenericExport } from "./app/export-normalization";
import { downloadBlobFile, downloadTextFile } from "./app/file-utils";
import { resolveTemplate } from "./app/model-templates";
import { getRouteFromHash, navigateToRoute, type AppRoute } from "./app/navigation";
import { formatParameterCount } from "./app/parameter-estimator";
import { graphPresets } from "./app/presets";
import { readProjectFromInput, validateProjectDocument } from "./app/project";
import type { CopyStatus, ExportPreview, ProjectDocument, SafeExport, SelectedState } from "./app/types";
import { formatValidationIssue } from "./app/validation";
import { CanvasPanel } from "./components/CanvasPanel";
import { ExportPanel } from "./components/ExportPanel";
import { NodeInspector } from "./components/NodeInspector";
import { PaletteSidebar } from "./components/PaletteSidebar";
import { PrunePage } from "./components/PrunePage";
import { TrainingInspector } from "./components/TrainingInspector";

type HistorySnapshot = {
  nodes: BlockNode[];
  edges: BlockEdge[];
  training: ModelGraph["training"];
};

type JsonEditorValidation = {
  ok: boolean;
  message: string;
  graph: ModelGraph | null;
};

const UNIVERSAL_EXPORT_BLOCKING_CODES = new Set([
  "graph_cycle",
  "export_input_count",
  "export_output_count",
  "decoder_output_head",
  "dangling_edge",
  "shape_mismatch",
  "dimension_mismatch",
  "missing_input",
  "missing_output",
  "missing_incoming_edge",
  "missing_outgoing_edge",
  "node_input_arity",
  "node_output_arity",
  "config_dim_mismatch",
  "config_vocab_mismatch",
  "hidden_before_embedding",
  "hidden_after_output",
  "output_before_embedding",
  "add_input_arity",
  "softmax_axis_invalid",
  "classifier_output_unimplemented"
]);

export function App() {
  const [route, setRoute] = useState<AppRoute>(() => getRouteFromHash(window.location.hash));
  const initialNodes = useMemo(
    () => [createNode("Input", 0), createNode("Embedding", 1), createNode("TransformerBlock", 2), createNode("Output", 3)],
    []
  );
  const initialEdges = useMemo<BlockEdge[]>(
    () => [
      { id: "edge-1", source: initialNodes[0]!.id, target: initialNodes[1]!.id },
      { id: "edge-2", source: initialNodes[1]!.id, target: initialNodes[2]!.id },
      { id: "edge-3", source: initialNodes[2]!.id, target: initialNodes[3]!.id }
    ],
    [initialNodes]
  );
  const initialTraining = useMemo(() => defaultTrainingConfig(), []);
  const [nodes, setNodes] = useState<BlockNode[]>(initialNodes);
  const [edges, setEdges] = useState<BlockEdge[]>(initialEdges);
  const [training, setTraining] = useState(initialTraining);
  const [selected, setSelected] = useState<SelectedState>({ kind: "training" });
  const [draggingNodeId, setDraggingNodeId] = useState<string | null>(null);
  const [selectedEdgeId, setSelectedEdgeId] = useState<string | null>(null);
  const [pendingConnectionSourceId, setPendingConnectionSourceId] = useState<string | null>(null);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const zoomRef = useRef(zoom);
  const panRef = useRef(pan);
  useEffect(() => { zoomRef.current = zoom; }, [zoom]);
  useEffect(() => { panRef.current = pan; }, [pan]);
  const [connectionError, setConnectionError] = useState("");
  const [exportPreview, setExportPreview] = useState<ExportPreview>(null);
  const [copyStatus, setCopyStatus] = useState<CopyStatus>("idle");
  const [projectStatus, setProjectStatus] = useState("");
  const [modelTemplateSelection, setModelTemplateSelection] = useState("GPT-2");
  const [templateBlockCount, setTemplateBlockCount] = useState(12);
  const [searchQuery, setSearchQuery] = useState("");
  const [jsonEditorText, setJsonEditorText] = useState("");
  const [jsonEditorValidation, setJsonEditorValidation] = useState<JsonEditorValidation>({
    ok: false,
    message: "Open Graph JSON to edit and apply changes.",
    graph: null
  });
  const dragStateRef = useRef<{ nodeId: string; pointerOffsetX: number; pointerOffsetY: number } | null>(null);
  const dragMovedRef = useRef(false);
  const canvasRef = useRef<HTMLDivElement | null>(null);
  const loadInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    function handleHashChange() {
      setRoute(getRouteFromHash(window.location.hash));
    }

    window.addEventListener("hashchange", handleHashChange);
    return () => window.removeEventListener("hashchange", handleHashChange);
  }, []);

  // History for undo/redo — all refs so event-listener closures are always fresh
  const historyRef = useRef<HistorySnapshot[]>([
    { nodes: initialNodes, edges: initialEdges, training: initialTraining }
  ]);
  const historyPosRef = useRef(0);
  const [historyPos, setHistoryPos] = useState(0); // drives canUndo/canRedo re-renders
  const nodesRef = useRef(initialNodes);
  const edgesRef = useRef(initialEdges);
  const trainingRef = useRef(initialTraining);
  useEffect(() => { nodesRef.current = nodes; }, [nodes]);
  useEffect(() => { edgesRef.current = edges; }, [edges]);
  useEffect(() => { trainingRef.current = training; }, [training]);

  const pushHistory = useCallback((nextNodes: BlockNode[], nextEdges: BlockEdge[], nextTraining: ModelGraph["training"] = trainingRef.current) => {
    const trimmed = historyRef.current.slice(0, historyPosRef.current + 1);
    trimmed.push({ nodes: nextNodes, edges: nextEdges, training: nextTraining });
    if (trimmed.length > 100) trimmed.shift();
    historyRef.current = trimmed;
    historyPosRef.current = trimmed.length - 1;
    setHistoryPos(trimmed.length - 1);
  }, []);

  const undo = useCallback(() => {
    if (historyPosRef.current <= 0) return;
    const newPos = historyPosRef.current - 1;
    const snap = historyRef.current[newPos]!;
    setNodes(snap.nodes);
    setEdges(snap.edges);
    setTraining(snap.training);
    historyPosRef.current = newPos;
    setHistoryPos(newPos);
    setSelected({ kind: "training" });
  }, []);

  const redo = useCallback(() => {
    if (historyPosRef.current >= historyRef.current.length - 1) return;
    const newPos = historyPosRef.current + 1;
    const snap = historyRef.current[newPos]!;
    setNodes(snap.nodes);
    setEdges(snap.edges);
    setTraining(snap.training);
    historyPosRef.current = newPos;
    setHistoryPos(newPos);
  }, []);

  const canUndo = historyPos > 0;
  const canRedo = historyPos < historyRef.current.length - 1;
  const graph: ModelGraph = useMemo(() => ({ nodes, edges, training }), [nodes, edges, training]);
  const parameterSummary = useMemo(() => formatParameterCount(graph), [graph]);
  const issues = useMemo(() => validateGraph(graph), [graph]);
  const normalizedExportGraph = useMemo(() => normalizeGraphForGenericExport(graph), [graph]);
  const pytorchExportIssues = useMemo(
    () => validateGraph(normalizedExportGraph.ok ? normalizedExportGraph.value : graph, "pytorch-export-valid"),
    [graph, normalizedExportGraph]
  );
  const decoderExportIssues = useMemo(
    () => validateGraph(normalizedExportGraph.ok ? normalizedExportGraph.value : graph, "decoder-training-valid"),
    [graph, normalizedExportGraph]
  );
  const universalExportIssues = useMemo(
    () => validateGraph(graph, "decoder-training-valid").filter((issue) => UNIVERSAL_EXPORT_BLOCKING_CODES.has(issue.code)),
    [graph]
  );
  const selectedNode = selected?.kind === "node" ? nodes.find((node) => node.id === selected.nodeId) ?? null : null;

  const trainingWarnings = useMemo(() => {
    const warnings: string[] = [];
    if (training.optimizer === "Custom" && !(training.optimizerCustomName ?? "").trim()) {
      warnings.push("Custom optimizer selected but no optimizer name is provided.");
    }
    if (training.loss === "Custom" && !(training.lossCustomName ?? "").trim()) {
      warnings.push("Custom loss selected but no loss name is provided.");
    }
    if (training.activation === "Custom" && !(training.activationCustomName ?? "").trim()) {
      warnings.push("Custom activation selected but no activation name is provided.");
    }
    return warnings;
  }, [training]);

  const gpt2Ir = useMemo(() => {
    try {
      return { ok: true as const, value: mapModelGraphToGPT2Ir(graph) };
    } catch (error) {
      return { ok: false as const, error: error instanceof Error ? error.message : "Unknown GPT-2 mapping error" };
    }
  }, [graph]);

  const llamaIr = useMemo(() => {
    try {
      return { ok: true as const, value: mapModelGraphToLlamaIr(graph) };
    } catch (error) {
      return { ok: false as const, error: error instanceof Error ? error.message : "Unknown LLaMA mapping error" };
    }
  }, [graph]);

  const hybridIr = useMemo(() => {
    try {
      return { ok: true as const, value: mapModelGraphToHybridIr(graph) };
    } catch (error) {
      return { ok: false as const, error: error instanceof Error ? error.message : "Unknown hybrid mapping error" };
    }
  }, [graph]);

  // Merge export-level validation only when no specialized IR can handle the graph.
  // Specialized IRs (GPT-2, LLaMA, Hybrid) have their own embedding/topology rules
  // so generic "pytorch-export-valid" checks would produce false positives on them.
  const displayIssues = useMemo(() => {
    if (gpt2Ir.ok || llamaIr.ok || hybridIr.ok) return issues;
    const exportLevel = validateGraph(graph, "pytorch-export-valid");
    const seen = new Set<string>();
    const merged = [];
    for (const issue of [...issues, ...exportLevel]) {
      const key = `${issue.code}-${issue.message}-${issue.nodeId ?? ""}-${issue.edgeId ?? ""}`;
      if (!seen.has(key)) { seen.add(key); merged.push(issue); }
    }
    return merged;
  }, [graph, issues, gpt2Ir.ok, llamaIr.ok, hybridIr.ok]);

  const formattedIssues = useMemo(() => displayIssues.map((i) => ({ raw: i, fmt: formatValidationIssue(i) })), [displayIssues]);
  const errorCount = formattedIssues.filter((i) => i.fmt.severity === "error").length;
  const warningCount = formattedIssues.filter((i) => i.fmt.severity === "warning").length;

  function summarizeValidationIssues(validationIssues: ValidationIssue[]): string {
    if (validationIssues.length === 0) {
      return "";
    }

    if (validationIssues.length === 1) {
      return validationIssues[0]!.message;
    }

    return `${validationIssues[0]!.message} (${validationIssues.length - 1} more issue${validationIssues.length > 2 ? "s" : ""})`;
  }

  function validateJsonEditorContents(contents: string): JsonEditorValidation {
    const trimmed = contents.trim();
    if (!trimmed) {
      return { ok: false, message: "JSON editor is empty.", graph: null };
    }

    let parsed: unknown;
    try {
      parsed = JSON.parse(trimmed);
    } catch (error) {
      return {
        ok: false,
        message: error instanceof Error ? `JSON parse error: ${error.message}` : "JSON parse error.",
        graph: null
      };
    }

    try {
      const document = validateProjectDocument(parsed);
      const validationIssues = validateGraph(document.graph);
      const validationErrors = validationIssues.filter((issue) => formatValidationIssue(issue).severity === "error");
      if (validationErrors.length > 0) {
        return {
          ok: false,
          message: `Graph validation failed: ${summarizeValidationIssues(validationErrors)}`,
          graph: null
        };
      }

      return {
        ok: true,
        message: validationIssues.length > 0
          ? `JSON is valid and can be applied (${validationIssues.length} warning${validationIssues.length === 1 ? "" : "s"}).`
          : "JSON is valid and can be applied.",
        graph: document.graph
      };
    } catch (error) {
      return {
        ok: false,
        message: error instanceof Error ? error.message : "Invalid project JSON.",
        graph: null
      };
    }
  }

  const exportedPyTorch = useMemo<SafeExport<string>>(() => {
    if (universalExportIssues.length > 0) {
      return {
        ok: false,
        error: summarizeValidationIssues(universalExportIssues)
      };
    }
    if (!gpt2Ir.ok && !llamaIr.ok && !hybridIr.ok && (!normalizedExportGraph.ok || pytorchExportIssues.length > 0)) {
      return {
        ok: false,
        error: normalizedExportGraph.ok ? summarizeValidationIssues(pytorchExportIssues) : normalizedExportGraph.error
      };
    }

    try {
      if (gpt2Ir.ok) {
        return { ok: true, value: exportGPT2IrToPyTorch(gpt2Ir.value) };
      }
      if (llamaIr.ok) {
        return { ok: true, value: exportLlamaIrToPyTorch(llamaIr.value) };
      }
      if (hybridIr.ok) {
        return { ok: true, value: exportHybridIrToPyTorch(hybridIr.value) };
      }
      return { ok: true, value: exportModelGraphToPyTorch(normalizedExportGraph.ok ? normalizedExportGraph.value : graph) };
    } catch (error) {
      return {
        ok: false,
        error: error instanceof Error ? error.message : "Unknown export error"
      };
    }
  }, [graph, pytorchExportIssues, gpt2Ir, llamaIr, hybridIr, normalizedExportGraph, universalExportIssues]);

  const exportedJson = useMemo(() => JSON.stringify(graph, null, 2), [graph]);
  const exportedProject = useMemo<SafeExport<ReturnType<typeof exportProjectFiles>>>(() => {
    if (universalExportIssues.length > 0) {
      return {
        ok: false,
        error: summarizeValidationIssues(universalExportIssues)
      };
    }
    if (trainingWarnings.length > 0) {
      return { ok: false, error: trainingWarnings[0]! };
    }
    if (!gpt2Ir.ok && !llamaIr.ok && !hybridIr.ok && (!normalizedExportGraph.ok || decoderExportIssues.length > 0)) {
      return {
        ok: false,
        error: normalizedExportGraph.ok ? summarizeValidationIssues(decoderExportIssues) : normalizedExportGraph.error
      };
    }
    try {
      if (gpt2Ir.ok) {
        return { ok: true, value: exportGPT2IrProjectFiles(gpt2Ir.value, training) };
      }
      if (llamaIr.ok) {
        return { ok: true, value: exportLlamaIrProjectFiles(llamaIr.value, training) };
      }
      if (hybridIr.ok) {
        return { ok: true, value: exportHybridIrProjectFiles(hybridIr.value, training) };
      }
      return { ok: true, value: exportProjectFiles(normalizedExportGraph.ok ? normalizedExportGraph.value : graph) };
    } catch (error) {
      return { ok: false, error: error instanceof Error ? error.message : "Unknown project export error" };
    }
  }, [graph, trainingWarnings, decoderExportIssues, gpt2Ir, llamaIr, hybridIr, training, normalizedExportGraph, universalExportIssues]);

  useEffect(() => {
    const template = resolveTemplate(modelTemplateSelection);
    if (template) {
      setTemplateBlockCount(template.defaultBlockCount);
    }
  }, [modelTemplateSelection]);

  useEffect(() => {
    function handlePointerMove(event: PointerEvent) {
      const dragState = dragStateRef.current;
      const canvas = canvasRef.current;
      if (!dragState || !canvas) {
        return;
      }

      const bounds = canvas.getBoundingClientRect();
      const currentZoom = zoomRef.current;
      const currentPan = panRef.current;
      const nextX = (event.clientX - bounds.left - currentPan.x) / currentZoom - dragState.pointerOffsetX;
      const nextY = (event.clientY - bounds.top - currentPan.y) / currentZoom - dragState.pointerOffsetY;

      dragMovedRef.current = true;
      setNodes((current) =>
        current.map((node) =>
          node.id === dragState.nodeId
            ? {
                ...node,
                position: {
                  x: Math.max(16, nextX),
                  y: Math.max(16, nextY)
                }
              }
            : node
        )
      );
    }

    function handlePointerUp() {
      if (dragStateRef.current && dragMovedRef.current) {
        pushHistory(nodesRef.current, edgesRef.current);
        dragMovedRef.current = false;
      }
      dragStateRef.current = null;
      setDraggingNodeId(null);
    }

    window.addEventListener("pointermove", handlePointerMove);
    window.addEventListener("pointerup", handlePointerUp);
    return () => {
      window.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("pointerup", handlePointerUp);
    };
  }, []);

  useEffect(() => {
    function handleKeyDown(event: KeyboardEvent) {
      const ctrl = event.metaKey || event.ctrlKey;
      if (!ctrl) return;
      if (event.key === "z" && !event.shiftKey) { event.preventDefault(); undo(); }
      else if ((event.key === "z" && event.shiftKey) || event.key === "y") { event.preventDefault(); redo(); }
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [undo, redo]);

  function addNode(type: BlockNode["type"]) {
    const nextNode = createNode(type, nodes.length);
    const nextNodes = [...nodes, nextNode];
    setNodes(nextNodes);
    pushHistory(nextNodes, edges);
    setSelected({ kind: "node", nodeId: nextNode.id });
    setConnectionError("");
    setProjectStatus("");
  }

  function applyPreset(presetId: string) {
    const preset = graphPresets.find((entry) => entry.id === presetId);
    if (!preset) {
      setProjectStatus("Preset not found.");
      return;
    }

    const presetGraph = preset.build();
    setNodes(presetGraph.nodes);
    setEdges(presetGraph.edges);
    setTraining(presetGraph.training);
    historyRef.current = [{ nodes: presetGraph.nodes, edges: presetGraph.edges, training: presetGraph.training }];
    historyPosRef.current = 0;
    setHistoryPos(0);
    setSelected({ kind: "training" });
    setPendingConnectionSourceId(null);
    setConnectionError("");
    setExportPreview(null);
    setCopyStatus("idle");
    setProjectStatus(`Loaded preset: ${preset.name}.`);
  }

  function importModelConfig() {
    const template = resolveTemplate(modelTemplateSelection);
    if (!template) {
      setProjectStatus("Choose a supported architecture template.");
      return;
    }

    try {
      const blockCount = Math.max(1, Math.floor(templateBlockCount || 0));
      if (template.family === "gpt2") {
        const spec = buildGPT2ArchitectureSpec({
          name: `${template.label} ${blockCount}-block`,
          modelId: template.modelId,
          numHiddenLayers: blockCount
        });
        const importedGraph = projectGPT2IrToModelGraph(spec);
        setNodes(importedGraph.nodes);
        setEdges(importedGraph.edges);
        setTraining(importedGraph.training);
        historyRef.current = [{ nodes: importedGraph.nodes, edges: importedGraph.edges, training: importedGraph.training }];
        historyPosRef.current = 0;
        setHistoryPos(0);
        setSelected({ kind: "training" });
        setPendingConnectionSourceId(null);
        setConnectionError("");
        setExportPreview(null);
        setCopyStatus("idle");
        setProjectStatus(`Loaded ${template.label} template with ${blockCount} block${blockCount === 1 ? "" : "s"}.`);
        return;
      }

      if (template.family === "llama") {
        const spec = buildLlamaArchitectureSpec({
          name: `${template.label} ${blockCount}-block`,
          modelId: template.modelId,
          numHiddenLayers: blockCount
        });
        const importedGraph = projectLlamaIrToModelGraph(spec);
        setNodes(importedGraph.nodes);
        setEdges(importedGraph.edges);
        setTraining(importedGraph.training);
        historyRef.current = [{ nodes: importedGraph.nodes, edges: importedGraph.edges, training: importedGraph.training }];
        historyPosRef.current = 0;
        setHistoryPos(0);
        setSelected({ kind: "training" });
        setPendingConnectionSourceId(null);
        setConnectionError("");
        setExportPreview(null);
        setCopyStatus("idle");
        setProjectStatus(`Loaded ${template.label} template with ${blockCount} block${blockCount === 1 ? "" : "s"}.`);
        return;
      }

      throw new Error(`Unsupported template family: ${template.family}`);
    } catch (error) {
      setProjectStatus(error instanceof Error ? error.message : "Failed to load template.");
    }
  }

  function updateNodeConfig(nodeId: string, field: BlockField, rawValue: string) {
    let nextValue: string | number | boolean = rawValue;
    if (field.type === "number") nextValue = Number(rawValue);
    else if (field.type === "boolean") nextValue = rawValue === "true";
    const nextNodes = nodes.map((node) =>
      node.id !== nodeId ? node : { ...node, config: { ...node.config, [field.key]: nextValue } }
    );
    setNodes(nextNodes);
    pushHistory(nextNodes, edges);
  }

  function removeNode(nodeId: string) {
    const nextNodes = nodes.filter((node) => node.id !== nodeId);
    const nextEdges = edges.filter((edge) => edge.source !== nodeId && edge.target !== nodeId);
    setNodes(nextNodes);
    setEdges(nextEdges);
    pushHistory(nextNodes, nextEdges);
    setSelected({ kind: "training" });
    setProjectStatus("");
  }

  function removeEdge(edgeId: string) {
    const nextEdges = edges.filter((edge) => edge.id !== edgeId);
    setEdges(nextEdges);
    pushHistory(nodes, nextEdges);
    setProjectStatus("");
  }

  function beginConnection(nodeId: string) {
    setConnectionError("");
    setPendingConnectionSourceId(nodeId);
  }

  function completeConnection(targetNodeId: string) {
    if (!pendingConnectionSourceId) {
      return;
    }

    if (pendingConnectionSourceId === targetNodeId) {
      setPendingConnectionSourceId(null);
      setConnectionError("Source and target must be different blocks.");
      return;
    }

    const sourceNode = nodes.find((node) => node.id === pendingConnectionSourceId);
    const targetNode = nodes.find((node) => node.id === targetNodeId);
    if (!sourceNode || !targetNode) {
      setPendingConnectionSourceId(null);
      setConnectionError("Connection failed because one of the blocks no longer exists.");
      return;
    }

    const sourceDefinition = getBlockDefinition(sourceNode.type);
    const targetDefinition = getBlockDefinition(targetNode.type);
    const alreadyExists = edges.some((edge) => edge.source === pendingConnectionSourceId && edge.target === targetNodeId);
    if (alreadyExists) {
      setConnectionError("That connection already exists.");
      setPendingConnectionSourceId(null);
      return;
    }

    const nextEdge = {
      id: `edge-${crypto.randomUUID().slice(0, 8)}`,
      source: pendingConnectionSourceId,
      target: targetNodeId
    };
    const nextEdges = [...edges, nextEdge];
    const tentativeGraph: ModelGraph = { nodes, edges: nextEdges, training };
    const nodeIds = new Set(nodes.map((node) => node.id));
    const inferredSequenceDims = inferSequenceDimensions(tentativeGraph);
    const topologyIssues = validateTopology(tentativeGraph, "playground-valid");
    const edgeIssues = validateEdge(tentativeGraph, nextEdge, inferredSequenceDims, nodeIds);
    const configIssues = validateConfigCompatibility(tentativeGraph).filter(
      (issue) => issue.edgeId === nextEdge.id || issue.nodeId === targetNode.id
    );
    const connectTimeIssues = [...topologyIssues, ...edgeIssues, ...configIssues];

    if (connectTimeIssues.length > 0) {
      const primaryIssue = connectTimeIssues[0]!;
      let message = primaryIssue.message;

      if (primaryIssue.code === "shape_mismatch") {
        message = `${sourceDefinition.label} cannot connect to ${targetDefinition.label}.`;
      } else if (primaryIssue.code === "graph_cycle") {
        message = "That connection would create a cycle, which is not supported.";
      }

      setConnectionError(message);
      setPendingConnectionSourceId(null);
      return;
    }

    setEdges(nextEdges);
    pushHistory(nodes, nextEdges);
    setConnectionError("");
    setPendingConnectionSourceId(null);
  }

  function cancelPendingConnection() {
    setPendingConnectionSourceId(null);
    setConnectionError("");
  }

  function preventFocusScroll(event: ReactPointerEvent<HTMLElement>) {
    event.preventDefault();
    event.stopPropagation();
  }

  function saveProject() {
    const document: ProjectDocument = {
      version: 1,
      graph
    };
    downloadTextFile("neural-playground.project.json", JSON.stringify(document, null, 2));
    setProjectStatus("Project saved.");
  }

  async function loadProjectFromFile(event: ChangeEvent<HTMLInputElement>) {
    try {
      const result = await readProjectFromInput(event);
      if (!result) {
        return;
      }

      setNodes(result.document.graph.nodes);
      setEdges(result.document.graph.edges);
      setTraining(result.document.graph.training);
      historyRef.current = [{
        nodes: result.document.graph.nodes,
        edges: result.document.graph.edges,
        training: result.document.graph.training
      }];
      historyPosRef.current = 0;
      setHistoryPos(0);
      setSelected({ kind: "training" });
      setPendingConnectionSourceId(null);
      setConnectionError("");
      setExportPreview(null);
      setCopyStatus("idle");
      setProjectStatus(`Loaded ${result.fileName}.`);
    } catch (error) {
      setProjectStatus(error instanceof Error ? error.message : "Failed to load project.");
    } finally {
      event.target.value = "";
    }
  }

  async function copyText(contents: string, artifact: "json" | "pytorch") {
    try {
      await navigator.clipboard.writeText(contents);
      setCopyStatus(artifact === "json" ? "json-copied" : "pytorch-copied");
    } catch {
      setCopyStatus("copy-failed");
    }
  }

  async function downloadProjectArchive() {
    if (!exportedProject.ok) {
      setCopyStatus("copy-failed");
      return;
    }

    try {
      const zip = new JSZip();
      for (const [path, contents] of Object.entries(exportedProject.value)) {
        zip.file(path, contents);
      }
      const blob = await zip.generateAsync({ type: "blob" });
      await downloadBlobFile("neural-playground-export.zip", blob);
      setCopyStatus("project-downloaded");
    } catch {
      setCopyStatus("copy-failed");
    }
  }

  function openExportPreview(preview: ExportPreview) {
    if (preview === "json") {
      setJsonEditorText(exportedJson);
      setJsonEditorValidation(validateJsonEditorContents(exportedJson));
    }
    setExportPreview(preview);
  }

  function handleJsonEditorChange(nextContents: string) {
    setJsonEditorText(nextContents);
    setJsonEditorValidation(validateJsonEditorContents(nextContents));
  }

  function applyJsonEditorChanges() {
    if (!jsonEditorValidation.ok || !jsonEditorValidation.graph) {
      return;
    }

    const nextGraph = jsonEditorValidation.graph;
    setNodes(nextGraph.nodes);
    setEdges(nextGraph.edges);
    setTraining(nextGraph.training);
    pushHistory(nextGraph.nodes, nextGraph.edges, nextGraph.training);
    setSelected({ kind: "training" });
    setSelectedEdgeId(null);
    setPendingConnectionSourceId(null);
    setConnectionError("");
    setCopyStatus("idle");
    setProjectStatus("Applied Graph JSON changes.");

    const normalized = JSON.stringify(nextGraph, null, 2);
    setJsonEditorText(normalized);
    setJsonEditorValidation(validateJsonEditorContents(normalized));
  }

  function updateTraining(nextTraining: ModelGraph["training"]) {
    setTraining(nextTraining);
    pushHistory(nodesRef.current, edgesRef.current, nextTraining);
  }

  function fitToScreen() {
    const canvas = canvasRef.current;
    if (!canvas || nodes.length === 0) return;

    const NODE_W = 200;
    const NODE_H = 80;
    const PADDING = 48;

    const minX = Math.min(...nodes.map((n) => n.position.x));
    const minY = Math.min(...nodes.map((n) => n.position.y));
    const maxX = Math.max(...nodes.map((n) => n.position.x + NODE_W));
    const maxY = Math.max(...nodes.map((n) => n.position.y + NODE_H));

    const contentW = maxX - minX;
    const contentH = maxY - minY;
    const canvasW = canvas.clientWidth;
    const canvasH = canvas.clientHeight;

    const newZoom = Math.min(
      (canvasW - PADDING * 2) / contentW,
      (canvasH - PADDING * 2) / contentH,
      3
    );
    const clampedZoom = Math.max(0.15, newZoom);

    const centerX = minX + contentW / 2;
    const centerY = minY + contentH / 2;

    setZoom(clampedZoom);
    setPan({
      x: canvasW / 2 - centerX * clampedZoom,
      y: canvasH / 2 - centerY * clampedZoom
    });
  }

  function startDraggingNode(event: ReactPointerEvent<HTMLButtonElement>, node: BlockNode) {
    if (!canvasRef.current) {
      return;
    }

    const nodeBounds = event.currentTarget.getBoundingClientRect();
    const currentZoom = zoomRef.current;
    dragStateRef.current = {
      nodeId: node.id,
      pointerOffsetX: (event.clientX - nodeBounds.left) / currentZoom,
      pointerOffsetY: (event.clientY - nodeBounds.top) / currentZoom
    };
    setDraggingNodeId(node.id);
    setSelected({ kind: "node", nodeId: node.id });
    event.currentTarget.setPointerCapture(event.pointerId);
  }

  if (route === "prune") {
    return <PrunePage onBackToBuilder={() => navigateToRoute("builder")} />;
  }

  return (
    <div className="app-shell">
      <PaletteSidebar
        projectStatus={projectStatus}
        modelTemplateSelection={modelTemplateSelection}
        templateBlockCount={templateBlockCount}
        searchQuery={searchQuery}
        loadInputRef={loadInputRef}
        onAddNode={addNode}
        onApplyPreset={applyPreset}
        onModelTemplateSelectionChange={setModelTemplateSelection}
        onImportModel={importModelConfig}
        onTemplateBlockCountChange={setTemplateBlockCount}
        onSearchQueryChange={setSearchQuery}
        onSave={saveProject}
        onLoadClick={() => loadInputRef.current?.click()}
        onLoadFile={loadProjectFromFile}
        onOpenPruningTool={() => navigateToRoute("prune")}
      />

      <main className="workspace">
        <CanvasPanel
          canvasRef={canvasRef}
          parameterSummary={parameterSummary}
          nodes={nodes}
          edges={edges}
          selected={selected}
          draggingNodeId={draggingNodeId}
          pendingConnectionSourceId={pendingConnectionSourceId}
          connectionError={connectionError}
          selectedEdgeId={selectedEdgeId}
          zoom={zoom}
          pan={pan}
          onZoomChange={setZoom}
          onPanChange={setPan}
          canUndo={canUndo}
          canRedo={canRedo}
          onUndo={undo}
          onRedo={redo}
          onFitToScreen={fitToScreen}
          onCancelConnection={cancelPendingConnection}
          onShowTraining={() => setSelected({ kind: "training" })}
          onSelectNode={(nodeId) => { setSelected({ kind: "node", nodeId }); setSelectedEdgeId(null); }}
          onSelectEdge={setSelectedEdgeId}
          onRemoveEdge={removeEdge}
          onStartDrag={startDraggingNode}
          onBeginConnection={beginConnection}
          onCompleteConnection={completeConnection}
          onPreventHandleFocus={preventFocusScroll}
        />
      </main>

      <aside className="sidebar inspector">
        {selectedNode ? (
          <NodeInspector
            node={selectedNode}
            definition={getBlockDefinition(selectedNode.type)}
            onChange={updateNodeConfig}
            onDelete={removeNode}
          />
        ) : (
          <TrainingInspector training={training} warnings={trainingWarnings} onChange={updateTraining} />
        )}

        <section className="issues-panel">
          <div className="panel-header row">
            <div>
              <p className="eyebrow">Validation</p>
              <h2>Issues</h2>
            </div>
            {issues.length > 0 && (
              <div className="issues-counts">
                {errorCount > 0 && <span className="issues-count error">{errorCount} error{errorCount !== 1 ? "s" : ""}</span>}
                {warningCount > 0 && <span className="issues-count warning">{warningCount} warning{warningCount !== 1 ? "s" : ""}</span>}
              </div>
            )}
          </div>
          {issues.length === 0 ? <p className="success-copy">Graph looks valid at this layer.</p> : null}
          {formattedIssues.map(({ raw, fmt }) => (
            <div
              key={`${raw.code}-${raw.message}-${raw.nodeId ?? ""}-${raw.edgeId ?? ""}`}
              className={`issue-card ${fmt.severity}`}
            >
              <div className="issue-header">
                <strong>{fmt.title}</strong>
                <span className={`issue-badge ${fmt.severity}`}>{fmt.severity}</span>
              </div>
              <span>{fmt.message}</span>
              {(raw.nodeId ?? raw.edgeId) ? (
                <span className="issue-location">{raw.nodeId ? `Node: ${raw.nodeId}` : `Edge: ${raw.edgeId}`}</span>
              ) : null}
            </div>
          ))}
        </section>

        <ExportPanel
          exportedJson={exportedJson}
          exportedPyTorch={exportedPyTorch}
          exportedProject={exportedProject}
          exportPreview={exportPreview}
          copyStatus={copyStatus}
          jsonEditorValue={jsonEditorText}
          jsonEditorStatus={jsonEditorValidation.message}
          jsonEditorCanApply={jsonEditorValidation.ok}
          onJsonEditorChange={handleJsonEditorChange}
          onApplyJsonEditor={applyJsonEditorChanges}
          onOpenPreview={openExportPreview}
          onCopy={copyText}
          onDownloadText={downloadTextFile}
          onDownloadProject={downloadProjectArchive}
        />
      </aside>
    </div>
  );
}
