import { useState } from "react";
import { getBlockDefinition } from "@neural-playground/block-schema";
import type { useHistory } from "./useHistory";

type GraphState = ReturnType<typeof useHistory>["state"];

type UseConnectionsOptions = {
  state: GraphState;
  push: (next: GraphState) => void;
};

export function useConnections({ state, push }: UseConnectionsOptions) {
  const { nodes, edges } = state;
  const [pendingConnectionSourceId, setPendingConnectionSourceId] = useState<string | null>(null);
  const [connectionError, setConnectionError] = useState("");

  function beginConnection(nodeId: string) {
    setConnectionError("");
    setPendingConnectionSourceId(nodeId);
  }

  function completeConnection(targetNodeId: string) {
    if (!pendingConnectionSourceId) return;

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
    const compatible = sourceDefinition.outputs.some((shape) => targetDefinition.inputs.includes(shape));
    if (!compatible) {
      setConnectionError(`${sourceDefinition.label} cannot connect to ${targetDefinition.label}.`);
      setPendingConnectionSourceId(null);
      return;
    }

    const alreadyExists = edges.some((edge) => edge.source === pendingConnectionSourceId && edge.target === targetNodeId);
    if (alreadyExists) {
      setConnectionError("That connection already exists.");
      setPendingConnectionSourceId(null);
      return;
    }

    push({
      ...state,
      edges: [
        ...edges,
        {
          id: `edge-${crypto.randomUUID().slice(0, 8)}`,
          source: pendingConnectionSourceId,
          target: targetNodeId
        }
      ]
    });
    setConnectionError("");
    setPendingConnectionSourceId(null);
  }

  function cancelConnection() {
    setPendingConnectionSourceId(null);
    setConnectionError("");
  }

  function resetConnections() {
    setPendingConnectionSourceId(null);
    setConnectionError("");
  }

  return {
    pendingConnectionSourceId,
    connectionError,
    beginConnection,
    completeConnection,
    cancelConnection,
    resetConnections
  };
}
