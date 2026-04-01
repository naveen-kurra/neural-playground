import type { BlockEdge } from "@neural-playground/block-schema";

type ConnectionsPanelProps = {
  edges: BlockEdge[];
  onRemoveEdge: (edgeId: string) => void;
};

export function ConnectionsPanel(props: ConnectionsPanelProps) {
  const { edges, onRemoveEdge } = props;

  return (
    <section className="connections-panel">
      <div className="panel-header row">
        <div>
          <p className="eyebrow">Graph</p>
          <h2>Connections</h2>
        </div>
      </div>

      <p className="panel-copy">Connections are now created directly on the canvas using block handles.</p>

      <div className="edge-list">
        {edges.length === 0 ? <p className="muted">No connections yet.</p> : null}
        {edges.map((edge) => (
          <div key={edge.id} className="edge-row">
            <span>
              {edge.source} → {edge.target}
            </span>
            <button type="button" className="inline-button" onClick={() => onRemoveEdge(edge.id)}>
              Remove
            </button>
          </div>
        ))}
      </div>
    </section>
  );
}
