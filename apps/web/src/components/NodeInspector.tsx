import { type BlockDefinition, type BlockField, type BlockNode } from "@neural-playground/block-schema";
import { fieldValueAsString } from "../app/canvas";

type NodeInspectorProps = {
  node: BlockNode;
  definition: BlockDefinition;
  onChange: (nodeId: string, field: BlockField, rawValue: string) => void;
  onDelete: (nodeId: string) => void;
};

export function NodeInspector(props: NodeInspectorProps) {
  const { node, definition, onChange, onDelete } = props;
  const feedforwardType = String(node.config.feedforwardType ?? "mlp");

  const visibleFields = definition.fields.filter((field) => {
    if (node.type === "GPT2Block") {
      if (field.key === "ffnHidden") {
        return feedforwardType !== "moe";
      }
      if (field.key === "numExperts" || field.key === "topK" || field.key === "expertHidden") {
        return feedforwardType === "moe";
      }
    }

    return true;
  });

  return (
    <section className="inspector-panel">
      <div className="panel-header">
        <p className="eyebrow">Inspector</p>
        <h2>{definition.label}</h2>
      </div>
      <p className="panel-copy">{definition.description}</p>

      <div className="form-stack">
        {visibleFields.map((field) => {
          const value = node.config[field.key] ?? field.defaultValue;
          return (
            <label key={field.key} className="field">
              <span>{field.label}</span>
              {field.type === "select" ? (
                <select value={fieldValueAsString(value)} onChange={(event) => onChange(node.id, field, event.target.value)}>
                  {field.options?.map((option) => (
                    <option key={option} value={option}>
                      {option}
                    </option>
                  ))}
                </select>
              ) : field.type === "boolean" ? (
                <select value={fieldValueAsString(value)} onChange={(event) => onChange(node.id, field, event.target.value)}>
                  <option value="true">True</option>
                  <option value="false">False</option>
                </select>
              ) : (
                <input
                  type={field.type === "number" ? "number" : "text"}
                  value={fieldValueAsString(value)}
                  onChange={(event) => onChange(node.id, field, event.target.value)}
                />
              )}
            </label>
          );
        })}
      </div>

      <div className="node-summary">
        <strong>Inputs</strong>
        <span>{definition.inputs.join(", ") || "none"}</span>
        <strong>Outputs</strong>
        <span>{definition.outputs.join(", ") || "none"}</span>
      </div>

      <button type="button" className="danger-button" onClick={() => onDelete(node.id)}>
        Delete Node
      </button>
    </section>
  );
}
