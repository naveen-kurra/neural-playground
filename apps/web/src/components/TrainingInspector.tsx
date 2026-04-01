import type { TrainingConfig } from "@neural-playground/block-schema";

type TrainingInspectorProps = {
  training: TrainingConfig;
  warnings: string[];
  onChange: (training: TrainingConfig) => void;
};

export function TrainingInspector(props: TrainingInspectorProps) {
  const { training, warnings, onChange } = props;

  return (
    <section className="inspector-panel">
      <div className="panel-header">
        <p className="eyebrow">Inspector</p>
        <h2>Training Config</h2>
      </div>
      <p className="panel-copy">These settings describe how the graph would be trained or exported later.</p>
      {warnings.map((warning) => (
        <p key={warning} className="training-warning">
          {warning}
        </p>
      ))}

      <div className="form-stack">
        <label className="field">
          <span>Optimizer</span>
          <select value={training.optimizer} onChange={(event) => onChange({ ...training, optimizer: event.target.value as TrainingConfig["optimizer"] })}>
            <option value="AdamW">AdamW</option>
            <option value="SGD">SGD</option>
            <option value="Custom">Custom</option>
          </select>
        </label>
        {training.optimizer === "Custom" ? (
          <label className="field">
            <span>Custom Optimizer Name</span>
            <input
              type="text"
              value={training.optimizerCustomName ?? ""}
              placeholder="my_optimizer"
              onChange={(event) => onChange({ ...training, optimizerCustomName: event.target.value })}
            />
          </label>
        ) : null}

        <label className="field">
          <span>Loss</span>
          <select value={training.loss} onChange={(event) => onChange({ ...training, loss: event.target.value as TrainingConfig["loss"] })}>
            <option value="CrossEntropy">CrossEntropy</option>
            <option value="Custom">Custom</option>
          </select>
        </label>
        {training.loss === "Custom" ? (
          <label className="field">
            <span>Custom Loss Name</span>
            <input
              type="text"
              value={training.lossCustomName ?? ""}
              placeholder="my_loss"
              onChange={(event) => onChange({ ...training, lossCustomName: event.target.value })}
            />
          </label>
        ) : null}

        <label className="field">
          <span>Activation</span>
          <select
            value={training.activation}
            onChange={(event) => onChange({ ...training, activation: event.target.value as TrainingConfig["activation"] })}
          >
            <option value="GELU">GELU</option>
            <option value="ReLU">ReLU</option>
            <option value="SiLU">SiLU</option>
            <option value="Custom">Custom</option>
          </select>
        </label>
        {training.activation === "Custom" ? (
          <label className="field">
            <span>Custom Activation Name</span>
            <input
              type="text"
              value={training.activationCustomName ?? ""}
              placeholder="my_activation"
              onChange={(event) => onChange({ ...training, activationCustomName: event.target.value })}
            />
          </label>
        ) : null}

        <label className="field">
          <span>Learning Rate</span>
          <input
            type="number"
            step="0.0001"
            value={training.learningRate}
            onChange={(event) => onChange({ ...training, learningRate: Number(event.target.value) })}
          />
        </label>
      </div>
    </section>
  );
}
