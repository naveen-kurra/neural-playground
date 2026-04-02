import type { CopyStatus, ExportPreview, SafeExport } from "../app/types";

type ExportPanelProps<TProjectFiles> = {
  exportedJson: string;
  exportedPyTorch: SafeExport<string>;
  exportedProject: SafeExport<TProjectFiles>;
  exportPreview: ExportPreview;
  copyStatus: CopyStatus;
  onOpenPreview: (preview: ExportPreview) => void;
  onCopy: (contents: string, artifact: "json" | "pytorch") => void;
  onDownloadText: (filename: string, contents: string) => void;
  onDownloadProject: () => void;
};

export function ExportPanel<TProjectFiles>(props: ExportPanelProps<TProjectFiles>) {
  const {
    exportedJson,
    exportedPyTorch,
    exportedProject,
    exportPreview,
    copyStatus,
    onOpenPreview,
    onCopy,
    onDownloadText,
    onDownloadProject
  } = props;

  return (
    <section className="export-panel">
      <div className="panel-header row">
        <div>
          <p className="eyebrow">Export</p>
          <h2>Artifacts</h2>
        </div>
      </div>
      <div className="export-actions">
        <div className="export-card">
          <strong>Graph JSON</strong>
          <div className="export-buttons">
            <button type="button" className="ghost-button export-btn" onClick={() => onOpenPreview("json")}>View</button>
            <button type="button" className="ghost-button export-btn" onClick={() => onCopy(exportedJson, "json")}>Copy</button>
            <button type="button" className="ghost-button export-btn" onClick={() => onDownloadText("model-graph.json", exportedJson)}>Download</button>
          </div>
        </div>

        <div className="export-card">
          <strong>PyTorch Model</strong>
          {!exportedPyTorch.ok ? <span className="export-warning">Unavailable: {exportedPyTorch.error}</span> : null}
          <div className="export-buttons">
            <button type="button" className="ghost-button export-btn" onClick={() => onOpenPreview("pytorch")} disabled={!exportedPyTorch.ok}>View</button>
            <button type="button" className="ghost-button export-btn" onClick={() => exportedPyTorch.ok && onCopy(exportedPyTorch.value, "pytorch")} disabled={!exportedPyTorch.ok}>Copy</button>
            <button type="button" className="ghost-button export-btn" onClick={() => exportedPyTorch.ok && onDownloadText("model.py", exportedPyTorch.value)} disabled={!exportedPyTorch.ok}>Download</button>
          </div>
        </div>

        <div className="export-card">
          <strong>Full Project</strong>
          {!exportedProject.ok ? <span className="export-warning">Unavailable: {exportedProject.error}</span> : null}
          <div className="export-buttons">
            <button type="button" className="ghost-button export-btn" onClick={onDownloadProject} disabled={!exportedProject.ok}>Download Project</button>
          </div>
        </div>
      </div>
      {copyStatus !== "idle" ? (
        <p className={`copy-status${copyStatus === "copy-failed" ? " error" : ""}`}>
          {copyStatus === "json-copied" ? "Graph JSON copied." : null}
          {copyStatus === "pytorch-copied" ? "PyTorch model copied." : null}
          {copyStatus === "project-downloaded" ? "Project archive downloaded." : null}
          {copyStatus === "copy-failed" ? "Copy failed in this browser." : null}
        </p>
      ) : null}
      {exportPreview ? (
        <div className="export-preview">
          <div className="panel-header row export-subheader">
            <div>
              <p className="eyebrow">Preview</p>
              <h2>{exportPreview === "json" ? "Graph JSON" : "PyTorch Model"}</h2>
            </div>
            <button type="button" className="ghost-button" onClick={() => onOpenPreview(null)}>
              Close
            </button>
          </div>
          <pre>{exportPreview === "json" ? exportedJson : exportedPyTorch.ok ? exportedPyTorch.value : exportedPyTorch.error}</pre>
        </div>
      ) : null}
    </section>
  );
}
