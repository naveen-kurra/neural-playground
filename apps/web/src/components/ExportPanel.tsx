import { createPortal } from "react-dom";
import type { CopyStatus, ExportPreview, SafeExport } from "../app/types";

type ExportPanelProps<TProjectFiles> = {
  exportedJson: string;
  exportedPyTorch: SafeExport<string>;
  exportedProject: SafeExport<TProjectFiles>;
  exportPreview: ExportPreview;
  copyStatus: CopyStatus;
  jsonEditorValue: string;
  jsonEditorStatus: string;
  jsonEditorCanApply: boolean;
  onJsonEditorChange: (contents: string) => void;
  onApplyJsonEditor: () => void;
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
    jsonEditorValue,
    jsonEditorStatus,
    jsonEditorCanApply,
    onJsonEditorChange,
    onApplyJsonEditor,
    onOpenPreview,
    onCopy,
    onDownloadText,
    onDownloadProject
  } = props;

  const previewTitle = exportPreview === "json" ? "Graph JSON" : "PyTorch Model";
  const previewContent = exportedPyTorch.ok ? exportedPyTorch.value : exportedPyTorch.error;

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

      {exportPreview ? createPortal(
        <div className="preview-modal-overlay" onClick={() => onOpenPreview(null)}>
          <div className="preview-modal" onClick={(e) => e.stopPropagation()}>
            <div className="preview-modal-header">
              <div>
                <p className="eyebrow">Preview</p>
                <h2>{previewTitle}</h2>
              </div>
              <div className="preview-modal-actions">
                {exportPreview === "json" && (
                  <>
                    <button type="button" className="ghost-button" onClick={() => onCopy(jsonEditorValue, "json")}>Copy</button>
                    <button type="button" className="ghost-button" onClick={() => onDownloadText("model-graph.json", jsonEditorValue)}>Download</button>
                    <button type="button" className="ghost-button" onClick={onApplyJsonEditor} disabled={!jsonEditorCanApply}>Apply to Graph</button>
                  </>
                )}
                {exportPreview === "pytorch" && exportedPyTorch.ok && (
                  <>
                    <button type="button" className="ghost-button" onClick={() => onCopy(exportedPyTorch.value, "pytorch")}>Copy</button>
                    <button type="button" className="ghost-button" onClick={() => onDownloadText("model.py", exportedPyTorch.value)}>Download</button>
                  </>
                )}
                <button type="button" className="ghost-button" onClick={() => onOpenPreview(null)}>Close</button>
              </div>
            </div>
            {exportPreview === "json" ? (
              <div className="preview-modal-content json-editor-wrap">
                <p className={`json-editor-status${jsonEditorCanApply ? " success" : " error"}`}>{jsonEditorStatus}</p>
                <textarea
                  className="json-editor"
                  value={jsonEditorValue}
                  spellCheck={false}
                  onChange={(event) => onJsonEditorChange(event.target.value)}
                />
              </div>
            ) : (
              <pre className="preview-modal-content">{previewContent}</pre>
            )}
          </div>
        </div>,
        document.body
      ) : null}
    </section>
  );
}
