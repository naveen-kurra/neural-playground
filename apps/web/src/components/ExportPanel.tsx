import { createPortal } from "react-dom";
import { lazy, Suspense, useRef } from "react";
import type { CopyStatus, ExportPreview, SafeExport } from "../app/types";
import type { CodeViewerHandle } from "./CodeViewer";

const CodeViewer = lazy(() => import("./CodeViewer").then((m) => ({ default: m.CodeViewer })));

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

  const codeViewerRef = useRef<CodeViewerHandle | null>(null);

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
            <button type="button" className="ghost-button export-btn" disabled>Download</button>
          </div>
        </div>

        <div className="export-card">
          <strong>PyTorch Model</strong>
          {!exportedPyTorch.ok ? <span className="export-warning">Unavailable: {exportedPyTorch.error}</span> : null}
          <div className="export-buttons">
            <button type="button" className="ghost-button export-btn" onClick={() => onOpenPreview("pytorch")} disabled={!exportedPyTorch.ok}>View</button>
            <button type="button" className="ghost-button export-btn" onClick={() => exportedPyTorch.ok && onCopy(exportedPyTorch.value, "pytorch")} disabled={!exportedPyTorch.ok}>Copy</button>
            <button type="button" className="ghost-button export-btn" disabled>Download</button>
          </div>
        </div>

        <div className="export-card">
          <strong>Full Project</strong>
          {!exportedProject.ok ? <span className="export-warning">Unavailable: {exportedProject.error}</span> : null}
          <div className="export-buttons">
            <button type="button" className="ghost-button export-btn" disabled>Download Project</button>
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
                    <button type="button" className="ghost-button" disabled>Download</button>
                    <button type="button" className="ghost-button" onClick={onApplyJsonEditor} disabled={!jsonEditorCanApply}>Apply to Graph</button>
                  </>
                )}
                {exportPreview === "pytorch" && exportedPyTorch.ok && (
                  <>
                    <button type="button" className="ghost-button" onClick={() => codeViewerRef.current?.openSearch()}>Search</button>
                    <button type="button" className="ghost-button" onClick={() => onCopy(exportedPyTorch.value, "pytorch")}>Copy</button>
                    <button type="button" className="ghost-button" disabled>Download</button>
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
              <Suspense fallback={(
                <div className="preview-modal-content code-viewer-loading" role="status" aria-live="polite">
                  <div className="code-viewer-loading-spinner" aria-hidden="true" />
                  <p className="code-viewer-loading-title">Loading code viewer...</p>
                  <p className="code-viewer-loading-subtitle">Syntax highlighting and search are initializing.</p>
                </div>
              )}>
                <CodeViewer ref={codeViewerRef} code={previewContent ?? ""} />
              </Suspense>
            )}
          </div>
        </div>,
        document.body
      ) : null}
    </section>
  );
}
