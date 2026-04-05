import { blockDefinitions } from "@neural-playground/block-schema";
import type { BlockType } from "@neural-playground/block-schema";
import { useMemo } from "react";
import { modelTemplates } from "../app/model-templates";
import { graphPresets } from "../app/presets";

type PaletteSidebarProps = {
  projectStatus: string;
  modelTemplateSelection: string;
  templateBlockCount: number;
  searchQuery: string;
  loadInputRef: React.RefObject<HTMLInputElement | null>;
  onAddNode: (type: BlockType) => void;
  onApplyPreset: (presetId: string) => void;
  onModelTemplateSelectionChange: (value: string) => void;
  onImportModel: () => void;
  onTemplateBlockCountChange: (value: number) => void;
  onSearchQueryChange: (value: string) => void;
  onSave: () => void;
  onLoadClick: () => void;
  onLoadFile: React.ChangeEventHandler<HTMLInputElement>;
  onOpenPruningTool: () => void;
};

export function PaletteSidebar(props: PaletteSidebarProps) {
  const {
    projectStatus,
    modelTemplateSelection,
    templateBlockCount,
    searchQuery,
    loadInputRef,
    onAddNode,
    onApplyPreset,
    onModelTemplateSelectionChange,
    onImportModel,
    onTemplateBlockCountChange,
    onSearchQueryChange,
    onSave,
    onLoadClick,
    onLoadFile,
    onOpenPruningTool
  } = props;

  const normalizedQuery = searchQuery.trim().toLowerCase();
  const filteredPresets = useMemo(
    () =>
      graphPresets.filter((preset) =>
        normalizedQuery === ""
          ? true
          : `${preset.name} ${preset.description}`.toLowerCase().includes(normalizedQuery)
      ),
    [normalizedQuery]
  );
  const filteredBlocks = useMemo(
    () =>
      blockDefinitions.filter((definition) =>
        normalizedQuery === ""
          ? true
          : `${definition.label} ${definition.category} ${definition.description}`.toLowerCase().includes(normalizedQuery)
      ),
    [normalizedQuery]
  );

  return (
    <aside className="sidebar palette">
      <div className="panel-header row">
        <div>
          <p className="eyebrow">Neural Playground</p>
          <h1>Block Library</h1>
        </div>
        <div className="canvas-actions">
          <button type="button" className="ghost-button" onClick={onSave}>
            Save
          </button>
          <button type="button" className="ghost-button" onClick={onLoadClick}>
            Load
          </button>
          <button type="button" className="ghost-button accent-button" onClick={onOpenPruningTool}>
            Try Pruning Tool
          </button>
        </div>
      </div>
      <input ref={loadInputRef} className="hidden-file-input" type="file" accept=".json" onChange={onLoadFile} />
      <div className="palette-scroll">
        <p className="panel-copy">Add architecture pieces to the canvas and wire them into a graph.</p>
        {projectStatus ? <p className="project-status">{projectStatus}</p> : null}
        <div className="preset-section">
          <div className="section-header">
            <p className="eyebrow">Search</p>
          </div>
          <label className="field">
            <span>Find blocks or presets</span>
            <input
              type="text"
              value={searchQuery}
              placeholder="Search block library"
              onChange={(event) => onSearchQueryChange(event.target.value)}
            />
          </label>
        </div>
        <div className="preset-section">
          <div className="section-header">
            <p className="eyebrow">Import</p>
          </div>
          <div className="form-stack">
            <label className="field">
              <span>Architecture Template</span>
              <input
                type="text"
                list="model-template-options"
                value={modelTemplateSelection}
            placeholder="Search GPT-2, LLaMA, Phi-3, or Gemma 4"
                onChange={(event) => onModelTemplateSelectionChange(event.target.value)}
              />
            </label>
            <datalist id="model-template-options">
              {modelTemplates.map((template) => (
                <option key={template.id} value={template.label} />
              ))}
            </datalist>
            <label className="field">
              <span>Template Blocks</span>
              <input
                type="number"
                min="1"
                max="96"
                value={templateBlockCount}
                onChange={(event) => onTemplateBlockCountChange(Number(event.target.value))}
              />
            </label>
            <button type="button" className="ghost-button" onClick={onImportModel}>
              Load Template
            </button>
          </div>
        </div>
        <div className="preset-section">
          <div className="section-header">
            <p className="eyebrow">Presets</p>
          </div>
          <div className="preset-grid">
            {filteredPresets.map((preset) => (
              <button type="button" key={preset.id} className="block-card preset-card" onClick={() => onApplyPreset(preset.id)}>
                <strong>{preset.name}</strong>
                <span>{preset.description}</span>
              </button>
            ))}
            {filteredPresets.length === 0 ? <p className="muted">No presets match this search.</p> : null}
          </div>
        </div>
        <div className="block-grid">
          {filteredBlocks.map((definition) => (
            <button type="button" key={definition.type} className="block-card" onClick={() => onAddNode(definition.type)}>
              <span className="block-category">{definition.category}</span>
              <strong>{definition.label}</strong>
              <span>{definition.description}</span>
            </button>
          ))}
          {filteredBlocks.length === 0 ? <p className="muted">No blocks match this search.</p> : null}
        </div>
      </div>
    </aside>
  );
}
