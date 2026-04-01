import { blockDefinitions } from "@neural-playground/block-schema";
import type { BlockType } from "@neural-playground/block-schema";

type PaletteSidebarProps = {
  projectStatus: string;
  loadInputRef: React.RefObject<HTMLInputElement | null>;
  onAddNode: (type: BlockType) => void;
  onSave: () => void;
  onLoadClick: () => void;
  onLoadFile: React.ChangeEventHandler<HTMLInputElement>;
};

export function PaletteSidebar(props: PaletteSidebarProps) {
  const { projectStatus, loadInputRef, onAddNode, onSave, onLoadClick, onLoadFile } = props;

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
        </div>
      </div>
      <input ref={loadInputRef} className="hidden-file-input" type="file" accept=".json" onChange={onLoadFile} />
      <p className="panel-copy">Add architecture pieces to the canvas and wire them into a graph.</p>
      {projectStatus ? <p className="project-status">{projectStatus}</p> : null}
      <div className="block-grid">
        {blockDefinitions.map((definition) => (
          <button type="button" key={definition.type} className="block-card" onClick={() => onAddNode(definition.type)}>
            <span className="block-category">{definition.category}</span>
            <strong>{definition.label}</strong>
            <span>{definition.description}</span>
          </button>
        ))}
      </div>
    </aside>
  );
}
