import { useEffect, useRef, useImperativeHandle, forwardRef } from "react";
import { EditorView, basicSetup } from "codemirror";
import { EditorState } from "@codemirror/state";
import { python } from "@codemirror/lang-python";
import { search, openSearchPanel } from "@codemirror/search";
import { keymap } from "@codemirror/view";

const darkTheme = EditorView.theme({
  "&": {
    backgroundColor: "transparent",
    height: "100%",
    fontSize: "0.82rem"
  },
  ".cm-scroller": {
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Courier New", monospace',
    lineHeight: "1.6"
  },
  ".cm-content": {
    padding: "16px 0",
    caretColor: "#76c1ff"
  },
  ".cm-gutters": {
    backgroundColor: "rgba(10, 20, 34, 0.6)",
    borderRight: "1px solid rgba(139, 162, 194, 0.12)",
    color: "#4a607a",
    minWidth: "48px"
  },
  ".cm-lineNumbers .cm-gutterElement": {
    padding: "0 12px 0 8px"
  },
  ".cm-line": {
    padding: "0 16px"
  },
  ".cm-activeLine": {
    backgroundColor: "rgba(118, 193, 255, 0.06)"
  },
  ".cm-activeLineGutter": {
    backgroundColor: "rgba(118, 193, 255, 0.08)",
    color: "#76c1ff"
  },
  ".cm-selectionBackground, ::selection": {
    backgroundColor: "rgba(118, 193, 255, 0.2) !important"
  },
  ".cm-searchMatch": {
    backgroundColor: "rgba(255, 200, 80, 0.3)",
    outline: "1px solid rgba(255, 200, 80, 0.6)"
  },
  ".cm-searchMatch-selected": {
    backgroundColor: "rgba(255, 200, 80, 0.55)"
  },
  ".cm-panels": {
    backgroundColor: "rgba(10, 20, 34, 0.95)",
    borderTop: "1px solid rgba(139, 162, 194, 0.15)",
    color: "#b8c4d6",
    padding: "8px 10px"
  },
  ".cm-search": {
    display: "flex",
    flexWrap: "wrap",
    alignItems: "center",
    gap: "8px"
  },
  ".cm-search label": {
    display: "inline-flex",
    alignItems: "center",
    gap: "6px",
    whiteSpace: "nowrap",
    fontSize: "0.78rem",
    color: "#9db2cf"
  },
  ".cm-search input[type='text'], .cm-search input:not([type])": {
    width: "min(360px, 100%)",
    boxSizing: "border-box",
    backgroundColor: "rgba(16, 28, 44, 0.9)",
    border: "1px solid rgba(139, 162, 194, 0.25)",
    borderRadius: "6px",
    color: "#d3ddec",
    padding: "4px 8px",
    minWidth: "220px",
    fontSize: "0.82rem"
  },
  ".cm-search input[type='checkbox']": {
    width: "14px",
    height: "14px",
    boxSizing: "border-box",
    margin: 0,
    accentColor: "#76c1ff",
    cursor: "pointer"
  },
  ".cm-search button": {
    backgroundColor: "rgba(118, 193, 255, 0.12)",
    border: "1px solid rgba(118, 193, 255, 0.25)",
    borderRadius: "6px",
    color: "#76c1ff",
    padding: "4px 10px",
    fontSize: "0.8rem",
    cursor: "pointer"
  },
  ".cm-search button:hover": {
    backgroundColor: "rgba(118, 193, 255, 0.2)"
  },
  ".cm-search button:focus-visible, .cm-search input:focus-visible": {
    outline: "2px solid rgba(118, 193, 255, 0.45)",
    outlineOffset: "1px"
  }
}, { dark: true });

type CodeViewerProps = {
  code: string;
};

export type CodeViewerHandle = {
  openSearch: () => void;
};

export const CodeViewer = forwardRef<CodeViewerHandle, CodeViewerProps>(function CodeViewer({ code }, ref) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const viewRef = useRef<EditorView | null>(null);

  useImperativeHandle(ref, () => ({
    openSearch() {
      if (viewRef.current) openSearchPanel(viewRef.current);
    }
  }));

  useEffect(() => {
    if (!containerRef.current) return;

    const state = EditorState.create({
      doc: code,
      extensions: [
        basicSetup,
        python(),
        search({ top: false }),
        keymap.of([
          { key: "Mod-f", run: (view) => { openSearchPanel(view); return true; } }
        ]),
        EditorState.readOnly.of(true),
        EditorView.editable.of(false),
        darkTheme
      ]
    });

    const view = new EditorView({ state, parent: containerRef.current });
    viewRef.current = view;

    return () => {
      view.destroy();
      viewRef.current = null;
    };
  }, []);

  // Update content when code changes without recreating the editor
  useEffect(() => {
    const view = viewRef.current;
    if (!view) return;
    const current = view.state.doc.toString();
    if (current !== code) {
      view.dispatch({
        changes: { from: 0, to: current.length, insert: code }
      });
    }
  }, [code]);

  return <div ref={containerRef} className="code-viewer" />;
});
