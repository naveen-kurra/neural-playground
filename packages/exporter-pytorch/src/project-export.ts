import { type ModelGraph } from "@neural-playground/block-schema";
import { exportConfigPy, exportModelYaml, exportTrainYaml } from "./config-export";
import { exportModelGraphToPyTorch } from "./model-export";
import {
  exportCheckpointPy,
  exportCustomHookNotesPy,
  exportDataPy,
  exportEvalPy,
  exportLoggingUtilsPy,
  exportReadme,
  exportRequirementsTxt,
  exportSchedulePy,
  exportScriptTrainPy,
  exportTrainModulePy
} from "./runtime-templates";
import type { ProjectFileMap } from "./types";

function withGenericBuildModel(modelPy: string): string {
  return `${modelPy}


def build_model(cfg, seq_len_override: int | None = None) -> DecoderLM:
    seq_len = seq_len_override or cfg.model.seq_len
    return DecoderLM(
        vocab_size=cfg.model.vocab_size,
        n_layers=cfg.model.n_layers,
        d_model=cfg.model.d_model,
        n_heads=cfg.model.n_heads,
        ffn_hidden=cfg.model.ffn_hidden,
        seq_len=seq_len,
        activation_name=cfg.model.activation_name,
    )
`;
}

export function exportProjectFiles(graph: ModelGraph): ProjectFileMap {
  return {
    "README.md": exportReadme(graph),
    "requirements.txt": exportRequirementsTxt(),
    "configs/model.yaml": exportModelYaml(graph),
    "configs/train.yaml": exportTrainYaml(graph),
    "scripts/train.py": exportScriptTrainPy(graph),
    "src/kurra_ai_cb/__init__.py": "",
    "src/kurra_ai_cb/model.py": withGenericBuildModel(exportModelGraphToPyTorch(graph)),
    "src/kurra_ai_cb/config.py": exportConfigPy(),
    "src/kurra_ai_cb/checkpoint.py": exportCheckpointPy(),
    "src/kurra_ai_cb/data.py": exportDataPy(),
    "src/kurra_ai_cb/eval.py": exportEvalPy(),
    "src/kurra_ai_cb/logging_utils.py": exportLoggingUtilsPy(),
    "src/kurra_ai_cb/schedule.py": exportSchedulePy(),
    "src/kurra_ai_cb/train.py": exportTrainModulePy(graph),
    "CUSTOM_HOOKS.md": exportCustomHookNotesPy(graph)
  };
}
