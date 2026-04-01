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

export function exportProjectFiles(graph: ModelGraph): ProjectFileMap {
  return {
    "README.md": exportReadme(graph),
    "requirements.txt": exportRequirementsTxt(),
    "configs/model.yaml": exportModelYaml(graph),
    "configs/train.yaml": exportTrainYaml(graph),
    "scripts/train.py": exportScriptTrainPy(graph),
    "src/kurra_ai_cb/__init__.py": "",
    "src/kurra_ai_cb/model.py": exportModelGraphToPyTorch(graph),
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
