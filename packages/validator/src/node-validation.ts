import { getBlockDefinition, type BlockRuleSpec, type ModelGraph } from "@neural-playground/block-schema";

import { inferNodeSequenceDim, numberConfig } from "./inference";
import type { ValidationIssue } from "./issues";
import { getRuleMessage } from "./schema-rules";

type NodeRuleHandler = (
  node: ModelGraph["nodes"][number],
  inferredSequenceDims: Map<string, number | null>,
  spec: BlockRuleSpec
) => ValidationIssue[];

const nodeRuleHandlers: Record<string, NodeRuleHandler> = {
  invalid_d_model(node) {
    const dModel = numberConfig(node.config.dModel);
    if (dModel !== null && dModel > 0) {
      return [];
    }
    return [
      {
        code: "invalid_d_model",
        message: getRuleMessage(node.type, "invalid_d_model", "TransformerBlock requires a positive model dimension."),
        nodeId: node.id
      }
    ];
  },
  invalid_num_heads(node) {
    const numHeads = numberConfig(node.config.numHeads);
    if (numHeads !== null && numHeads > 0) {
      return [];
    }
    return [
      {
        code: "invalid_num_heads",
        message: getRuleMessage(node.type, "invalid_num_heads", "TransformerBlock requires a positive number of heads."),
        nodeId: node.id
      }
    ];
  },
  heads_dimension_mismatch(node) {
    const dModel = numberConfig(node.config.dModel);
    const numHeads = numberConfig(node.config.numHeads);
    if (dModel === null || numHeads === null || dModel <= 0 || numHeads <= 0 || dModel % numHeads === 0) {
      return [];
    }
    return [
      {
        code: "heads_dimension_mismatch",
        message: `${getRuleMessage(
          node.type,
          "heads_dimension_mismatch",
          "Model dimension must be divisible by the number of heads."
        )} Found d_model=${dModel}, numHeads=${numHeads}.`,
        nodeId: node.id
      }
    ];
  },
  invalid_ffn_hidden(node) {
    const ffnHidden = numberConfig(node.config.ffnHidden);
    if (ffnHidden !== null && ffnHidden > 0) {
      return [];
    }
    return [
      {
        code: "invalid_ffn_hidden",
        message: getRuleMessage(node.type, "invalid_ffn_hidden", "TransformerBlock requires a positive FFN hidden size."),
        nodeId: node.id
      }
    ];
  },
  invalid_expert_count(node) {
    const numExperts = numberConfig(node.config.numExperts);
    if (numExperts !== null && numExperts > 0) {
      return [];
    }
    return [
      {
        code: "invalid_expert_count",
        message: getRuleMessage(node.type, "invalid_expert_count", "MoE requires a positive number of experts."),
        nodeId: node.id
      }
    ];
  },
  invalid_top_k(node) {
    const topK = numberConfig(node.config.topK);
    const numExperts = numberConfig(node.config.numExperts);
    if (topK !== null && numExperts !== null && topK > 0 && numExperts > 0 && topK <= numExperts) {
      return [];
    }
    return [
      {
        code: "invalid_top_k",
        message: getRuleMessage(
          node.type,
          "invalid_top_k",
          "MoE requires top-k to be positive and no greater than the number of experts."
        ),
        nodeId: node.id
      }
    ];
  },
  invalid_expert_hidden(node) {
    const expertHidden = numberConfig(node.config.expertHidden);
    if (expertHidden !== null && expertHidden > 0) {
      return [];
    }
    return [
      {
        code: "invalid_expert_hidden",
        message: getRuleMessage(node.type, "invalid_expert_hidden", "MoE requires a positive expert hidden size."),
        nodeId: node.id
      }
    ];
  },
  invalid_vocab_size(node) {
    const vocabSize = numberConfig(node.config.vocabSize);
    if (vocabSize !== null && vocabSize > 0) {
      return [];
    }
    return [
      {
        code: "invalid_vocab_size",
        message: getRuleMessage(node.type, "invalid_vocab_size", "Embedding requires a positive vocab size."),
        nodeId: node.id
      }
    ];
  },
  invalid_embedding_dim(node) {
    const embeddingDim = numberConfig(node.config.embeddingDim);
    if (embeddingDim !== null && embeddingDim > 0) {
      return [];
    }
    return [
      {
        code: "invalid_embedding_dim",
        message: getRuleMessage(node.type, "invalid_embedding_dim", "Embedding requires a positive embedding dimension."),
        nodeId: node.id
      }
    ];
  },
  invalid_hidden_dim(node) {
    const hiddenDim = numberConfig(node.config.hiddenDim);
    if (hiddenDim !== null && hiddenDim > 0) {
      return [];
    }
    return [
      {
        code: "invalid_hidden_dim",
        message: getRuleMessage(node.type, "invalid_hidden_dim", "MLP requires a positive hidden dimension."),
        nodeId: node.id
      }
    ];
  },
  unknown_mlp_input_dim(node, inferredSequenceDims) {
    const inferredDim = inferNodeSequenceDim(node, inferredSequenceDims);
    if (inferredDim !== null) {
      return [];
    }
    return [
      {
        code: "unknown_mlp_input_dim",
        message: getRuleMessage(
          node.type,
          "unknown_mlp_input_dim",
          "MLP input dimension could not be inferred from incoming connections."
        ),
        nodeId: node.id
      }
    ];
  },
  unknown_moe_input_dim(node, inferredSequenceDims) {
    const inferredDim = inferNodeSequenceDim(node, inferredSequenceDims);
    if (inferredDim !== null) {
      return [];
    }
    return [
      {
        code: "unknown_moe_input_dim",
        message: getRuleMessage(
          node.type,
          "unknown_moe_input_dim",
          "MoE input dimension could not be inferred from incoming connections."
        ),
        nodeId: node.id
      }
    ];
  },
  invalid_sequence_length(node) {
    const sequenceLength = numberConfig(node.config.sequenceLength);
    if (sequenceLength !== null && sequenceLength > 1) {
      return [];
    }
    return [
      {
        code: "invalid_sequence_length",
        message: getRuleMessage(node.type, "invalid_sequence_length", "Input requires a sequence length greater than 1."),
        nodeId: node.id
      }
    ];
  },
  unknown_layernorm_dim(node, inferredSequenceDims) {
    const inferredDim = inferNodeSequenceDim(node, inferredSequenceDims);
    if (inferredDim !== null) {
      return [];
    }
    return [
      {
        code: "unknown_layernorm_dim",
        message: getRuleMessage(
          node.type,
          "unknown_layernorm_dim",
          "LayerNorm dimension could not be inferred from incoming connections."
        ),
        nodeId: node.id
      }
    ];
  },
  invalid_kv_heads(node) {
    const numHeads = numberConfig(node.config.numHeads);
    const kvHeads = numberConfig(node.config.numKeyValueHeads);
    if (numHeads === null || kvHeads === null) return [];
    if (kvHeads <= 0) {
      return [{ code: "invalid_kv_heads", message: `KV heads must be positive. Found ${kvHeads}.`, nodeId: node.id }];
    }
    if (kvHeads > numHeads) {
      return [{ code: "invalid_kv_heads", message: `KV heads (${kvHeads}) cannot exceed attention heads (${numHeads}).`, nodeId: node.id }];
    }
    if (numHeads % kvHeads !== 0) {
      return [{ code: "invalid_kv_heads", message: `Attention heads (${numHeads}) must be divisible by KV heads (${kvHeads}).`, nodeId: node.id }];
    }
    return [];
  },
  head_dim_mismatch(node) {
    const dModel = numberConfig(node.config.dModel);
    const numHeads = numberConfig(node.config.numHeads);
    const headDim = numberConfig(node.config.headDim);
    if (dModel === null || numHeads === null || numHeads <= 0 || headDim === null) return [];
    const expected = Math.floor(dModel / numHeads);
    if (headDim !== expected) {
      return [{ code: "head_dim_mismatch", message: `Head dim should be ${expected} (${dModel} / ${numHeads}), but is set to ${headDim}.`, nodeId: node.id }];
    }
    return [];
  },
  unknown_output_dim(node, inferredSequenceDims) {
    const incomingDim = inferNodeSequenceDim(node, inferredSequenceDims);
    const headType = String(node.config.headType ?? "LanguageModel");
    if (headType !== "LanguageModel" || incomingDim !== null) {
      return [];
    }
    return [
      {
        code: "unknown_output_dim",
        message: getRuleMessage(
          node.type,
          "unknown_output_dim",
          "Output head dimension could not be inferred from incoming connections."
        ),
        nodeId: node.id
      }
    ];
  }
};

export function validateNodeConfig(
  node: ModelGraph["nodes"][number],
  inferredSequenceDims: Map<string, number | null>
): ValidationIssue[] {
  const definition = getBlockDefinition(node.type);
  const issues: ValidationIssue[] = [];

  for (const spec of definition.ruleSpecs) {
    const handler = nodeRuleHandlers[spec.code];
    if (handler) {
      issues.push(...handler(node, inferredSequenceDims, spec));
    }
  }

  return issues;
}
