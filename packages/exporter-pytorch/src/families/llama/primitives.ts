import {
  renderActivationRegistry,
  renderLlamaAttention,
  renderLlamaDecoderLayer,
  renderLlamaMLP,
  renderLlamaNormAndRotaryHelpers
} from "../../shared/python-snippets";

export function renderLlamaPrimitives(): string {
  return [
    renderActivationRegistry("llama"),
    renderLlamaNormAndRotaryHelpers(),
    renderLlamaAttention("LlamaConfig"),
    renderLlamaMLP("LlamaConfig", "get_activation"),
    renderLlamaDecoderLayer("LlamaConfig", { hybrid: false })
  ].join("\n");
}
