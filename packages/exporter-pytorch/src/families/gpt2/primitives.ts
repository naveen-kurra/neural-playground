import {
  renderActivationRegistry,
  renderConv1D,
  renderGpt2Attention,
  renderGpt2Block,
  renderGpt2MLP,
  renderNewGeluActivation
} from "../../shared/python-snippets";

export function renderGpt2Primitives(): string {
  return [
    renderNewGeluActivation(),
    renderActivationRegistry("gpt2"),
    renderConv1D(),
    renderGpt2Attention("GPT2Config"),
    renderGpt2MLP("GPT2Config", "get_activation"),
    renderGpt2Block("GPT2Config", { hybrid: false })
  ].join("\n");
}
