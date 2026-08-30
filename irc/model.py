"""Model loading and residual-stream activation capture for Gemma 3 27B-it."""

from irc import env  # must run before transformers import

import torch
from torch import nn
from transformers import AutoTokenizer

MODEL_ID = "google/gemma-3-27b-it"


def load_tokenizer(model_id: str = MODEL_ID):
    env.require_hf_token()
    return AutoTokenizer.from_pretrained(model_id)


def load_model(model_id: str = MODEL_ID, device: str = "cuda"):
    """Load the model in bfloat16 (never fp32/quantized — we measure activations)."""
    from transformers import AutoModelForCausalLM

    env.require_hf_token()

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map=device
        )
    except ValueError:
        # gemma-3-27b-it is a multimodal checkpoint; fall back to the
        # conditional-generation class and use its text stack.
        from transformers import Gemma3ForConditionalGeneration

        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map=device
        )
    model.eval()
    return model


def get_decoder_layers(model: nn.Module) -> list[nn.Module]:
    """Return the list of decoder layers regardless of wrapper class."""
    for path in ("model.layers", "language_model.layers", "model.language_model.layers"):
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
        except AttributeError:
            continue
        return list(obj)
    raise AttributeError(f"could not locate decoder layers on {type(model).__name__}")


class ResidualCapture:
    """Context manager capturing resid_post (decoder-layer outputs) at given layers.

    After a forward pass, `self.acts[layer]` is (batch, seq, d_model).

    `to_cpu=True` (the default) moves each layer to CPU and upcasts to fp32
    inside the hook. That is one synchronizing transfer per layer -- 62 pipeline
    stalls in a single forward pass -- and it moves twice the bytes by upcasting
    first. Measured on gemma-3-27b-it, it makes the capture pass ~6.6s against
    ~1.0s for the generation it follows.

    `to_cpu=False` leaves activations on-device in their native dtype, so the
    caller slices to the tokens it actually wants and makes ONE transfer. Prefer
    it for bulk runs, and wrap the forward in torch.no_grad(): without it the
    pass builds an autograd graph across every layer, which is the other half of
    the cost (model.generate() applies no_grad internally, which is why
    generation looks fast by comparison).
    """

    def __init__(self, model: nn.Module, layers: list[int], to_cpu: bool = True):
        self.layers = layers
        self.to_cpu = to_cpu
        self._decoder_layers = get_decoder_layers(model)
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self.acts: dict[int, torch.Tensor] = {}

    def _make_hook(self, layer_idx: int):
        def hook(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden = hidden.detach()
            self.acts[layer_idx] = hidden.float().cpu() if self.to_cpu else hidden

        return hook

    def __enter__(self):
        for i in self.layers:
            self._handles.append(
                self._decoder_layers[i].register_forward_hook(self._make_hook(i))
            )
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return False


def sentence_token_ids(tokenizer, sentence: str) -> list[int]:
    """Token ids of a bare sentence (no BOS/specials) — the unit that stored
    response activations are aligned and trimmed to."""
    return tokenizer(sentence, add_special_tokens=False)["input_ids"]


def sentence_display_tokens(tokenizer, sentence: str) -> list[str]:
    """Human-readable per-token strings (for axis labels / the viewer)."""
    ids = sentence_token_ids(tokenizer, sentence)
    return [t.replace("▁", " ") for t in tokenizer.convert_ids_to_tokens(ids)]


def chat_ids(tokenizer, user_message: str, device: str = "cuda") -> torch.Tensor:
    """Tokenize a single-turn user message with the chat template."""
    enc = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_message}],
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    return enc["input_ids"].to(device)
