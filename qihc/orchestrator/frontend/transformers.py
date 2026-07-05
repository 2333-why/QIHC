"""Tier-B/C frontend using HuggingFace Transformers."""

from __future__ import annotations

import numpy as np

from qihc.orchestrator.frontend.base import BaseFrontend
from qihc.orchestrator.types import RoutingContext


class TransformersFrontend(BaseFrontend):
    """
    Small LM frontend (default: DistilGPT-2).

    Pipeline
    --------
    1. Tokenize input text
    2. Run one forward pass, take last-token hidden state
    3. Linear routing head -> expert logits

    Upgrade path
    ------------
    - Change ``config.model_name`` to ``gpt2``, ``gpt2-medium``, etc. (tier_c)
    - Replace routing head with trained checkpoint later
    - Swap ``encode_batch`` internals to use MoE layer hidden states
    """

    def __init__(self, config):
        super().__init__(config)
        self._model = None
        self._tokenizer = None
        self._routing_head = None
        self._device = None

    def _lazy_init(self) -> None:
        if self._model is not None:
            return

        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Tier-B frontend requires optional deps: pip install -e \".[llm]\""
            ) from exc

        cfg = self.config
        self._device = cfg.device
        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

        self._tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModel.from_pretrained(cfg.model_name)
        self._model.eval()
        self._model.to(self._device)

        hidden = int(self._model.config.hidden_size)
        rng = np.random.default_rng(cfg.seed)
        # Fixed random routing head for PoC; replace with trained weights later.
        weight = rng.normal(scale=0.02, size=(cfg.num_experts, hidden)).astype(np.float32)
        bias = rng.normal(scale=0.01, size=(cfg.num_experts,)).astype(np.float32)

        import torch

        self._routing_head = torch.nn.Linear(hidden, cfg.num_experts, bias=True)
        with torch.no_grad():
            self._routing_head.weight.copy_(torch.from_numpy(weight))
            self._routing_head.bias.copy_(torch.from_numpy(bias))
        self._routing_head.eval()
        self._routing_head.to(self._device)

    def encode_batch(self, texts: list[str]) -> list[RoutingContext]:
        self._lazy_init()
        import torch

        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        encoded = {k: v.to(self._device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = self._model(**encoded)
            hidden = outputs.last_hidden_state  # (B, T, H)
            lengths = encoded["attention_mask"].sum(dim=1) - 1
            batch_idx = torch.arange(hidden.size(0), device=self._device)
            last_hidden = hidden[batch_idx, lengths, :]
            logits_t = self._routing_head(last_hidden)

        logits_np = logits_t.detach().cpu().numpy()
        contexts: list[RoutingContext] = []
        for i, text in enumerate(texts):
            contexts.append(
                RoutingContext(
                    text=text,
                    logits=logits_np[i],
                    hidden=last_hidden[i].detach().cpu().numpy(),
                    metadata={
                        "tier": "transformers",
                        "model_name": self.config.model_name,
                        "device": self._device,
                    },
                )
            )
        return contexts
