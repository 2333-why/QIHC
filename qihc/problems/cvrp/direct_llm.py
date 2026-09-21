"""Local full-solution LLM generator for the unrepaired direct-solve baseline.

The class deliberately does not import p-bit, repair, neighbourhood, or solver
components.  It supports both normal causal LMs and recent text-capable
multimodal Qwen models used with text-only messages.
"""

from __future__ import annotations


class LocalDirectLLMGenerator:
    """Generate complete JSON routes from a local model without any repair."""

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cuda:0",
        temperature: float = 0.2,
        max_new_tokens: int = 8192,
        max_input_tokens: int = 32768,
    ):
        try:
            import torch
            from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Direct local LLM solving requires torch and transformers") from exc
        self.torch = torch
        self.device = device
        self.temperature = float(temperature)
        self.max_new_tokens = int(max_new_tokens)
        self.max_input_tokens = int(max_input_tokens)
        config = AutoConfig.from_pretrained(model_path, local_files_only=True)
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        load_kwargs = {"local_files_only": True, "torch_dtype": dtype, "low_cpu_mem_usage": True}
        if device.startswith("cuda"):
            load_kwargs["device_map"] = {"": device}

        self.processor = None
        model_type = str(getattr(config, "model_type", ""))
        if model_type.startswith("qwen3_5"):
            try:
                from transformers import AutoModelForMultimodalLM, AutoProcessor
            except ImportError as exc:  # pragma: no cover - depends on current Transformers
                raise RuntimeError(
                    "Qwen3.5 requires a current Transformers release with "
                    "AutoModelForMultimodalLM; install transformers>=4.57"
                ) from exc
            self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
            self.model = AutoModelForMultimodalLM.from_pretrained(model_path, **load_kwargs)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            self.model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        if not device.startswith("cuda"):
            self.model = self.model.to(device)
        self.model.eval()

    def _inputs(self, prompt: str):
        messages = [{"role": "user", "content": prompt}]
        if self.processor is not None:
            # Qwen3.5 is multimodal, but this benchmark is intentionally
            # text-only. Its processor still provides the correct chat template.
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            return self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt",
            )
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            rendered = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            rendered = prompt
        return self.tokenizer(rendered, return_tensors="pt", truncation=False)

    def _generate(self, prompt: str) -> str:
        inputs = self._inputs(prompt)
        input_length = int(inputs["input_ids"].shape[1])
        if input_length > self.max_input_tokens:
            raise ValueError(
                f"Direct-LLM prompt has {input_length} tokens, exceeding explicit "
                f"limit {self.max_input_tokens}; refusing silent truncation"
            )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        generate_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
            "pad_token_id": getattr(self.model.config, "pad_token_id", None)
            or getattr(self.model.config, "eos_token_id", None),
        }
        if self.temperature > 0:
            generate_kwargs.update({"temperature": self.temperature, "top_p": 0.9})
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, **generate_kwargs)
        generated = output[0, input_length:]
        if self.processor is not None:
            return self.processor.decode(generated, skip_special_tokens=True)
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def generate(self, prompt: str) -> str:
        """Public alias kept separate from the implementation detail used by old runners."""

        return self._generate(prompt)
