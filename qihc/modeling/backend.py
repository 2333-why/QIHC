"""Local model backend extracted from the legacy loader."""
from __future__ import annotations


class LocalModel:
    """Offline model loading and text tokenization; independent of problem family."""

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
        sharded = device in ("auto", "balanced")
        self.device = "cuda:0" if sharded else device
        self.temperature = float(temperature)
        self.max_new_tokens = int(max_new_tokens)
        self.max_input_tokens = int(max_input_tokens)
        config = AutoConfig.from_pretrained(model_path, local_files_only=True)
        dtype = torch.bfloat16 if sharded or device.startswith("cuda") else torch.float32
        load_kwargs = {"local_files_only": True, "dtype": dtype, "low_cpu_mem_usage": True}
        if sharded:
            load_kwargs.update(device_map=device, attn_implementation="sdpa",
                               max_memory={i: int(torch.cuda.mem_get_info(i)[0] * .8)
                                           for i in range(torch.cuda.device_count())})
        elif device.startswith("cuda"):
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
            self.tokenizer = self.processor.tokenizer
            self.model = AutoModelForMultimodalLM.from_pretrained(model_path, **load_kwargs)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            self.model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        if not sharded and not device.startswith("cuda"):
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
                return_dict=True, return_tensors="pt", enable_thinking=False,
            )
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            rendered = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        else:
            rendered = prompt
        return self.tokenizer(rendered, return_tensors="pt", truncation=False)
