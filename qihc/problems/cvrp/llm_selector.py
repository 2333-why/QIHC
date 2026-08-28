"""Offline local-LLM neighborhood selection for NL-CVRP."""

from __future__ import annotations

import json
import re
from pathlib import Path

from qihc.problems.cvrp.instance import CVRPInstance, ConstraintSpec, RouteSolution
from qihc.problems.cvrp.neighborhood import (
    KNNNeighborhoodSelector,
    NeighborhoodProposal,
)
from qihc.problems.cvrp.verifier import verify_solution


def _extract_json(text: str) -> dict:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    candidates = [fenced.group(1)] if fenced else []
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            continue
    raise ValueError("LLM output contains no valid JSON object")


class LocalLLMNeighborhoodSelector:
    """Generate a structured destroy/candidate-route proposal from an offline model."""

    def __init__(
        self,
        model_path: str,
        destroy_size: int = 8,
        routes_per_customer: int = 3,
        device: str = "cuda:0",
        max_new_tokens: int = 384,
        temperature: float = 0.2,
        refresh_interval: int = 5,
        audit_path: str | Path | None = None,
    ):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional formal dependency
            raise RuntimeError("Local LLM selection requires torch and transformers") from exc
        self.torch = torch
        self.destroy_size = int(destroy_size)
        self.routes_per_customer = int(routes_per_customer)
        self.device = device
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.refresh_interval = max(1, int(refresh_interval))
        self.audit_path = Path(audit_path) if audit_path else None
        if self.audit_path:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        self.fallback = KNNNeighborhoodSelector(destroy_size, routes_per_customer)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(device)
        self.model.eval()
        self._cache: NeighborhoodProposal | None = None
        self._cache_instance: str | None = None

    def _prompt(self, instance: CVRPInstance, solution: RouteSolution, iteration: int) -> str:
        verification = verify_solution(instance, solution)
        route_summary = [
            {
                "route": idx,
                "customers": route,
                "load": verification.route_loads[idx] if idx < len(verification.route_loads) else 0,
            }
            for idx, route in enumerate(solution.routes)
        ]
        constraints = [
            {"type": c.type, "hard": c.hard, "weight": c.weight, **c.params}
            for c in instance.constraints
        ]
        return (
            "You select a small large-neighborhood-search subproblem for CVRP. "
            "Do not solve the full route. Return JSON only. Select customers whose reassignment may reduce "
            "distance while respecting hard constraints. Every selected customer needs candidate vehicle route IDs.\n"
            f"Iteration: {iteration}\n"
            f"Vehicle capacity: {instance.vehicle_capacity}\n"
            f"Constraints: {json.dumps(constraints, ensure_ascii=False)}\n"
            f"Routes: {json.dumps(route_summary, ensure_ascii=False)}\n"
            f"Required destroy count: {self.destroy_size}\n"
            f"Maximum candidate routes per customer: {self.routes_per_customer}\n"
            "Schema: {\"destroy_customers\":[int],\"candidate_routes\":{\"customer_id\":[route_id]},"
            "\"confidence\":number,\"reason\":string}"
        )

    def _generate(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            rendered = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            rendered = prompt
        inputs = self.tokenizer(rendered, return_tensors="pt", truncation=True, max_length=8192)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        do_sample = self.temperature > 0
        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs.update({"temperature": self.temperature, "top_p": 0.9})
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, **generation_kwargs)
        generated = output[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def parse_constraint_ir(
        self,
        description: str,
        customer_ids: list[int],
    ) -> tuple[list[ConstraintSpec], dict]:
        """Parse natural-language constraints into normalized Constraint IR."""

        prompt = (
            "Convert the following vehicle-routing requirements to JSON Constraint IR. "
            "Return JSON only. Supported types are capacity, same_resource, mutual_exclusion, "
            "precedence, time_window, max_route_distance, and preferred_time. "
            "Use hard=true for words equivalent to must/cannot, otherwise hard=false.\n"
            f"Valid customer IDs: {customer_ids}\n"
            f"Requirements: {description}\n"
            "Schema: {\"constraints\":[{\"type\":string,\"hard\":boolean,\"weight\":number,"
            "\"params\":object,\"source_text\":string,\"confidence\":number}]}"
        )
        raw_output = self._generate(prompt)
        value = _extract_json(raw_output)
        raw_constraints = value.get("constraints", [])
        if not isinstance(raw_constraints, list):
            raise ValueError("constraints must be a list")
        allowed = {
            "capacity",
            "same_resource",
            "same_vehicle",
            "mutual_exclusion",
            "precedence",
            "time_window",
            "max_route_distance",
            "preferred_time",
        }
        normalized = []
        for raw in raw_constraints:
            if not isinstance(raw, dict) or raw.get("type") not in allowed:
                continue
            spec = ConstraintSpec.from_dict(raw)
            mentioned: list[int] = []
            if spec.type in {"same_resource", "same_vehicle", "mutual_exclusion"}:
                mentioned = [int(x) for x in spec.params.get("entities", [])]
            elif spec.type == "precedence":
                mentioned = [int(spec.params["before"]), int(spec.params["after"])]
            elif spec.type in {"time_window", "preferred_time"}:
                entity = spec.params.get("entity", spec.params.get("customer"))
                mentioned = [int(entity)] if entity is not None else []
            if mentioned and any(x not in customer_ids for x in mentioned):
                raise ValueError(f"Constraint contains unknown customer ID: {spec.params}")
            normalized.append(spec)
        return normalized, {"prompt": prompt, "output": raw_output, "parsed": value}

    def _normalize(
        self,
        instance: CVRPInstance,
        solution: RouteSolution,
        value: dict,
    ) -> NeighborhoodProposal:
        valid_ids = set(instance.customer_ids)
        destroyed = []
        for raw in value.get("destroy_customers", []):
            customer = int(raw)
            if customer in valid_ids and customer not in destroyed:
                destroyed.append(customer)
            if len(destroyed) >= self.destroy_size:
                break
        if len(destroyed) < self.destroy_size:
            fallback = self.fallback.propose(instance, solution, iteration=0, seed=0)
            for customer in fallback.destroy_customers:
                if customer not in destroyed:
                    destroyed.append(customer)
                if len(destroyed) >= min(self.destroy_size, len(valid_ids)):
                    break
        raw_routes = value.get("candidate_routes", {})
        current = {
            customer: route_idx for route_idx, route in enumerate(solution.routes) for customer in route
        }
        candidate_routes: dict[int, list[int]] = {}
        for customer in destroyed:
            routes = raw_routes.get(str(customer), raw_routes.get(customer, []))
            normalized = []
            for raw in routes:
                route = int(raw)
                if 0 <= route < instance.vehicle_count and route not in normalized:
                    normalized.append(route)
                if len(normalized) >= self.routes_per_customer:
                    break
            if current[customer] not in normalized:
                normalized.append(current[customer])
            candidate_routes[customer] = normalized[: self.routes_per_customer]
        proposal = NeighborhoodProposal(
            destroy_customers=destroyed,
            candidate_routes=candidate_routes,
            confidence=float(value.get("confidence", 0.5)),
            source="llm",
            raw=value,
        )
        proposal.validate(instance)
        return proposal

    def _audit(self, record: dict) -> None:
        if not self.audit_path:
            return
        with self.audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def propose(self, instance: CVRPInstance, solution: RouteSolution, iteration: int, seed: int) -> NeighborhoodProposal:
        if (
            self._cache is not None
            and self._cache_instance == instance.name
            and iteration % self.refresh_interval != 0
        ):
            return NeighborhoodProposal(
                destroy_customers=list(self._cache.destroy_customers),
                candidate_routes={k: list(v) for k, v in self._cache.candidate_routes.items()},
                confidence=self._cache.confidence,
                source="llm_cached",
                raw=dict(self._cache.raw),
            )
        prompt = self._prompt(instance, solution, iteration)
        raw_output = ""
        try:
            raw_output = self._generate(prompt)
            parsed = _extract_json(raw_output)
            proposal = self._normalize(instance, solution, parsed)
            self._cache, self._cache_instance = proposal, instance.name
            self._audit(
                {
                    "instance": instance.name,
                    "iteration": iteration,
                    "ok": True,
                    "prompt": prompt,
                    "output": raw_output,
                    "proposal": parsed,
                }
            )
            return proposal
        except Exception as exc:
            fallback = self.fallback.propose(instance, solution, iteration, seed)
            fallback.source = "llm_fallback_knn"
            fallback.raw = {"error": repr(exc), "output": raw_output}
            self._audit(
                {
                    "instance": instance.name,
                    "iteration": iteration,
                    "ok": False,
                    "prompt": prompt,
                    "output": raw_output,
                    "error": repr(exc),
                }
            )
            return fallback
