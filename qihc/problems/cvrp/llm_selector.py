"""Offline local-LLM neighborhood selection for NL-CVRP."""

from __future__ import annotations

import json
import re
import math
import time
from pathlib import Path

from qihc.problems.cvrp.instance import CVRPInstance, ConstraintSpec, RouteSolution
from qihc.problems.cvrp.neighborhood import (
    KNNNeighborhoodSelector,
    NeighborhoodProposal,
)
from qihc.problems.cvrp.verifier import verify_solution
from qihc.problems.cvrp.json_output import answer_text, extract_answer_json


def bounded_number(value, lower: float, upper: float) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Non-finite LLM numeric output")
    return max(lower, min(upper, number))


class PBitRouteTokenLogitsProcessor:
    """Bias *generation* token logits only while writing a candidate route ID.

    Route IDs with multi-token encodings are left unchanged. The residual is
    customer-specific, so applying it to every occurrence of a digit would be
    incorrect (and would corrupt unrelated JSON numbers).
    """

    def __init__(self, tokenizer, prompt_length: int, feedback: dict[int, dict[int, float]], strength: float = 1.0):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.strength = float(strength)
        self.feedback = feedback
        self.applied_steps = 0
        self.token_ids: dict[int, dict[int, float]] = {}
        for customer, routes in feedback.items():
            mapped = {}
            for route, value in routes.items():
                encoded = tokenizer.encode(str(route), add_special_tokens=False)
                if len(encoded) == 1:
                    mapped[encoded[0]] = float(value)
            self.token_ids[customer] = mapped

    def __call__(self, input_ids, scores):
        for row in range(input_ids.shape[0]):
            generated = self.tokenizer.decode(input_ids[row, self.prompt_length:], skip_special_tokens=True)
            marker = generated.rfind('"candidate_routes"')
            if marker < 0:
                continue
            tail = generated[marker:]
            match = re.search(r'"(\d+)"\s*:\s*\[\s*(?:\d+\s*,\s*)*$', tail)
            if not match:
                continue
            for token_id, value in self.token_ids.get(int(match.group(1)), {}).items():
                scores[row, token_id] += self.strength * value
                self.applied_steps += 1
        return scores


def _extract_json(text: str) -> dict:
    return extract_answer_json(text)


def _extract_json_member(text: str, key: str):
    """Decode one complete top-level member from an otherwise truncated object."""

    text = answer_text(text)
    matches = list(re.finditer(rf'"{re.escape(key)}"\s*:', text))
    match = matches[-1] if matches else None
    if not match:
        raise ValueError(f"LLM output is missing {key}")
    start = match.end()
    while start < len(text) and text[start].isspace():
        start += 1
    value, _ = json.JSONDecoder().raw_decode(text, start)
    return value


def _extract_partial_neighborhood_json(text: str) -> dict:
    """Recover the complete candidate domain when only trailing logits were cut off."""

    destroyed = _extract_json_member(text, "destroy_customers")
    routes = _extract_json_member(text, "candidate_routes")
    if not isinstance(destroyed, list) or not isinstance(routes, dict):
        raise ValueError("Truncated LLM output has no recoverable candidate domain")
    return {
        "destroy_customers": destroyed,
        "candidate_routes": routes,
        "candidate_route_logits": {},
        "confidence": 0.5,
        "recovered_truncated_logits": True,
    }


class LocalLLMNeighborhoodSelector:
    """Generate a structured destroy/candidate-route proposal from an offline model."""

    def __init__(
        self,
        model_path: str,
        adapter_path: str | None = None,
        destroy_size: int = 8,
        routes_per_customer: int = 3,
        device: str = "cuda:0",
        max_new_tokens: int = 768,
        temperature: float = 0.2,
        refresh_interval: int = 5,
        audit_path: str | Path | None = None,
        token_feedback_strength: float = 1.0,
        max_input_tokens: int = 32768,
    ):
        try:
            import torch
            from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
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
        self.token_feedback_strength = float(token_feedback_strength)
        self.max_input_tokens = int(max_input_tokens)
        if self.audit_path:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        self.fallback = KNNNeighborhoodSelector(destroy_size, routes_per_customer)
        config = AutoConfig.from_pretrained(model_path, local_files_only=True)
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        load_kwargs = {
            "local_files_only": True,
            # torch_dtype remains compatible with the full supported
            # Transformers >=4.57 range (newer releases alias it to dtype).
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
        }
        if device.startswith("cuda"):
            # Load directly onto this torchrun rank's GPU.  This avoids first
            # materializing a 30B model in host memory and then copying it.
            load_kwargs["device_map"] = {"": device}
        self.processor = None
        if str(getattr(config, "model_type", "")).startswith("qwen3_5"):
            try:
                from transformers import AutoModelForMultimodalLM, AutoProcessor
            except ImportError as exc:  # pragma: no cover - depends on installed Transformers
                raise RuntimeError(
                    "Qwen3.5 requires Transformers with AutoModelForMultimodalLM; "
                    "install transformers>=4.57"
                ) from exc
            self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
            # The token-logit feedback processor works on the underlying text
            # tokenizer, while the processor formats Qwen3.5 text-only prompts.
            self.tokenizer = self.processor.tokenizer
            model = AutoModelForMultimodalLM.from_pretrained(model_path, **load_kwargs)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        if adapter_path:
            try:
                from peft import PeftModel
            except ImportError as exc:  # pragma: no cover - formal dependency
                raise RuntimeError("Loading a trained adapter requires peft") from exc
            model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
        self.model = model if device.startswith("cuda") else model.to(device)
        self.model.eval()
        self._cache: NeighborhoodProposal | None = None
        self._cache_instance: str | None = None
        self._pbit_logit_feedback: dict[int, dict[int, float]] = {}
        self._token_logit_feedback: dict[int, dict[int, float]] = {}
        self._distribution_diagnostics: dict = {}
        self._last_token_feedback_steps = 0

    def reset_search_state(self) -> None:
        self._cache = None
        self._cache_instance = None
        self._pbit_logit_feedback = {}
        self._token_logit_feedback = {}
        self._distribution_diagnostics = {}
        self._generation_totals = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "elapsed_s": 0.0}

    def set_feedback_channels(self, prompt_feedback: dict, token_feedback: dict, diagnostics: dict) -> None:
        self._pbit_logit_feedback = prompt_feedback
        self._token_logit_feedback = token_feedback
        self._distribution_diagnostics = diagnostics

    def set_pbit_logit_feedback(self, feedback: dict[int, dict[int, float]]) -> None:
        """Inject the previous p-bit posterior residuals into the next LLM turn."""
        self._pbit_logit_feedback = {
            int(customer): {int(route): float(value) for route, value in routes.items()}
            for customer, routes in feedback.items()
        }
        self._token_logit_feedback = dict(self._pbit_logit_feedback)

    def _move_features(
        self, instance: CVRPInstance, solution: RouteSolution
    ) -> list[dict]:
        """Build a compact objective-aware shortlist without choosing the final move."""

        route_loads = [
            sum(instance.customers[customer].demand for customer in route)
            for route in solution.routes
        ]
        constrained: set[int] = set()
        for spec in instance.constraints:
            if spec.type in {"same_resource", "same_vehicle", "mutual_exclusion"}:
                constrained.update(int(x) for x in spec.params.get("entities", []))
            elif spec.type == "precedence":
                constrained.update(
                    int(spec.params[key]) for key in ("before", "after") if key in spec.params
                )
            elif spec.type in {"time_window", "preferred_time"}:
                entity = spec.params.get("entity", spec.params.get("customer"))
                if entity is not None:
                    constrained.add(int(entity))

        rows = []
        for current_route, route in enumerate(solution.routes):
            extended = [instance.depot.id, *route, instance.depot.id]
            for position, customer in enumerate(route):
                node = instance.customers[customer]
                previous = extended[position]
                following = extended[position + 2]
                removal_saving = (
                    instance.distance(previous, customer)
                    + instance.distance(customer, following)
                    - instance.distance(previous, following)
                )
                moves = []
                for target_route, target in enumerate(solution.routes):
                    if target_route == current_route:
                        continue
                    remaining = instance.vehicle_capacity - route_loads[target_route]
                    target_extended = [instance.depot.id, *target, instance.depot.id]
                    insertion = min(
                        instance.distance(target_extended[index], customer)
                        + instance.distance(customer, target_extended[index + 1])
                        - instance.distance(target_extended[index], target_extended[index + 1])
                        for index in range(len(target_extended) - 1)
                    )
                    moves.append(
                        {
                            "route": target_route,
                            "net_delta": round(insertion - removal_saving, 2),
                            "remaining": int(remaining - node.demand),
                            "capacity_release_required": max(0, int(node.demand - remaining)),
                        }
                    )
                moves.sort(key=lambda item: (item["net_delta"], item["route"]))
                moves = moves[: max(1, self.routes_per_customer - 1)]
                rows.append(
                    {
                        "customer": customer,
                        "current_route": current_route,
                        "position": position,
                        "demand": node.demand,
                        "removal_saving": round(removal_saving, 2),
                        "moves": moves,
                        "best_net_delta": moves[0]["net_delta"] if moves else 0.0,
                        "constraint_related": customer in constrained,
                    }
                )

        rows.sort(
            key=lambda row: (
                not row["constraint_related"],
                row["best_net_delta"],
                row["customer"],
            )
        )
        shortlist_size = max(24, self.destroy_size * 4)
        required = [row for row in rows if row["constraint_related"]]
        selected_ids = {row["customer"] for row in required}
        selected = list(required)
        for row in sorted(rows, key=lambda item: (item["best_net_delta"], item["customer"])):
            if row["customer"] not in selected_ids:
                selected.append(row)
                selected_ids.add(row["customer"])
            if len(selected) >= shortlist_size:
                break
        return selected

    def _prompt(self, instance: CVRPInstance, solution: RouteSolution, iteration: int) -> str:
        verification = verify_solution(instance, solution)
        route_summary = [
            {
                "route": idx,
                "load": verification.route_loads[idx] if idx < len(verification.route_loads) else 0,
                "remaining": instance.vehicle_capacity
                - (verification.route_loads[idx] if idx < len(verification.route_loads) else 0),
            }
            for idx, _route in enumerate(solution.routes)
        ]
        move_features = self._move_features(instance, solution)
        feature_ids = {row["customer"] for row in move_features}
        constraints = [
            {"type": c.type, "hard": c.hard, "weight": c.weight, **c.params}
            for c in instance.constraints
        ]
        compact_feedback = {
            str(customer): {
                str(route): round(float(value), 3)
                for route, value in routes.items()
            }
            for customer, routes in self._pbit_logit_feedback.items()
            if customer in feature_ids
        }
        return (
            "You understand the complete constrained CVRP and propose a candidate solution/search region. "
            "Do not create an energy function. Return JSON only. Select customers whose reassignment may reduce "
            "distance while respecting hard constraints. Every selected customer needs candidate vehicle route IDs.\n"
            "The move feature net_delta is the exact geometric objective change before secondary interactions; "
            "negative is promising. Choose destroy_customers from the supplied feature shortlist and choose route "
            "IDs only from that customer's moves plus its current_route. p-bit will perform the final joint search.\n"
            "A positive capacity_release_required means another selected customer must leave that route; "
            "such moves can be feasible jointly even though they are infeasible in isolation.\n"
            f"Iteration: {iteration}\n"
            f"Vehicle capacity: {instance.vehicle_capacity}\n"
            f"Constraints: {json.dumps(constraints, ensure_ascii=False)}\n"
            f"Route capacities: {json.dumps(route_summary, ensure_ascii=False, separators=(',', ':'))}\n"
            f"Objective-aware move features: {json.dumps(move_features, ensure_ascii=False, separators=(',', ':'))}\n"
            f"P-bit logit feedback from earlier iterations: "
            f"{json.dumps(compact_feedback, ensure_ascii=False, separators=(',', ':'))}\n"
            f"Empirical population diagnostics: {json.dumps({k: v for k, v in getattr(self, '_distribution_diagnostics', {}).items() if k != 'marginals'}, ensure_ascii=False, separators=(',', ':'))}\n"
            "Positive feedback means p-bit repeatedly selected that assignment in improving samples; "
            "negative feedback means it was unsupported or harmful. Use it as evidence, not a hard rule.\n"
            f"Required destroy count: {min(self.destroy_size, len(move_features))}\n"
            "Destroy means include in the search region, not force a route change. "
            "Each customer may keep its current route. Propose candidates, not a final "
            "joint assignment; p-bit and the verifier handle joint feasibility.\n"
            f"Maximum candidate routes per customer: {self.routes_per_customer}\n"
            "Return one compact JSON object with exactly these four fields; omit explanations, "
            "candidate_solution, candidate_edges, and reason. candidate_routes is the proposed "
            "candidate assignment/search region that p-bit will optimize.\n"
            "Write compact one-line JSON. Every candidate_route_logits value must be a short "
            "integer preference score from -2 to 2, where higher means more preferred; never "
            "copy long decimal feedback values.\n"
            "Schema: {\"destroy_customers\":[int],"
            "\"candidate_routes\":{\"customer_id\":[route_id]},"
            "\"candidate_route_logits\":{\"customer_id\":{\"route_id\":number}},"
            "\"confidence\":number}"
        )

    def _generate(self, prompt: str) -> str:
        started = time.perf_counter()
        self._last_generation_diagnostics = {"thinking_enabled": False}
        messages = [{"role": "user", "content": prompt}]
        if self.processor is not None:
            inputs = self.processor.apply_chat_template(
                [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
                add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
                enable_thinking=False,
            )
        elif hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            rendered = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            inputs = self.tokenizer(rendered, return_tensors="pt", truncation=False)
        else:
            rendered = prompt
            inputs = self.tokenizer(rendered, return_tensors="pt", truncation=False)
        input_length = int(inputs["input_ids"].shape[1])
        if input_length > self.max_input_tokens:
            raise ValueError(
                f"LLM prompt has {input_length} tokens, exceeding the explicit "
                f"limit of {self.max_input_tokens}; refusing silent truncation"
            )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        do_sample = self.temperature > 0
        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs.update({"temperature": self.temperature, "top_p": 0.9})
        processor = None
        token_feedback = getattr(self, "_token_logit_feedback", self._pbit_logit_feedback)
        if token_feedback and self.token_feedback_strength:
            processor = PBitRouteTokenLogitsProcessor(
                self.tokenizer, inputs["input_ids"].shape[1],
                token_feedback, self.token_feedback_strength,
            )
            generation_kwargs["logits_processor"] = [processor]
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, **generation_kwargs)
        generated = output[0, inputs["input_ids"].shape[1] :]
        self._last_generation_diagnostics = {
            "thinking_enabled": False, "generated_tokens": int(generated.shape[0]),
            "max_new_tokens": self.max_new_tokens,
            "reached_token_limit": int(generated.shape[0]) >= self.max_new_tokens,
        }
        statistics = {"calls": 1, "input_tokens": input_length, "output_tokens": int(generated.shape[0]),
                      "elapsed_s": time.perf_counter() - started}
        totals = getattr(self, "_generation_totals", {"calls": 0, "input_tokens": 0, "output_tokens": 0, "elapsed_s": 0.0})
        self._generation_totals = {key: totals[key] + statistics[key] for key in totals}
        self._last_token_feedback_steps = processor.applied_steps if processor else 0
        return (self.processor.decode(generated, skip_special_tokens=True)
                if self.processor is not None else self.tokenizer.decode(generated, skip_special_tokens=True))

    def propose_cold_batch(
        self, instance: CVRPInstance, partial: RouteSolution,
        domain: NeighborhoodProposal, iteration: int, seed: int,
    ) -> NeighborhoodProposal:
        """Ask for a bounded residual preference after exposing exact numeric features."""
        context = domain.raw.get("cold_context", {}) if isinstance(domain.raw, dict) else {}
        prompt = (
            "Construct a CVRP solution from empty routes, one batch at a time. "
            "The supplied insertion_delta is the trusted geometric baseline. Add only a small residual "
            "preference for future capacity scarcity and semantic constraints; p-bit decides the joint assignment. "
            "Return JSON with candidate_routes, candidate_route_scores in [-1,1], and confidence in [0,1]. "
            "Use every supplied route exactly once in each customer's ranked list; never invent or remove a route. "
            "A low-confidence answer should use zero scores.\n"
            f"Numeric construction state: {json.dumps(context, ensure_ascii=False, separators=(',', ':'))}\n"
            f"Allowed domains: {json.dumps(domain.candidate_routes, separators=(',', ':'))}\n"
            f"Constraints: {json.dumps([{'type': c.type, 'params': c.params} for c in instance.constraints], ensure_ascii=False)}"
        )
        try:
            raw_output = self._generate(prompt)
            value = _extract_json(raw_output)
            raw = value.get("candidate_routes", {})
            raw_scores = value.get("candidate_route_scores", value.get("candidate_route_logits", {}))
            candidates = {}
            logits = {}
            for customer in domain.destroy_customers:
                selected = [int(route) for route in raw.get(str(customer), raw.get(customer, []))]
                preferred = [
                    route for route in selected
                    if route in domain.candidate_routes[customer]
                ]
                candidates[customer] = list(dict.fromkeys(
                    preferred + domain.candidate_routes[customer]
                ))
                supplied = raw_scores.get(str(customer), raw_scores.get(customer, {}))
                logits[customer] = {
                    route: bounded_number(supplied.get(str(route), supplied.get(route, 0.0)), -1.0, 1.0)
                    for route in candidates[customer]
                }
            confidence = bounded_number(value.get("confidence", 0.5), 0.0, 1.0)
            trace_raw = dict(value)
            trace_raw.update({
                "cold_prompt": prompt,
                "cold_output": raw_output,
                "cold_context": context,
            })
            proposal = NeighborhoodProposal(
                list(domain.destroy_customers), candidates,
                source="llm_cold_domain", raw=trace_raw,
                confidence=confidence,
                candidate_route_logits=logits,
            )
            proposal.validate(instance)
            self._audit({
                "instance": instance.name,
                "search_seed": seed,
                "iteration": iteration,
                "phase": "cold_start",
                "ok": True,
                "prompt": prompt,
                "output": raw_output,
                "proposal": value,
            })
            return proposal
        except Exception as exc:
            domain.raw = {
                **(domain.raw if isinstance(domain.raw, dict) else {}),
                "cold_prompt": prompt,
                "cold_llm_error": repr(exc),
            }
            return domain

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
            "Use these exact params schemas: same_resource/mutual_exclusion => "
            "{\"entities\":[customer_a,customer_b]}; precedence => "
            "{\"before\":customer_a,\"after\":customer_b}. Do not rename these keys.\n"
            f"Valid customer IDs: {customer_ids}\n"
            f"Requirements: {description}\n"
            "Schema: {\"constraints\":[{\"type\":string,\"hard\":boolean,\"weight\":number,"
            "\"params\":object,\"source_text\":string,\"confidence\":number}]}"
        )
        raw_output = self._generate(prompt)
        value = _extract_json(raw_output)
        if "constraints" not in value:
            raise ValueError("Constraint IR response is missing the constraints field")
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
        dropped = []
        valid_ids = set(customer_ids)
        for index, raw in enumerate(raw_constraints):
            if not isinstance(raw, dict) or raw.get("type") not in allowed:
                dropped.append({"index": index, "reason": "unsupported_or_malformed", "value": raw})
                continue
            params = raw.get("params", {})
            if not isinstance(params, dict):
                dropped.append({"index": index, "reason": "params_not_object", "value": raw})
                continue
            params = dict(params)
            for key, item in raw.items():
                if key not in {"type", "hard", "weight", "params", "source_text", "confidence"}:
                    params.setdefault(key, item)
            kind = str(raw["type"])
            try:
                mentioned: list[int] = []
                if kind in {"same_resource", "same_vehicle", "mutual_exclusion"}:
                    raw_entities = params.get(
                        "entities",
                        params.get("customer_ids", params.get("customers", [])),
                    )
                    entities = list(dict.fromkeys(int(x) for x in raw_entities))
                    if len(entities) < 2:
                        raise ValueError("requires at least two entities")
                    params["entities"] = entities[:2]
                    params.pop("customer_ids", None)
                    params.pop("customers", None)
                    mentioned = params["entities"]
                elif kind == "precedence":
                    before_value = params.get("before", params.get("before_customer_id"))
                    after_value = params.get("after", params.get("after_customer_id"))
                    if before_value is None or after_value is None:
                        raise ValueError("requires before and after")
                    before, after = int(before_value), int(after_value)
                    if before == after:
                        raise ValueError("before and after must differ")
                    params.update({"before": before, "after": after})
                    params.pop("before_customer_id", None)
                    params.pop("after_customer_id", None)
                    mentioned = [before, after]
                elif kind in {"time_window", "preferred_time"}:
                    entity = params.get("entity", params.get("customer"))
                    if entity is None:
                        raise ValueError("requires entity/customer")
                    params["entity"] = int(entity)
                    mentioned = [params["entity"]]
                if mentioned and any(x not in valid_ids for x in mentioned):
                    raise ValueError(f"unknown customer IDs: {mentioned}")
                candidate = dict(raw)
                candidate["params"] = params
                spec = ConstraintSpec.from_dict(candidate)
            except (TypeError, ValueError, KeyError) as exc:
                dropped.append({"index": index, "reason": str(exc), "value": raw})
                continue
            normalized.append(spec)
        return normalized, {
            "backend": "local_llm",
            "prompt": prompt,
            "output": raw_output,
            "parsed": value,
            "dropped": dropped,
        }

    def _normalize(
        self,
        instance: CVRPInstance,
        solution: RouteSolution,
        value: dict,
    ) -> NeighborhoodProposal:
        valid_ids = set(instance.customer_ids)
        feature_rows = self._move_features(instance, solution)
        allowed_routes = {
            row["customer"]: {
                row["current_route"],
                *(move["route"] for move in row["moves"]),
            }
            for row in feature_rows
        }
        shortlist_ids = set(allowed_routes)
        destroyed = []
        for raw in value.get("destroy_customers", []):
            customer = int(raw)
            if customer in valid_ids and customer in shortlist_ids and customer not in destroyed:
                destroyed.append(customer)
            if len(destroyed) >= self.destroy_size:
                break
        if len(destroyed) < self.destroy_size:
            retained_model_customers = len(destroyed)
            fallback = self.fallback.propose(instance, solution, iteration=0, seed=0)
            for customer in fallback.destroy_customers:
                if customer not in destroyed and customer in shortlist_ids:
                    destroyed.append(customer)
                if len(destroyed) >= min(self.destroy_size, len(shortlist_ids)):
                    break
        else:
            retained_model_customers = len(destroyed)
        raw_routes = value.get("candidate_routes", {})
        current = {
            customer: route_idx for route_idx, route in enumerate(solution.routes) for customer in route
        }
        candidate_routes: dict[int, list[int]] = {}
        candidate_route_logits: dict[int, dict[int, float]] = {}
        raw_logits = value.get("candidate_route_logits", {})
        for customer in destroyed:
            routes = raw_routes.get(str(customer), raw_routes.get(customer, []))
            normalized = []
            for raw in routes:
                route = int(raw)
                if (
                    0 <= route < instance.vehicle_count
                    and route in allowed_routes.get(customer, {current[customer]})
                    and route not in normalized
                ):
                    normalized.append(route)
                if len(normalized) >= self.routes_per_customer:
                    break
            if current[customer] not in normalized:
                if len(normalized) >= self.routes_per_customer:
                    normalized[-1] = current[customer]
                else:
                    normalized.append(current[customer])
            candidate_routes[customer] = normalized[: self.routes_per_customer]
            supplied = raw_logits.get(str(customer), raw_logits.get(customer, {}))
            candidate_route_logits[customer] = {
                route: bounded_number(supplied.get(str(route), supplied.get(route, 0.0)), -2.0, 2.0)
                for route in candidate_routes[customer]
            }
        candidate_edges = []
        for edge in value.get("candidate_edges", []):
            if isinstance(edge, (list, tuple)) and len(edge) == 2:
                left, right = int(edge[0]), int(edge[1])
                if left in valid_ids and right in valid_ids and left != right:
                    candidate_edges.append((left, right))
        proposal = NeighborhoodProposal(
            destroy_customers=destroyed,
            candidate_routes=candidate_routes,
            confidence=bounded_number(value.get("confidence", 0.5), 0.0, 1.0),
            source="llm" if retained_model_customers else "llm_fallback_knn",
            raw={**value, "retained_model_customers": retained_model_customers,
                 "normalization_filled_customers": len(destroyed) - retained_model_customers},
            candidate_route_logits=candidate_route_logits,
            candidate_edges=candidate_edges,
        )
        proposal.validate(instance)
        return proposal

    def _audit(self, record: dict) -> None:
        record.setdefault("generation", getattr(self, "_last_generation_diagnostics", {}))
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
                source=self._cache.source + "_cached" if "fallback" in self._cache.source else "llm_cached",
                raw=dict(self._cache.raw),
                candidate_route_logits={
                    customer: dict(logits)
                    for customer, logits in self._cache.candidate_route_logits.items()
                },
                candidate_edges=list(self._cache.candidate_edges),
            )
        prompt = self._prompt(instance, solution, iteration)
        raw_output = ""
        try:
            raw_output = self._generate(prompt)
            try:
                parsed = _extract_json(raw_output)
            except ValueError:
                parsed = _extract_partial_neighborhood_json(raw_output)
            proposal = self._normalize(instance, solution, parsed)
            self._cache, self._cache_instance = proposal, instance.name
            self._audit(
                {
                    "instance": instance.name,
                    "search_seed": seed,
                    "iteration": iteration,
                    "ok": True,
                    "prompt": prompt,
                    "output": raw_output,
                    "proposal": parsed,
                    "token_feedback_applied_steps": self._last_token_feedback_steps,
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
                    "search_seed": seed,
                    "iteration": iteration,
                    "ok": False,
                    "prompt": prompt,
                    "output": raw_output,
                    "error": repr(exc),
                }
            )
            return fallback
