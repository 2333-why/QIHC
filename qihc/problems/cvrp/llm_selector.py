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


def _extract_json_member(text: str, key: str):
    """Decode one complete top-level member from an otherwise truncated object."""

    match = re.search(rf'"{re.escape(key)}"\s*:', text)
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
        self.token_feedback_strength = float(token_feedback_strength)
        if self.audit_path:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        self.fallback = KNNNeighborhoodSelector(destroy_size, routes_per_customer)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        if adapter_path:
            try:
                from peft import PeftModel
            except ImportError as exc:  # pragma: no cover - formal dependency
                raise RuntimeError("Loading a trained adapter requires peft") from exc
            model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
        self.model = model.to(device)
        self.model.eval()
        self._cache: NeighborhoodProposal | None = None
        self._cache_instance: str | None = None
        self._pbit_logit_feedback: dict[int, dict[int, float]] = {}
        self._last_token_feedback_steps = 0

    def set_pbit_logit_feedback(self, feedback: dict[int, dict[int, float]]) -> None:
        """Inject the previous p-bit posterior residuals into the next LLM turn."""
        self._pbit_logit_feedback = {
            int(customer): {int(route): float(value) for route, value in routes.items()}
            for customer, routes in feedback.items()
        }

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
                    if remaining < node.demand:
                        continue
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
                        }
                    )
                moves.sort(key=lambda item: (item["net_delta"], item["route"]))
                moves = moves[: max(1, self.routes_per_customer - 1)]
                if not moves:
                    continue
                rows.append(
                    {
                        "customer": customer,
                        "current_route": current_route,
                        "position": position,
                        "demand": node.demand,
                        "removal_saving": round(removal_saving, 2),
                        "moves": moves,
                        "best_net_delta": moves[0]["net_delta"],
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
            f"Iteration: {iteration}\n"
            f"Vehicle capacity: {instance.vehicle_capacity}\n"
            f"Constraints: {json.dumps(constraints, ensure_ascii=False)}\n"
            f"Route capacities: {json.dumps(route_summary, ensure_ascii=False, separators=(',', ':'))}\n"
            f"Objective-aware move features: {json.dumps(move_features, ensure_ascii=False, separators=(',', ':'))}\n"
            f"P-bit logit feedback from earlier iterations: "
            f"{json.dumps(compact_feedback, ensure_ascii=False, separators=(',', ':'))}\n"
            "Positive feedback means p-bit repeatedly selected that assignment in improving samples; "
            "negative feedback means it was unsupported or harmful. Use it as evidence, not a hard rule.\n"
            f"Required destroy count: {self.destroy_size}\n"
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
        processor = None
        if self._pbit_logit_feedback and self.token_feedback_strength:
            processor = PBitRouteTokenLogitsProcessor(
                self.tokenizer, inputs["input_ids"].shape[1],
                self._pbit_logit_feedback, self.token_feedback_strength,
            )
            generation_kwargs["logits_processor"] = [processor]
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, **generation_kwargs)
        generated = output[0, inputs["input_ids"].shape[1] :]
        self._last_token_feedback_steps = processor.applied_steps if processor else 0
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def propose_cold_batch(
        self, instance: CVRPInstance, partial: RouteSolution,
        domain: NeighborhoodProposal, iteration: int, seed: int,
    ) -> NeighborhoodProposal:
        """LLM narrows a cold-start batch; p-bit still chooses the assignment."""
        loads = [sum(instance.customers[c].demand for c in route) for route in partial.routes]
        prompt = (
            "Construct a CVRP solution from empty routes, one batch at a time. "
            "Choose candidate vehicle IDs for each unassigned customer; p-bit will decide the joint assignment. "
            "Return compact JSON {\"candidate_routes\":{\"customer_id\":[route_id]}}. "
            "Only use route IDs from the supplied domains and include at least one per customer.\n"
            f"Capacity: {instance.vehicle_capacity}; route_loads: {json.dumps(loads)}\n"
            f"Customers: {json.dumps({c: instance.customers[c].demand for c in domain.destroy_customers})}\n"
            f"Allowed domains: {json.dumps(domain.candidate_routes)}\n"
            f"Constraints: {json.dumps([{'type': c.type, 'params': c.params} for c in instance.constraints], ensure_ascii=False)}"
        )
        try:
            value = _extract_json(self._generate(prompt))
            raw = value.get("candidate_routes", {})
            candidates = {}
            for customer in domain.destroy_customers:
                selected = [int(route) for route in raw.get(str(customer), raw.get(customer, []))]
                preferred = [
                    route for route in selected
                    if route in domain.candidate_routes[customer]
                ][:self.routes_per_customer]
                candidates[customer] = list(dict.fromkeys(
                    preferred + domain.candidate_routes[customer]
                ))
            proposal = NeighborhoodProposal(
                list(domain.destroy_customers), candidates,
                source="llm_cold_domain", raw=value,
                candidate_route_logits={
                    customer: {route: float(len(candidates[customer]) - index)
                               for index, route in enumerate(candidates[customer])}
                    for customer in candidates
                },
            )
            proposal.validate(instance)
            return proposal
        except Exception as exc:
            domain.raw = {"cold_llm_error": repr(exc)}
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
                normalized.append(current[customer])
            candidate_routes[customer] = normalized[: self.routes_per_customer]
            supplied = raw_logits.get(str(customer), raw_logits.get(customer, {}))
            candidate_route_logits[customer] = {
                route: float(supplied.get(str(route), supplied.get(route, 0.0)))
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
            confidence=float(value.get("confidence", 0.5)),
            source="llm",
            raw=value,
            candidate_route_logits=candidate_route_logits,
            candidate_edges=candidate_edges,
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
                    "iteration": iteration,
                    "ok": False,
                    "prompt": prompt,
                    "output": raw_output,
                    "error": repr(exc),
                }
            )
            return fallback
