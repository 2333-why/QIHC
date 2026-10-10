"""Shared answer parsing for selectors, direct baselines and training rewards."""

import json
import re


def answer_text(text: str) -> str:
    return re.sub(r"<think>.*?(?:</think>|$)", "", text, flags=re.DOTALL | re.IGNORECASE)


def extract_answer_json(text: str) -> dict:
    text = answer_text(text)
    decoder = json.JSONDecoder()
    candidates = []
    covered_until = -1
    for match in re.finditer(r"\{", text):
        if match.start() < covered_until:
            continue
        try:
            value, end = decoder.raw_decode(text, match.start())
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            candidates.append(value)
            covered_until = end  # Do not mistake a nested object for the final answer.
    if not candidates:
        raise ValueError("LLM output contains no complete answer JSON object")
    root_keys = {"constraints", "candidate_routes", "destroy_customers", "routes"}
    relevant = [value for value in candidates if root_keys & value.keys()]
    if not relevant and re.search(r'"(?:constraints|candidate_routes|destroy_customers|routes)"\s*:', text):
        raise ValueError("Expected optimization answer is incomplete; nested objects are not complete answers")
    return (relevant or candidates)[-1]
