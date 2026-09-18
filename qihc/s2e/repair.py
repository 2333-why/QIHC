"""Deterministic, auditable CPP repair after verifier failures."""

from __future__ import annotations

from dataclasses import replace

from qihc.s2e.cpp import ConstraintProgram, ConstraintProgramPackage, SUPPORTED_CONSTRAINTS
from qihc.s2e.validator import CPPValidationReport


def repair_cpp(cpp: ConstraintProgramPackage, report: CPPValidationReport) -> ConstraintProgramPackage:
    valid_ids = set(cpp.customer_ids); repaired: list[ConstraintProgram] = []; actions = []
    for index, program in enumerate(cpp.programs):
        original = program
        if program.type == "same_vehicle":
            program = replace(program, type="same_resource"); actions.append({"program": original.id, "action": "canonicalize_type", "to": "same_resource"})
        if program.type not in SUPPORTED_CONSTRAINTS:
            actions.append({"program": original.id, "action": "drop_unsupported"}); continue
        params = dict(program.params)
        if program.type in {"same_resource", "mutual_exclusion"}:
            entities = [int(x) for x in params.get("entities", params.get("customers", [])) if int(x) in valid_ids]
            if len(entities) < 2:
                actions.append({"program": original.id, "action": "drop_invalid_arity"}); continue
            params = {"entities": entities[:2]}
        elif program.type == "precedence":
            before, after = int(params.get("before", -1)), int(params.get("after", -1))
            if before not in valid_ids or after not in valid_ids or before == after:
                actions.append({"program": original.id, "action": "drop_invalid_precedence"}); continue
            params = {"before": before, "after": after}
        encodings = tuple(x for x in program.candidate_encodings if x in {"qubo", "pdit", "mfc"}) or ("qubo", "pdit", "mfc")
        tests = program.tests
        if not tests and program.type in {"same_resource", "mutual_exclusion"}:
            a, b = params["entities"]
            expected = program.type == "same_resource"
            tests = (
                {"routes": [[a, b]], "expected": expected},
                {"routes": [[a], [b]], "expected": not expected},
            )
            actions.append({"program": original.id, "action": "add_metamorphic_tests"})
        program = replace(program, id=f"c{index:03d}-{original.id.split('-')[-1]}", params=params, candidate_encodings=encodings, tests=tests)
        repaired.append(program)
    history = [*cpp.repair_history, {"input_checksum": cpp.checksum, "failed_stage": report.highest_stage, "counterexamples": report.counterexamples, "actions": actions}]
    return ConstraintProgramPackage(cpp.instance_name, cpp.description, list(cpp.customer_ids), repaired, cpp.schema_version, dict(cpp.generator), history)
