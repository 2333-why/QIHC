from experiments.prepare_joint_training_data import pbit_teacher_proposal


def test_pbit_teacher_proposal_adds_residuals_only_to_candidate_routes():
    proposal = {
        "destroy_customers": [1],
        "candidate_routes": {"1": [0, 2]},
        "candidate_route_logits": {"1": {"0": 1.0, "2": -1.0}},
        "confidence": 0.8,
    }
    teacher = pbit_teacher_proposal(
        proposal,
        {"1": {"0": -0.25, "2": 0.75, "3": 4.0}},
    )
    assert teacher["candidate_route_logits"]["1"] == {"0": 0.75, "2": -0.25}
    assert proposal["candidate_route_logits"]["1"] == {"0": 1.0, "2": -1.0}
