import pytest
from circuitry.mechanistic_interpretability.examples import TracrProportionTask, TracrReverseTask


@pytest.mark.parametrize(
    "task_cls, settings",
    [
        (TracrReverseTask, {"device": "cpu", "zero_ablation": False}),
        (TracrReverseTask, {"device": "cpu", "zero_ablation": True}),
        (TracrProportionTask, {"device": "cpu", "n_examples": 10, "zero_ablation": False}),
        (TracrProportionTask, {"device": "cpu", "n_examples": 10, "zero_ablation": True}),
    ],
)
def test_tracrs_is_perfect(task_cls, settings):
    """
    Because tracr is a constructed circuit, it should be perfect.
    and hence the score should be exactly 0.
    """
    task = task_cls(**settings)
    score_complete_circuit_1, _ = task.score_and_logits(task.complete_circuit)
    _, _ = task.score_and_logits(task.canonical_circuit)
    score_complete_circuit_2, _ = task.score_and_logits(task.complete_circuit)

    assert abs(score_complete_circuit_1.mean() - score_complete_circuit_2.mean()) < 1e-6
    assert abs(score_complete_circuit_1.mean()) < 1e-6
