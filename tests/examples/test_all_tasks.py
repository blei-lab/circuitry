import pytest
from circuitry.mechanistic_interpretability.examples import (
    DocstringTask,
    GreaterThanTask,
    InductionTask,
    IOITask,
    TracrProportionTask,
    TracrReverseTask,
)


@pytest.mark.parametrize(
    "task_cls, settings",
    [
        (InductionTask, {"n_examples": 40, "seq_len": 300}),
        (GreaterThanTask, {"device": "cpu", "n_examples": 15, "zero_ablation": False}),
        (GreaterThanTask, {"device": "cpu", "n_examples": 15, "zero_ablation": True}),
        (IOITask, {"n_examples": 3, "device": "cpu", "zero_ablation": False}),
        (IOITask, {"n_examples": 3, "device": "cpu", "zero_ablation": True}),
        (TracrReverseTask, {"device": "cpu", "zero_ablation": False}),
        (TracrReverseTask, {"device": "cpu", "zero_ablation": True}),
        (TracrProportionTask, {"device": "cpu", "n_examples": 10, "zero_ablation": False}),
        (TracrProportionTask, {"device": "cpu", "n_examples": 10, "zero_ablation": True}),
        (DocstringTask, {"device": "cpu", "n_examples": 10, "zero_ablation": False}),
        (DocstringTask, {"device": "cpu", "n_examples": 10, "zero_ablation": True}),
    ],
)
def test_sanity_full_circuit(task_cls, settings):
    """
    Check circuit are deterministic.
    """
    task = task_cls(**settings)
    score_complete_circuit_1, _ = task.score_and_logits(task.complete_circuit)
    _, _ = task.score_and_logits(task.canonical_circuit)
    score_complete_circuit_2, _ = task.score_and_logits(task.complete_circuit)

    # Induction may have different number of examples
    if "n_examples" in settings and task_cls != InductionTask:
        assert score_complete_circuit_1.shape[0] == settings["n_examples"]
        assert score_complete_circuit_2.shape[0] == settings["n_examples"]

    assert abs(score_complete_circuit_1.mean() - score_complete_circuit_2.mean()) < 1e-6
