import pytest
from circuitry.hypothesis_testing.non_equivalence import non_equivalence_test
from circuitry.mechanistic_interpretability.examples import (
    InductionTask,
)


@pytest.mark.parametrize(
    "task_cls, settings, should_pass, epsilon",
    [
        # should not pass: epsilon is really small.
        (InductionTask, {"n_examples": 40, "seq_len": 300}, False, 0.01),
        # should not pass: epsilon includes all of the possible interval
        (InductionTask, {"n_examples": 40, "seq_len": 300}, False, 0.5),
    ],
)
def test_sanity_non_equivalence_test(task_cls, settings, should_pass, epsilon):
    """
    Check circuit are deterministic.

    should_pass: bool
        Indicates whether the test should pass based on the task_cls and settings.
        In the case of induction we know that it should not pass due to previous
        experience with the task.
        to be exact.

    epsilon: float
        The epsilon value to use for the test. See `non_equivalence_test`
        for more information
    """
    task = task_cls(**settings)
    results = non_equivalence_test(
        task=task,
        candidate_circuit=task.canonical_circuit,
        epsilon=epsilon,
    )

    if should_pass:
        assert results["p_value"] > 0.05
    else:
        assert results["p_value"] < 0.05
