import pytest
import torch
from circuitry.hypothesis_testing.non_independence import (
    hsic_independence_test,
    non_independence_test,
)
from circuitry.mechanistic_interpretability.examples import (
    TracrProportionTask,
)


# TODO: Induction takes too long to run maybe find an alternative way for
# testing this.
@pytest.mark.parametrize(
    "task_cls, settings, is_independent",
    [
        # should not pass: we observed this emprirically - hence this only
        # checks for consistency of the test
        # (InductionTask, {"n_examples": 110, "seq_len": 300}, False),  #
        # should pass: Tracr is a perfect circuit
        (TracrProportionTask, {"device": "cpu", "n_examples": 20, "zero_ablation": False}, True),
    ],
)
def test_sanity_full_circuit(task_cls, settings, is_independent):
    """
    is_indpendent: bool
        Indicates whether we expect the test to satisfy the independence
        criterion. The null hypothesis is that the output of the complement of the
        candidate and the original model are independent.
    """
    task = task_cls(**settings)
    n_permutations = 100

    results = non_independence_test(
        task=task,
        candidate_circuit=task.canonical_circuit,
        n_permutations=n_permutations,
    )

    # If it is independent, the
    if is_independent:
        assert results["p_value"] > 0.05
    else:
        assert results["p_value"] < 0.05


def test_sanity_hisc_independence_test():
    """
    We check that for very simple tasks the function used
    for the non-independence test is working correctly
    """
    torch.manual_seed(0)

    x = torch.arange(100).float()
    y_dependent = x + torch.randn(100)
    y_independent = torch.randn(100)

    y_dependent_results = hsic_independence_test(x, y_dependent, num_permutations=1000)
    y_independent_results = hsic_independence_test(x, y_independent, num_permutations=1000)

    assert y_dependent_results["p_value"] < 0.05
    assert y_independent_results["p_value"] > 0.05
