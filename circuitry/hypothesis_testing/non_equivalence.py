import math

from circuitry.circuit import Circuit
from circuitry.mechanistic_interpretability import BaseTask
from circuitry.utils import to_numpy


def non_equivalence_test(
    task: BaseTask,
    candidate_circuit: Circuit = None,
    epsilon: float = 0.1,
    verbose=False,
):
    """
    Checks that the candidate circuit has an equal chance of outperforming the original model.

    Intutiviely if the candidate circuit C is a good approximation
    of the original model M, then should perform as well as M on any
    random task input. Hence, the difference in task performance between M and C
    should be indistinguishable from chance.

    We formalize this intuition with an equivalence test: the circuit
    and the original model should have the same chance of outperforming each other.

    We write the difference in the task performance between the candidate
    circuit and the original model on one task datapoint (x, y) as
    Δ(x, y) = s(C(x); y) - s(M(x); y), and let the null hypothesis be

        H0 : | P_{(X,Y) ~ D} (Δ(X,Y) > 0) - 1/2 | < ε

    Note
    ----
    The test only works if Δ(x, y) is not equal to 0 for all (x, y) in the task.
    If this is the case this test is not adequate thought the hypothesis is still valid.

    Parameters
    ----------
    task : BaseTask
        The task to test.
    candidate_circuit : Circuit, optional
        The candidate circuit to test. If None, the canonical circuit of the task is used.
    epsilon : float, optional
        The threshold for the test. The default is 0.1.
    """
    if candidate_circuit is None:
        candidate_circuit = task.canonical_circuit

    complete_circuit_scores, _ = task.score_and_logits(task.complete_circuit)
    candidate_circuit_scores, _ = task.score_and_logits(candidate_circuit)

    assert complete_circuit_scores.shape == candidate_circuit_scores.shape
    assert len(complete_circuit_scores.shape) == 1  # should have only the n_examples dimension

    empirical_diff: float = (complete_circuit_scores - candidate_circuit_scores).mean().item()
    n_examples: int = complete_circuit_scores.shape[0]

    p_val = _compute_pval(n_examples, epsilon, empirical_diff)

    results = {
        "test_name": "non_equivalence_test",
        "task_name": task.name,
        "empirical_difference": empirical_diff,
        "epsilon": epsilon,
        "n_examples": n_examples,
        "p_value": p_val,
        "complete_circuit_score": to_numpy(complete_circuit_scores),
        "candidate_circuit_score": to_numpy(candidate_circuit_scores),
    }

    return results


def _compute_pval(n: int, epsilon: float, empirical_diff: float) -> float:
    """
    Computes the summation:
    sum_{k∈[n]} binom{n}{k} (1/2 + ε)^k (1 - 1/2 - ε)^{n-k}
    where the sum is taken over all k such that |k/n - 1/2| >= t_obs.

    Parameters:
    n (int): The upper limit of the summation. Essentially the number of examples
    epsilon (float or int): The epsilon threshold value for the condition.
    empirical_diff (float or int): The empirical mean value for the outperformance.
    t_obs = |empirical_mean - 0.5|

    Returns:
    float: p_value
    """
    result = 0.0
    t_obs = abs(empirical_diff - 0.5)

    for k in range(n + 1):
        if abs(k / n - 0.5) >= t_obs:
            binom_coeff = math.comb(n, k)
            term = binom_coeff * (0.5 + epsilon) ** k * (0.5 - epsilon) ** (n - k)
            result += term

    return result
