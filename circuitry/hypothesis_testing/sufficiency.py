import warnings

import numpy as np
from scipy.stats import binomtest
from tqdm import tqdm

from circuitry.circuit import Circuit
from circuitry.mechanistic_interpretability import BaseTask
from circuitry.utils import to_numpy


def sufficiency_test(
    task: BaseTask,
    candidate_circuit: Circuit = None,
    size_random_circuits: int | float | None = None,
    n_random_circuits: int = 100,
    quantile=0.9,
    use_mean=False,
    verbose=False,
):
    """
    Perform the faithfulness test on the given task.

    Parameters
    ----------
    task : BaseTask
        The task to test.
    candidate_circuit : Circuit, optional
        The candidate circuit to test. If None, the canonical circuit of the task is used.
    n_random_circuits : int, optional
        The number of random circuits to sample. The default is 100.
    size_random_circuits : int | float | None, optional
        The size of the random circuits. If int, the size is set to the int. If float, the size is set to the size of
        the full circuit times the float (fraction of the full circuit).
        If None, the size is set to the size of the candidate circuit.
    quantile : float, optional
        The quantile to test against. The default is 0.9.

    the logic of the test is as follows:
    The alternative hypothesis we want to test is the candidate is more faithful than the qth most faithful random circuit
    i.e., P( candidate_circuit_metric < random_circuit_metric) > 0.9
    The test statistics is how many times the candidate_circuit_metric has lower faithfulness loss than the random_circuit_metric in the n_random_circuits experiments
    If the test statistics/n_samples is a lot bigger than 0.9, we can reject the null hypothesis
    """
    if size_random_circuits == 1:
        warnings.warn(
            f"size_random_circuits is {size_random_circuits}. Make sure that it is of the type you want (float for "
            f"a full circuit, or int for a circuit with 1 edge)",
            stacklevel=2,
        )
    if candidate_circuit is None:
        candidate_circuit = task.canonical_circuit
    if size_random_circuits is None:
        size_random_circuits = len(candidate_circuit)
    elif isinstance(size_random_circuits, float):
        size_random_circuits = int(len(task.complete_circuit) * size_random_circuits)

    complete_circuit_score, _ = task.score_and_logits(task.complete_circuit)
    candidate_circuit_score, _ = task.score_and_logits(candidate_circuit)

    random_circuit_scores = []
    random_circuit_metrics = []
    candidate_circuit_metric = task.eval_metric(
        complete_circuit_score, candidate_circuit_score, use_mean=use_mean
    ).item()

    # Sample random circuits
    random_circuit_sizes = []
    for _ in tqdm(range(n_random_circuits), disable=(not verbose)):
        random_circuit = candidate_circuit.sample_circuit(size_random_circuits)
        random_circuit_score, _ = task.score_and_logits(random_circuit)
        random_circuit_metric = task.eval_metric(
            complete_circuit_score, random_circuit_score, use_mean=use_mean
        )
        random_circuit_scores.append(random_circuit_score.cpu().detach().numpy())
        random_circuit_metrics.append(random_circuit_metric.item())
        random_circuit_sizes.append(len(random_circuit))

    test_statistics = np.sum(candidate_circuit_metric <= np.array(random_circuit_metrics))
    empirical_quantile = test_statistics / len(random_circuit_metrics)

    test_result = binomtest(
        int(test_statistics), n=len(random_circuit_metrics), p=quantile, alternative="greater"
    )

    results = {
        "test_name": "sufficiency",
        "task_name": task.name,
        "use_mean": use_mean,
        "test_quantile": quantile,
        "empirical_quantile": empirical_quantile,
        "p_value": test_result.pvalue,
        "complete_circuit_score": to_numpy(complete_circuit_score),
        "complete_circuit_size": len(task.complete_circuit),
        "candidate_circuit_score": to_numpy(candidate_circuit_score),
        "random_circuit_scores": to_numpy(random_circuit_scores),
        "candidate_circuit_metric": candidate_circuit_metric,
        "random_circuit_metrics": random_circuit_metrics,
        "candidate_circuit_size": len(candidate_circuit),
        "random_circuit_size_expected": size_random_circuits,
        "random_circuit_sizes": random_circuit_sizes,
    }

    return results
