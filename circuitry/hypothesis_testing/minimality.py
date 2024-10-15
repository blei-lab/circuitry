import random

import numpy as np
from scipy.stats import binomtest
from tqdm import tqdm

from circuitry.circuit import Circuit
from circuitry.circuit.circuit import Edge
from circuitry.mechanistic_interpretability import BaseTask
from circuitry.utils import to_numpy


def minimality_test(
    task: BaseTask,
    candidate_circuit: Circuit = None,
    n_random_circuits: int = 100,
    quantile=0.9,
    max_edges_to_test=None,
    use_mean=False,
    verbose=False,
):
    """
    Perform the minimality test on the given task.

    Parameters
    ----------
    task : BaseTask
        The task to test.
    candidate_circuit : Circuit, optional
        The candidate circuit to test. If None, the canonical circuit of the task is used.
    n_random_circuits : int, optional
        The number of random circuits to sample. The default is 100.
    quantile : float, optional
        The quantile to test against. The default is 0.9.
    max_edges_to_test: int, optional
        Number to optionally limit the number of edges from the candidate circuit to test.

    The alternative hypothesis for an edge in the candidate circuit is
    that the edge has a more significant effect than a random edge in the inflated circuit less than q^* of the time.
    The test statistics is defined as the number of times the candidate_edge_knockout_metric is more than the random_inflated_knockout_metrics.
    If for any edge, we reject the null, we can say the circuit is not minimal.
    TODO: still need to add bonferroni correction

    """
    if candidate_circuit is None:
        candidate_circuit = task.canonical_circuit

    # 1) compute the scores for the inflated circuits
    random_inflated_scores = []
    random_knockout_scores = []
    random_inflated_knockout_metrics = []
    for _ in range(n_random_circuits):
        inflated_circuit = candidate_circuit.sample_inflated(1)
        edge_to_remove = _sample_different_edge(candidate_circuit, inflated_circuit)
        knocked_out_circuit = inflated_circuit.remove_edge(
            edge_to_remove.src_node, edge_to_remove.dst_node
        )

        inflated_score, _ = task.score_and_logits(inflated_circuit)
        knockout_score, _ = task.score_and_logits(knocked_out_circuit)
        inflated_knockout_metric = task.eval_metric(inflated_score, knockout_score)
        random_inflated_scores.append(inflated_score.mean().item())
        random_knockout_scores.append(knockout_score.mean().item())
        random_inflated_knockout_metrics.append(inflated_knockout_metric.item())
    random_inflated_knockout_metrics = np.array(random_inflated_knockout_metrics)

    # 2) Compute the scores for the edges
    candidate_score, _ = task.score_and_logits(candidate_circuit)
    edges_knockout_scores = []
    candidate_edge_knockout_metrics = []
    edges = [edge for edge in candidate_circuit.get_present_edges() if not edge.is_placeholder][
        :max_edges_to_test
    ]
    for edge in tqdm(edges, disable=not verbose):
        if edge.is_placeholder:
            raise ValueError("Should nt be there")
        knocked_out_circuit = candidate_circuit.remove_edge(edge.src_node, edge.dst_node)
        knockout_score, _ = task.score_and_logits(knocked_out_circuit)
        edges_knockout_scores.append(knockout_score.mean().item())
        candidate_edge_knockout_metric = task.eval_metric(candidate_score, knockout_score)
        candidate_edge_knockout_metrics.append(candidate_edge_knockout_metric.item())
    candidate_edge_knockout_metrics = np.array(candidate_edge_knockout_metrics)

    # Finally we compute the p-value
    p_values = []
    empirical_quantiles = []
    for candidate_edge_knockout_metric in candidate_edge_knockout_metrics:
        test_statistics = (candidate_edge_knockout_metric > random_inflated_knockout_metrics).sum()
        empirical_quantile = test_statistics / len(random_inflated_knockout_metrics)
        p_val = binomtest(
            test_statistics, n=len(random_inflated_knockout_metrics), p=quantile, alternative="less"
        )

        p_values.append(p_val)
        empirical_quantiles.append(empirical_quantile)

    results = {
        "test_name": "minimality",
        "task_name": task.name,
        "use_mean": use_mean,
        "test_quantile": quantile,
        "empirical_quantiles": empirical_quantiles,
        "p_values": p_values,
        "candidate_circuit_score": to_numpy(candidate_score).mean(),
        "candidate_edge_knockout_metrics": candidate_edge_knockout_metrics,
        "random_inflated_knockout_metrics": random_inflated_knockout_metrics,
    }

    return results


def _sample_different_edge(
    base_circuit: Circuit,
    inflated_circuit: Circuit,
) -> Edge:
    """
    Randomly selects a non-placeholder edge in inflated_circuit but not candidate_circuit.
    """
    possible_edges = [
        edge
        for edge in inflated_circuit.get_present_edges()
        if not base_circuit.edge_lookup[edge.src_node][edge.dst_node].present
        and not edge.is_placeholder
    ]
    return random.choice(possible_edges)
