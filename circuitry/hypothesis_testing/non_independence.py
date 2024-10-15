import torch
from jaxtyping import Float
from torch import Tensor

from circuitry.circuit import Circuit
from circuitry.mechanistic_interpretability import BaseTask
from circuitry.utils import to_numpy


def non_independence_test(
    task: BaseTask,
    candidate_circuit: Circuit = None,
    n_permutations: int = 1000,
    verbose=False,
):
    """
    Checks that the output of the complement candidate candidate circuit
    is independent of the output of the original model.

    If a circuit, C,  is solely responsible for the operations
    relevant to a task, then knocking it out would render the complement
    circuit unable to perform the task. An implication is that the
    performance of the complement circuit is independent of the original
    model on the task.

    To formalize this claim, we define the null hypothesis as

        H0 : s(C(X); Y) ⫫ s(M(X); Y)

    where the randomness is over X and Y.

    To test this hypothesis, we use a permutation test. Specifically, we
    measure the independence between the performance of the complement
    circuit and the performance of the original model by using the Hilbert
    Schmidt Independence Criterion (HSIC) (Gretton et al., 2007), a
    nonparametric test of independence.

    Parameters
    ----------
    task : BaseTask
        The task to test.
    candidate_circuit : Circuit, optional
        The candidate circuit to test. If None, the canonical circuit of the task is used.
    n_permutations : int, optional
        The number of permutations to use for permutation test in the HSIC test. The default is 1000.
    """
    if candidate_circuit is None:
        candidate_circuit = task.canonical_circuit

    complete_circuit_scores, _ = task.score_and_logits(task.complete_circuit)
    inverted_candidate_circuit_scores, _ = task.score_and_logits(candidate_circuit, invert=True)

    assert complete_circuit_scores.shape == inverted_candidate_circuit_scores.shape
    assert len(complete_circuit_scores.shape) == 1  # should have only the n_examples dimension

    independence_results = hsic_independence_test(
        complete_circuit_scores,
        inverted_candidate_circuit_scores,
        num_permutations=n_permutations,
    )

    results = {
        "test_name": "non_independence_test",
        "task_name": task.name,
        "hisc-statistic": independence_results["hsic"],
        "p_value": independence_results["p_value"],
        "simulated_statistics": independence_results["simulated_statistics"],
        "complete_circuit_score": to_numpy(complete_circuit_scores),
        "candidate_circuit_score": to_numpy(inverted_candidate_circuit_scores),
    }

    return results


## these are for the permutation test
def _gaussian_kernel(
    X: Float[Tensor, " n_examples 1"], Y: Float[Tensor, " n_examples 1"], sigma: float
) -> Float[Tensor, " n_examples n_examples"]:
    X = X.view(-1, 1, X.size(-1))
    Y = Y.view(1, -1, Y.size(-1))
    beta = 1 / (2 * sigma**2)
    dist = torch.sum((X - Y) ** 2, dim=2)
    return torch.exp(-beta * dist)


def _hsic(
    X: Float[Tensor, " n_examples 1"], Y: Float[Tensor, " n_examples 1"], sigma: float
) -> Float[Tensor, " "]:
    n = X.size(0)

    K = _gaussian_kernel(X, X, sigma)
    L = _gaussian_kernel(Y, Y, sigma)

    H = torch.eye(n) - torch.ones((n, n)) / n
    H = H.to(X.device)
    K_centered = torch.mm(torch.mm(H, K), H)
    L_centered = torch.mm(torch.mm(H, L), H)

    hsic_statistic = torch.trace(torch.mm(K_centered, L_centered)) / (n - 1) ** 2
    return hsic_statistic


def hsic_independence_test(
    X: Float[Tensor, " n_examples"], Y: Float[Tensor, " n_examples"], num_permutations=1000
) -> dict:
    """
    Check whether X and Y are independent using the Hilbert Schmidt Independence Criterion
    (HSIC) test.

    This test essentially checks whether the joint distribution of X and Y is the product
    by doing a permutation test. See Gretton et al. (2007) for more details.

    X: torch.Tensor
        Samples from the first distribution
    Y: torch.Tensor
        Samples from the second distribution
    num_permutations: int
        Number of permutations to use for the permutation test
    """
    # check the shape of X and Y
    assert X.shape == Y.shape, "X and Y should have the same shape"
    assert len(X.shape) == 1, "X and Y should be 1D arrays"

    X = X.view(X.shape[0], -1)  # convert to 2D tensor
    Y = Y.view(Y.shape[0], -1)  # convert to 2D tensor

    # take the median of the pairwise distance as the sigma
    sigma = torch.cdist(X, Y, p=2).median()
    n = X.size(0)
    hsic_observed = _hsic(X, Y, sigma)

    # set the device to device of X
    hsic_permutations = torch.zeros(num_permutations).to(X.device)

    for i in range(num_permutations):
        Y_permuted = Y[torch.randperm(n)]
        hsic_permutations[i] = _hsic(X, Y_permuted, sigma)

    p_value = (hsic_permutations >= hsic_observed).float().mean()

    return {
        "hsic": hsic_observed.item(),
        "p_value": p_value.item(),
        "simulated_statistics": hsic_permutations.cpu().detach().numpy(),
    }
