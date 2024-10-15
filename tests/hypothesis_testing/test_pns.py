from circuitry.hypothesis_testing.partial_necessity import partial_necessity_test
from circuitry.mechanistic_interpretability.examples import TracrProportionTask


def test_pns_results_sanity_normal_reference_distribution():
    """
    We use PNS with the reference distribution where we sample from
    the whole model and check if the results are as expected.
    """
    size_random_circuits_small = 2
    size_random_circuits_large = 10
    # This is what determines the reference distribution
    use_complement_for_reference_distribution = False

    # Initialize TracrProportionTask with the given settings
    task = TracrProportionTask(
        device="cpu",
        n_examples=10,
        zero_ablation=True,
    )

    # Run the partial necessity test
    results_small = partial_necessity_test(
        task=task,
        candidate_circuit=task.canonical_circuit,
        n_random_circuits=100,
        size_random_circuits=size_random_circuits_small,
        use_complement_for_reference_distribution=use_complement_for_reference_distribution,
    )

    results_large = partial_necessity_test(
        task=task,
        candidate_circuit=task.canonical_circuit,
        n_random_circuits=100,
        size_random_circuits=size_random_circuits_large,
        use_complement_for_reference_distribution=use_complement_for_reference_distribution,
    )

    # Empirical quantile should be smaller for results large because
    # knocking out a larger circuit should hurt the performance more
    assert results_small["empirical_quantile"] > results_large["empirical_quantile"]


def test_pns_results_sanity_complement_reference_distribution():
    """
    We use PNS with the reference distribution where we sample from
    the whole model and check if the results are as expected.
    """
    size_random_circuits_small = 2
    size_random_circuits_large = 10
    # This is what determines the reference distribution
    use_complement_for_reference_distribution = True

    # Initialize TracrProportionTask with the given settings
    task = TracrProportionTask(
        device="cpu",
        n_examples=10,
        zero_ablation=True,
    )

    # Run the partial necessity test
    results_small = partial_necessity_test(
        task=task,
        candidate_circuit=task.canonical_circuit,
        n_random_circuits=100,
        size_random_circuits=size_random_circuits_small,
        use_complement_for_reference_distribution=use_complement_for_reference_distribution,
    )

    results_large = partial_necessity_test(
        task=task,
        candidate_circuit=task.canonical_circuit,
        n_random_circuits=100,
        size_random_circuits=size_random_circuits_large,
        use_complement_for_reference_distribution=use_complement_for_reference_distribution,
    )

    # They all should be 1.0 because:
    # 1. We are knocking out the complement of the complement of the circuit.
    # 2. The original circuit should be intact.
    # 3. The score should be the same as the original circuit.
    assert results_small["empirical_quantile"] == results_large["empirical_quantile"]
    assert results_small["empirical_quantile"] == 1.0
