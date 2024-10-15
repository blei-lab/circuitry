from circuitry.circuit import Circuit
from circuitry.mechanistic_interpretability.examples.induction import get_induction_model


def test_hash():
    model = get_induction_model(device="cpu")
    circuit1 = Circuit.make_circuit(model)
    circuit2 = Circuit.make_circuit(model)
    assert hash(circuit1) == hash(circuit2)

    circuit2.nodes[2].present = False
    assert hash(circuit1) != hash(circuit2)
