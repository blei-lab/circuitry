"""
The canonical circuits below are the same as those used in the ACDC library (https://github.com/ArthurConmy/Automatic-Circuit-Discovery/), which accompanies the ACDC paper (https://arxiv.org/abs/2304.14997).
"""

from dataclasses import dataclass

from circuitry.circuit import Circuit, Node, NodeType
from circuitry.utils import DummyModel


def manual_add_edge(circuit, src_node, dst_node):
    circuit.node_lookup[dst_node].present = True
    circuit.node_lookup[src_node].present = True
    circuit.edge_lookup[src_node][dst_node].present = True

    if dst_node.node_type in {NodeType.ATTN_Q, NodeType.ATTN_K, NodeType.ATTN_V}:
        attn_out_node = Node(NodeType.ATTN_OUT, dst_node.layer_idx, dst_node.head_idx)
        circuit.node_lookup[attn_out_node].present = True


@dataclass(frozen=True)
class Conn:
    inp: str
    out: str
    qkv: tuple[str, ...]


IOI_CIRCUIT = {
    "name mover": [
        (9, 9),  # by importance
        (10, 0),
        (9, 6),
    ],
    "backup name mover": [
        (10, 10),
        (10, 6),
        (10, 2),
        (10, 1),
        (11, 2),
        (9, 7),
        (9, 0),
        (11, 9),
    ],
    "negative": [(10, 7), (11, 10)],
    "s2 inhibition": [(7, 3), (7, 9), (8, 6), (8, 10)],
    "induction": [(5, 5), (5, 8), (5, 9), (6, 9)],
    "duplicate token": [
        (0, 1),
        (0, 10),
        (3, 0),
    ],
    "previous token": [
        (2, 2),
        (4, 11),
    ],
}


special_connections: set[Conn] = {
    Conn("INPUT", "previous token", ("q", "k", "v")),
    Conn("INPUT", "duplicate token", ("q", "k", "v")),
    Conn("INPUT", "s2 inhibition", ("q",)),
    Conn("INPUT", "negative", ("k", "v")),
    Conn("INPUT", "name mover", ("k", "v")),
    Conn("INPUT", "backup name mover", ("k", "v")),
    Conn("previous token", "induction", ("k", "v")),
    Conn("induction", "s2 inhibition", ("k", "v")),
    Conn("duplicate token", "s2 inhibition", ("k", "v")),
    Conn("s2 inhibition", "negative", ("q",)),
    Conn("s2 inhibition", "name mover", ("q",)),
    Conn("s2 inhibition", "backup name mover", ("q",)),
    Conn("negative", "OUTPUT", ()),
    Conn("name mover", "OUTPUT", ()),
    Conn("backup name mover", "OUTPUT", ()),
}


def make_ioi_circuit(node_dict, special_connections):
    model = DummyModel(12, 12, False)
    circuit = Circuit.make_circuit(model)
    circuit.set_all_nodes_edges(False)
    manual_add_edge(circuit, circuit.sources[0], circuit.sink)

    t_to_name = {}
    for name, lt in node_dict.items():
        for t in lt:
            t_to_name[t] = name

    present_heads = t_to_name.keys()
    conn_map = {}
    for conn in special_connections:
        conn_map[(conn.inp, conn.out)] = "".join(conn.qkv)

    for i in range(model.cfg.n_layers):
        # input -> MLP
        manual_add_edge(circuit, circuit.sources[0], Node(NodeType.MLP, i))
        # MLP -> output
        manual_add_edge(circuit, Node(NodeType.MLP, i), circuit.sink)
        # MLP -> MLP
        for j in range(i):
            manual_add_edge(circuit, Node(NodeType.MLP, j), Node(NodeType.MLP, i))

    for t in present_heads:
        # input -> head
        for letter in "qkv":
            c = ("INPUT", t_to_name[t])
            if c in conn_map and letter in conn_map[c]:
                manual_add_edge(
                    circuit, circuit.sources[0], Node(NodeType(f"attn_{letter}"), t[0], t[1])
                )

        # head -> output
        c = (t_to_name[t], "OUTPUT")
        if c in conn_map:
            manual_add_edge(circuit, Node(NodeType.ATTN_OUT, t[0], t[1]), circuit.sink)

        # MLP -> head
        for i in range(t[0]):
            for letter in "qkv":
                manual_add_edge(
                    circuit, Node(NodeType.MLP, i), Node(NodeType(f"attn_{letter}"), t[0], t[1])
                )

        # head -> MLP
        for i in range(t[0], model.cfg.n_layers):
            manual_add_edge(circuit, Node(NodeType.ATTN_OUT, t[0], t[1]), Node(NodeType.MLP, i))

        # head1 -> head2
        src_node = Node(NodeType.ATTN_OUT, t[0], t[1])
        for out_edge in circuit.node_lookup[src_node].out_edges:
            dst_node = out_edge.dst_node
            t2 = (dst_node.layer_idx, dst_node.head_idx)
            if t2 not in present_heads:
                continue
            for letter in "qkv":
                if dst_node.node_type == NodeType(f"attn_{letter}"):
                    c = (t_to_name[t], t_to_name[t2])
                    if c in conn_map and letter in conn_map[c]:
                        manual_add_edge(circuit, src_node, dst_node)

    # print(f"Nodes: {circuit.count_present_nodes()}, edges: {circuit.count_present_edges()}")

    return circuit


def get_ioi_canonical_circuit():
    return make_ioi_circuit(IOI_CIRCUIT, special_connections)


def get_docstring_canonical_circuit():
    model = DummyModel(4, 8, True)  # attn-only-4l tl_model
    circuit = Circuit.make_circuit(model)
    circuit.set_all_nodes_edges(False)

    manual_add_edge(circuit, Node(NodeType.ATTN_OUT, 0, 5), Node(NodeType.ATTN_V, 1, 4))
    manual_add_edge(circuit, Node(NodeType.TOK_EMBED, 0), Node(NodeType.ATTN_V, 0, 5))
    manual_add_edge(circuit, Node(NodeType.TOK_EMBED, 0), Node(NodeType.ATTN_Q, 2, 0))
    manual_add_edge(circuit, Node(NodeType.ATTN_OUT, 0, 5), Node(NodeType.ATTN_Q, 2, 0))
    manual_add_edge(circuit, Node(NodeType.TOK_EMBED, 0), Node(NodeType.ATTN_K, 2, 0))
    manual_add_edge(circuit, Node(NodeType.ATTN_OUT, 0, 5), Node(NodeType.ATTN_K, 2, 0))
    manual_add_edge(circuit, Node(NodeType.ATTN_OUT, 1, 4), Node(NodeType.ATTN_V, 2, 0))
    manual_add_edge(circuit, Node(NodeType.TOK_EMBED, 0), Node(NodeType.ATTN_V, 1, 4))
    manual_add_edge(circuit, Node(NodeType.TOK_EMBED, 0), Node(NodeType.ATTN_Q, 1, 2))
    manual_add_edge(circuit, Node(NodeType.TOK_EMBED, 0), Node(NodeType.ATTN_K, 1, 2))
    manual_add_edge(circuit, Node(NodeType.ATTN_OUT, 0, 5), Node(NodeType.ATTN_Q, 1, 2))
    manual_add_edge(circuit, Node(NodeType.ATTN_OUT, 0, 5), Node(NodeType.ATTN_K, 1, 2))

    for head_idx in [0, 6]:
        manual_add_edge(circuit, Node(NodeType.ATTN_OUT, 3, head_idx), Node(NodeType.LOGITS, 3))
        manual_add_edge(circuit, Node(NodeType.ATTN_OUT, 1, 4), Node(NodeType.ATTN_Q, 3, head_idx))
        manual_add_edge(circuit, Node(NodeType.TOK_EMBED, 0), Node(NodeType.ATTN_V, 3, head_idx))
        manual_add_edge(circuit, Node(NodeType.ATTN_OUT, 0, 5), Node(NodeType.ATTN_V, 3, head_idx))
        manual_add_edge(circuit, Node(NodeType.ATTN_OUT, 2, 0), Node(NodeType.ATTN_K, 3, head_idx))
        manual_add_edge(circuit, Node(NodeType.ATTN_OUT, 1, 2), Node(NodeType.ATTN_K, 3, head_idx))

    return circuit


GREATERTHAN_CIRCUIT = {
    "0305": [(0, 3), (0, 5)],
    "01": [(0, 1)],
    "MEARLY": [(0, None), (1, None), (2, None), (3, None)],
    "AMID": [(5, 5), (6, 1), (6, 9), (7, 10), (8, 11), (9, 1)],
    "MLATE": [(8, None), (9, None), (10, None), (11, None)],
}


def get_greaterthan_canonical_circuit():
    model = DummyModel(12, 12, False)
    circuit = Circuit.make_circuit(model)
    circuit.set_all_nodes_edges(False)

    # connect input
    for GROUP in ["0305", "01", "MEARLY"]:
        for i, j in GREATERTHAN_CIRCUIT[GROUP]:
            if j is None:
                manual_add_edge(circuit, circuit.sources[0], Node(NodeType.MLP, i))
            else:
                for letter in "qkv":
                    manual_add_edge(
                        circuit,
                        circuit.sources[0],
                        Node(NodeType(f"attn_{letter}"), i, j),
                    )

    for GROUP in ["AMID", "MLATE"]:
        for i, j in GREATERTHAN_CIRCUIT[GROUP]:
            if j is None:
                manual_add_edge(circuit, Node(NodeType.MLP, i), circuit.sink)
            else:
                manual_add_edge(circuit, Node(NodeType.ATTN_OUT, i, j), circuit.sink)

    # MLPs are interconnected
    for GROUP in GREATERTHAN_CIRCUIT.keys():
        if GREATERTHAN_CIRCUIT[GROUP][0][1] is not None:
            continue
        for i1, _ in GREATERTHAN_CIRCUIT[GROUP]:
            for i2, _ in GREATERTHAN_CIRCUIT[GROUP]:
                if i1 >= i2:
                    continue
                manual_add_edge(circuit, Node(NodeType.MLP, i1), Node(NodeType.MLP, i2))

    connected_pairs = [
        ("01", "MEARLY"),
        ("01", "AMID"),
        ("0305", "AMID"),
        ("MEARLY", "AMID"),
        ("AMID", "MLATE"),
    ]

    # connected pairs
    for GROUP1, GROUP2 in connected_pairs:
        for i1, j1 in GREATERTHAN_CIRCUIT[GROUP1]:
            for i2, j2 in GREATERTHAN_CIRCUIT[GROUP2]:
                if i1 >= i2 and not (i1 == i2 and j1 is not None and j2 is None):
                    continue
                src_node = (
                    Node(NodeType.ATTN_OUT, i1, j1) if j1 is not None else Node(NodeType.MLP, i1)
                )
                if j2 is None:
                    manual_add_edge(circuit, src_node, Node(NodeType.MLP, i2))
                else:
                    for letter in "qkv":
                        manual_add_edge(circuit, src_node, Node(NodeType(f"attn_{letter}"), i2, j2))

    # Hanna et al have totally clean query inputs to AMID heads
    # this is A LOT of edges so we add the MLP -> AMID Q edges
    MAX_AMID_LAYER = max([layer_idx for layer_idx, _ in GREATERTHAN_CIRCUIT["AMID"]])
    # connect all MLPs before the AMID heads
    for mlp_sender_layer in range(MAX_AMID_LAYER):
        for mlp_receiver_layer in range(1 + mlp_sender_layer, MAX_AMID_LAYER):
            manual_add_edge(
                circuit,
                Node(NodeType.MLP, mlp_sender_layer),
                Node(NodeType.MLP, mlp_receiver_layer),
            )

    # connect all early MLPs to AMID heads
    for layer_idx, head_idx in GREATERTHAN_CIRCUIT["AMID"]:
        for mlp_sender_layer in range(layer_idx):
            manual_add_edge(
                circuit,
                Node(NodeType.MLP, mlp_sender_layer),
                Node(NodeType.ATTN_Q, layer_idx, head_idx),
            )

    return circuit


def get_induction_canonical_circuit():
    model = DummyModel(2, 8, True)
    circuit = Circuit.make_circuit(model)
    circuit.set_all_nodes_edges(False)

    manual_add_edge(circuit, circuit.sources[0], Node(NodeType.ATTN_V, 0, 0))
    manual_add_edge(circuit, circuit.sources[0], Node(NodeType.ATTN_Q, 1, 6))
    manual_add_edge(circuit, Node(NodeType.ATTN_OUT, 0, 0), Node(NodeType.ATTN_K, 1, 6))
    manual_add_edge(circuit, circuit.sources[0], Node(NodeType.ATTN_V, 1, 6))
    manual_add_edge(circuit, circuit.sources[0], circuit.sink)
    manual_add_edge(circuit, Node(NodeType.ATTN_OUT, 1, 6), circuit.sink)

    return circuit


def get_tracr_proportion_canonical_circuit():
    """
    We double checked results from ACDC and found that the canonical circuit is
    incorrect. We have fixed it here and commented out the incorrect version.

    We think that it is incorrect because if you remove the edges from the circuit
    the score is the same. Moreover, if you manually examine the WQ, WK WE matrices
    you see that they don't really interact with each other.

    """
    model = DummyModel(2, 1, False)
    circuit = Circuit.make_circuit(model, use_pos_embed=True)
    circuit.set_all_nodes_edges(False)

    manual_add_edge(circuit, Node(NodeType.TOK_EMBED, 0), Node(NodeType.MLP, 0))
    manual_add_edge(circuit, Node(NodeType.POS_EMBED, 0), Node(NodeType.ATTN_Q, 1, 0))
    # manual_add_edge(circuit, Node(NodeType.TOK_EMBED, 0), Node(NodeType.ATTN_Q, 1, 0))
    manual_add_edge(circuit, Node(NodeType.POS_EMBED, 0), Node(NodeType.ATTN_K, 1, 0))
    # manual_add_edge(circuit, Node(NodeType.TOK_EMBED, 0), Node(NodeType.ATTN_K, 1, 0))
    manual_add_edge(circuit, Node(NodeType.MLP, 0), Node(NodeType.ATTN_V, 1, 0))
    manual_add_edge(circuit, Node(NodeType.ATTN_OUT, 1, 0), circuit.sink)

    return circuit


def get_tracr_reverse_canonical_circuit():
    model = DummyModel(4, 1, False)
    circuit = Circuit.make_circuit(model, use_pos_embed=True)
    circuit.set_all_nodes_edges(False)

    manual_add_edge(circuit, Node(NodeType.TOK_EMBED, 0), Node(NodeType.ATTN_V, 0, 0))
    manual_add_edge(circuit, Node(NodeType.TOK_EMBED, 0), Node(NodeType.MLP, 0))
    manual_add_edge(circuit, Node(NodeType.ATTN_OUT, 0, 0), Node(NodeType.MLP, 0))
    manual_add_edge(circuit, Node(NodeType.POS_EMBED, 0), Node(NodeType.MLP, 1))
    manual_add_edge(circuit, Node(NodeType.TOK_EMBED, 0), Node(NodeType.MLP, 1))
    manual_add_edge(circuit, Node(NodeType.MLP, 0), Node(NodeType.MLP, 1))
    manual_add_edge(circuit, Node(NodeType.MLP, 1), Node(NodeType.MLP, 2))
    manual_add_edge(circuit, Node(NodeType.MLP, 2), Node(NodeType.ATTN_Q, 3, 0))
    manual_add_edge(circuit, Node(NodeType.POS_EMBED, 0), Node(NodeType.ATTN_K, 3, 0))
    manual_add_edge(circuit, Node(NodeType.TOK_EMBED, 0), Node(NodeType.ATTN_V, 3, 0))
    manual_add_edge(circuit, Node(NodeType.ATTN_OUT, 3, 0), circuit.sink)

    return circuit
