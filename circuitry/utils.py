import random
from collections import defaultdict
from typing import Union

import numpy as np
import torch

from circuitry.circuit import Node, NodeType


def organize_nodelist(nodes):
    d = {}

    for node in nodes:
        if node.layer_idx not in d:
            d[node.layer_idx] = defaultdict(list)

        if node.node_type in {NodeType.ATTN_Q, NodeType.ATTN_K, NodeType.ATTN_V}:
            d[node.layer_idx][node.node_type.value].append(node.head_idx)
        else:
            d[node.layer_idx][node.node_type.value] = True

    for d_sub in d.values():
        for v in d_sub.values():
            if isinstance(v, list):
                v.sort()

    return d


def organize_circuit_nodes(circuit):
    return organize_nodelist(circuit.get_present_nodes())


def hook_name_to_node(hook_name):
    if "." in hook_name:
        layer_idx = int(hook_name.split(".")[1])
    else:
        layer_idx = 0
    node_type = None

    if hook_name == "hook_embed" or "resid_pre" in hook_name:
        node_type = NodeType.TOK_EMBED
    elif hook_name == "hook_pos_embed":
        node_type = NodeType.POS_EMBED
    elif "resid_post" in hook_name:
        node_type = NodeType.LOGITS
    elif "mlp" in hook_name:
        node_type = NodeType.MLP
    elif "attn.hook_result" in hook_name:
        node_type = NodeType.ATTN_OUT
    else:
        for letter in "qkv":
            if f"hook_{letter}" in hook_name:
                node_type = NodeType(f"attn_{letter}")

    if node_type is None:
        raise ValueError(f"Unknown hook name: {hook_name}")

    return Node(node_type, layer_idx, None)


class DummyConfig:
    def __init__(self, n_layers, n_heads, attn_only):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.attn_only = attn_only


class DummyModel:
    def __init__(self, n_layers, n_heads, attn_only):
        self.cfg = DummyConfig(n_layers, n_heads, attn_only)


class rng_context:
    """
    Utility context manager for locally deterministic code.
    It sets the random seed to a fixed value when entering the context and restores
    the previous state when exiting.

    Example
    -------
    ```python
        with rng_context(seed):
            perm = torch.randperm(n)
    ```
    """

    def __init__(self, seed):
        self.seed = seed
        self.state = None

    def __enter__(self):
        self.state = torch.get_rng_state()
        torch.manual_seed(self.seed)

    def __exit__(self, exc_type, exc_value, traceback):
        torch.set_rng_state(self.state)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    num_gpus = torch.cuda.device_count()

    if num_gpus > 0:
        torch.cuda.manual_seed_all(seed)


def to_numpy(x) -> Union[np.ndarray, list[np.ndarray]]:
    if isinstance(x, list):
        return [to_numpy(xi) for xi in x]
    elif isinstance(x, torch.Tensor):
        return x.cpu().detach().numpy()
    else:
        return x
