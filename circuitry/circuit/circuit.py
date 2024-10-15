from __future__ import annotations

import random
from copy import deepcopy
from enum import Enum


class NodeType(str, Enum):
    """Specifies the possible node types in a Circuit graph."""

    POS_EMBED = "pos_embed"
    TOK_EMBED = "tok_embed"
    MLP = "mlp"
    ATTN_Q = "attn_q"
    ATTN_K = "attn_k"
    ATTN_V = "attn_v"
    ATTN_OUT = "attn_out"
    LOGITS = "logits"


class Node:
    """The implementation of a node in a Circuit graph."""

    hook_name: str  #: the name(s) of the corresponding TransformerLens hook(s)
    node_type: NodeType  #: the node's type
    layer_idx: int  #: the layer index in the model
    head_idx: int | None  #: the head index in the model for attention nodes, otherwise None

    in_edges: list[Edge]  # the list of incoming edges
    out_edges: list[Edge]  # the list of outgoing edges
    present: bool  # whether the node will be ablated when we run the circuit

    def __init__(self, node_type, layer_idx, head_idx=None, is_dummy=True):
        """Creates a new Node instance.

        Parameters
        ----------
        node_type : NodeType
            The node's type.
        layer_idx : int
            The layer index of the node.
        head_idx : Optional[int]
            The head index of the node, or None if the node is not an attention node.
        is_dummy : Optional[bool]
            Whether the node is a dummy node.

        """
        self.node_type = node_type
        self.layer_idx = layer_idx
        self.head_idx = head_idx
        self.is_dummy = is_dummy
        self.set_hook_names()

        self.in_edges = []
        self.out_edges = []
        self.present = True

    def to_dummy(self):
        """Creates a dummy node with the same attributes as the current node.

        Returns
        -------
        Node
            A dummy node with the same attributes as the current node.

        """
        return Node(self.node_type, self.layer_idx, self.head_idx)

    def is_attn_input(self):
        """Returns whether the node is an attention input node.

        Returns
        -------
        bool
            True if the node is an attention input node, False otherwise.

        """
        return self.node_type in {NodeType.ATTN_Q, NodeType.ATTN_K, NodeType.ATTN_V}

    def set_hook_names(self):
        """Sets the `hook_names` attribute for the node based on its node type and layer index."""
        match self.node_type:
            case NodeType.TOK_EMBED:
                self.hook_names = [
                    f"blocks.{self.layer_idx}.hook_resid_pre",
                    "hook_embed",
                ]
            case NodeType.POS_EMBED:
                self.hook_names = [
                    f"blocks.{self.layer_idx}.hook_resid_pre",
                    "hook_pos_embed",
                ]
            case NodeType.MLP:
                self.hook_names = [
                    f"blocks.{self.layer_idx}.hook_mlp_in",
                    f"blocks.{self.layer_idx}.hook_mlp_out",
                ]
            case NodeType.LOGITS:
                self.hook_names = [f"blocks.{self.layer_idx}.hook_resid_post"]
            case NodeType.ATTN_OUT:
                self.hook_names = [f"blocks.{self.layer_idx}.attn.hook_result"]
            case NodeType.ATTN_Q:
                self.hook_names = [
                    f"blocks.{self.layer_idx}.attn.hook_q",
                    f"blocks.{self.layer_idx}.attn.hook_q_input",
                ]
            case NodeType.ATTN_K:
                self.hook_names = [
                    f"blocks.{self.layer_idx}.attn.hook_k",
                    f"blocks.{self.layer_idx}.attn.hook_k_input",
                ]
            case NodeType.ATTN_V:
                self.hook_names = [
                    f"blocks.{self.layer_idx}.attn.hook_v",
                    f"blocks.{self.layer_idx}.attn.hook_v_input",
                ]

    def __eq__(self, other):
        if not isinstance(other, Node):
            return False

        return (
            self.node_type == other.node_type
            and self.layer_idx == other.layer_idx
            and self.head_idx == other.head_idx
        )

    def __hash__(self):
        return hash((self.node_type, self.layer_idx, self.head_idx))

    def __str__(self):
        return f"{self.node_type}({self.layer_idx}, {self.head_idx})"


class Edge:
    """The implementation of an edge in a Circuit graph."""

    dst_node: Node  #: the node for which the edge is an incoming edge
    src_node: Node  #: the node for which the edge is an outgoing edge
    is_placeholder: bool  #: whether the edge is a placeholder edge (always present, if so)
    present: bool  #: whether the edge will be ablated when we run the circuit

    def __init__(self, src_node, dst_node, is_placeholder):
        """Creates a new Edge instance.

        Parameters
        ----------
        src_node : Node
            The node the edge originates from.
        dst_node : Node
            The node the edge points to.
        is_placeholder : bool
            Whether the edge is a placeholder edge.

        """
        self.src_node = src_node
        self.dst_node = dst_node
        self.is_placeholder = is_placeholder
        self.present = True

    def __eq__(self, other):
        if not isinstance(other, Edge):
            return False

        return self.src_node == other.src_node and self.dst_node == other.dst_node

    def __hash__(self):
        return hash((self.src_node, self.dst_node))

    def __str__(self):
        return f"{self.src_node!s} -> {self.dst_node!s}"


class Circuit:
    """The implementation of a Circuit graph."""

    nodes: list[Node]  #: the list of nodes
    edges: list[Edge]  #: the list of edges
    n_nodes: int  #: the number of nodes
    n_edges: int  #: the number of edges
    node_lookup: dict[Node, Node]  # a mapping of dummy nodes to their corresponding real nodes
    edge_lookup: dict[
        Node, dict[Node, Edge]
    ]  # a nested mapping of parent node to child node to their edge
    sources: list[Node]  # the list of source nodes
    sink: Node  # the sink node

    def __init__(self):
        """Creates a new empty Circuit instance.

        This method is not meant to be called directly. Use the `make_circuit` class method instead.

        """
        self.nodes = []
        self.edges = []
        self.n_nodes = 0
        self.n_edges = 0
        self.node_lookup = {}
        self.edge_lookup = {}
        self.sources = []
        self.sink = None

    def __hash__(self):
        return hash(
            (
                tuple([n for n in self.nodes if n.present]),
                tuple([e for e in self.edges if e.present]),
            )
        )

    @classmethod
    def make_circuit(self, model, use_pos_embed=False):
        """Creates a new Circuit instance based on the given TransformerLens model.

        The relevant model attributes are its number of layers and heads, and whether it is attention only.

        Parameters
        ----------
        model : HookedTransformer
            The TransformerLens model to use.
        use_pos_embed : Optional[bool]
            Whether to include the positional embedding node.

        Returns
        -------
        Circuit
            A Circuit instantiated from the model.

        """
        circuit = self()

        downstream_residual_nodes = []
        logits_node = Node(NodeType.LOGITS, model.cfg.n_layers - 1, is_dummy=False)
        circuit._setup_node(logits_node)
        circuit.sink = logits_node
        downstream_residual_nodes.append(logits_node)

        for layer_idx in range(model.cfg.n_layers - 1, -1, -1):
            if not model.cfg.attn_only:
                cur_mlp = Node(NodeType.MLP, layer_idx, is_dummy=False)
                circuit._setup_node(cur_mlp)

                for residual_stream_node in downstream_residual_nodes:
                    circuit._setup_edge(cur_mlp, residual_stream_node)

                downstream_residual_nodes.append(cur_mlp)

            new_downstream_residual_nodes = []

            for head_idx in range(model.cfg.n_heads - 1, -1, -1):
                cur_attn_out = Node(NodeType.ATTN_OUT, layer_idx, head_idx=head_idx, is_dummy=False)
                circuit._setup_node(cur_attn_out)

                for residual_stream_node in downstream_residual_nodes:
                    circuit._setup_edge(cur_attn_out, residual_stream_node)

                for letter in ("q", "k", "v"):
                    cur_attn_in = Node(
                        NodeType(f"attn_{letter}"),
                        layer_idx,
                        head_idx=head_idx,
                        is_dummy=False,
                    )
                    circuit._setup_node(cur_attn_in)
                    circuit._setup_edge(cur_attn_in, cur_attn_out)

                    new_downstream_residual_nodes.append(cur_attn_in)

            downstream_residual_nodes.extend(new_downstream_residual_nodes)

        token_embedding_node = Node(NodeType.TOK_EMBED, 0, is_dummy=False)
        circuit._setup_node(token_embedding_node)
        circuit.sources.append(token_embedding_node)

        if use_pos_embed:
            position_embedding_node = Node(NodeType.POS_EMBED, 0, is_dummy=False)
            circuit._setup_node(position_embedding_node)
            circuit.sources.append(position_embedding_node)

        for source_node in circuit.sources:
            for residual_stream_node in downstream_residual_nodes:
                circuit._setup_edge(source_node, residual_stream_node)

        return circuit

    def _setup_edge(self, src_node, dst_node):
        """Internal method called by `make_circuit` to add an edge to the circuit.

        Parameters
        ----------
        src_node : Node
            The node the edge originates from.
        dst_node : Node
            The node the edge points to.

        """
        is_placeholder = dst_node.node_type == NodeType.ATTN_OUT
        edge = Edge(src_node, dst_node, is_placeholder)
        self.edges.append(edge)
        self.edge_lookup[src_node][dst_node] = edge
        self.node_lookup[src_node].out_edges.append(edge)
        self.node_lookup[dst_node].in_edges.append(edge)
        if not is_placeholder:
            self.n_edges += 1

    def _setup_node(self, node):
        """Internal method called by `make_circuit` to add a node to the circuit.

        Parameters
        ----------
        node : Node
            The dummy node to add to the circuit.

        """
        self.nodes.append(node)
        self.node_lookup[node.to_dummy()] = node
        self.edge_lookup[node] = {}
        if not node.is_attn_input():
            self.n_nodes += 1

    def remove_edge(self, src_node, dst_node, in_place=False):
        """Removes an edge from the circuit.

        Parameters
        ----------
        src_node : Node
            The node the edge originates from.
        dst_node : Node
            The node the edge points to.
        in_place : Optional[bool]
            Whether to remove the edge in place or return a new Circuit.

        """
        if src_node not in self.node_lookup:
            raise ValueError(f"{src_node!s} not in the circuit")
        if dst_node not in self.edge_lookup[src_node]:
            raise ValueError(f"Edge {src_node} -> {dst_node} not in the circuit")
        elif self.edge_lookup[src_node][dst_node].is_placeholder:
            raise ValueError("Cannot remove a placeholder edge")

        if in_place:
            self.edge_lookup[src_node][dst_node].present = False
        else:
            circuit = deepcopy(self)
            circuit.edge_lookup[src_node][dst_node].present = False
            return circuit

    def get_successors(self, node):
        """Gets the *present* successors (children) of a node.

        Parameters
        ----------
        node : Node
            The node to get the successors. Can be a dummy node.

        Returns
        -------
        list[Node]
            The list of the node's present successors.

        """
        if node not in self.node_lookup:
            print(f"Couldn't find {node} in the graph!")

        return [edge.dst_node for edge in self.node_lookup[node].out_edges if edge.present]

    def get_predecessors(self, node):
        """Gets the *present* predecessors (parents) of a node.

        Parameters
        ----------
        node : Node
            The node to get the predecessors. Can be a dummy node.

        Returns
        -------
        list[Node]
            The list of the node's predecessors.

        """
        if node not in self.node_lookup:
            print(f"Couldn't find {node} in the graph!")

        return [edge.src_node for edge in self.node_lookup[node].in_edges if edge.present]

    def get_present_nodes(self):
        """Gets the present nodes in the circuit.

        Returns
        -------
        list[Node]
            The list of the circuit's present nodes.

        """
        return [node for node in self.nodes if node.present]

    def get_present_edges(self):
        """Gets the present edges in the circuit.

        Returns
        -------
        list[Edge]
            The list of the circuit's present edges.

        """
        return [edge for edge in self.edges if edge.present]

    def count_present_nodes(self, expand_attn=False):
        """Counts the present nodes in the circuit.

        Parameters
        ----------
        expand_attn : Optional[bool]
            Whether to count the attention input nodes.

        Returns
        -------
        int
            The number of present nodes in the circuit.

        """
        return sum(
            node.present if not node.is_attn_input() or expand_attn else False
            for node in self.nodes
        )

    def count_present_edges(self):
        """Counts the present non-placeholder edges in the graph.

        Returns
        -------
        int
            The number of present non-placeholder edges in the circuit.

        """
        return sum(edge.present for edge in self.edges if not edge.is_placeholder)

    def __len__(self):
        return self.count_present_edges()

    def get_total_nodes(self, expand_attn=False):
        """Returns the total number of nodes in the circuit.

        Parameters
        ----------
        expand_attn : Optional[bool]
            Whether to count the attention input nodes.

        Returns
        -------
        int
            The number of nodes in the circuit.

        """
        return len(self.nodes) if expand_attn else self.n_nodes

    def get_total_edges(self):
        """Returns the total number of non-placeholder edges in the circuit.

        Returns
        -------
        int
            The number of non-placeholder edges in the circuit.

        """
        return self.n_edges

    def is_connected(self):
        """
        Runs a DFS from each source node to see if the sink is reachable using present edges.

        Returns
        -------
        bool
            Whether the circuit graph is connected from source to sink.

        """
        seen = set()

        def dfs(node):
            nonlocal seen
            if node == self.sink:
                return True
            for nei in self.get_successors(node):
                if nei not in seen:
                    seen.add(nei)
                    if dfs(nei):
                        return True
            return False

        return any(dfs(source) for source in self.sources)

    # ****************
    # Sampling methods
    # ****************
    def set_all_nodes_edges(self, present: bool):
        """
        Sets all nodes and edges to be present or not present.

        Parameters
        ----------
        present : bool
            Whether to set all nodes and edges to be present or not present.

        """
        for node in self.nodes:
            node.present = present
        for edge in self.edges:
            if not edge.is_placeholder:
                edge.present = present
            else:
                edge.present = True  # shouldn't need this

    def sample_circuit(self, minimum_number_of_edges: int) -> Circuit:
        """Samples a circuit with at least minimum_number_of_edges such that all edges are along connected paths.

        Parameters
        ----------
        minimum_number_of_edges : int
            The minimum number of edges to sample.

        Returns
        -------
        Circuit
            A new circuit with connected paths and at least minimum_number_of_edges.

        """
        if minimum_number_of_edges < 0:
            raise ValueError("Minimum number of edges must be non-negative")
        elif minimum_number_of_edges > self.get_total_edges():
            raise ValueError("Minimum number of edges cannot exceed total edges")

        circuit = deepcopy(self)
        circuit.set_all_nodes_edges(False)

        num_edges = 0
        while num_edges < minimum_number_of_edges:
            path = circuit._sample_path()

            for edge in path:
                if not edge.present:
                    edge.present = edge.dst_node.present = edge.src_node.present = True
                    if not edge.is_placeholder:
                        num_edges += 1

        return circuit

    def _sample_path(self) -> list[Edge]:
        """Internal method to sample a path from source to sink, disregarding node and edge presence.

        Returns
        -------
        list[Edge]
            The list of edges forming the sampled path.

        """
        edges_in_path = []
        current_node = random.choice(self.sources)

        while current_node != self.sink:
            possible_next_nodes = [edge.dst_node for edge in current_node.out_edges]
            next_node = random.choice(possible_next_nodes)
            next_edge = self.edge_lookup[current_node][next_node]
            edges_in_path.append(next_edge)
            current_node = next_edge.dst_node

        return edges_in_path

    def sample_circuit_complement(self, minimum_number_of_edges: int) -> Circuit:
        """Samples a circuit with connected paths and minimum_number_of_edges such that it contains no edges in common with self.

        Parameters
        ----------
        minimum_number_of_edges : int
            The minimum number of edges to sample.

        Returns
        -------
        Circuit
            A new circuit with connected paths, at least `minimum_number_of_edges`, and no edges in common with `self`.

        """
        if minimum_number_of_edges < 0:
            raise ValueError("Minimum number of edges must be non-negative")
        elif minimum_number_of_edges > self.get_total_edges() - self.count_present_edges():
            raise ValueError("Minimum number of edges cannot exceed total-present edges")

        circuit = deepcopy(self)
        circuit.set_all_nodes_edges(False)

        n_edges = 0
        while n_edges < minimum_number_of_edges:
            # We sample from self here because we need to know which edges are present
            path = self._sample_path_complement()

            for edge in path:
                # Get the corresponding edge on the complement circuit
                edge = circuit.edge_lookup[edge.src_node][edge.dst_node]
                if not edge.present:
                    edge.present = edge.src_node.present = edge.dst_node.present = True
                    if not edge.is_placeholder:  # shouldn't need this check
                        n_edges += 1

        for edge in circuit.get_present_edges():
            if self.edge_lookup[edge.src_node][edge.dst_node].present and not edge.is_placeholder:
                raise ValueError("Edge in the circuit is in the original circuit")

        return circuit

    def _sample_path_complement(self) -> list[Edge]:
        """Internal method that samples a path from source to sink in the complement circuit.

        Notes
        -----
        Runs a random-order dfs from source to sink.
        Only non-present edges in self or placeholder edges are traversable.
        Keeps track of parent nodes to get the path.

        Returns
        -------
        list[Edge]
            The list of edges forming the sampled path.

        """

        def _dfs(node):
            if node == self.sink:
                return node
            possible_next_nodes = [
                edge.dst_node for edge in node.out_edges if not edge.present or edge.is_placeholder
            ]
            random.shuffle(possible_next_nodes)

            for in_nei in possible_next_nodes:
                parent[in_nei] = node
                found_node = _dfs(in_nei)
                if found_node is not None:
                    return found_node

            return None

        end = None
        parent = {}
        for source_node in random.sample(self.sources, len(self.sources)):
            parent = {source_node: None}
            end = _dfs(source_node)
            if end == self.sink:
                break

        if end != self.sink:
            raise ValueError("Couldn't find a path from source to sink")

        edges = []

        while parent[end]:
            edges.append(self.edge_lookup[parent[end]][end])
            end = parent[end]

        return edges

    def sample_inflated(self, inflate_size: int) -> Circuit:
        """Creates a new circuit that is inflated by at least `inflate_size` edges, such that all edges are along connected paths.

        Parameters
        ----------
        inflate_size : int
            The number of new edges to inflate the circuit by.

        Returns
        -------
        Circuit
            A new circuit with connected paths, at least `inflate_size` new edges, and no edges in common with `self`.

        """
        if inflate_size < 0:
            raise ValueError("inflate size edges must be non-negative")
        elif inflate_size + self.count_present_edges() > self.get_total_edges():
            raise ValueError("Inflated circuit cannot have more edges than total edges")

        circuit = deepcopy(self)
        n_edges = n_original_edges = circuit.count_present_edges()

        while n_edges < inflate_size + n_original_edges:
            path = circuit._sample_path()

            for edge in path:
                if not edge.present:
                    edge.present = edge.dst_node.present = edge.src_node.present = True
                    if not edge.is_placeholder:  # shouldn't need this check
                        n_edges += 1

        return circuit

    def make_original_circuit(self) -> Circuit:
        """Creates a new circuit from `self` where all nodes and edges are present.

        Returns
        -------
        Circuit
            A new circuit with all nodes and edges present.

        """
        circuit = deepcopy(self)
        circuit.set_all_nodes_edges(True)
        return circuit

    def __deepcopy__(self, memo) -> Circuit:
        nodes = []
        edges = []
        node_lookup = {}
        edge_lookup = {}

        for node in self.nodes:
            node_copy = Node(node.node_type, node.layer_idx, node.head_idx, is_dummy=False)
            node_copy.present = node.present
            node_lookup[node_copy.to_dummy()] = node_copy
            edge_lookup[node_copy.to_dummy()] = {}
            nodes.append(node_copy)

        for edge in self.edges:
            src_node = node_lookup[edge.src_node]
            dst_node = node_lookup[edge.dst_node]
            edge_copy = Edge(src_node, dst_node, edge.is_placeholder)
            edge_copy.present = edge.present
            edges.append(edge_copy)

            src_node.out_edges.append(edge_copy)
            dst_node.in_edges.append(edge_copy)
            edge_lookup[src_node][dst_node] = edge_copy

        circuit_copy = Circuit()
        circuit_copy.nodes = nodes
        circuit_copy.edges = edges
        circuit_copy.node_lookup = node_lookup
        circuit_copy.edge_lookup = edge_lookup
        circuit_copy.n_nodes = self.n_nodes
        circuit_copy.n_edges = self.n_edges
        circuit_copy.sources = [node_lookup[source] for source in self.sources]
        circuit_copy.sink = node_lookup[self.sink]

        return circuit_copy

    def is_subset(self, other: Circuit) -> bool:
        """Checks if `self` is a subset of `other`.

        Parameters
        ----------
        other : Circuit
            The other circuit to compare against.

        Returns
        -------
        bool
            True if `self` is a subset of `other`, False otherwise.

        """
        if self.get_total_edges() > other.get_total_edges():
            return False

        for edge in self.edges:
            if edge.is_placeholder:
                continue
            if edge.present:
                # find edge in the other:
                if (
                    edge.src_node in other.edge_lookup
                    and edge.dst_node in other.edge_lookup[edge.src_node]
                    and other.edge_lookup[edge.src_node][edge.dst_node].present
                ):
                    continue
                return False

        return True
