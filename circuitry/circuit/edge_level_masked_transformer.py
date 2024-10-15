"""
This file is based on code provided by the authors of the ACDC library
(https://github.com/ArthurConmy/Automatic-Circuit-Discovery/), which
accompanies the ACDC paper (https://arxiv.org/abs/2304.14997).

It is licensed under the MIT License.
"""

import logging
import math
from collections.abc import Iterable, Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager
from enum import Enum
from typing import (
    Callable,
    Literal,
    TypeAlias,
    Union,
    cast,
)

import torch
from einops import rearrange
from jaxtyping import Float, Num
from transformer_lens import ActivationCache, HookedTransformer
from transformer_lens.hook_points import HookPoint, NamesFilter
from transformer_lens.HookedTransformer import Loss

from circuitry.circuit import Circuit, NodeType
from circuitry.utils import hook_name_to_node

logger = logging.getLogger(__name__)

HookPointName: TypeAlias = str
PatchData: TypeAlias = Num[torch.Tensor, "batch pos"] | None  # use None if you want zero ablation


class CircuitStartingPointType(str, Enum):
    """We have two conventions for where to start the circuit: either at hook_embed and hook_pos_embed, or at
    hook_resid_pre.

    In older pieces of code this concept might also be referred to as 'use_pos_embed: bool' (where True corresponds
    to CircuitStartingPointType.POS_EMBED).
    """

    POS_EMBED = "pos_embed"  # uses hook_embed and hook_pos_embed as starting point
    RESID_PRE = "resid_pre"  # uses blocks.0.hook_resid_pre as starting point


def create_mask_parameters_and_forward_cache_hook_points(
    circuit_start_type: CircuitStartingPointType,
    num_heads: int,
    num_layers: int,
    device: str | torch.device,
    mask_init_constant: float,
    attn_only: bool,
):
    """
    Given the relevant configuration for a transformer, this function produces two things:

    1. The parameters for the masks, in a dict that maps a hook point name to its mask parameter
    2. The hook points that need to be cached on a forward pass.

    TODO: why don't we keep a friggin dict or something to keep track of how each IndexedHookPoint maps to an index?
    basically dict[IndexedHookPoint, int] or something like that
    """

    ordered_forward_cache_hook_points: list[HookPointName] = []

    # we need to track the number of outputs so far, because this will be the number of parents
    # of all the following units.
    num_output_units_so_far = 0

    hook_point_to_parents: dict[HookPointName, list[HookPointName]] = {}
    mask_parameter_list = torch.nn.ParameterList()
    mask_parameter_dict: dict[HookPointName, torch.nn.Parameter] = {}

    # Implementation details:
    #
    # We distinguish hook points that are used for input, and hook points that are used for output.
    #
    # The input hook points are the ones that get mask parameters.
    # The output hook points are the ones that we cache on a forward pass, so that we can later mask them.

    def setup_output_hook_point(mask_name: str, num_instances: int):
        ordered_forward_cache_hook_points.append(mask_name)

        nonlocal num_output_units_so_far
        num_output_units_so_far += num_instances

    def setup_input_hook_point(mask_name: str, num_instances: int):
        """
        Adds a mask logit for the given mask name and parent nodes
        Parent nodes are (attention, MLP)

        We need to add a parameter to mask the input to these units
        """
        nonlocal num_output_units_so_far
        hook_point_to_parents[mask_name] = ordered_forward_cache_hook_points[
            :
        ]  # everything that has come before

        new_mask_parameter = torch.nn.Parameter(
            torch.full(
                (num_output_units_so_far, num_instances),
                mask_init_constant,
                device=device,
            )
        )
        mask_parameter_list.append(new_mask_parameter)
        mask_parameter_dict[mask_name] = (
            new_mask_parameter  # pyright: ignore reportArgumentType  # seems to be an issue with pyright or with torch.nn.Parameter?
        )

    match circuit_start_type:
        case CircuitStartingPointType.POS_EMBED:
            starting_points = ["hook_embed", "hook_pos_embed"]
        case CircuitStartingPointType.RESID_PRE:
            starting_points = ["blocks.0.hook_resid_pre"]
        case _:
            raise ValueError(f"Unknown circuit_start_type: {circuit_start_type}")

    for embedding_hook_point in starting_points:
        setup_output_hook_point(embedding_hook_point, 1)

    # Add mask logits for ablation cache
    # Mask logits have a variable dimension depending on the number of in-edges (increases with layer)
    for layer_i in range(num_layers):
        for q_k_v in ["q", "k", "v"]:
            setup_input_hook_point(
                mask_name=f"blocks.{layer_i}.hook_{q_k_v}_input", num_instances=num_heads
            )

        setup_output_hook_point(f"blocks.{layer_i}.attn.hook_result", num_heads)

        if not attn_only:
            setup_input_hook_point(mask_name=f"blocks.{layer_i}.hook_mlp_in", num_instances=1)
            setup_output_hook_point(f"blocks.{layer_i}.hook_mlp_out", num_instances=1)

    # why does this get a mask? isn't it pointless to mask this?
    setup_input_hook_point(mask_name=f"blocks.{num_layers - 1}.hook_resid_post", num_instances=1)

    return (
        ordered_forward_cache_hook_points,
        hook_point_to_parents,
        mask_parameter_list,
        mask_parameter_dict,
    )


class EdgeLevelMaskedTransformer(torch.nn.Module):
    """
    A wrapper around HookedTransformer that allows edge-level subnetwork probing.

    There are two sets of hooks:
    - `activation_mask_hook`s change the input to a node. The input to a node is the sum
      of several residual stream terms; ablated edges are looked up from `ablation_cache`
      and non-ablated edges from `forward_cache`, then the sum is taken.
    - `caching_hook`s save the output of a node to `forward_cache` for use in later layers.

    Qs:

    - what are the names of the mask parameters in the mask parameter dict? We just use the names of the hook point
    - how are all the mask params laid out as a tensor?
    - does everything use the HookName as a kind ...? if so, use that in types
    """

    model: HookedTransformer
    ablation_cache: ActivationCache
    forward_cache: ActivationCache
    hook_point_to_parents: dict[
        HookPointName, list[HookPointName]
    ]  # the parents of each hook point
    mask_parameter_list: (
        torch.nn.ParameterList
    )  # the parameters that we use to mask the input to each node
    _mask_parameter_dict: dict[
        str, torch.nn.Parameter
    ]  # same parameters, but indexed by the hook point that they are applied to
    forward_cache_hook_points: list[
        HookPointName
    ]  # the hook points where we need to cache the output on a forward pass

    def __init__(
        self,
        model: HookedTransformer,
        beta=2 / 3,
        gamma=-0.1,
        zeta=1.1,
        mask_init_p=0.9,
        starting_point_type: CircuitStartingPointType = CircuitStartingPointType.POS_EMBED,
        no_ablate=False,
        verbose=False,
    ):
        """
        - 'use_pos_embed': if set to True, create masks for edges from 'hook_embed' and 'hook_pos_embed'; othererwise,
            create masks for edges from 'blocks.0.hook_resid_pre'.
        """
        super().__init__()

        self.model = model
        self.n_heads = model.cfg.n_heads
        self.n_mlp = 0 if model.cfg.attn_only else 1
        self.no_ablate = no_ablate
        if no_ablate:
            print("WARNING: no_ablate is True, this is for testing only")
        self.device = self.model.parameters().__next__().device
        self.starting_point_type = starting_point_type
        self.verbose = verbose

        self.ablation_cache = ActivationCache({}, self.model)
        self.forward_cache = ActivationCache({}, self.model)
        # Hyperparameters
        self.beta = beta
        self.gamma = gamma
        self.zeta = zeta
        self.mask_init_p = mask_init_p

        # Copied from subnetwork probing code. Similar to log odds (but not the same)
        p = (self.mask_init_p - self.gamma) / (self.zeta - self.gamma)

        model.cfg.use_hook_mlp_in = True  # We need to hook the MLP input to do subnetwork probing

        (
            self.forward_cache_hook_points,
            self.hook_point_to_parents,
            self.mask_parameter_list,
            self._mask_parameter_dict,
        ) = create_mask_parameters_and_forward_cache_hook_points(
            circuit_start_type=self.starting_point_type,
            num_heads=self.n_heads,
            num_layers=model.cfg.n_layers,
            device=self.device,
            mask_init_constant=math.log(p / (1 - p)),
            attn_only=model.cfg.attn_only,
        )

        # Ensure model is used strictly for eval
        for param in self.parameters():
            param.requires_grad = False
        self.model.eval()

    def run_circuit(
        self,
        circuit: Circuit,
        base_dataset,  # TODO: Add type
        ablation_dataset=None,  # TODO: Add type
        zero_ablation: bool = True,
        return_type: Literal["logits | both"] = "logits",
        loss_per_token: bool = False,
        invert: bool = False,
    ):
        """
        Runs the model forward on the base_dataset with the given circuit.

        This is one one of the main functions of this class.

        Parameters
        ----------
        circuit : Circuit
            The circuit to run the model with.
        base_dataset : torch.Tensor
            TODO: Add type
        ablation_dataset : torch.Tensor
            TODO: Add type
        zero_ablation : bool
            Whether to use zero ablation or ablate with a corrupted dataset.
        return_type : Literal["logits | both"]
            Whether to return logits, loss, or both.
        loss_per_token : bool
            Whether to return the loss per token or not.
        invert : bool
            Whether to run the model on the given circuit or its inverse. Default is False.
        """

        if zero_ablation is False and ablation_dataset is None:
            msg = "zero_ablation is False, but ablation_dataset is None. "
            msg += "Please provide a valid ablation dataset."
            raise ValueError(msg)

        # This sets the circuit in the model
        self.set_binary_mask(circuit, invert=invert)

        # TODO: This is a quirk of the class which should be fixed
        # setting the patch_ds to None does zero ablation
        patch_ds = None if zero_ablation else ablation_dataset
        self.calculate_and_store_ablation_cache(patch_ds)

        with self.with_fwd_hooks_and_new_ablation_cache(patch_ds) as hooked_model:
            with torch.no_grad():
                result = hooked_model(
                    base_dataset, return_type=return_type, loss_per_token=loss_per_token
                )

        return result

    @property
    def mask_parameter_names(self) -> Iterable[str]:
        return self._mask_parameter_dict.keys()

    def sample_mask(self, mask_name: str) -> torch.Tensor:
        """Samples a binary-ish mask from the mask_scores for the particular `mask_name` activation"""
        mask_parameters = self._mask_parameter_dict[mask_name]
        uniform_sample = (
            torch.zeros_like(mask_parameters, requires_grad=False).uniform_().clamp_(0.0001, 0.9999)
        )
        s = torch.sigmoid(
            (uniform_sample.log() - (1 - uniform_sample).log() + mask_parameters) / self.beta
        )
        s_bar = s * (self.zeta - self.gamma) + self.gamma
        mask = s_bar.clamp(min=0.0, max=1.0)

        return mask

    def regularization_loss(self) -> torch.Tensor:
        center = self.beta * math.log(-self.gamma / self.zeta)
        per_parameter_loss = [
            torch.sigmoid(scores - center).mean() for scores in self.mask_parameter_list
        ]
        return torch.mean(torch.stack(per_parameter_loss))

    def _calculate_and_store_zero_ablation_cache(self) -> None:
        """Caches zero for every possible mask point."""
        patch_data = torch.zeros((1, 1), device=self.device, dtype=torch.int64)  # batch pos
        self._calculate_and_store_resampling_ablation_cache(
            patch_data
        )  # wtf? is this just to initialize the cache object? if we had tests, I would refactor this
        self.ablation_cache.cache_dict = {
            name: torch.zeros_like(scores)
            for name, scores in self.ablation_cache.cache_dict.items()
        }

    def _calculate_and_store_resampling_ablation_cache(
        self,
        patch_data: Num[torch.Tensor, "batch pos"],
        retain_cache_gradients: bool = False,
    ) -> None:
        # Only cache the tensors needed to fill the masked out positions
        if not retain_cache_gradients:
            with torch.no_grad():
                model_out, self.ablation_cache = self.model.run_with_cache(
                    patch_data,
                    names_filter=lambda name: name in self.forward_cache_hook_points,
                    return_cache_object=True,
                )
        else:
            model_out, self.ablation_cache = self.run_with_attached_cache(
                patch_data,
                names_filter=lambda name: name in self.forward_cache_hook_points,
            )

    def run_with_attached_cache(self, *model_args, names_filter: NamesFilter = None) -> tuple[
        Union[
            None,
            Float[torch.Tensor, "batch pos d_vocab"],
            Loss,
            tuple[Float[torch.Tensor, "batch pos d_vocab"], Loss],
        ],
        ActivationCache,
    ]:
        """An adaptation of HookedTransformer.run_with_cache that does not
        detach the tensors in the cache. This means we can calculate the gradient
        of the patch data from the cache."""
        cache_dict = {}

        def save_hook(tensor, hook):
            # This is the essential difference with HookedTransformer.run_with_cache:
            # it uses a hook that detaches the tensor rather than just storing it
            cache_dict[hook.name] = tensor

        names_filter = self._convert_names_filter(names_filter)
        fwd_hooks = cast(
            list[tuple[str | Callable, Callable]],
            [(name, save_hook) for name in self.model.hook_dict.keys() if names_filter(name)],
        )

        with self.hooks(fwd_hooks=fwd_hooks) as runner_with_cache:
            out = runner_with_cache.model(*model_args)

        return out, ActivationCache(
            cache_dict=cache_dict,
            model=self,
        )

    @staticmethod
    def _convert_names_filter(names_filter: NamesFilter) -> Callable[[str], bool]:
        """This is extracted from HookedRootModule.get_caching_hooks."""
        if names_filter is None:
            names_filter = lambda name: True  # noqa: E731
        elif isinstance(names_filter, str):
            filter_str = names_filter
            names_filter = lambda name: name == filter_str  # noqa: E731
        elif isinstance(names_filter, Sequence):
            filter_list = names_filter
            names_filter = lambda name: name in filter_list  # noqa: E731

        return names_filter

    def calculate_and_store_ablation_cache(
        self, patch_data: PatchData, retain_cache_gradients: bool = True
    ):
        """Use None for the patch data for zero ablation.

        If you want to calculate gradients on the patch data/the cache, set retain_cache_gradients=True.
        Otherwise set to False for performance."""
        if patch_data is None:
            self._calculate_and_store_zero_ablation_cache()
        else:
            assert isinstance(patch_data, torch.Tensor)
            self._calculate_and_store_resampling_ablation_cache(
                patch_data, retain_cache_gradients=retain_cache_gradients
            )

    def get_activation_values(
        self, parent_names: list[str], cache: ActivationCache
    ) -> Num[torch.Tensor, "batch seq parentindex d"]:
        """
        Returns a single tensor of the mask values used for a given hook.
        Attention is shape batch, seq, heads, head_size while MLP out is batch, seq, d_model
        so we need to reshape things to match.

        The output is "batch seq parentindex d_model", where "parentindex" is the index in the list
        of `parent_names`.
        """
        result = []
        for name in parent_names:
            value = cache[name]  # b s n_heads d, or b s d
            if value.ndim == 3:
                value = value.unsqueeze(2)  # b s 1 d
            result.append(value)
        return torch.cat(result, dim=2)

    def compute_weighted_values(
        self, hook: HookPoint
    ) -> Float[torch.Tensor, "batch pos head_index d_resid"]:
        parent_names = self.hook_point_to_parents[
            hook.name
        ]  # pyright: ignore # hook.name is not typed correctly
        ablation_values = self.get_activation_values(
            parent_names, self.ablation_cache
        )  # b s i d (i = parentindex)
        forward_values = self.get_activation_values(parent_names, self.forward_cache)  # b s i d
        mask = self.sample_mask(
            hook.name  # pyright: ignore # hook.name is not typed correctly
        )  # in_edges, nodes_per_mask, ...

        weighted_ablation_values = torch.einsum(
            "b s i d, i o -> b s o d", ablation_values, 1 - mask
        )
        weighted_forward_values = torch.einsum("b s i d, i o -> b s o d", forward_values, mask)
        return weighted_ablation_values + weighted_forward_values

    def activation_mask_hook(self, hook_point_out: torch.Tensor, hook: HookPoint, verbose=False):
        """
        For edge-level SP, we discard the hook_point_out value and resum the residual stream.
        """
        show = print if verbose else lambda *args, **kwargs: None
        show(f"Doing ablation of {hook.name}")
        mem1 = torch.cuda.memory_allocated()
        show(f"Using memory {mem1:_} bytes at hook start")
        is_attn = (
            "mlp" not in hook.name and "resid_post" not in hook.name
        )  # pyright: ignore # hook.name is not typed correctly

        # To trade off CPU against memory, you can use
        #    out = checkpoint(self.compute_weighted_values, hook, use_reentrant=False)
        # However, that messes with the backward pass, so I've disabled it for now.
        out = self.compute_weighted_values(hook)
        if not is_attn:
            out = rearrange(out, "b s 1 d -> b s d")

        # add back attention bias
        # Explanation: the attention bias is not part of the cached values (why not?), so we need to add it back here
        # we need to iterate over all attention layers that come before current layer
        current_block_index = int(
            hook.name.split(".")[1]
        )  # pyright: ignore # hook.name is not typed correctly
        last_attention_block_index = (
            current_block_index + 1
            if ("resid_post" in hook.name or "mlp" in hook.name)
            else current_block_index  # pyright: ignore # hook.name is not typed correctly
        )
        for layer in self.model.blocks[:last_attention_block_index]:  # pyright: ignore
            out += layer.attn.b_O

        if self.no_ablate and not torch.allclose(hook_point_out, out, atol=1e-4):
            print(f"Warning: hook_point_out and out are not close for {hook.name}")
            print(f"{hook_point_out.mean()=}, {out.mean()=}")

        if self.verbose:
            no_change = torch.allclose(hook_point_out, out)
            absdiff = (hook_point_out - out).abs().mean()
            print(
                f"Ablation hook {'did NOT' if no_change else 'DID'} change {hook.name} by {absdiff:.3f}"
            )
        torch.cuda.empty_cache()
        show(f"Using memory {torch.cuda.memory_allocated():_} bytes after clearing cache")
        return out

    def caching_hook(self, hook_point_out: torch.Tensor, hook: HookPoint):
        assert hook.name is not None
        self.forward_cache.cache_dict[hook.name] = hook_point_out
        return hook_point_out

    def fwd_hooks(self) -> list[tuple[str | Callable, Callable]]:
        return cast(
            list[tuple[str | Callable, Callable]],
            [(hook_point, self.activation_mask_hook) for hook_point in self.mask_parameter_names]
            + [(hook_point, self.caching_hook) for hook_point in self.forward_cache_hook_points],
        )

    def with_fwd_hooks(self) -> AbstractContextManager[HookedTransformer]:
        return self.model.hooks(self.fwd_hooks())

    def with_fwd_hooks_and_new_ablation_cache(
        self, patch_data: PatchData
    ) -> AbstractContextManager[HookedTransformer]:
        self.calculate_and_store_ablation_cache(patch_data, retain_cache_gradients=False)
        return self.with_fwd_hooks()

    @contextmanager
    def hooks(
        self,
        fwd_hooks: list[tuple[str | Callable, Callable]] | None = None,
        bwd_hooks: list[tuple[str | Callable, Callable]] | None = None,
        reset_hooks_end: bool = True,
        clear_contexts: bool = False,
    ) -> Iterator["EdgeLevelMaskedTransformer"]:
        """Imitates the 'hooks' context manager in HookedTransformer."""
        with self.model.hooks(
            fwd_hooks=fwd_hooks or [],
            bwd_hooks=bwd_hooks or [],
            reset_hooks_end=reset_hooks_end,
            clear_contexts=clear_contexts,
        ):
            # the with-statement above updates the hooks in self.model
            # so we can simply yield self
            yield self

    def freeze_weights(self):
        for p in self.model.parameters():
            p.requires_grad = False

    def num_edges(self):
        values = []
        for name, _mask in self._mask_parameter_dict.items():
            mask_value = self.sample_mask(name)
            values.extend(mask_value.flatten().tolist())
        values = torch.tensor(values)
        return (values > 0.5).sum().item()

    def num_params(self):
        return sum(p.numel() for p in self.mask_parameter_list)

    """
    An internal function that sets up a correspondence between circuit edges and mask parameters
    and applies the edge-level function f.
    Args:
       f: a function to apply, with inputs (src_node, dst_node, p, row_idx, col_idx)
          The inputs are the edge-level correspondence, which f can read or write to.
       sample_mask: whether to sample the mask before applying f.
    """

    def _mask_circuit_correspondence_iter(self, f, sample_mask=False):
        def _node_level_iter(hook_name, dst_node):
            num_output_units_so_far = 0
            col_idx = dst_node.head_idx if dst_node.head_idx is not None else 0

            for previous_hook_name in self.hook_point_to_parents[hook_name]:
                src_node = hook_name_to_node(previous_hook_name)

                if src_node.node_type == NodeType.ATTN_OUT:
                    for out_head_idx in range(self.n_heads):
                        src_node.head_idx = out_head_idx
                        f(src_node, dst_node, p, num_output_units_so_far, col_idx)
                        num_output_units_so_far += 1
                else:
                    f(src_node, dst_node, p, num_output_units_so_far, col_idx)
                    num_output_units_so_far += 1

        for hook_name, p in self._mask_parameter_dict.items():
            dst_node = hook_name_to_node(hook_name)
            is_attn_hook = dst_node.node_type in {
                NodeType.ATTN_OUT,
                NodeType.ATTN_Q,
                NodeType.ATTN_K,
                NodeType.ATTN_V,
            }

            if sample_mask:
                p = self.sample_mask(hook_name)

            if is_attn_hook:
                for dst_head_idx in range(self.n_heads):
                    dst_node.head_idx = dst_head_idx
                    _node_level_iter(hook_name, dst_node)
            else:
                _node_level_iter(hook_name, dst_node)

    def set_binary_mask(self, circuit, invert=False):
        mult = -1 if invert else 1

        def set_edge_mask(src_node, dst_node, p, row_idx, col_idx):
            edge = circuit.edge_lookup[src_node][dst_node]
            # print(f"Setting {edge} to {edge.present}")
            # print(hook_name, x, y)
            with torch.no_grad():
                p[row_idx, col_idx] = mult * 1e5 if edge.present else mult * -1e5

        self._mask_circuit_correspondence_iter(set_edge_mask)

    def sample_circuit_from_mask(self):
        def get_edge_mask(src_node, dst_node, p, row_idx, col_idx):
            if p[row_idx, col_idx] <= 0.5:
                circuit.remove_edge(src_node, dst_node, in_place=True)

        circuit = Circuit.make_circuit(self.model)
        self._mask_circuit_correspondence_iter(get_edge_mask, sample_mask=True)

        return circuit
