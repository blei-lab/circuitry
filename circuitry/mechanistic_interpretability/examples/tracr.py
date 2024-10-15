"""
This module implements the tracr reverse task from the tracr paper.
The code is heavily based on ACDC.

For more details, please see: https://tinyurl.com/acdc-tracr
"""

import itertools
from typing import ClassVar, Literal, NamedTuple

import einops
import numpy as np
import torch
from jaxtyping import Float, Int
from torch import Tensor
from tracr.compiler import compiling
from tracr.rasp import rasp
from transformer_lens import HookedTransformer, HookedTransformerConfig

from circuitry.circuit import Circuit
from circuitry.mechanistic_interpretability import BaseTask
from circuitry.mechanistic_interpretability.examples.canonical_circuits import (
    get_tracr_proportion_canonical_circuit,
    get_tracr_reverse_canonical_circuit,
)
from circuitry.utils import rng_context

BOS_TOKEN = 3


class _ReverseTracrTaskSettings(NamedTuple):
    """
    Settings for the reverse task.
    """

    device: str


class _ReverseTracrMetadata(NamedTuple):
    """
    Metadata for the reverse task. The
    """

    # -1 because we remove the BOS token
    original_model_output: Float[Tensor, "n_examples seq_len-1 n_tokens"]


class _ProportionTracrTaskSettings(NamedTuple):
    """
    Settings for the proportion task.
    """

    device: str
    n_examples: int


class _ProportionTracrMetadata(NamedTuple):
    """
    Metadata for the proportion task. The
    """

    # -1 because we remove the BOS token
    original_model_output: Float[Tensor, "n_examples seq_len-1"]


class TracrReverseTask(BaseTask):
    """
    Implements the reverse task from the tracr paper

    - This task is to reverse a list of 3 elements.
        "BOS 1 2 3" -> "BOS 3 2 1"
    - The model is a compiled RASP program that reverses lists.
    - The dataset is a list of 3 elements with the BOS token at the start
      in all possible permutations.
    - The score is the mean squared difference between the original
      model logits and the model logits produced by the model.

    Parameters
    ----------
    zero_ablation : bool
        Whether to use zero ablation or ablate with a corrupted dataset.
    device : str
        Device to run the experiment on.
    """

    SEQ_LEN: ClassVar[int] = 4
    N_EXAMPLES: ClassVar[int] = 6

    def __init__(
        self,
        zero_ablation: bool = False,
        device: str = "cpu",
    ):

        super().__init__(
            zero_ablation=zero_ablation,
            use_pos_embed=True,
            task_settings=_ReverseTracrTaskSettings(device=device),
        )

    def _load_model(self, task_settings: _ReverseTracrTaskSettings) -> HookedTransformer:
        """
        Loads the model for the reverse task which is a compiled RASP program.
        """
        model = get_tracr_model_input_and_tl_model("reverse", task_settings.device)
        return model

    def _load_canonical_circuit(self, task_settings: _ReverseTracrTaskSettings) -> Circuit:
        """
        Load the canonical circuit.
        """
        return get_tracr_reverse_canonical_circuit()

    def _load_dataset_metadata(
        self, task_settings: _ReverseTracrTaskSettings
    ) -> _ReverseTracrMetadata:
        """
        Load the dataset metadata which is simply the original model output.
        """
        model = get_tracr_model_input_and_tl_model("reverse", task_settings.device)
        base_dataset = self._load_base_dataset(task_settings)

        with torch.no_grad():
            orignal_model_output: Float[Tensor, "n_examples seq_len n_tokens"] = model(base_dataset)

        return _ReverseTracrMetadata(
            original_model_output=orignal_model_output,
        )

    def _load_base_dataset(self, task_settings: _ReverseTracrTaskSettings) -> Int[Tensor, "n_examples seq_len"]:  # type: ignore
        """
        Returns the dataset where
        3 is the BOS token, and 0, 1, 2 are the tokens to permute.
        [
            [3, 0, 1, 2],
            [3, 0, 2, 1],
            [3, 1, 0, 2],
            [3, 1, 2, 0],
            [3, 2, 0, 1],
            [3, 2, 1, 0]
        ]
        """
        data_tens = torch.zeros(
            (self.N_EXAMPLES, self.SEQ_LEN), device=task_settings.device, dtype=torch.long
        )

        vals = [0, 1, 2]

        for perm_idx, perm in enumerate(itertools.permutations(vals)):
            data_tens[perm_idx] = torch.tensor([BOS_TOKEN, perm[0], perm[1], perm[2]])

        return data_tens

    def _load_ablation_dataset(
        self, task_settings: _ReverseTracrTaskSettings
    ) -> Int[Tensor, "n_examples seq_len"]:
        """
        Returns a permuted version of the base dataset.
        """
        base_dataset = self._load_base_dataset(task_settings)
        patch_data_indices = get_perm(len(base_dataset), seed=42)
        ablation_dataset = base_dataset[patch_data_indices]
        return ablation_dataset

    def _compute_score_from_output_logits(
        self,
        logits: Float[Tensor, "n_examples seq_len n_tokens"],
        loss: Float[Tensor, "n_examples seq_len"],
        dataset_metadata: _ReverseTracrMetadata,
    ) -> Float[Tensor, " n_examples"]:
        """
        This function computes the score of the model output logits
        by taking the mean of the squared difference between the logits and
        the logits produced by the original model.
        """

        # Get rid of logits for the BOS token
        logits = logits[:, 1:, :]
        original = dataset_metadata.original_model_output[:, 1:, :]
        diff: Float[Tensor, "n_examples seq_len-1 n_tokens"] = (logits - original) ** 2
        diff = diff.mean(-1).mean(-1)  # Taken the mean along the seq_len and n_tokens axis
        return diff


class TracrProportionTask(BaseTask):
    """
    Implements the proportion task from the tracr paper
    This task consists in estimating the proportion of x in a sequence of tokens.

    For example
    ``BOS x x y z" -> "BOS 1 1 0.66 0.5"``
    Additionally these task has the following characteristics:
    - The model is a compiled RASP program that estimates the proportion of x in a sequence.
    - The dataset consists of strings of length four of characters drawn from  (w, x, y, z)
      with replacement. An additional BOS token is added at the start.
    - The model is ground truth, rasp program.
    - The score is the mean squared difference between the original predicted proportion
      and the circuit's predicted proportion.

    Parameters
    ----------
    zero_ablation : bool
        Whether to use zero ablation or ablate with a corrupted dataset.
    device : str
        Device to run the experiment on.
    n_examples : int
        Number of examples to use in the dataset.
    """

    SEQ_LEN: ClassVar[int] = 5

    def __init__(
        self,
        zero_ablation: bool = False,
        device: str = "cpu",
        n_examples: int = 128,
    ):

        super().__init__(
            zero_ablation=zero_ablation,
            use_pos_embed=True,
            task_settings=_ProportionTracrTaskSettings(device=device, n_examples=n_examples),
        )

    def score_and_logits(
        self, circuit: Circuit, invert: bool = False
    ) -> tuple[Float[Tensor, " n_examples"], Float[Tensor, " n_examples seq_len n_tokens"]]:
        """
        Compute the score of the given circuit on the task.
        """
        # We neeed to rewrite this function because we omit the computation of
        # the loss if not the model crashes. Weird stuff with RASP models.
        logits = self._masked_model.run_circuit(
            circuit=circuit,
            base_dataset=self._base_dataset,
            ablation_dataset=self._ablation_dataset,
            zero_ablation=self._zero_ablation,
            return_type="logits",
            loss_per_token=False,
            invert=invert,
        )

        score = self._compute_score_from_output_logits(
            logits=logits,
            dataset_metadata=self._dataset_metadata,
        )

        return score, logits

    def _load_model(self, task_settings: _ReverseTracrTaskSettings) -> HookedTransformer:
        """
        Load compiled RASP model for the proportion task.
        """
        model = get_tracr_model_input_and_tl_model("proportion", task_settings.device)
        return model

    def _load_canonical_circuit(self, task_settings: _ReverseTracrTaskSettings) -> Circuit:
        """
        Load the canonical circuit.
        """
        return get_tracr_proportion_canonical_circuit()

    def _load_dataset_metadata(
        self, task_settings: _ReverseTracrTaskSettings
    ) -> _ReverseTracrMetadata:
        """
        Load the dataset metadata. This consists of the original model output.
        """
        model = get_tracr_model_input_and_tl_model("proportion", task_settings.device)
        base_dataset = self._load_base_dataset(task_settings)
        with torch.no_grad():
            orignal_model_output: Float[Tensor, "n_examples seq_len n_tokens"] = model(base_dataset)

        return _ProportionTracrMetadata(
            original_model_output=orignal_model_output,
        )

    def _load_base_dataset(self, task_settings: _ProportionTracrTaskSettings) -> Int[Tensor, "n_examples seq_len"]:  # type: ignore
        """
        Returns the dataset:
        Each row consists of a draws from (2, 3, 4, 5) with replacement
        but every example is different. These numbers are the token id corresponding
        to "w", "x", "y", "z" respectively. The number 0 is added at the start
        to represent the BOS token.
        """

        n_examples = task_settings.n_examples
        device = task_settings.device

        alphabet = "wxyz"
        # values picked by manually looking at the encoder
        mapping = {"w": 2, "x": 3, "y": 4, "z": 5, "BOS": 0}

        # All possible permutations of the alphabet of length SEQ_LEN-1 (excluding BOS token)
        all_permutations = list(itertools.product(alphabet, repeat=self.SEQ_LEN - 1))
        rand_choices = get_perm(len(all_permutations), no_fp=False, seed=42)
        data_tens = torch.zeros((n_examples, self.SEQ_LEN), dtype=torch.long, device=device)

        for i in range(len(data_tens)):
            selected_perm = all_permutations[rand_choices[i]]
            selected_perm = ["BOS", *list(selected_perm)]
            data_tens[i] = torch.tensor([mapping[c] for c in selected_perm]).int()

        return data_tens

    def _load_ablation_dataset(
        self, task_settings: _ProportionTracrTaskSettings
    ) -> Int[Tensor, "n_examples seq_len"]:
        """
        Returns a permuted version of the base dataset.
        """
        base_dataset = self._load_base_dataset(task_settings)
        patch_data_indices = get_perm(len(base_dataset), seed=42)
        ablation_dataset = base_dataset[patch_data_indices]
        return ablation_dataset

    def _compute_score_from_output_logits(
        self,
        logits: Float[Tensor, "n_examples seq_len n_tokens"],
        dataset_metadata: _ReverseTracrMetadata,
    ) -> Float[Tensor, " n_examples"]:
        """
        Computes the squared difference between the predictions for
        for the proportion task and the original model output (which is correct).
        """
        # Get rid of logits for the BOS token and only consider the first token.
        # First token contains the proportion of token x in the sequence
        logits = logits[:, 1:, 0]
        original = dataset_metadata.original_model_output[:, 1:, 0]

        diff: Float[Tensor, "n_examples seq_len-1"] = (logits - original) ** 2
        diff = diff.mean(-1)
        return diff


########################
# Helper functions
########################


# get some random permutation with no fixed points
def get_perm(n: int, no_fp=True, seed=42) -> Int[Tensor, " n"]:
    if no_fp:
        assert n > 1

    with rng_context(seed=seed):
        perm = torch.randperm(n)
        while (perm == torch.arange(n)).any().item():
            perm = torch.randperm(n)
        return perm


def get_tracr_model_input_and_tl_model(
    task: Literal["reverse", "proportion"], device: str
) -> HookedTransformer:
    """
    This function adapts Neel's TransformerLens porting of tracr
    """
    bos = "BOS"

    # Loads an example RASP program model. This program reverses lists.
    # The model takes as input a list of pre-tokenization elements (here `["BOS", 1, 2, 3]`),
    # these are tokenized (`[3, 0, 1, 2]`), the transformer is applied, and then an argmax is taken over the output and it is detokenized
    # - this can be seen on the `out.decoded` attribute of the output

    def make_length():
        all_true_selector = rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.TRUE)
        return rasp.SelectorWidth(all_true_selector)

    if task == "reverse":
        length = make_length()  # `length` is not a primitive in our implementation.
        opp_index = length - rasp.indices - 1
        flip = rasp.Select(rasp.indices, opp_index, rasp.Comparison.EQ)
        reverse = rasp.Aggregate(flip, rasp.tokens)
        model = compiling.compile_rasp_to_model(
            reverse,
            vocab={1, 2, 3},
            max_seq_len=5,
            compiler_bos=bos,
        )

    elif task == "proportion":
        from tracr.compiler.lib import make_frac_prevs

        model = compiling.compile_rasp_to_model(
            make_frac_prevs(rasp.tokens == "x"),
            vocab={"w", "x", "y", "z"},
            max_seq_len=5,
            compiler_bos="BOS",
        )

    # Extract the model config from the Tracr model, and create a blank HookedTransformer object
    n_heads = model.model_config.num_heads
    n_layers = model.model_config.num_layers
    d_head = model.model_config.key_size
    d_mlp = model.model_config.mlp_hidden_size
    act_fn = "relu"
    normalization_type = "LN" if model.model_config.layer_norm else None
    attention_type = "causal" if model.model_config.causal else "bidirectional"

    n_ctx = model.params["pos_embed"]["embeddings"].shape[0]
    # Equivalent to length of vocab, with BOS and PAD at the end
    d_vocab = model.params["token_embed"]["embeddings"].shape[0]
    # Residual stream width, I don't know of an easy way to infer it from the above config.
    d_model = model.params["token_embed"]["embeddings"].shape[1]

    # Equivalent to length of vocab, WITHOUT BOS and PAD at the end because we never care about these outputs
    d_vocab_out = model.params["token_embed"]["embeddings"].shape[0] - 2

    cfg = HookedTransformerConfig(
        n_layers=n_layers,
        d_model=d_model,
        d_head=d_head,
        n_ctx=n_ctx,
        d_vocab=d_vocab,
        d_vocab_out=d_vocab_out,
        d_mlp=d_mlp,
        n_heads=n_heads,
        act_fn=act_fn,
        attention_dir=attention_type,
        normalization_type=normalization_type,
        use_attn_result=True,
        use_split_qkv_input=True,
        device=device,
    )
    tl_model = HookedTransformer(cfg)
    if "use_hook_mlp_in" in tl_model.cfg.to_dict():  # both tracr models include MLPs
        tl_model.set_use_hook_mlp_in(True)

    # Extract the state dict, and do some reshaping so that everything has a n_heads dimension
    sd = {}
    sd["pos_embed.W_pos"] = model.params["pos_embed"]["embeddings"]
    sd["embed.W_E"] = model.params["token_embed"]["embeddings"]
    # Equivalent to max_seq_len plus one, for the BOS

    # The unembed is just a projection onto the first few elements of the residual stream, these store output tokens
    # This is a NumPy array, the rest are Jax Arrays, but w/e it's fine.
    sd["unembed.W_U"] = np.eye(d_model, d_vocab_out)

    for l in range(n_layers):  # noqa
        sd[f"blocks.{l}.attn.W_K"] = einops.rearrange(
            model.params[f"transformer/layer_{l}/attn/key"]["w"],
            "d_model (n_heads d_head) -> n_heads d_model d_head",
            d_head=d_head,
            n_heads=n_heads,
        )
        sd[f"blocks.{l}.attn.b_K"] = einops.rearrange(
            model.params[f"transformer/layer_{l}/attn/key"]["b"],
            "(n_heads d_head) -> n_heads d_head",
            d_head=d_head,
            n_heads=n_heads,
        )
        sd[f"blocks.{l}.attn.W_Q"] = einops.rearrange(
            model.params[f"transformer/layer_{l}/attn/query"]["w"],
            "d_model (n_heads d_head) -> n_heads d_model d_head",
            d_head=d_head,
            n_heads=n_heads,
        )
        sd[f"blocks.{l}.attn.b_Q"] = einops.rearrange(
            model.params[f"transformer/layer_{l}/attn/query"]["b"],
            "(n_heads d_head) -> n_heads d_head",
            d_head=d_head,
            n_heads=n_heads,
        )
        sd[f"blocks.{l}.attn.W_V"] = einops.rearrange(
            model.params[f"transformer/layer_{l}/attn/value"]["w"],
            "d_model (n_heads d_head) -> n_heads d_model d_head",
            d_head=d_head,
            n_heads=n_heads,
        )
        sd[f"blocks.{l}.attn.b_V"] = einops.rearrange(
            model.params[f"transformer/layer_{l}/attn/value"]["b"],
            "(n_heads d_head) -> n_heads d_head",
            d_head=d_head,
            n_heads=n_heads,
        )
        sd[f"blocks.{l}.attn.W_O"] = einops.rearrange(
            model.params[f"transformer/layer_{l}/attn/linear"]["w"],
            "(n_heads d_head) d_model -> n_heads d_head d_model",
            d_head=d_head,
            n_heads=n_heads,
        )
        sd[f"blocks.{l}.attn.b_O"] = model.params[f"transformer/layer_{l}/attn/linear"]["b"]

        sd[f"blocks.{l}.mlp.W_in"] = model.params[f"transformer/layer_{l}/mlp/linear_1"]["w"]
        sd[f"blocks.{l}.mlp.b_in"] = model.params[f"transformer/layer_{l}/mlp/linear_1"]["b"]
        sd[f"blocks.{l}.mlp.W_out"] = model.params[f"transformer/layer_{l}/mlp/linear_2"]["w"]
        sd[f"blocks.{l}.mlp.b_out"] = model.params[f"transformer/layer_{l}/mlp/linear_2"]["b"]

    # Convert weights to tensors and load into the tl_model

    for k, v in sd.items():
        # I cannot figure out a neater way to go from a Jax array to a numpy array lol
        sd[k] = torch.tensor(np.array(v))

    tl_model.load_state_dict(sd, strict=False)

    return tl_model
