"""
Heavily inspired by:
- https://github.com/ArthurConmy/Automatic-Circuit-Discovery/blob/main/acdc/ioi/utils.py
"""

from functools import cache
from typing import NamedTuple

import torch
import torch as t
from jaxtyping import Float
from transformer_lens import HookedTransformer

from circuitry.circuit import Circuit
from circuitry.mechanistic_interpretability import BaseTask
from circuitry.mechanistic_interpretability.examples.canonical_circuits import (
    get_ioi_canonical_circuit,
)
from circuitry.mechanistic_interpretability.examples.ioi.ioi_dataset import IOIDataset


class _IOITaskSettings(NamedTuple):
    device: str
    n_examples: int


class _IOIMetadata(NamedTuple):
    labels: t.Tensor
    wrong_labels: t.Tensor


class IOITask(BaseTask):
    """
    IOI task from the paper: https://arxiv.org/pdf/2211.00593

    The task is to complte senteces of the form.
    "When Mary and John went to the store John gave a bottle of milk
    to _____________." (Mary)

    The taks is measured by comparing the logit difference between
    the correct (Mary) and incorrect (John) completions of the sentence.

    Parameters
    ----------
    zero_ablation: bool
        Whether to use zero ablation or not. Default is True.
    device: str
        Device to use for the model. Default is "cpu".
    n_examples: int or None
        Number of examples to use in the dataset. If None, all examples will be used.

    """

    def __init__(
        self,
        zero_ablation: bool = False,
        device: str = "cpu",
        n_examples: int = 100,
    ):
        """
        Parameters
        ----------
        zero_ablation: bool
            Whether to use zero ablation or not. Default is True.
        device: str
            Device to use for the model. Default is "cpu".
        n_examples: int or None
            Number of examples to use in the dataset. If None, all examples will be used.
        """
        task_settings: NamedTuple = _IOITaskSettings(device=device, n_examples=n_examples)

        # use_pos_embed=False because this is what ACDC does: https://tinyurl.com/acdc-colab-demo
        super().__init__(
            zero_ablation=zero_ablation, task_settings=task_settings, use_pos_embed=False
        )

    def _load_model(self, task_settings: _IOITaskSettings) -> HookedTransformer:
        return get_gpt2_small(device=task_settings.device)

    def _load_canonical_circuit(self, task_settings: _IOITaskSettings) -> Circuit:
        return get_ioi_canonical_circuit()

    def _load_dataset_metadata(self, task_settings: _IOITaskSettings) -> _IOIMetadata:
        _, _, labels, wrong_labels = get_validation_data(
            n_examples=task_settings.n_examples,
            device=task_settings.device,
        )
        return _IOIMetadata(labels=labels, wrong_labels=wrong_labels)

    def _load_base_dataset(
        self, task_settings: _IOITaskSettings
    ) -> Float[t.Tensor, "n_examples seq_len"]:
        base_data, _, _, _ = get_validation_data(
            n_examples=task_settings.n_examples,
            device=task_settings.device,
        )
        return base_data

    def _load_ablation_dataset(
        self, task_settings: _IOITaskSettings
    ) -> Float[t.Tensor, "n_examples seq_len"]:
        _, ablation_data, _, _ = get_validation_data(
            n_examples=task_settings.n_examples,
            device=task_settings.device,
        )
        return ablation_data

    def _compute_score_from_output_logits(
        self,
        logits: Float[t.Tensor, "n_examples seq_len d_vocab"],
        loss: Float[t.Tensor, "n_examples seq_len-1"],
        dataset_metadata: _IOIMetadata,
    ):
        range = torch.arange(len(logits))
        correct_logits = logits[range, -1, dataset_metadata.labels]
        incorrect_logits = logits[range, -1, dataset_metadata.wrong_labels]

        result = -(correct_logits - incorrect_logits).view(-1)
        return result


# --------------------------------------------------------------
#  ACDC code
# --------------------------------------------------------------
def get_gpt2_small(device="cuda") -> HookedTransformer:
    tl_model = HookedTransformer.from_pretrained("gpt2")
    tl_model = tl_model.to(device)
    tl_model.set_use_attn_result(True)
    tl_model.set_use_split_qkv_input(True)
    if "use_hook_mlp_in" in tl_model.cfg.to_dict():
        tl_model.set_use_hook_mlp_in(True)
    return tl_model


@cache
def get_validation_data(
    n_examples: int, device: str
) -> tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor]:
    """
    Returns validation data for the IOI task.
    """
    ioi_dataset = IOIDataset(
        prompt_type="ABBA",
        N=n_examples * 2,
        nb_templates=1,
        seed=0,
    )

    abc_dataset = (
        ioi_dataset.gen_flipped_prompts(("IO", "RAND"), seed=1)
        .gen_flipped_prompts(("S", "RAND"), seed=2)
        .gen_flipped_prompts(("S1", "RAND"), seed=3)
    )

    seq_len = ioi_dataset.toks.shape[1]
    assert seq_len == 16, f"Well, I thought ABBA #1 was 16 not {seq_len} tokens long..."

    default_data = ioi_dataset.toks.long()[: n_examples * 2, : seq_len - 1].to(device)
    patch_data = abc_dataset.toks.long()[: n_examples * 2, : seq_len - 1].to(device)
    labels = ioi_dataset.toks.long()[: n_examples * 2, seq_len - 1]
    wrong_labels = torch.as_tensor(
        ioi_dataset.s_tokenIDs[: n_examples * 2], dtype=torch.long, device=device
    )

    assert torch.equal(labels, torch.as_tensor(ioi_dataset.io_tokenIDs, dtype=torch.long))
    labels = labels.to(device)

    validation_data = default_data[:n_examples, :]
    validation_patch_data = patch_data[:n_examples, :]
    validation_labels = labels[:n_examples]
    validation_wrong_labels = wrong_labels[:n_examples]

    return validation_data, validation_patch_data, validation_labels, validation_wrong_labels
