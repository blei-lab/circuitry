from typing import NamedTuple

import huggingface_hub
import torch
import torch as t
from jaxtyping import Bool, Float
from transformer_lens import HookedTransformer

from circuitry.circuit import Circuit
from circuitry.mechanistic_interpretability import BaseTask
from circuitry.mechanistic_interpretability.examples.canonical_circuits import (
    get_induction_canonical_circuit,
)


class _InductionTaskSettings(NamedTuple):
    device: str
    seq_len: int
    n_examples: int


class _InductionMetadata(NamedTuple):
    validation_mask: t.Tensor


class InductionTask(BaseTask):
    """
    Induction task from the paper: TODO

    Parameters
    ----------
    zero_ablation: bool
        Whether to use zero ablation or not. Default is True.
    device: str
        Device to use for the model. Default is "cpu".
    seq_len: int or None
        Maximum length of the sequences to use in the dataset. If None, the sequences will
        not be truncated.
    n_examples: int or None
        Number of examples to use in the dataset. If None, all examples will be used.
        This will be the batch size of the model.
    """

    def __init__(
        self,
        zero_ablation: bool = True,
        device: str = "cpu",
        seq_len: int | None = 300,
        n_examples: int | None = 40,
    ):
        task_settings: NamedTuple = _InductionTaskSettings(
            device=device, seq_len=seq_len, n_examples=n_examples
        )

        # use_pos_embed=False because this is what ACDC does
        # https://tinyurl.com/acdc-colab-demo
        super().__init__(
            zero_ablation=zero_ablation,
            use_pos_embed=False,
            task_settings=task_settings,
        )

    def _load_model(self, task_settings: _InductionTaskSettings) -> HookedTransformer:
        return get_induction_model(task_settings.device)

    def _load_canonical_circuit(self, task_settings: _InductionTaskSettings) -> Circuit:
        return get_induction_canonical_circuit()

    def _load_base_dataset(
        self, task_settings: _InductionTaskSettings
    ) -> Float[t.Tensor, "n_examples seq_len"]:
        return get_validation_data(
            device=task_settings.device,
            seq_len=task_settings.seq_len,
            n_examples=task_settings.n_examples,
        )

    def _load_dataset_metadata(self, task_settings: _InductionTaskSettings) -> _InductionMetadata:
        validation_mask = get_mask_repeat_candidates(
            n_examples=task_settings.n_examples,
            seq_len=task_settings.seq_len,
            device=task_settings.device,
        )
        dataset_metadata = _InductionMetadata(validation_mask=validation_mask)
        return dataset_metadata

    def _compute_score_from_output_logits(
        self,
        logits: Float[t.Tensor, "n_examples seq_len d_vocab"],
        loss: Float[t.Tensor, "n_examples seq_len-1"],
        dataset_metadata: NamedTuple,
    ) -> Float[t.Tensor, " n_examples"]:
        """
        Compute the score of the induction model on the dataset. The score for each prompt
        is the average negative log-likelihood of the model for each token in the sequence
        where induction should be happening. For example (assuming each letter is a token):

        a b j k l m n [a] [b] q r s

        [a] and [b] are the tokens where induction should be happening. The score for this
        prompt would be the average negative log-likelihood of the model for the tokens [a] and [b].


        Parameters
        ----------
        logits : Tensor
            The logits of the model after running the model on the base dataset
            after having applied the circuit and the ablation scheme. This is not
            used in this task.
        loss : Tensor
            The loss of the model on the base dataset after having applied the circuit
            to the model. The tensor contains for each example the negative log likelihood
            of the model for each token in the sequence. It has shape (batch_size, seq_len-1)
            because the last token is not used to compute the loss.
        dataset_metadata : NamedTuple
            Any metadata about the dataset which is used to compute the score. For e
            example masks for the dataset.

        Returns
        -------
        avg_loss_per_prompt: t.Tensor
            The average loss per prompt or the averaged.

        Notes
        -----
        When `seq_len` is small, there might be fewer than `n_examples` returned, because
        examples where the induction is happening outside the `seq_len` range are discarded.
        """
        total_loss = (loss * dataset_metadata.validation_mask[:, :-1].int()).sum(dim=-1)
        avg_loss_per_prompt = total_loss / dataset_metadata.validation_mask[:, :-1].int().sum(
            dim=-1
        )
        nan_indices = t.isnan(avg_loss_per_prompt)
        avg_loss_per_prompt = avg_loss_per_prompt[~nan_indices]

        return avg_loss_per_prompt


# From acdc:
def get_induction_model(device: str) -> HookedTransformer:
    tl_model = HookedTransformer.from_pretrained(
        "redwood_attn_2l",  # load Redwood's model
        center_writing_weights=False,  # these are needed as this model is a Shortformer; this is a technical detail
        center_unembed=False,
        fold_ln=False,
        device=device,
    )
    tl_model.set_use_attn_result(True)
    tl_model.set_use_split_qkv_input(True)
    return tl_model


def get_validation_data(
    n_examples=None, seq_len=None, device="cpu"
) -> Float[t.Tensor, "n_examples seq_len"]:
    validation_fname = huggingface_hub.hf_hub_download(
        repo_id="ArthurConmy/redwood_attn_2l", filename="validation_data.pt"
    )
    validation_data = torch.load(validation_fname, map_location=device).long()
    return validation_data[:n_examples, :seq_len]


def get_mask_repeat_candidates(
    n_examples=None, seq_len=None, device="cpu"
) -> Bool[t.Tensor, "n_examples seq_len"]:
    """
    Returns a mask such that mask[i,j] is true if the j-th token in the i-th example is
    such that the token right afterwards should be predictable by using "induction" i.e
    looking back to the context and copying exactly the same sequence.
    """
    mask_repeat_candidates_fname = huggingface_hub.hf_hub_download(
        repo_id="ArthurConmy/redwood_attn_2l", filename="mask_repeat_candidates.pkl"
    )
    mask_repeat_candidates = torch.load(mask_repeat_candidates_fname, map_location=device)
    mask_repeat_candidates.requires_grad = False
    return mask_repeat_candidates[:n_examples, :seq_len]
