"""
Implements the docstring task adapting the code from
https://tinyurl.com/acdc-docstring
"""

from typing import NamedTuple

import torch
import torch as t
from jaxtyping import Float, Int
from transformer_lens import HookedTransformer

from circuitry.circuit import Circuit
from circuitry.mechanistic_interpretability import BaseTask
from circuitry.mechanistic_interpretability.examples.canonical_circuits import (
    get_docstring_canonical_circuit,
)
from circuitry.mechanistic_interpretability.examples.docstring import prompts

# ---------------------------------------------
#  Task specific functions
# ---------------------------------------------


class _DocstringTaskSettings(NamedTuple):
    device: str
    n_examples: int


class _DocstringnMetadata(NamedTuple):
    correct_labels: Int[t.Tensor, " n_examples"]  # There's only one correct label
    wrong_labels: Int[t.Tensor, "n_examples seq_len"]


class DocstringTask(BaseTask):
    """
    This is the docstring task from: https://tinyurl.com/docstring-circuit

    The task is essentially to take a docstring of the form:
    ```
    def old(self, first, files, page, names, size, read):
        ""sector gap population

        :param page: message tree
        :param names: detail mine
        :param
    ```
    and predict the following word(s) in the docstring (for example: "size" or "read")
    The metric used is the maximum difference between the negative log likelihood
    of the correct word  minus a set of incorrect words.

    The ablation dataset consists of the same docstring but with a word changed for
    something different (for example: "size" -> "new") and with the previous
    elements in the docstring removed.

    For example:
    ```
    def old(self, first, files, project, target, new, read):
        ""sector gap population

        :param image: message tree
        :param update: detail mine
        :param

    ```

    Parameters
    ----------
    zero_ablation : bool
        Whether to use zero ablation or ablate with a corrupted dataset.
    device : str
        Device to run the experiment on. Default is "cuda".
    n_examples : int
        Number of examples to use in the dataset.
    """

    def __init__(
        self,
        zero_ablation: bool = False,
        device: str = "cpu",
        n_examples: int = 100,
    ):
        """
        n_examples:
            Number of examples to use in the dataset.
        """
        task_settings: NamedTuple = _DocstringTaskSettings(
            device=device,
            n_examples=n_examples,
        )

        # use_pos_embed=False because this is what ACDC does
        # https://tinyurl.com/acdc-colab-demo
        super().__init__(
            zero_ablation=zero_ablation,
            use_pos_embed=False,
            task_settings=task_settings,
        )

    def _load_model(self, task_settings: _DocstringTaskSettings) -> HookedTransformer:
        return get_docstring_model(task_settings.device)

    def _load_canonical_circuit(self, task_settings: _DocstringTaskSettings) -> Circuit:
        return get_docstring_canonical_circuit()

    def _load_base_dataset(
        self, task_settings: _DocstringTaskSettings
    ) -> Int[t.Tensor, "n_examples seq_len"]:
        """
        Load the base dataset for the greater than task. Each element is a list of tokens
        that represent the docstring. For example:
        ```
        def old(self, first, files, page, names, size, read):
            ""sector gap population

            :param page: message tree
            :param names: detail mine
            :param
        ```
        """
        model = self._load_model(task_settings)
        raw_prompts = get_raw_prompts(n_examples=task_settings.n_examples)

        clean_prompt = [p.clean_prompt for p in raw_prompts]

        clean_tokens = torch.stack(
            [model.to_tokens(batch, prepend_bos=True)[0] for batch in clean_prompt]
        )

        toks_int_values = clean_tokens

        validation_data = toks_int_values[: task_settings.n_examples]

        return validation_data

    def _load_ablation_dataset(
        self, task_settings: _DocstringTaskSettings
    ) -> Int[t.Tensor, "n_examples seq_len"]:
        """
        Returns the ablation dataset for the greater than task. Each element is a list of tokens
        that represent the docstring. Both the "correct" and
        the names of the arguments in the docstring are changed

        For example:
        ```
        def old(self, first, files, project, target, new, read):
            ""sector gap population

            :param image: message tree
            :param update: detail mine
            :param
        ```
        """
        dataset_version = "random_random"  # acdc does this
        model = self._load_model(task_settings)
        raw_prompts = get_raw_prompts(n_examples=task_settings.n_examples)

        corrupt_prompt = [p.corrupt_prompt[dataset_version] for p in raw_prompts]

        corrupt_tokens = torch.stack(
            [model.to_tokens(batch, prepend_bos=True)[0] for batch in corrupt_prompt]
        )
        validation_patch_data = corrupt_tokens[: task_settings.n_examples]

        return validation_patch_data

    def _load_dataset_metadata(self, task_settings: _DocstringTaskSettings) -> _DocstringnMetadata:
        """
        Returns the metadata which consists of tensors with the correct
        tokens and lists of incorrect tokens.
        """
        model = self._load_model(task_settings)
        raw_prompts = get_raw_prompts(n_examples=task_settings.n_examples)

        # Correct answer each element is a list of the form [ " word"]
        # Wrong answer each element is a list of the form [ " word1", " word2"]
        correct_answers: list[list[str]] = [p.correct_answers for p in raw_prompts]
        wrong_answers: list[list[str]] = [p.wrong_answers for p in raw_prompts]

        # [batch, n_correct_tokens]
        correct_tokens = torch.stack(
            [model.to_tokens(batch, prepend_bos=False)[:, 0] for batch in correct_answers]
        )
        # [batch, n_wrong_tokens]
        wrong_tokens = torch.stack(
            [model.to_tokens(batch, prepend_bos=False)[:, 0] for batch in wrong_answers]
        )

        toks_int_labels = correct_tokens.squeeze(-1)  # all words are a single token
        toks_int_wrong_labels = wrong_tokens

        assert toks_int_labels.ndim == 1
        assert toks_int_wrong_labels.ndim == 2

        validation_labels = toks_int_labels[: task_settings.n_examples]
        validation_wrong_labels = toks_int_wrong_labels[: task_settings.n_examples]

        return _DocstringnMetadata(
            correct_labels=validation_labels, wrong_labels=validation_wrong_labels
        )

    def _compute_score_from_output_logits(
        self,
        logits: Float[t.Tensor, "n_examples seq_len vocab"],
        loss: Float[t.Tensor, "n_examples seq_len-1"],
        dataset_metadata: _DocstringnMetadata,
    ) -> Float[torch.Tensor, " n_examples"]:
        """
        For each example the score is the maximum difference between
        logit of the correct word minus the logit of the incorrect word,
        where the maximum is taken over all incorrect words.


        Parameters
        ----------
        logits : Float[t.Tensor, "n_examples seq_len vocab"]
            The logits of the model after being run on the dataset.
        loss : Float[t.Tensor, "n_examples seq_len-1"]
            The loss of the model after being run on the dataset.
        dataset_metadata : _GreaterThanMetadata
            The metadata of the dataset.

        Returns
        -------
        score : Float[torch.Tensor, " n_examples"]
            For each example the score of the model on the dataset. where the
            score is defined as above.
        """
        correct_labels = dataset_metadata.correct_labels
        wrong_labels = dataset_metadata.wrong_labels

        # With neg sign so we minimize this
        correct_logits = logits[torch.arange(len(logits)), -1, correct_labels]
        incorrect_logits = logits[torch.arange(len(logits)).unsqueeze(-1), -1, wrong_labels]

        # note neg sign!!!
        answer = -(correct_logits - incorrect_logits.max(dim=-1).values)
        return answer


# ---------------------------------------------
#  ACDC specific functions
# ---------------------------------------------
def get_docstring_model(device: str) -> HookedTransformer:
    tl_model = HookedTransformer.from_pretrained(
        "attn-only-4l",
    )
    tl_model.set_use_attn_result(True)
    tl_model.set_use_split_qkv_input(True)
    tl_model.to(device)
    return tl_model


def get_raw_prompts(n_examples: int) -> list[prompts.Prompt]:
    # DEFAULTS BY ACDC
    docstring_ind_prompt_kwargs = {
        "n_matching_args": 3,
        "n_def_prefix_args": 2,
        "n_def_suffix_args": 1,
        "n_doc_prefix_args": 0,
        "met_desc_len": 3,
        "arg_desc_len": 2,
    }

    raw_prompts = [
        prompts.docstring_induction_prompt_generator("rest", **docstring_ind_prompt_kwargs, seed=i)
        for i in range(n_examples * 2)
    ]
    return raw_prompts
