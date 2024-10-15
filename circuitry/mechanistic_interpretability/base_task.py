"""
This file contains the abstract class for a mechanistic interpretability task.
"""

import abc
import warnings
from typing import NamedTuple, Union

import torch as t
from jaxtyping import Float
from torch import Tensor
from transformer_lens import HookedTransformer

from circuitry.circuit import Circuit, EdgeLevelMaskedTransformer
from circuitry.circuit.edge_level_masked_transformer import CircuitStartingPointType


class BaseTask(abc.ABC):
    """
    Base class for a mechanistic interpretability task.

    A mechanistic interpretability task defines a model, a dataset and a score function that
    evaluates the model on the dataset. The task can be evaluated on different circuits of the
    model, that are set before with the `set_circuit` method.

    Parameters
    ----------
    zero_ablation : bool
        Whether to use zero ablation or ablate with a corrupted dataset.
    device : str
        Device to run the experiment on. Default is "cpu".
    use_pos_embed : bool
        Whether to use positional embedding. Default is False.
    task_settings : NamedTuple
        Additional settings for the task that are passed directly
        to the each _load_* method. This named tuple is not directly used
        or stored by the BaseTask class.

    Attributes
    ----------
    complete_circuit : Circuit
        The complete circuit of the model.
    canonical_circuit : Circuit
        The canonical circuit of the model.
    """

    def __init__(
        self,
        zero_ablation: bool,
        use_pos_embed: bool,
        task_settings: NamedTuple,
    ):
        self._zero_ablation = zero_ablation

        self._base_dataset = self._load_base_dataset(task_settings)
        self._ablation_dataset = self._load_ablation_dataset(task_settings)
        self._model = self._load_model(task_settings)
        self._dataset_metadata = self._load_dataset_metadata(task_settings)
        self.canonical_circuit = self._load_canonical_circuit(task_settings)
        self.complete_circuit = Circuit.make_circuit(self._model, use_pos_embed=use_pos_embed)

        self._masked_model = EdgeLevelMaskedTransformer(
            self._model,
            mask_init_p=1.0,
            starting_point_type=(
                CircuitStartingPointType.POS_EMBED
                if use_pos_embed
                else CircuitStartingPointType.RESID_PRE
            ),
        )

    @staticmethod
    def eval_metric(
        original_score: Union[Float[Tensor, " batch"], Float[Tensor, ""]],
        candidate_score: Union[Float[Tensor, " batch"], Float[Tensor, ""]],
        use_mean: bool = False,
        distance: str = "l2",
    ) -> t.Tensor:
        """
        TODO: clean this up
        Returns the squared difference between the original score and the candidate score.
        """
        # Check that if per prompt then the batch dimension is the same
        if use_mean:
            candidate_score = candidate_score.float().mean()
            original_score = original_score.float().mean()
        if not use_mean:
            assert candidate_score.shape[0] == original_score.shape[0]

        # Compute the difference
        if distance == "l2":
            diff = (candidate_score - original_score) ** 2
        elif distance == "l1":
            diff = t.abs(candidate_score - original_score)
        else:
            raise ValueError(f"Unknown distance {distance}, try input l1 or l2")
        return diff.float().mean()

    def score_and_logits(self, circuit: Circuit, invert: bool = False):
        """
        Compute the score of the given circuit on the task.
        """
        logits, loss = self._masked_model.run_circuit(
            circuit=circuit,
            base_dataset=self._base_dataset,
            ablation_dataset=self._ablation_dataset,
            zero_ablation=self._zero_ablation,
            return_type="both",
            loss_per_token=True,
            invert=invert,
        )

        score = self._compute_score_from_output_logits(
            logits=logits,
            loss=loss,
            dataset_metadata=self._dataset_metadata,
        )

        return score, logits

    @abc.abstractmethod
    def _load_model(self, task_settings: NamedTuple) -> HookedTransformer:
        """
        Returns the model to be used in the task.
        """

    @abc.abstractmethod
    def _load_canonical_circuit(self, task_settings: NamedTuple) -> Circuit:
        """
        Returns the canonical circuit of the model.
        """

    @abc.abstractmethod
    def _load_base_dataset(self, task_settings: NamedTuple):
        """
        Returns the base dataset of the task on which the model
        is going to be evaluated.
        """

    def _load_ablation_dataset(self, task_settings: NamedTuple):
        """
        Returns the ablation dataset of the task which is used
        for doing activation patching.
        """
        # Add a warning if this is not implemented
        warnings.warn(
            f"No ablation dataset provided for the class {self.__class__.__name__}. ",
            stacklevel=2,
        )

        return None

    def _load_dataset_metadata(self, task_settings: NamedTuple) -> dict:
        """
        Returns metadata about the dataset which is used to compute the score.
        For example masks for the dataset.

        All metadata should be returned as a dictionary.
        By default, returns an empty dictionary.
        """
        return {}

    @abc.abstractmethod
    def _compute_score_from_output_logits(
        self,
        logits: Tensor,
        loss: Tensor,
        dataset_metadata: NamedTuple,
    ):
        """
        Computes the score of the model on the dataset. This function is used
        to evaluate the model on the dataset after having applied the circuit
        and the ablation scheme. See `score_and_logits` for more details.

        Parameters
        ----------
        logits : Tensor
            The logits of the model after running the model on the base dataset
            after having applied the circuit and the ablation scheme.
        loss : Union[Tensor, Float]
            The loss of the model on the base dataset after having applied the circuit
            and the ablation scheme.
        dataset_metadata : NamedTuple
            Any metadata about the dataset which is used to compute the score. For e
            example masks for the dataset.
        """

    @property
    def name(self):
        """
        Returns the name of the class that inherits from BaseTask.
        """
        return self.__class__.__name__
