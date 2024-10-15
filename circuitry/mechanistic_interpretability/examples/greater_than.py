from functools import cache
from typing import ClassVar, NamedTuple, Optional

import torch
import torch as t
from jaxtyping import Float, Int
from transformer_lens import HookedTransformer

from circuitry.circuit import Circuit
from circuitry.mechanistic_interpretability import BaseTask
from circuitry.mechanistic_interpretability.examples.canonical_circuits import (
    get_greaterthan_canonical_circuit,
)

# ---------------------------------------------
#  ACDC specific functions
# ---------------------------------------------

TERMINATION_POS = 7
TOKEN_ID_01 = 486

NOUNS = [
    "abduction",
    "accord",
    "affair",
    "agreement",
    "appraisal",
    "assaults",
    "assessment",
    "attack",
    "attempts",
    "campaign",
    "captivity",
    "case",
    "challenge",
    "chaos",
    "clash",
    "collaboration",
    "coma",
    "competition",
    "confrontation",
    "consequence",
    "conspiracy",
    "construction",
    "consultation",
    "contact",
    "contract",
    "convention",
    "cooperation",
    "custody",
    "deal",
    "decline",
    "decrease",
    "demonstrations",
    "development",
    "disagreement",
    "disorder",
    "dispute",
    "domination",
    "dynasty",
    "effect",
    "effort",
    "employment",
    "endeavor",
    "engagement",
    "epidemic",
    "evaluation",
    "exchange",
    "existence",
    "expansion",
    "expedition",
    "experiments",
    "fall",
    "fame",
    "flights",
    "friendship",
    "growth",
    "hardship",
    "hostility",
    "illness",
    "impact",
    "imprisonment",
    "improvement",
    "incarceration",
    "increase",
    "insurgency",
    "invasion",
    "investigation",
    "journey",
    "kingdom",
    "marriage",
    "modernization",
    "negotiation",
    "notoriety",
    "obstruction",
    "operation",
    "order",
    "outbreak",
    "outcome",
    "overhaul",
    "patrols",
    "pilgrimage",
    "plague",
    "plan",
    "practice",
    "process",
    "program",
    "progress",
    "project",
    "pursuit",
    "quest",
    "raids",
    "reforms",
    "reign",
    "relationship",
    "retaliation",
    "riot",
    "rise",
    "rivalry",
    "romance",
    "rule",
    "sanctions",
    "shift",
    "siege",
    "slump",
    "stature",
    "stint",
    "strikes",
    "study",
    "test",
    "testing",
    "tests",
    "therapy",
    "tour",
    "tradition",
    "treaty",
    "trial",
    "trip",
    "unemployment",
    "voyage",
    "warfare",
    "work",
]


class GreaterThanConstants:
    YEARS: list[str]
    YEARS_BY_CENTURY: dict[str, list[str]]
    TOKENS: list[int]
    INV_TOKENS: dict[int, int]
    TOKENS_TENSOR: torch.Tensor
    INV_TOKENS_TENSOR: torch.Tensor

    _instance: ClassVar[Optional["GreaterThanConstants"]] = None

    @classmethod
    def get(cls: type["GreaterThanConstants"], device) -> "GreaterThanConstants":
        if cls._instance is None:
            cls._instance = cls(device)
        return cls._instance

    def __init__(self, device):
        model = get_gpt2_small(device=device)
        _TOKENIZER = model.tokenizer
        del model

        self.YEARS = []
        self.YEARS_BY_CENTURY = {}

        for century in range(11, 18):
            all_success = []
            for year in range(century * 100 + 2, (century * 100) + 99):
                a = _TOKENIZER.encode(f" {year}")
                if a == [
                    _TOKENIZER.encode(f" {str(year)[:2]}")[0],
                    _TOKENIZER.encode(str(year)[2:])[0],
                ]:
                    all_success.append(str(year))
                    continue
            self.YEARS.extend(all_success[1:-1])
            self.YEARS_BY_CENTURY[century] = all_success[1:-1]

        TOKENS = {i: _TOKENIZER.encode(f"{'0' if i<=9 else ''}{i}")[0] for i in range(100)}
        self.INV_TOKENS = {v: k for k, v in TOKENS.items()}
        self.TOKENS = TOKENS

        TOKENS_TENSOR = torch.as_tensor([TOKENS[i] for i in range(100)], dtype=torch.long)
        INV_TOKENS_TENSOR = torch.zeros(50290, dtype=torch.long)
        for i, v in enumerate(TOKENS_TENSOR):
            INV_TOKENS_TENSOR[v] = i

        self.TOKENS_TENSOR = TOKENS_TENSOR
        self.INV_TOKENS_TENSOR = INV_TOKENS_TENSOR


@cache
def get_gpt2_small(device="cuda") -> HookedTransformer:
    tl_model = HookedTransformer.from_pretrained("gpt2")
    tl_model = tl_model.to(device)
    tl_model.set_use_attn_result(True)
    tl_model.set_use_split_qkv_input(True)
    if "use_hook_mlp_in" in tl_model.cfg.to_dict():
        tl_model.set_use_hook_mlp_in(True)
    return tl_model


def get_year_data(num_examples, model):
    constants = GreaterThanConstants.get(model.cfg.device)

    template = "The {noun} lasted from the year {year1} to "

    # set some random seed
    torch.random.manual_seed(54)
    nouns_perm = torch.randint(0, len(NOUNS), (num_examples,))
    years_perm = torch.randint(0, len(constants.YEARS), (num_examples,))

    prompts = []
    prompts_tokenized = []
    for i in range(num_examples):
        year = constants.YEARS[years_perm[i]]
        prompts.append(
            template.format(
                noun=NOUNS[nouns_perm[i]],
                year1=year,
            )
            + year[:2]
        )
        prompts_tokenized.append(
            model.tokenizer.encode(prompts[-1], return_tensors="pt").to(model.cfg.device)
        )
        assert prompts_tokenized[-1].shape == prompts_tokenized[0].shape, (
            prompts_tokenized[-1].shape,
            prompts_tokenized[0].shape,
        )
    prompts_tokenized = torch.cat(prompts_tokenized, dim=0)
    assert len(prompts_tokenized.shape) == 2, prompts_tokenized.shape

    return prompts_tokenized, prompts


# ---------------------------------------------
#  Task specific functions
# ---------------------------------------------


class _GreaterThanTaskSettings(NamedTuple):
    device: str
    n_examples: int


class _GreaterThanMetadata(NamedTuple):
    """
    years_to_tokens_as_tensor: t.Tensor
        A tensor such that years_to_tokens_as_tensor[year_term] = token
    tokens_to_years_as_tensor: t.Tensor
        A tensor of the vocab length such that
        tokens_to_years_as_tensor[token] = year_term
        if token is a token for a year term, and 0 otherwise
    """

    year_termination_to_tokens_as_tensor: Int[
        t.Tensor, " n_year_terminations"
    ]  # 100 year terminations
    tokens_to_year_termination_as_tensor: Int[t.Tensor, " vocab"]
    base_dataset: Int[t.Tensor, "n_examples seq_len"]


class GreaterThanTask(BaseTask):
    """
    Greater than task from:
     - How does GPT-2 compute greater-than?:
     Interpreting mathematical abilities in a pre-trained language model
    https://openreview.net/pdf?id=p4PckNQR8k

    This task is about predicting the year termination of a sentence. In particular
    if the data have the structure The {noun} lasted from the year XXYY to XX__ where
    the model should predict the year termination XX__ such that __ is greater than or
    equal to YY.

    We measure if the model performs this task by computing the difference between
    the probabilities for the correct year terminations and the sum of the probabilities
    for the incorrect year terminations. That is
    score = sum_{i >= YY} p(i) - sum_{i < YY} p(i)
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
        task_settings: NamedTuple = _GreaterThanTaskSettings(device=device, n_examples=n_examples)

        # use_pos_embed=False because this is what ACDC does
        # https://tinyurl.com/acdc-colab-demo
        super().__init__(
            zero_ablation=zero_ablation,
            use_pos_embed=False,
            task_settings=task_settings,
        )

    def _load_model(self, task_settings: _GreaterThanTaskSettings) -> HookedTransformer:
        return get_gpt2_small(task_settings.device)

    def _load_canonical_circuit(self, task_settings: _GreaterThanTaskSettings) -> Circuit:
        return get_greaterthan_canonical_circuit()

    def _load_base_dataset(
        self, task_settings: _GreaterThanTaskSettings
    ) -> Int[t.Tensor, "n_examples seq_len"]:
        """
        Load the base dataset for the greater than task.
        The base dataset is a list of examples of the form
        "The {noun} lasted from the year XXYY to XX__"
        """
        model = self._load_model(task_settings)
        base_dataset, _ = get_year_data(task_settings.n_examples * 2, model)
        return base_dataset[: task_settings.n_examples]

    def _load_ablation_dataset(
        self, task_settings: _GreaterThanTaskSettings
    ) -> Int[t.Tensor, "n_examples seq_len"]:
        """
        Load the ablation dataset for the greater than task.
        The ablation dataset is the same as the he base dataset but with
        the last token replaced with 01.
        """
        model = self._load_model(task_settings)
        base_dataset, _ = get_year_data(task_settings.n_examples * 2, model)
        ablation_dataset = base_dataset.clone()
        ablation_dataset[:, TERMINATION_POS] = TOKEN_ID_01  # replace with 01
        return ablation_dataset[: task_settings.n_examples]

    def _load_dataset_metadata(
        self, task_settings: _GreaterThanTaskSettings
    ) -> _GreaterThanMetadata:
        model = self._load_model(task_settings)
        base_dataset, _ = get_year_data(task_settings.n_examples * 2, model)
        constants = GreaterThanConstants.get(model.cfg.device)

        return _GreaterThanMetadata(
            year_termination_to_tokens_as_tensor=constants.TOKENS_TENSOR,
            tokens_to_year_termination_as_tensor=constants.INV_TOKENS_TENSOR,
            base_dataset=base_dataset[: task_settings.n_examples],
        )

    def _compute_score_from_output_logits(
        self,
        logits: Float[t.Tensor, "n_examples seq_len vocab"],
        loss: Float[t.Tensor, "n_examples seq_len-1"],
        dataset_metadata: _GreaterThanMetadata,
    ) -> Float[torch.Tensor, " n_examples"]:
        """
        For each example of the form "The {noun} lasted from the year XXYY to XX__"
        we return the score:
            score = sum_{i >= YY} p(i) - sum_{i < YY} p(i)
        where p(i) is the probability of token i.

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
        tok_to_year = dataset_metadata.tokens_to_year_termination_as_tensor
        year_to_tok = dataset_metadata.year_termination_to_tokens_as_tensor
        base_dataset = dataset_metadata.base_dataset

        last_logits: Float[t.Tensor, "n_examples vocab"] = logits[:, -1]
        last_probs = torch.nn.functional.softmax(last_logits, dim=-1)

        # Gets the cumulative sum of terminations starting from 0 to 99
        # cumsum[i] = sum_{j=0}^{i} p(j) where p(j) is the probability of token j
        csum: Float[t.Tensor, "n_examples n_year_terminations"] = torch.cumsum(
            last_probs[:, year_to_tok], dim=-1
        )
        # Gets the year termination for each example
        yearend: Float[t.Tensor, " n_examples"] = tok_to_year[
            base_dataset[:, TERMINATION_POS].cpu()
        ].to(logits.device)

        range_tensor = torch.arange(len(yearend))
        total = csum[:, -1]

        # If the true termination is 00, then the negative sum is 0
        # if the true termination is different then we select sum_{i < YY} p(i)
        negative = torch.where(
            yearend == 0, torch.zeros((), device=csum.device), csum[range_tensor, yearend]
        )
        positive = total - negative

        return -(positive - negative)
