"""
This file is totally thanks to Stefan and Kajetan

See here https://github.com/jettmayzner/mi_utils_public

(this file comes from https://github.com/mlab-account/mlab2/blob/acdc/docstring_induction/acdc/gen_docstring_prompts.py)
"""

import random
from dataclasses import dataclass
from typing import Optional, Union

import torch
from jaxtyping import Float
from torch import Tensor
from transformer_lens import HookedTransformer
from typeguard import typechecked

from circuitry.mechanistic_interpretability.examples.docstring.constants import (
    COMMON_SINGLE_TOKEN_NOUNS,
    VARIABLE_NAMES,
)


@dataclass
class Prompt:
    clean_prompt: str
    corrupt_prompt: Union[str, dict[str, str]]
    correct_answers: list[str]
    wrong_answers: list[str]

    def __post_init__(self):
        assert not self.clean_prompt.startswith("<|BOS|>")
        if isinstance(self.corrupt_prompt, str):
            assert not self.corrupt_prompt.startswith("<|BOS|>")
        elif isinstance(self.corrupt_prompt, dict):
            for v in self.corrupt_prompt.values():
                assert not v.startswith("<|BOS|>")
        else:
            raise TypeError("corrupt_prompt has to be str or dict")

    def _get_corrupt_propmt(self, key=None):
        if key is None:
            assert isinstance(self.corrupt_prompt, str)
            return self.corrupt_prompt
        return self.corrupt_prompt[key]

    def _print_prompt(self, prompt):
        print(prompt, end="")
        print(f"|{self.correct_answers[0]}| <===")

    def print_clean(self):
        self._print_prompt(self.clean_prompt)

    def print_corrupt(self, key=None):
        self._print_prompt(self._get_corrupt_propmt(key))

    def print_all_corrupt(self):
        assert isinstance(self.corrupt_prompt, dict)
        for key in self.corrupt_prompt.keys():
            print()
            print(key)
            self.print_corrupt(key)

    def _print_tokenized(self, model, prompt):
        print("|", end="")
        for t in model.to_str_tokens(prompt):
            print(t, end="|")
        print(f"\nAnswer: |{self.correct_answers[0]}|")

    def print_clean_tokenized(self, model):
        self._print_tokenized(model, self.clean_prompt)

    def print_corrupt_tokenized(self, model, key=None):
        self._print_tokenized(model, self._get_corrupt_propmt(key))

    def print_all_corrupt_tokenized(self, model):
        assert isinstance(self.corrupt_prompt, dict)
        for key in self.corrupt_prompt.keys():
            print()
            print(key)
            self.print_corrupt_tokenized(model, key)

    def print_tokenized(self, model):
        """Alias for backward compatibility"""
        self.print_clean_tokenized(model)


class BatchedPrompts:
    @typechecked
    def __init__(self, prompts: list[Prompt], model: HookedTransformer):
        self.clean_prompt = [p.clean_prompt for p in prompts]
        if isinstance(prompts[0].corrupt_prompt, str):
            self.corrupt_prompt = [p.corrupt_prompt for p in prompts]
        else:
            self.corrupt_prompt = {
                key: [p.corrupt_prompt[key] for p in prompts]
                for key in prompts[0].corrupt_prompt.keys()
            }
        self.correct_answers = [p.correct_answers for p in prompts]
        if self.correct_answers[0][0][0] != " ":
            print("THE CORRECT ANSWER DOES NOT START WITH A SPACE -- ARE YOU SURE ABOUT THAT?")
        self.wrong_answers = [p.wrong_answers for p in prompts]

        self.clean_tokens = torch.stack(
            [model.to_tokens(batch, prepend_bos=True)[0] for batch in self.clean_prompt]
        )
        if isinstance(prompts[0].corrupt_prompt, str):
            self.corrupt_tokens = torch.stack(
                [model.to_tokens(batch, prepend_bos=True)[0] for batch in self.corrupt_prompt]
            )
        else:
            self.corrupt_tokens = {
                key: torch.stack([model.to_tokens(batch, prepend_bos=True)[0] for batch in prompts])
                for key, prompts in self.corrupt_prompt.items()
            }

        # [batch, n_correct_tokens]
        self.correct_tokens = torch.stack(
            [model.to_tokens(batch, prepend_bos=False)[:, 0] for batch in self.correct_answers]
        )
        # [batch, n_wrong_tokens]
        self.wrong_tokens = torch.stack(
            [model.to_tokens(batch, prepend_bos=False)[:, 0] for batch in self.wrong_answers]
        )

    def get_prompt(self, index):
        if isinstance(self.corrupt_prompt, list):
            corrupt_prompt = self.corrupt_prompt[index]
        else:
            corrupt_prompt = {k: v[index] for k, v in self.corrupt_prompt.items()}
        return Prompt(
            self.clean_prompt[index],
            corrupt_prompt,
            self.correct_answers[index],
            self.wrong_answers[index],
        )

    def correct_prob(
        self, logits: Float[Tensor, "batch pos d_vocab"], pos: int = -1
    ) -> Float[Tensor, " batch"]:
        pos_logits = logits[:, pos, :]
        pos_log_probs = torch.log_softmax(pos_logits, dim=1)
        correct_log_probs = torch.gather(pos_log_probs, index=self.correct_tokens, dim=1)
        correct_probs = correct_log_probs.exp()
        return correct_probs.mean(dim=1)

    def correct_rank(
        self, logits: Float[Tensor, "batch pos d_vocab"], pos: int = -1
    ) -> Float[Tensor, " batch"]:
        pos_logits = logits[:, pos, :]
        best_correct_logits, _ = torch.gather(pos_logits, index=self.correct_tokens, dim=1).max(
            dim=1, keepdim=True
        )
        return (pos_logits > best_correct_logits).long().sum(dim=1)


def docstring_prompt_templ(
    style: str,
    *,
    met_name: str,
    met_desc_words: list[str],
    def_args: list[str],
    doc_args: list[str],
    doc_args_desc_words: list[list[str]],
    default: bool = False,
) -> str:
    def_args_str = ", ".join(arg + ("=None" if default else "") for arg in def_args)
    met_desc_str = " ".join(met_desc_words)
    ind4 = 4 * " "
    args_line = (
        f"""
{ind4}Args:"""
        if style == "goog"
        else ""
    )
    def_and_desc = f'''def {met_name}(self, {def_args_str}):
{ind4}"""{met_desc_str}
{args_line}'''
    param_prefix = f"{ind4}:param" if style == "rest" else f"{ind4}   "
    doc_args_desc = [" ".join(arg_desc_words) for arg_desc_words in doc_args_desc_words]
    doc_lines = [f"{param_prefix} {arg}: {desc}" for arg, desc in zip(doc_args, doc_args_desc)]
    doc_lines_str = "\n".join(doc_lines)
    return f"""
{def_and_desc}
{doc_lines_str}
{param_prefix}"""


def docstring_prompt_gen(
    style: str,  # rest or goog,
    *,
    n_args: int,  # num of args in method def
    pred_nth_arg: Optional[int] = None,  # arg index to predict, last if None
    met_desc_len: int = 10,  # number of words in first doc line
    arg_desc_len: int = 4,  # number of words in each arg desc
    seed: int = 42,
) -> Prompt:

    random.seed(seed)

    assert style in ["rest", "goog"]
    if pred_nth_arg is None:
        # predict the last arg
        pred_nth_arg = n_args - 1

    met_name, *def_args = random.sample(VARIABLE_NAMES, 1 + n_args)
    clean_doc_args = def_args[:pred_nth_arg]
    met_desc_words = random.sample(COMMON_SINGLE_TOKEN_NOUNS, met_desc_len)
    doc_args_desc_words = [
        random.sample(COMMON_SINGLE_TOKEN_NOUNS, arg_desc_len) for _ in clean_doc_args
    ]

    clean_prompt = docstring_prompt_templ(
        style,
        met_name=met_name,
        met_desc_words=met_desc_words,
        def_args=def_args,
        doc_args=clean_doc_args,
        doc_args_desc_words=doc_args_desc_words,
    )

    correct_answers = [" " + def_args[pred_nth_arg]]
    wrong_answers = [" " + arg for arg in def_args if arg != def_args[pred_nth_arg]]

    random_order_doc_args = list(clean_doc_args)
    while random_order_doc_args == clean_doc_args:
        random.shuffle(random_order_doc_args)
    random_order_prompt = docstring_prompt_templ(
        style,
        met_name=met_name,
        met_desc_words=met_desc_words,
        def_args=def_args,
        doc_args=random_order_doc_args,
        doc_args_desc_words=doc_args_desc_words,
    )

    shift1_doc_args = def_args[pred_nth_arg - 1 : pred_nth_arg] + def_args[: pred_nth_arg - 1]
    shift1_prompt = docstring_prompt_templ(
        style,
        met_name=met_name,
        met_desc_words=met_desc_words,
        def_args=def_args,
        doc_args=shift1_doc_args,
        doc_args_desc_words=doc_args_desc_words,
    )

    rand_def_args = random.sample(VARIABLE_NAMES, n_args)
    while rand_def_args in def_args:
        rand_def_args = random.sample(VARIABLE_NAMES, n_args)
    shift1_doc_random_def = docstring_prompt_templ(
        style,
        met_name=met_name,
        met_desc_words=met_desc_words,
        def_args=rand_def_args,
        doc_args=shift1_doc_args,
        doc_args_desc_words=doc_args_desc_words,
    )

    random_doc_doc_args = random.sample(VARIABLE_NAMES, pred_nth_arg)
    random_doc_prompt = docstring_prompt_templ(
        style,
        met_name=met_name,
        met_desc_words=met_desc_words,
        def_args=def_args,
        doc_args=random_doc_doc_args,
        doc_args_desc_words=doc_args_desc_words,
    )

    random_def_arg = random.choice(VARIABLE_NAMES)
    while random_def_arg in def_args:
        random_def_arg = random.choice(VARIABLE_NAMES)
    swap_random_def_args = def_args[:pred_nth_arg] + [random_def_arg] + def_args[pred_nth_arg + 1 :]
    swap_random_prompt = docstring_prompt_templ(
        style,
        met_name=met_name,
        met_desc_words=met_desc_words,
        def_args=swap_random_def_args,
        doc_args=clean_doc_args,
        doc_args_desc_words=doc_args_desc_words,
    )

    swap_def_random_doc_prompt = docstring_prompt_templ(
        style,
        met_name=met_name,
        met_desc_words=met_desc_words,
        def_args=swap_random_def_args,
        doc_args=random_doc_doc_args,
        doc_args_desc_words=doc_args_desc_words,
    )

    return Prompt(
        clean_prompt=clean_prompt,
        corrupt_prompt={
            "random_order": random_order_prompt,
            "random_doc": random_doc_prompt,
            "swap_random": swap_random_prompt,
            "shift1": shift1_prompt,
            "swap_def_random_doc": swap_def_random_doc_prompt,
            "shift1_doc_random_def": shift1_doc_random_def,
        },
        correct_answers=correct_answers,
        wrong_answers=wrong_answers,
    )


def docstring_induction_prompt_generator(
    style: str,  # rest or goog,
    *,
    n_matching_args: int,
    n_def_prefix_args: int,
    n_def_suffix_args: int,
    n_doc_prefix_args: int,
    met_desc_len: int = 10,  # number of words in first doc line
    arg_desc_len: int = 4,  # number of words in each arg desc,
    default=False,
    seed: int = 42,
) -> Prompt:
    assert style in ["rest", "goog"]

    random.seed(seed)

    n_not_matching_args = n_matching_args - 1
    met_name, *all_args = random.sample(
        VARIABLE_NAMES,
        2
        + n_matching_args
        + n_not_matching_args
        + n_not_matching_args
        + n_matching_args
        + n_def_prefix_args
        + n_def_suffix_args
        + n_doc_prefix_args,
    )
    args_ = list(all_args)
    random_answer, *args_ = args_
    random_random_mid_def_args, args_ = args_[:n_matching_args], args_[n_matching_args:]
    random_random_mid_doc_args, args_ = args_[:n_not_matching_args], args_[n_not_matching_args:]
    not_matching_args, args_ = args_[:n_not_matching_args], args_[n_not_matching_args:]
    matching_args, args_ = args_[:n_matching_args], args_[n_matching_args:]
    def_prefix_args, args_ = args_[:n_def_prefix_args], args_[n_def_prefix_args:]
    def_suffix_args, args_ = args_[:n_def_suffix_args], args_[n_def_suffix_args:]
    doc_prefix_args, args_ = args_[:n_doc_prefix_args], args_[n_doc_prefix_args:]
    assert len(args_) == 0

    clean_def_args = def_prefix_args + matching_args + def_suffix_args
    clean_doc_args = doc_prefix_args + matching_args[:-1]
    met_desc_words = random.sample(COMMON_SINGLE_TOKEN_NOUNS, met_desc_len)
    doc_args_desc_words = [
        random.sample(COMMON_SINGLE_TOKEN_NOUNS, arg_desc_len) for _ in clean_doc_args
    ]

    clean_prompt = docstring_prompt_templ(
        style,
        met_name=met_name,
        met_desc_words=met_desc_words,
        def_args=clean_def_args,
        doc_args=clean_doc_args,
        doc_args_desc_words=doc_args_desc_words,
        default=default,
    )

    correct_answers = [" " + matching_args[-1]]
    wrong_answers = [" " + arg for arg in all_args if arg != matching_args[-1]]

    # corruptions

    random_doc_doc_args = doc_prefix_args + not_matching_args
    random_doc_prompt = docstring_prompt_templ(
        style,
        met_name=met_name,
        met_desc_words=met_desc_words,
        def_args=clean_def_args,
        doc_args=random_doc_doc_args,
        doc_args_desc_words=doc_args_desc_words,
        default=default,
    )

    random_def_def_args = def_prefix_args + not_matching_args + matching_args[-1:] + def_suffix_args
    random_def_prompt = docstring_prompt_templ(
        style,
        met_name=met_name,
        met_desc_words=met_desc_words,
        def_args=random_def_def_args,
        doc_args=clean_doc_args,
        doc_args_desc_words=doc_args_desc_words,
        default=default,
    )

    random_answer_def_args = (
        def_prefix_args + matching_args[:-1] + [random_answer] + def_suffix_args
    )
    random_answer_prompt = docstring_prompt_templ(
        style,
        met_name=met_name,
        met_desc_words=met_desc_words,
        def_args=random_answer_def_args,
        doc_args=clean_doc_args,
        doc_args_desc_words=doc_args_desc_words,
        default=default,
    )

    random_def_doc_prompt = docstring_prompt_templ(
        style,
        met_name=met_name,
        met_desc_words=met_desc_words,
        def_args=random_def_def_args,
        doc_args=random_doc_doc_args,
        doc_args_desc_words=doc_args_desc_words,
        default=default,
    )

    random_answer_doc_prompt = docstring_prompt_templ(
        style,
        met_name=met_name,
        met_desc_words=met_desc_words,
        def_args=random_answer_def_args,
        doc_args=random_doc_doc_args,
        doc_args_desc_words=doc_args_desc_words,
        default=default,
    )

    random_random_def_args = def_prefix_args + random_random_mid_def_args + def_suffix_args
    random_random_doc_args = doc_prefix_args + random_random_mid_doc_args
    random_random_prompt = docstring_prompt_templ(
        style,
        met_name=met_name,
        met_desc_words=met_desc_words,
        def_args=random_random_def_args,
        doc_args=random_random_doc_args,
        doc_args_desc_words=doc_args_desc_words,
        default=default,
    )

    vary_length_doc_desc = [[doc_args_desc_words[i][0]] for i in range(len(clean_doc_args))]
    for i in range(len(clean_doc_args)):
        for j in range(1, len(doc_args_desc_words[i])):
            k = random.randint(0, len(clean_doc_args) - 1)
            vary_length_doc_desc[k].append(doc_args_desc_words[i][j])

    vary_length_doc_desc_prompt = docstring_prompt_templ(
        style,
        met_name=met_name,
        met_desc_words=met_desc_words,
        def_args=clean_def_args,
        doc_args=clean_doc_args,
        doc_args_desc_words=vary_length_doc_desc,
        default=default,
    )

    vary_length_doc_desc_random_doc_prompt = docstring_prompt_templ(
        style,
        met_name=met_name,
        met_desc_words=met_desc_words,
        def_args=clean_def_args,
        doc_args=random_doc_doc_args,
        doc_args_desc_words=vary_length_doc_desc,
        default=default,
    )

    return Prompt(
        clean_prompt=clean_prompt,
        corrupt_prompt={
            "random_doc": random_doc_prompt,
            "random_def": random_def_prompt,
            "random_answe": random_answer_prompt,
            "random_def_doc": random_def_doc_prompt,
            "random_answer_doc": random_answer_doc_prompt,
            "random_random": random_random_prompt,
            "vary_length_doc_desc": vary_length_doc_desc_prompt,
            "vary_length_doc_desc_random_doc": vary_length_doc_desc_random_doc_prompt,
        },
        correct_answers=correct_answers,
        wrong_answers=wrong_answers,
    )
