# %%
import copy
import os.path
import pickle
from typing import Optional

import numpy as np
import pandas as pd
import seaborn as sns
import torch
import tqdm
import yaml
from matplotlib import pyplot as plt
from scipy.stats import binomtest


# %%
def gather_results(
    path: str,
    task_names: list[str],
    test_name: str,
    dumped_fields: list[str],
    test_config_fields: Optional[list[str]] = None,
):
    """
    Gather results from the experiments in the path.

    Parameters
    ----------
    path : str
        The path to the results.
    task_names : list[str]
        The names of the tasks. These will be the subdirectories in the path.
    test_name : str
        The name of the test.
    dumped_fields : list[str]
        The fields to extract from the dumped results.
    test_config_fields : Optional[list[str]], optional
        The fields to extract from the test configuration. The default is None.

    Returns
    -------
    pd.DataFrame
        The results in a DataFrame.

    Examples
    --------
    >>> path = "path/to/results"
    >>> task_names = ["docstring", "induction"]
    >>> test_name = "sufficiency"
    >>> dumped_fields = ["empirical_quantile", "candidate_circuit_size", "complete_circuit_size"]
    >>> test_config_fields = ["size_random_circuits"]
    >>> results = gather_results(path, task_names, test_name, dumped_fields, test_config_fields)

    In this example, the folders in the path should be organized as follows:
    path
    ├── docstring
    │   ├── sufficiency
    │   │   ├── my_run (or any other name)
    │   │   │   ├── docstring_sufficiency.pkl
    │   │   │   ├── sufficiency.yaml
    │   │   ├── another_run
    │   │   ...
    ├── induction
    │   ...
    """
    if test_config_fields is None:
        test_config_fields = []
    results = []
    for task_name in task_names:
        sufficiency_path = os.path.join(path, task_name, test_name)
        run_dirs = [os.path.join(sufficiency_path, d) for d in sorted(os.listdir(sufficiency_path))]
        for run_dir in run_dirs:
            with open(os.path.join(run_dir, f"{task_name}_{test_name}.pkl"), "rb") as f:
                dumped = pickle.load(f)  # noqa S301
            with open(os.path.join(run_dir, f"{test_name}.yaml")) as f:
                config = yaml.safe_load(f)

            run_results = {"task_name": dumped["task_name"]}
            for field in dumped_fields:
                run_results[field] = dumped[field]
            for field in test_config_fields:
                run_results[field] = config[field]
            results.append(run_results)

    return pd.DataFrame(results)


# %%


def plot_sufficiency(
    sufficiency_results,
    task_name_labels=None,
    palette=None,
    figsize=(4, 3),
):
    if palette is None:
        palette = sns.color_palette("tab10", n_colors=len(task_name_labels))

    fig, axes = plt.subplots(1, 1, figsize=figsize)
    ax = axes
    for i, t in enumerate(task_name_labels.keys()):
        d = sufficiency_results[sufficiency_results["task_name"] == t].copy()
        d["random_circuit_size"] = d["size_random_circuits"]
        d = d.sort_values(by="random_circuit_size")
        ax.plot(
            d["random_circuit_size"],
            d["empirical_quantile"],
            c=palette[i],
            label=task_name_labels[t],
            lw=1 if t == "GreaterThanTask" else 3 if t == "IoITask" else 2,
            marker="o",
            markersize=5,
        )
        print(d)
        ax.axvline(
            d["candidate_circuit_size"].iloc[0] / d["complete_circuit_size"].iloc[0],
            0,
            1,
            c=palette[i],
            ls="--",
        )
    ax.set_ylabel(
        r"$\mathrm{\mathbb{P}}(C^*$ more faithful than $ C^r)$",
    )
    ax.set_xlabel("Size $|C^r|$ as a fraction of $|M|$")

    ax.legend(
        fontsize=9,
        ncols=3,
        bbox_to_anchor=(0.45, 1.3),
        loc="upper center",
        labelspacing=0.1,
        borderpad=0.4,
        fancybox=False,
    )

    sns.despine()
    fig.tight_layout()


def plot_minimality(
    minimality_results,
    task_name_labels=None,
    palette=None,
):
    CANDIDATE = "candidate_edge_knockout_metrics"
    RANDOM = "random_inflated_knockout_metrics"
    minimality_results = minimality_results.set_index("task_name")
    if palette is None:
        palette = sns.color_palette("tab10", n_colors=len(task_name_labels))

    fig, axes = plt.subplots(3, 6, figsize=(5, 5), width_ratios=[3, 0.7, 1, 3, 0.7, 0.0])

    for i, task in enumerate(task_name_labels):
        task_name = task_name_labels[task]
        effects = minimality_results.loc[task].copy()

        n_edges = len(effects[CANDIDATE])
        n_random = len(effects[RANDOM])

        # recompute empirical quantiles and p-values
        effects["empirical_quantiles"] = (
            effects[RANDOM][:, None] < effects[CANDIDATE][None, :]
        ).sum(axis=0) / len(effects[RANDOM])

        edge_idx = np.argsort(effects[CANDIDATE])
        effects["empirical_quantiles"] = effects["empirical_quantiles"][edge_idx]
        effects[CANDIDATE] = effects[CANDIDATE][edge_idx]

        effects["pvalues"] = np.array(
            [
                binomtest(int(q * n_random), n=n_random, p=0.9, alternative="less").pvalue
                for q in effects["empirical_quantiles"]
            ]
        )
        corrected_significance = 0.05 / len(effects["pvalues"])
        reject = effects["pvalues"] < corrected_significance
        threshold = np.sum(reject)

        ax = axes.ravel()[i * 3]
        ax_hist = axes.ravel()[i * 3 + 1]
        axes.ravel()[i * 3 + 2].set_axis_off()
        # plot the change in score of each edge (ordered)
        ax.plot(
            np.arange(n_edges) + 1,
            #         effects["empirical_quantiles"],
            effects[CANDIDATE],
            label=task_name,
            lw=1.5,
            c=palette[i],
            markersize=2.5,
            marker="o",
        )

        ax.set_ylabel("")
        tmp = ax.axvline(max(0, threshold), 0, 1, color="blue", linestyle="dashed", lw=1)

        if i == 0:
            legend_line = ax.legend(
                [tmp],
                ["Threshold under which individual edges are not significant"],
                bbox_to_anchor=(1.25, 1.5),
                loc="upper center",
                handlelength=1,
                fontsize=10,
            )

        ax.set_xlabel("")

        if task_name in ["IOI", "G-T"]:
            ax.set_yscale("log")
            # take min of non-zero values
            y_min = min(
                effects[RANDOM][effects[RANDOM] > 0].min(),
                effects[CANDIDATE][effects[CANDIDATE] > 0].min(),
            )
            y_max = max(
                effects[RANDOM].max(),
                effects[CANDIDATE].max(),
            )
            tmp_hist_data = np.log10(effects[RANDOM])
            y_minh = np.log10(y_min)
            y_maxh = np.log10(y_max)

        else:
            y_min = min(
                effects[RANDOM].min(),
                effects[CANDIDATE].min(),
            )
            y_max = max(
                effects[RANDOM].max(),
                effects[CANDIDATE].max(),
            )
            tmp_hist_data = effects[RANDOM]
            y_minh = y_min
            y_maxh = y_max
            y_min = y_min - y_max * 0.04
            y_max = y_max * 1.04

        ax.fill_between([0, threshold], y_min, y_max, alpha=0.2, color=palette[2])

        ax_hist.hist(
            tmp_hist_data,
            orientation="horizontal",
            color=palette[i],
            bins=np.linspace(y_minh, y_maxh, 20),
        )
        ax_hist.set_axis_off()
        ax.set_xticks([1, (n_edges) // 2, n_edges])
        ax.set_ylim(y_min, y_max)
        print(task_name, y_min, y_max, y_minh, y_maxh)
        ax_hist.set_ylim(y_minh, y_maxh)

        leg_loc = "best"
        ax.legend(fancybox=False, loc=leg_loc, handlelength=1, fontsize=9)

    axes[2, 0].set_xlabel(
        (" " * 40) + "Edge $e$ in the circuit (sorted by their y-axis value)", fontsize=11
    )
    axes[1, 0].set_ylabel(
        r"Change in score, $\delta(e, C^*)$, when removing $e$ from $C^*$ ",
        labelpad=10,
        fontsize=11,
    )

    axes[0, 0].add_artist(legend_line)
    fig.suptitle("    ")
    plt.subplots_adjust(wspace=0.015, hspace=0.3)


def _get_task(name):
    from circuitry.mechanistic_interpretability.examples import (
        DocstringTask,
        GreaterThanTask,
        InductionTask,
        IOITask,
        TracrProportionTask,
        TracrReverseTask,
    )

    if name == "InductionTask":
        return InductionTask(device="cpu")
    elif name == "DocstringTask":
        return DocstringTask(device="mps")
    elif name == "IOITask":
        return IOITask(device="mps")
    elif name == "GreaterThanTask":
        return GreaterThanTask(device="mps")
    elif name == "TracrProportionTask":
        return TracrProportionTask(device="cpu")
    elif name == "TracrReverseTask":
        return TracrReverseTask(device="cpu")


def _compute_superset_probability(n_samples=100):
    import gc

    import tqdm

    task_names = [
        "InductionTask",
        "IOITask",
        "GreaterThanTask",
        "DocstringTask",
    ]

    res_tasks = []
    for task_name in task_names:
        if task_name in ["IOITask", "GreaterThanTask"]:
            # we know the results for these tasks, and it is extremly slow to compute them
            res = pd.DataFrame({"p": np.linspace(0, 0.95, 21), "prob": 0, "task": task_name})
            res_tasks.append(res)
            continue
        task = _get_task(task_name)
        complete_circuit = task.complete_circuit
        canonical_circuit = task.canonical_circuit

        complete_circuit_size = len(complete_circuit)

        d = {}

        for p in tqdm.tqdm(np.linspace(0, 0.95, 21)):
            circuit_size = complete_circuit_size * p
            counts = 0
            for _ in range(n_samples):
                random_circuit = complete_circuit.sample_circuit(circuit_size)
                if canonical_circuit.is_subset(random_circuit):
                    counts += 1
            d[p] = counts / n_samples

        res = pd.DataFrame(d.items(), columns=["p", "prob"])
        print(res)
        res["task"] = task_name
        res_tasks.append(res)

        # make sure to run garbage collection
        del task
        torch.cuda.empty_cache()
        gc.collect()

    res_tasks.append(
        pd.DataFrame(
            {"p": [1] * len(task_names), "prob": [1] * len(task_names), "task": task_names}
        )
    )
    # add a point at (0.99, 0) for IOITask and GreaterThanTask
    res_tasks.append(
        pd.DataFrame({"p": [0.999] * 2, "prob": [0] * 2, "task": ["IOITask", "GreaterThanTask"]})
    )

    res_tasks = pd.concat(res_tasks)
    return res_tasks


def plot_circuit_superset_probability(
    superset_probabilities, task_name_labels=None, palette=None, figsize=(4, 3)
):
    fig, axes = plt.subplots(1, 1, figsize=figsize)
    ax = axes

    for t in task_name_labels.keys():
        tmp_data = superset_probabilities[superset_probabilities["task"] == t]
        if len(tmp_data) == 0:
            continue
        tmp_data.sort_values("p", inplace=True)
        ax.plot(
            tmp_data["p"],
            tmp_data["prob"],
            label=task_name_labels[t],
            marker="o",
            markersize=5 if "reater" not in t else 3,
            lw=2 if "reater" not in t else 0.75,
            zorder=1 if "reater" not in t else 2,
            c=palette[t],
        )
    ax.set_ylabel(
        r"$\mathrm{\mathbb{P}}(C^* \subset C^r)$",
    )
    ax.set_xlabel("Size $|C^r|$ as a fraction of $|M|$")

    ax.legend(
        fontsize=9,
        ncols=2,
        bbox_to_anchor=(0.45, 1.3),
        loc="upper center",
        labelspacing=0.1,
        borderpad=0.4,
        fancybox=False,
    )

    sns.despine()

    plt.savefig("prob_superset.pdf", bbox_inches="tight", pad_inches=0.1)


def plot_minimality_ablation(
    minimality_results: pd.DataFrame,
    palette=None,
):
    for _, row in minimality_results.iterrows():
        task_name = row["task_name"]
        candidate_edge_knockout_metrics = row["candidate_edge_knockout_metrics"]
        edges_idx = np.argsort(candidate_edge_knockout_metrics)

        task = _get_task(task_name)
        candidate_circuit = task.canonical_circuit
        assert len(candidate_circuit) == len(candidate_edge_knockout_metrics)

        complete_circuit = task.complete_circuit
        complete_circuit_score, _ = task.score_and_logits(complete_circuit)

        canonical_circuit = task.canonical_circuit
        canonical_score, _ = task.score_and_logits(canonical_circuit)
        canonical_faithfulness = task.eval_metric(canonical_score, complete_circuit_score)

        empty_circuit = task.complete_circuit.sample_circuit(0)
        empty_score, _ = task.score_and_logits(empty_circuit)
        empty_faithfulness = task.eval_metric(empty_score, complete_circuit_score)

        edges = [edge for edge in candidate_circuit.get_present_edges() if not edge.is_placeholder]
        faithfulnesses = []
        for i in tqdm.tqdm(range(len(edges) + 1)):
            sub_circuit = copy.deepcopy(task.canonical_circuit)
            for j in range(i):
                edge = edges[edges_idx[j]]
                sub_circuit.remove_edge(edge.src_node, edge.dst_node, in_place=True)
            sub_circuit_score, _ = task.score_and_logits(sub_circuit)
            sub_circuit_faithfulness = task.eval_metric(sub_circuit_score, complete_circuit_score)
            faithfulnesses.append(sub_circuit_faithfulness.item())

        plt.figure(figsize=(3, 2), dpi=200)
        plt.plot(
            faithfulnesses,
            # label=task_name,
            marker="o",
            lw=2,
            markersize=3,
            c=palette[task_name],
        )
        plt.axhline(
            canonical_faithfulness.item(), ls=":", color="#444444"
        )  # , label="Canonical circuit")
        plt.axhline(empty_faithfulness.item(), color="#444444")  # , label="empty circuit")
        plt.title(task_name)
        plt.xlabel("Number of edges removed")  # , from\nleast minimal to most minimal")
        plt.ylabel("Faithfulness")
        # add 5 ticks total, make sure they are integers
        steps = len(faithfulnesses) // 3
        plt.xticks(
            [
                0,
                steps,
                2 * steps,
                3 * steps,
            ],
        )
        plt.tight_layout()
        sns.despine()
        plt.savefig(f"minimality_{task_name}.pdf", bbox_inches="tight")
        plt.show()

    return faithfulnesses


# %%

if __name__ == "__main__":
    task_dirs = [
        "docstring",
        "induction",
        "ioi",
        "tracr-proportion",
        "tracr-reverse",
        "greater-than",
    ]
    task_name_labels = {
        "TracrProportionTask": r"Tracr-P",
        "TracrReverseTask": "Tracr-R",
        "InductionTask": "Induction",
        "IOITask": "IOI",
        "GreaterThanTask": "G-T",
        "DocstringTask": "DS",
    }
    palette_mapping = {
        "TracrProportionTask": "#34495e",
        "TracrReverseTask": "#2ecc71",
        "InductionTask": "#3498db",
        "IOITask": "#95a5a6",
        "GreaterThanTask": "#e74c3c",
        "DocstringTask": "#9b59b6",
    }
    palette = sns.color_palette(palette_mapping.values())
    sns.set_palette(palette)

    path = "../hypo-interp-arxiv/"
    # path = "./paper_experiments/hypo-interp-arxiv"
    sufficiency_results = gather_results(
        path,
        task_dirs,
        "sufficiency",
        ["empirical_quantile", "candidate_circuit_size", "complete_circuit_size"],
        test_config_fields=["size_random_circuits"],
    )

    plot_sufficiency(
        sufficiency_results,
        task_name_labels=task_name_labels,
        palette=palette,
        figsize=(4, 3),
    )
    plt.savefig("sufficiency.pdf", bbox_inches="tight")

    minimality_results = gather_results(
        path,
        task_dirs,
        "minimality",
        ["task_name", "candidate_edge_knockout_metrics", "random_inflated_knockout_metrics"],
    )

    plot_minimality(
        minimality_results,
        task_name_labels=task_name_labels,
        palette=palette,
    )
    plt.savefig("minimality.pdf", bbox_inches="tight")

    tmp = _compute_superset_probability(n_samples=500)

    plot_circuit_superset_probability(tmp, task_name_labels, palette_mapping, figsize=(4, 2.2))
    plt.savefig("superset.pdf", bbox_inches="tight")

    plot_minimality_ablation(minimality_results, palette_mapping)
