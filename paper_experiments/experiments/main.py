import argparse
import pickle
from pathlib import Path

import yaml
from circuitry.hypothesis_testing import (
    minimality_test,
    non_equivalence_test,
    non_independence_test,
    partial_necessity_test,
    sufficiency_test,
)
from circuitry.mechanistic_interpretability.examples import (
    DocstringTask,
    GreaterThanTask,
    InductionTask,
    IOITask,
    TracrProportionTask,
    TracrReverseTask,
)
from circuitry.utils import seed_everything

task_name_to_task = {
    "induction": InductionTask,
    "tracr-proportion": TracrProportionTask,
    "tracr-reverse": TracrReverseTask,
    "greater-than": GreaterThanTask,
    "ioi": IOITask,
    "docstring": DocstringTask,
}

test_name_to_test = {
    "sufficiency": sufficiency_test,
    "minimality": minimality_test,
    "partial-necessity": partial_necessity_test,
    "non-equivalence": non_equivalence_test,
    "non-independence": non_independence_test,
}


def int_or_float(value):
    try:
        return int(value) if "." not in value else float(value)
    except ValueError as err:
        raise argparse.ArgumentTypeError(
            f"Invalid value: {value}. Must be an int or a float."
        ) from err


def run_task_test(
    task_name, test_name, task_config, test_config, seed, save_dir, device, verbose, sweep_str=""
):
    print(f"Task: {task_name}")
    print(yaml.dump(task_config, default_flow_style=True))

    print(f"Test: {test_name}")
    print(yaml.dump(test_config, default_flow_style=True))
    print(f"Test seed: {seed}")

    task = task_name_to_task[task_name](**task_config, device=device)
    if seed is not None:
        seed_everything(seed)
    hypo_test = test_name_to_test[test_name]
    results = hypo_test(task, **test_config, verbose=verbose)

    if save_dir is not None:
        print(f"Saving configs and results to {save_dir}")
        task_save_dir = save_dir / task_name

        if not task_save_dir.exists():
            task_save_dir.mkdir(parents=True)

        task_test_save_dir = task_save_dir / test_name

        if not task_test_save_dir.exists():
            task_test_save_dir.mkdir(parents=True)

        n_run_dirs = len(
            [d for d in task_test_save_dir.iterdir() if d.is_dir() and d.name.startswith("run")]
        )

        task_test_run_save_dir = task_test_save_dir / f"run_{n_run_dirs:03}{sweep_str}"
        task_test_run_save_dir.mkdir(parents=True)

        with open(task_test_run_save_dir / f"{task_name}.yaml", "w") as f:
            yaml.dump(task_config, f)

        with open(task_test_run_save_dir / f"{test_name}.yaml", "w") as f:
            yaml.dump(test_config, f)

        with open(task_test_run_save_dir / f"{task_name}_{test_name}.pkl", "wb") as f:
            pickle.dump(results, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str)
    parser.add_argument("--test_name", type=str)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save_dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--sweep",
        action="store_true",
        default=False,
        help="Whether to sweep over a parameter.",
    )
    parser.add_argument(
        "--param_name",
        type=str,
        default=None,
        help="Name of parameter to sweep, prepended with 'task' or 'test', e.g., test.size_random_circuits.",
    )
    parser.add_argument(
        "--sweep_values",
        nargs="+",
        type=int_or_float,
        default=None,
        help="Values to sweep (type is int or float).",
    )
    args = parser.parse_args()

    print("*" * 10)
    print(args)
    print("*" * 10)

    if args.task_name not in task_name_to_task:
        raise ValueError(f"Task {args.task_name} not recognized.")

    if args.test_name not in test_name_to_test:
        raise ValueError(f"Test {args.test_name} not recognized.")

    with open(f"task_configs/{args.task_name}.yaml") as f:
        task_config = yaml.safe_load(f)

    with open(f"test_configs/{args.test_name}.yaml") as f:
        test_config = yaml.safe_load(f)

    if args.sweep:
        if args.param_name is None:
            raise ValueError("Must specify param_name when sweeping.")
        if not args.param_name.startswith("task.") and not args.param_name.startswith("test."):
            raise ValueError("param_name must start with 'task.' or 'test.'.")
        if args.sweep_values is None:
            raise ValueError("Must specify sweep_values when sweeping.")

        for sweep_value in args.sweep_values:
            param_name = args.param_name.split(".")[1]
            if args.param_name.startswith("task."):
                task_config[param_name] = sweep_value
            elif args.param_name.startswith("test."):
                test_config[param_name] = sweep_value

            sweep_str = f"_{param_name}_{sweep_value}"

            print("=" * 10)
            run_task_test(
                args.task_name,
                args.test_name,
                task_config,
                test_config,
                args.seed,
                args.save_dir,
                args.device,
                args.verbose,
                sweep_str,
            )

    else:
        run_task_test(
            args.task_name,
            args.test_name,
            task_config,
            test_config,
            args.seed,
            args.save_dir,
            args.device,
            args.verbose,
        )


if __name__ == "__main__":
    main()
