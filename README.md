# Circuitry

Circuitry is a package that implements the tests proposed in the 2024 NeurIPS paper Hypothesis Testing The Circuit Hypothesis in LLMs.
These tests aim to check:

1. Mechanism Preservation: The performance of an idealized circuit should match that of the original model.
2. Mechanism Localization: Removing the circuit should eliminate the model’s ability to perform the associated task.
3. Minimality: A circuit should not contain any redundant edges.

For a thorough description of each test please consult the paper.


# Installation

For installing the repo please clone the project and inside of the main directory run:

```
cd circuitry
pip3 install -e .
```

# Experiment Replication
To replicate a particular experiment in the paper please run:
```
python3 main.py --task_name [task] --test_name [test] --seed 1 --save_dir [path to save results] --device {cuda,cpu} [--verbose]
```
For convenience we have also provided a slurm script in `./paper_experiments/run_all.sh` that
runs all of the tests on all of the circuits of the paper.
For more details please refer to the `./paper_experiments` folder.

## Organisation of the repo
The repo is roughly organised as follows:
```
root
├─ circuitry
│  ├─ circuit
│  ├─ mechanistic_interpretability
│  │  └─ examples
│  └─ hypothesis_testing
├─ paper_experiments
│  ├─ experiments
│  └─ figures
│     └─ pdf
└─ tests
```

- `circuitry` contains the main package.
- `circuitry/circuit` defines and manages the circuits abstranctions.
- `circuitry/mechanistic_interpretability` defines an interpretability task.
- `circuitry/mechanistic_interpretability/examples` contains examples of tasks: Induction, Docstring, IoI, ...
- `circuitry/hypothesis_testing` defines our hypothesis tests.
- `paper_experiments` contains the code to run the experiments in the paper.
- `paper_experiments/experiments` contains the scripts and code to run the experiments in the paper.
- `paper_experiments/figures` contains the code to generate and reproduce the figures in the paper.
- `paper_experiments/figures/pdf` contains the pdf figures of the paper.
- `tests` contains unit tests for the code.


### Citation (APA)

Shi, C., Beltran-Velez, N., Nazaret, A., Zheng, C., Garriga-Alonso, A., Jesson, A., Makar, M., & Blei, D. (2024). Hypothesis testing the circuit hypothesis in LLMs. In *NeurIPS 2024*.

# BibTex
Please use the following BibTex entry when citing our work.
```
@inproceedings{
shi2024hypothesis,
title={Hypothesis Testing the Circuit Hypothesis in {LLM}s},
author={Claudia Shi and Nicolas Beltran-Velez and Achille Nazaret and Carolina Zheng and Adri{\`a} Garriga-Alonso and Andrew Jesson and Maggie Makar and David Blei},
booktitle={NeurIPS},
year={2024},
}
```
