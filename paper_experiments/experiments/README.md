# Experiments
The configs for the tasks are under `task_configs/`, and likewise the test configs are under `test_configs/`. We used a seed of 1 for all experiments. The default config values are the ones used in the paper, with the exception of `minimality.n_random_circuits`, which we increase to 10,000 for IOI and GT, and use 1,000 for all other tasks.

To launch a run of a single hypothesis test for a given task, run

```
python3 main.py --task_name [task] --test_name [test] --seed [seed] --save_dir [path to save results] --device {cuda,cpu} [--verbose]
```

To launch runs of all (test x task) combinations on a job scheduler such as slurm, you can run

```
bash run_all.sh
```

To obtain the data to generate Figure 2 in the paper, you can use the `run_sweep_job.sh` script.
