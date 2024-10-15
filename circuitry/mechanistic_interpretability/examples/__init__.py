from circuitry.mechanistic_interpretability import BaseTask
from circuitry.mechanistic_interpretability.examples.docstring.task import DocstringTask
from circuitry.mechanistic_interpretability.examples.greater_than import GreaterThanTask
from circuitry.mechanistic_interpretability.examples.induction import InductionTask
from circuitry.mechanistic_interpretability.examples.ioi import IOITask
from circuitry.mechanistic_interpretability.examples.tracr import (
    TracrProportionTask,
    TracrReverseTask,
)

_tasks = {
    "induction": InductionTask,
    "greater_than": GreaterThanTask,
    "tracr_reverse": TracrReverseTask,
    "tracr_proportion": TracrProportionTask,
    "docstring": DocstringTask,
    "ioi": IOITask,
}


def available_tasks():
    return list(_tasks.keys())


def get_task_cls(name: str) -> type[BaseTask]:
    if name not in _tasks:
        raise ValueError(f"Task {name} not found. Available tasks: {available_tasks()}")
    return _tasks[name]


__all__ = [
    "available_tasks",
    "get_task_cls",
    "InductionTask",
    "GreaterThanTask",
    "TracrProportionTask",
    "TracrReverseTask",
    "DocstringTask",
    "IOITask",
]
