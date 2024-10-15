import pytest
from circuitry.mechanistic_interpretability.examples.induction import (
    InductionTask,
    get_mask_repeat_candidates,
    get_validation_data,
)
from tests.utils import IN_GITHUB_ACTIONS, REASON_LOAD_BIG_DATA, REASON_LOAD_BIG_MODEL


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason=REASON_LOAD_BIG_DATA)
def test_data():
    validation_data_orig = get_validation_data(device="cpu")
    mask_orig = get_mask_repeat_candidates(device="cpu")
    assert validation_data_orig.shape == mask_orig.shape


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason=REASON_LOAD_BIG_MODEL)
def test_load_induction():
    InductionTask()
