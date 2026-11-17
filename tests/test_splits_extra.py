"""Extra splits for the arXiv v1 robustness work (plan A3, A7)."""
import numpy as np

from qeccal.data.splits_extra import blocked_indices


def test_blocked_split_is_contiguous_in_file_order():
    idx = blocked_indices(50000)
    assert {k: v.size for k, v in idx.items()} == {"train": 20000, "calibrate": 15000, "test": 15000}
    for v in idx.values():
        assert np.array_equal(v, np.arange(v[0], v[-1] + 1))
    assert idx["train"][-1] < idx["calibrate"][0] and idx["calibrate"][-1] < idx["test"][0]
    assert np.array_equal(np.concatenate([idx["train"], idx["calibrate"], idx["test"]]), np.arange(50000))
