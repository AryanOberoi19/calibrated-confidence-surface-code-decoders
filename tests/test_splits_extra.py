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


def test_confirmatory_split_is_a_seeded_40_30_30_permutation():
    from qeccal.data.splits_extra import confirmatory_indices
    a = confirmatory_indices("d3_at_q4_9/Z/r13", 50000)
    b = confirmatory_indices("d3_at_q4_9/Z/r13", 50000)
    c = confirmatory_indices("d3_at_q4_9/X/r13", 50000)
    assert {k: v.size for k, v in a.items()} == {"train": 20000, "calibrate": 15000, "test": 15000}
    assert all(np.array_equal(a[k], b[k]) for k in a)                     # reproducible
    assert not np.array_equal(a["test"], c["test"])                      # per-experiment seed
    assert np.array_equal(np.sort(np.concatenate(list(a.values()))), np.arange(50000))
    assert all(np.all(np.diff(v) > 0) for v in a.values())               # sorted, like split_indices
    assert not np.array_equal(a["test"], np.arange(35000, 50000))        # not blocked
