"""Learned decoder plumbing: the step layout keeps every detection event, the model runs and round-trips."""
import numpy as np
import pytest
import stim

torch = pytest.importorskip("torch")

from qeccal.learned.features import Layout  # noqa: E402
from qeccal.learned.model import GRUDecoder  # noqa: E402


def circuit(r):
    return stim.Circuit.generated("surface_code:rotated_memory_z", distance=3, rounds=r,
                                  after_clifford_depolarization=0.01, before_measure_flip_probability=0.01)


def test_layout_keeps_every_event():
    c = circuit(5)
    lay = Layout(c)
    assert (lay.steps, lay.slots) == (6, 8)
    det = c.compile_detector_sampler(seed=1).sample(500)
    seq = lay.sequence(det)
    assert seq.shape == (500, 6, 8)
    assert np.array_equal(seq.reshape(500, -1)[:, lay.position], det)
    assert seq.sum() == det.sum()
    packed = np.packbits(det, axis=1, bitorder="little")
    assert np.array_equal(lay.sequence(packed, packed=True), seq)
    one = Layout(circuit(1), slots=8)                          # r = 1 has 4 detectors per step; pad to 8
    assert one.slots == 8 and one.steps == 2
    with pytest.raises(ValueError):
        Layout(c, slots=4)


def test_model_learns_a_little_and_round_trips(tmp_path):
    torch.manual_seed(0)
    c = circuit(3)
    lay = Layout(c)
    det, obs = c.compile_detector_sampler(seed=2).sample(4096, separate_observables=True)
    x = torch.from_numpy(lay.sequence(det)).float()
    y = torch.from_numpy(obs[:, 0]).float()
    m = GRUDecoder(lay.slots, ["q0", "q1"], hidden=32)
    ids = (torch.zeros(len(y), dtype=torch.long), torch.ones(len(y), dtype=torch.long))
    assert m(x, *ids).shape == (len(y),)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    first = loss_fn(m(x, *ids), y).item()
    for _ in range(60):
        opt.zero_grad()
        loss = loss_fn(m(x, *ids), y)
        loss.backward()
        opt.step()
    assert loss_fn(m(x, *ids), y).item() < first
    m.save(tmp_path / "m.pt")
    m2 = GRUDecoder.load(tmp_path / "m.pt")
    with torch.no_grad():
        assert torch.allclose(m(x[:10], ids[0][:10], ids[1][:10]), m2(x[:10], ids[0][:10], ids[1][:10]))
