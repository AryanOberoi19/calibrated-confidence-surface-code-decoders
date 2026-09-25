"""A small recurrent decoder (spec section 8.3, M3): AlphaQubit's recipe at distance 3 (pre-train on simulated
shots, fine-tune on hardware), not its architecture.

Per time step the input is the step's detection events, two flags (first step, last step) and learned
embeddings of the patch and the basis. A two-layer GRU reads the steps in order; a linear readout of its last
state gives the logit of P(observable flipped). The prediction is logit > 0; its confidence is |logit|.
"""
from __future__ import annotations

import torch
from torch import nn


def device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GRUDecoder(nn.Module):
    def __init__(self, slots: int, patches: list[str], hidden: int = 128, layers: int = 2, embed: int = 16):
        super().__init__()
        self.config = {"slots": slots, "patches": list(patches), "hidden": hidden, "layers": layers, "embed": embed}
        self.patch_index = {p: i for i, p in enumerate(patches)}
        self.patch = nn.Embedding(len(patches), embed)
        self.basis = nn.Embedding(2, embed)
        self.inp = nn.Sequential(nn.Linear(slots + 2 + 2 * embed, hidden), nn.ReLU())
        self.gru = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True)
        self.out = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor, patch: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
        """x: (batch, steps, slots) in {0, 1}; patch, basis: (batch,) long. Returns logits (batch,)."""
        b, steps, _ = x.shape
        flags = torch.zeros(steps, 2, device=x.device, dtype=x.dtype)
        flags[0, 0] = 1
        flags[-1, 1] = 1
        ctx = torch.cat([self.patch(patch), self.basis(basis)], dim=1)
        z = torch.cat([x, flags.expand(b, -1, -1), ctx[:, None, :].expand(-1, steps, -1)], dim=2)
        h, _ = self.gru(self.inp(z))
        return self.out(h[:, -1]).squeeze(1)

    @classmethod
    def load(cls, path, map_location="cpu") -> "GRUDecoder":
        ck = torch.load(path, map_location=map_location, weights_only=False)
        m = cls(**ck["config"])
        m.load_state_dict(ck["state"])
        return m

    def save(self, path, **extra):
        torch.save({"config": self.config, "state": self.state_dict(), **extra}, path)
