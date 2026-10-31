"""Early abort as optimal stopping, learned from logged shots (spec section 8.8).

A shot of an r-round run passes checkpoints t_1 < ... < t_K < r. At checkpoint k the state is (k, b), with b the
bin of the prefix gap; the actions are abort (the shot ends, reward 0) and continue (reward -rho for each round
until the next checkpoint). A shot that runs to the end is kept if its final gap is in bin j or above (reward 1),
and discarded otherwise (reward 0). j is the keep threshold, the same kind of rule as the static thresholds of
qeccal.guarantees.

Shots are attempted over and over, so the quantity to maximise is kept shots per round of hardware time,
E[kept] / E[rounds] (a renewal-reward, or average-reward, problem). Dinkelbach's method finds it: solve the MDP with
a per-round cost rho, set rho to the rate of the policy found, and repeat until rho stops changing. At the fixed
point the policy maximises E[kept] - rho E[rounds] = 0, which is the same as maximising the rate. Whether the kept
shots are right is not in the reward: Learn-then-Test controls that, by choosing j (certify_family).

Aborting does not change what the qubits do, so every logged shot shows what would have followed had the policy
continued. The MDP is therefore learned offline from Train shots, and any policy is evaluated exactly on other
shots, with no simulator and no off-policy correction.

Solver: tabular Q-iteration on the logged transitions (batch, or fitted, Q-iteration). A sweep replaces
Q(s, continue) by the mean, over Train shots in state s, of  -rho * dt + max(0, V(s')),  where s' is that shot's
next state. The horizon is finite and layered, so one backward sweep is exact. q_learning_online reaches the same
table with sample-by-sample Q-learning updates and a target table refreshed every epoch.

Each Q-table is constrained to be non-decreasing in the gap bin (weighted isotonic regression of the per-bin
means): a larger prefix gap never makes continuing less worthwhile. Without it, sampling noise in sparse bins (the
MWPM gap has many ties in long distance-3 runs) would make abort sets non-contiguous.

The keep threshold j indexes a family of policies, from strict (high j) to loose; the family is ordered by Train
keep fraction and Learn-then-Test picks one on Calibrate, as qeccal.guarantees picks a static threshold.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from ..guarantees.ltt import cp_lower, p_value

@dataclasses.dataclass
class Binner:
    """Bin edges per column from Train quantiles (tied quantiles merged). The bin index grows with the gap."""
    edges: list

    @classmethod
    def fit(cls, X, nbins) -> "Binner":
        X = np.asarray(X, np.float64)
        nb = [int(nbins)] * X.shape[1] if np.isscalar(nbins) else [int(b) for b in nbins]
        return cls([np.unique(np.quantile(X[:, j], np.linspace(0, 1, nb[j] + 1)[1:-1])) for j in range(X.shape[1])])

    @property
    def sizes(self) -> list[int]:
        return [e.size + 1 for e in self.edges]

    def transform(self, X) -> np.ndarray:
        X = np.asarray(X, np.float64)
        return np.stack([np.searchsorted(e, X[:, j], side="right") for j, e in enumerate(self.edges)],
                        axis=1).astype(np.int64)


@dataclasses.dataclass
class Policy:
    rounds: np.ndarray          # (K+1,): checkpoint rounds t_1..t_K, then the run length r
    q_cont: list                # K arrays over bins: Q(k, b, continue); abort is worth 0
    keep_from: int              # a shot that runs to the end is kept if its final bin is >= keep_from
    rho: float                  # per-round cost at the optimum = kept shots per round on Train

    def decisions(self) -> tuple:
        """Hashable summary of what the policy does (abort sets and keep threshold)."""
        return tuple(tuple(np.flatnonzero(q < 0).tolist()) for q in self.q_cont) + (self.keep_from,)


def isotonic(y, w) -> np.ndarray:
    """Weighted least-squares non-decreasing fit (pool adjacent violators). Entries with zero weight are
    interpolated from their non-empty neighbours (constant beyond the ends)."""
    y, w = np.asarray(y, np.float64), np.asarray(w, np.float64)
    idx = np.flatnonzero(w > 0)
    if idx.size == 0:
        return np.zeros_like(y)
    vals, wts, lens = [], [], []
    for i in idx:
        vals.append(y[i]); wts.append(w[i]); lens.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v2, w2, l2 = vals.pop(), wts.pop(), lens.pop()
            v1, w1, l1 = vals.pop(), wts.pop(), lens.pop()
            vals.append((v1 * w1 + v2 * w2) / (w1 + w2)); wts.append(w1 + w2); lens.append(l1 + l2)
    return np.interp(np.arange(y.size), idx, np.repeat(vals, lens))


def _mean_by(b, y, size) -> np.ndarray:
    """Per-bin mean of y, made non-decreasing in the bin (isotonic, weighted by the bin counts)."""
    cnt = np.bincount(b, minlength=size)
    tot = np.bincount(b, weights=y, minlength=size)
    out = np.zeros(size)
    np.divide(tot, cnt, out=out, where=cnt > 0)
    return isotonic(out, cnt)


def _check(B, rounds, sizes):
    rounds = np.asarray(rounds, np.float64)
    if B.shape[1] != rounds.size or len(sizes) != rounds.size:
        raise ValueError("B needs one column per checkpoint plus the final state")
    return rounds


def fitted_q(B, rounds, keep_from: int, rho: float, sizes) -> Policy:
    """Exact Q-table for per-round cost rho by one backward sweep over the logged Train transitions."""
    rounds = _check(B, rounds, sizes)
    K = rounds.size - 1
    v_next = (B[:, K] >= keep_from).astype(np.float64)   # per shot: value of its next state under the policy
    q_cont = [None] * K
    for k in range(K - 1, -1, -1):
        q_cont[k] = _mean_by(B[:, k], -rho * (rounds[k + 1] - rounds[k]) + v_next, sizes[k])
        v_next = np.maximum(q_cont[k][B[:, k]], 0.0)
    return Policy(rounds, q_cont, int(keep_from), float(rho))


def q_learning_online(B, rounds, keep_from: int, rho: float, sizes, epochs: int | None = None,
                      seed: int = 0) -> Policy:
    """The same table by tabular Q-learning: one update per logged transition, step size 1/N(s, a) within an
    epoch, bootstrapping from a target table frozen at the start of the epoch, and the isotonic constraint applied
    at the end of the epoch with that epoch's visit counts. Equals fitted_q after K epochs (each epoch fixes one
    more layer, counting back from the end)."""
    rounds = _check(B, rounds, sizes)
    K = rounds.size - 1
    epochs = K + 1 if epochs is None else epochs
    rng = np.random.default_rng(seed)
    q_cont = [np.zeros(s) for s in sizes[:K]]
    for _ in range(epochs):
        target = [q.copy() for q in q_cont]
        n_cont = [np.zeros(s) for s in sizes[:K]]
        for i in rng.permutation(B.shape[0]):
            b = B[i]
            for k in range(K):
                nxt = float(b[K] >= keep_from) if k == K - 1 else max(target[k + 1][b[k + 1]], 0.0)
                n_cont[k][b[k]] += 1
                q_cont[k][b[k]] += (-rho * (rounds[k + 1] - rounds[k]) + nxt - q_cont[k][b[k]]) / n_cont[k][b[k]]
        q_cont = [isotonic(q, n) for q, n in zip(q_cont, n_cont)]
    return Policy(rounds, q_cont, int(keep_from), float(rho))


def run(policy: Policy, B) -> tuple[np.ndarray, np.ndarray]:
    """Play the policy on shots with state bins B: (kept bool, rounds used float)."""
    K = policy.rounds.size - 1
    alive = np.ones(B.shape[0], dtype=bool)
    stop = np.full(B.shape[0], policy.rounds[K])
    for k in range(K):
        ab = alive & (policy.q_cont[k][B[:, k]] < 0)
        stop[ab] = policy.rounds[k]
        alive &= ~ab
    return alive & (B[:, K] >= policy.keep_from), stop


def reward_rate(policy: Policy, B) -> float:
    """Kept shots per round used."""
    keep, stop = run(policy, B)
    return float(keep.sum() / stop.sum())


def solve_rate(B, rounds, keep_from: int, sizes, iters: int = 100) -> Policy:
    """Policy with the most kept shots per round for this keep threshold (Dinkelbach iteration on rho)."""
    rho = 0.0
    pol = fitted_q(B, rounds, keep_from, rho, sizes)
    for _ in range(iters):
        new = reward_rate(pol, B)
        if new <= rho * (1 + 1e-12) + 1e-15:
            break
        rho = new
        pol = fitted_q(B, rounds, keep_from, rho, sizes)
    return pol


def policy_family(B, rounds, sizes, abort: bool = True) -> list:
    """[(Train keep fraction, Policy)] for every keep threshold j, strict to loose, duplicates removed.
    abort=False fixes rho = 0: the same keep thresholds with no early abort (the ablation)."""
    K = len(sizes) - 1
    fam, seen = [], set()
    for j in range(sizes[K] - 1, -1, -1):
        pol = solve_rate(B, rounds, j, sizes) if abort else fitted_q(B, rounds, j, 0.0, sizes)
        keep, _ = run(pol, B)
        key = (pol.decisions() if abort else j)
        if key in seen or not keep.any():
            continue
        seen.add(key)
        fam.append((float(keep.mean()), pol))
    fam.sort(key=lambda x: (x[0], -x[1].keep_from))
    return fam


def certify_family(family, B_cal, wrong_cal, alpha: float, delta: float):
    """Learn-then-Test over a family ordered strict to loose: index of the loosest certified policy, or None.
    The sequence starts at the first policy whose Train keep fraction could certify alpha with the Calibrate
    size (the start rule of qeccal.guarantees); each policy is tested at level delta; stop at the first failure."""
    need = np.log(1 / delta) / -np.log1p(-alpha / 2)
    n_cal = B_cal.shape[0]
    wrong_cal = np.asarray(wrong_cal, bool)
    certified = None
    for i, (kf, pol) in enumerate(family):
        if kf * n_cal < need:
            continue
        keep, _ = run(pol, B_cal)
        if p_value(int(keep.sum()), int(wrong_cal[keep].sum()), alpha) > delta:
            break
        certified = i
    return certified


def evaluate(policy: Policy, B, wrong, alpha: float) -> dict:
    """Kept count, errors, rate, exceedance, keep fraction, mean rounds per attempt and kept shots per round."""
    keep, stop = run(policy, B)
    n, e = int(keep.sum()), int(np.asarray(wrong, bool)[keep].sum())
    rate = e / n if n else float("nan")
    K = policy.rounds.size - 1
    return {"n": n, "errors": e, "rate": rate, "exceed": int(n > 0 and rate > alpha),
            "sig_exceed": int(cp_lower(e, n) > alpha), "keep": n / B.shape[0], "rounds_mean": float(stop.mean()),
            "aborted": float((stop < policy.rounds[K]).mean()), "throughput": n / float(stop.sum())}


def abort_below(policy: Policy, binner: Binner) -> list:
    """Per checkpoint: the prefix gap below which the policy aborts (upper edge of the highest abort bin),
    -inf if it never aborts there. The tables are non-decreasing, so the abort bins are the low ones."""
    out = []
    for k, q in enumerate(policy.q_cont):
        idx = np.flatnonzero(q < 0)
        if idx.size == 0:
            out.append(-np.inf)
        else:
            e = binner.edges[k]
            out.append(float(e[idx.max()]) if idx.max() < e.size else np.inf)
    return out
