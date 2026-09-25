"""Belief-matching gap (spec section 8.3).

The `beliefmatching` package returns predictions only (and skips matching entirely when BP converges), so
the gap is reimplemented here:

1. Belief propagation on the full DEM: one variable per error mechanism, one check per detector. Flooding
   product-sum, a fixed number of iterations (20, the beliefmatching default), vectorised over shots. With
   no early stop, every shot's posteriors come from the same computation; the beliefmatching package also
   matches on 20-iteration posteriors whenever BP has not converged. Output: the posterior probability of
   each mechanism given the shot's detections.
2. Each mechanism is decomposed into graph-like components (the DEM's "^" hints). A component's posterior
   is the probability that an odd number of the mechanisms containing it fired, assuming independence:
   p_e = (1 - prod(1 - 2 p_h)) / 2.
3. Matching on those components with weights -ln(p_e), the observable turned into an extra detector node
   (as in qeccal.soft.gap), decoded with that node forced to 0 and to 1. The gap is the weight difference;
   the prediction is the lighter class. The -ln(p_e) weights follow the beliefmatching package; on stim
   test circuits they gave lower logical error than ln((1 - p_e) / p_e) (d3, 3 rounds: 2.1% vs 3.0%),
   because posteriors near 1/2 would otherwise get weights near zero.

Components are keyed by (detectors, observable), so a boundary edge that flips the observable stays
distinct from one that does not. The matching graph is rebuilt per shot, which is what makes this slower
than the plain gap.
"""
from __future__ import annotations

import numpy as np
import pymatching
import stim
from scipy.sparse import csc_matrix, csr_matrix

from .gap import PreconditionError

TINY = 1e-300
ONE = 1 - 1e-15
EDGES_PER_CHUNK = 4_000_000        # (Tanner edges x shots) per BP batch: bounds memory to ~0.5 GB


def _components(inst):
    comps, cur = [], []
    for t in inst.targets_copy():
        if t.is_separator():
            comps.append(cur)
            cur = []
        else:
            cur.append(t)
    comps.append(cur)
    return comps


def _csc(columns, n_rows):
    rows = np.concatenate([np.asarray(c, dtype=np.int64) for c in columns]) if columns else np.zeros(0, np.int64)
    cols = np.concatenate([np.full(len(c), j, dtype=np.int64) for j, c in enumerate(columns)]) if columns else rows
    return csc_matrix((np.ones(rows.size, dtype=np.uint8), (rows, cols)), shape=(n_rows, len(columns)))


def _signed_log(x):
    """(log|x|, x < 0) for combining products in the log domain."""
    return np.log(np.maximum(np.abs(x), TINY)), (x < 0)


class BatchBP:
    """Flooding product-sum BP with a fixed iteration count. Messages are arrays of shape (Tanner edges, shots)."""

    def __init__(self, check: csc_matrix, priors: np.ndarray, iters: int = 20):
        h = check.tocoo()
        order = np.lexsort((h.col, h.row))
        self.e_chk, self.e_var = h.row[order], h.col[order]
        n_edges = self.e_chk.size
        n_chk, n_var = check.shape
        one = np.ones(n_edges)
        self.to_chk = csr_matrix((one, (self.e_chk, np.arange(n_edges))), shape=(n_chk, n_edges))
        self.to_var = csr_matrix((one, (self.e_var, np.arange(n_edges))), shape=(n_var, n_edges))
        self.prior = np.log((1 - priors) / priors)
        self.iters = iters

    def llr(self, syndrome: np.ndarray) -> np.ndarray:
        """Posterior log((1 - p) / p) per mechanism, shape (mechanisms, shots), for syndrome (shots, checks)."""
        sign = np.where(np.asarray(syndrome, bool).T, -1.0, 1.0)[self.e_chk]
        vc = np.repeat(self.prior[self.e_var][:, None], sign.shape[1], axis=1)
        post = np.repeat(self.prior[:, None], sign.shape[1], axis=1)
        for _ in range(self.iters):
            la, neg = _signed_log(np.tanh(vc / 2))
            la_ext = (self.to_chk @ la)[self.e_chk] - la
            odd = ((self.to_chk @ neg.astype(np.float64))[self.e_chk] - neg) % 2 > 0.5
            cv = 2 * np.arctanh(np.where(odd, -sign, sign) * np.minimum(np.exp(la_ext), ONE))
            post = self.prior[:, None] + self.to_var @ cv
            vc = post[self.e_var] - cv
        return post


class BeliefMatchingGap:
    def __init__(self, dem: stim.DetectorErrorModel, bp_iters: int = 20):
        if dem.num_observables != 1:
            raise ValueError("expected one observable")
        n = self.num_detectors = dem.num_detectors
        mech_cols, priors, edge_ids, edge_cols, mech_to_edges = [], [], {}, [], []
        for inst in dem.flattened():
            if inst.type != "error":
                continue
            p = inst.args_copy()[0]
            full, eids = set(), []
            for comp in _components(inst):
                dets = frozenset(t.val for t in comp if t.is_relative_detector_id())
                obs = sum(1 for t in comp if t.is_logical_observable_id()) % 2
                if obs and len(dets) >= 2:
                    raise PreconditionError(f"component {comp} flips the observable and touches {len(dets)} detectors")
                if len(dets) > 2:
                    raise ValueError("undecomposed hyperedge component; the DEM needs decompose_errors=True")
                full ^= set(dets)
                key = (dets, obs)
                if key not in edge_ids:
                    edge_ids[key] = len(edge_cols)
                    edge_cols.append(sorted(dets) + ([n] if obs else []))
                eids.append(edge_ids[key])
            if not full or p <= 0:
                continue          # invisible to BP (or impossible); its components keep no posterior mass
            mech_cols.append(sorted(full))
            priors.append(min(p, 0.5))
            mech_to_edges.append(eids)
        self.check = _csc(mech_cols, n)                          # detectors x mechanisms, for BP
        self.priors = np.asarray(priors, dtype=np.float64)
        self.edges = _csc(edge_cols, n + 1)                      # (detectors + observable node) x components
        self.h2e = _csc(mech_to_edges, len(edge_cols)).tocsr().astype(np.float64)   # components x mechanisms
        self.bp = BatchBP(self.check, self.priors, bp_iters)

    def edge_probabilities(self, syndrome: np.ndarray) -> np.ndarray:
        """Component posteriors p_e, shape (components, shots), for syndrome (shots, detectors); 1-D in, 1-D out."""
        one = np.ndim(syndrome) == 1
        llr = self.bp.llr(np.atleast_2d(syndrome))
        la, neg = _signed_log(np.tanh(llr / 2))                   # 1 - 2 p_h = tanh(llr / 2)
        la = self.h2e @ la
        odd = (self.h2e @ neg.astype(np.float64)) % 2 > 0.5
        p_e = np.where(odd, (1 + np.exp(la)) / 2, -np.expm1(la) / 2)
        p_e = np.clip(p_e, 1e-15, 1 - 1e-15)
        return p_e[:, 0] if one else p_e

    def _match(self, syndrome, p_e):
        m = pymatching.Matching.from_check_matrix(self.edges, weights=-np.log(p_e), use_virtual_boundary_node=True)
        x = np.zeros(self.num_detectors + 1, dtype=np.uint8)
        x[:-1] = syndrome
        _, w0 = m.decode(x, return_weight=True)
        x[-1] = 1
        _, w1 = m.decode(x, return_weight=True)
        return bool(w1 < w0), float(abs(w1 - w0))

    def decode(self, det: np.ndarray, *, packed: bool = False, chunk: int | None = None):
        """(prediction bool, gap float64) for each shot (rows of det; bit-packed if packed=True)."""
        n = det.shape[0]
        chunk = chunk or max(1, min(512, EDGES_PER_CHUNK // max(1, self.check.nnz)))
        pred = np.empty(n, dtype=bool)
        gap = np.empty(n, dtype=np.float64)
        for i in range(0, n, chunk):
            block = det[i:i + chunk]
            if packed:
                block = np.unpackbits(block, axis=1, count=self.num_detectors, bitorder="little")
            p_e = self.edge_probabilities(block)
            for j in range(block.shape[0]):
                pred[i + j], gap[i + j] = self._match(block[j], p_e[:, j])
        return pred, gap
