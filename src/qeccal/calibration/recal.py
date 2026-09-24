"""Post-hoc calibrators (spec section 8.5), fitted on the Train split only.

Every soft output is reduced to a confidence score s = log((1 - q) / q), the log-odds that the prediction is
right; for the MWPM gap, s is the gap itself. A calibrator maps s to a new q = P(prediction wrong):

  raw          q = sigmoid(-s)                 (no fitting)
  temperature  q = sigmoid(-s / T)             (one parameter; T > 1 means raw q was overconfident)
  platt        q = sigmoid(-(a s + b))         (two parameters)
  isotonic     q = f(s), f non-increasing      (step function; clipped away from 0 and 1)

All are non-increasing in s, so none changes which shots a threshold on s keeps; they change only the
probabilities attached to them.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression


def score_from_q(q) -> np.ndarray:
    """s = log((1 - q) / q), with q clipped to [1e-300, 0.5]."""
    q = np.clip(np.asarray(q, dtype=np.float64), 1e-300, 0.5)
    return np.log1p(-q) - np.log(q)


def _nll_logit(z, y):
    """Mean negative log-likelihood of y under P(y=1) = sigmoid(z), computed stably."""
    return float(np.mean(np.where(y, np.logaddexp(0.0, -z), np.logaddexp(0.0, z))))


class Raw:
    name = "raw"

    def fit(self, s, wrong):
        return self

    def predict(self, s):
        return expit(-np.asarray(s, dtype=np.float64))

    def params(self):
        return {}


class Temperature:
    name = "temperature"

    def fit(self, s, wrong):
        s, y = np.asarray(s, np.float64), np.asarray(wrong, bool)
        res = minimize_scalar(lambda lt: _nll_logit(-s / np.exp(lt), y), bounds=(-5, 5), method="bounded")
        self.T = float(np.exp(res.x))
        return self

    def predict(self, s):
        return expit(-np.asarray(s, np.float64) / self.T)

    def params(self):
        return {"T": self.T}


class Platt:
    name = "platt"

    def fit(self, s, wrong):
        s, y = np.asarray(s, np.float64), np.asarray(wrong, bool)
        res = minimize(lambda ab: _nll_logit(-(ab[0] * s + ab[1]), y), x0=np.array([1.0, 0.0]), method="BFGS")
        self.a, self.b = map(float, res.x)
        return self

    def predict(self, s):
        return expit(-(self.a * np.asarray(s, np.float64) + self.b))

    def params(self):
        return {"a": self.a, "b": self.b}


class Isotonic:
    name = "isotonic"

    def fit(self, s, wrong):
        n = len(s)
        lo = 0.5 / (n + 1)                      # never exactly 0: one Test error would give infinite NLL
        self.model = IsotonicRegression(increasing=False, y_min=lo, y_max=1 - lo, out_of_bounds="clip")
        self.model.fit(np.asarray(s, np.float64), np.asarray(wrong, np.float64))
        return self

    def predict(self, s):
        return self.model.predict(np.asarray(s, np.float64))

    def params(self):
        return {"steps": int(len(self.model.X_thresholds_))}


CALIBRATORS = {c.name: c for c in (Raw, Temperature, Platt, Isotonic)}
