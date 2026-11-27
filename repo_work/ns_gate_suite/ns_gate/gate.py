"""Pre-fusion validity gate: metadata predicates + pairing-dependence predicate.

Gamma = AND_j pi_j(D_E, D_I); fuse() runs only if Gamma == 1  (draft Eq. 9).

Predicates pi_1..pi_8 check identifiers and metadata. pi_9 checks that the
joined feature payloads are statistically dependent the way correctly paired
records are. Two forms of pi_9 are provided:

  * "perm"  : permutation test of independence (H0: x_E independent of x_I).
              Block when H0 cannot be rejected at level alpha.
  * "calib" : reference-calibrated dependence-drop test. The statistic of the
              incoming batch is compared with its distribution over clean
              batches of the same size drawn from a trusted paired reference
              set; block when it falls below the conformal alpha-quantile.
              Under exchangeability this bounds the clean false-block rate by
              alpha, and it is sensitive to *partial* mispairing.

Statistics: "cca"  = Pillai trace (sum of squared canonical correlations)
            "hsic" = biased HSIC with Gaussian kernels, median bandwidth.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- statistics

def _center(x):
    return x - x.mean(axis=0, keepdims=True)


def _orth(x):
    q, _ = np.linalg.qr(_center(x))
    return q


def cca_pillai(x, y):
    m = _orth(x).T @ _orth(y)
    return float(np.sum(m * m))


def _gauss_gram(x):
    x = (x - x.mean(0)) / (x.std(0) + 1e-12)
    sq = np.sum(x * x, 1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2 * x @ x.T, 0.0)
    med = np.median(d2[np.triu_indices_from(d2, 1)])
    return np.exp(-d2 / (med + 1e-12))


def _hcenter(k):
    return k - k.mean(0, keepdims=True) - k.mean(1, keepdims=True) + k.mean()


def hsic(x, y):
    n = len(x)
    return float(np.sum(_hcenter(_gauss_gram(x)) * _gauss_gram(y)) / n**2)


STATS = {"cca": cca_pillai, "hsic": hsic}


def perm_pvalue(x, y, stat="cca", n_perm=199, rng=None):
    """Permutation p-value for H0: independence. p = (1+#{T*>=T})/(B+1)."""
    rng = rng or np.random.default_rng()
    n = len(x)
    if stat == "cca":
        qx, qy = _orth(x), _orth(y)
        t_obs = float(np.sum((qx.T @ qy) ** 2))
        null = np.empty(n_perm)
        for b in range(n_perm):
            m = qx.T @ qy[rng.permutation(n)]
            null[b] = np.sum(m * m)
    elif stat == "hsic":
        kc = _hcenter(_gauss_gram(x))
        l = _gauss_gram(y)
        t_obs = float(np.sum(kc * l) / n**2)
        null = np.empty(n_perm)
        for b in range(n_perm):
            p = rng.permutation(n)
            null[b] = np.sum(kc * l[np.ix_(p, p)]) / n**2
    else:
        raise ValueError(stat)
    p = (1 + np.sum(null >= t_obs)) / (n_perm + 1)
    return t_obs, float(p)


@dataclass
class Reference:
    """Clean-batch reference distribution of a statistic for batch size n."""
    stat: str
    n: int
    values: np.ndarray

    def threshold(self, alpha):
        b = len(self.values)
        k = math.floor(alpha * (b + 1))  # conformal lower rank
        if k < 1:
            return -np.inf
        return float(np.sort(self.values)[k - 1])

    def pvalue(self, t):
        return float((1 + np.sum(self.values <= t)) / (len(self.values) + 1))


def build_reference(x_ref, y_ref, n, stat="cca", n_batches=500, rng=None):
    rng = rng or np.random.default_rng()
    f = STATS[stat]
    vals = np.empty(n_batches)
    for b in range(n_batches):
        idx = rng.choice(len(x_ref), size=n, replace=False)
        vals[b] = f(x_ref[idx], y_ref[idx])
    return Reference(stat, n, vals)


# ---------------------------------------------------------------- metadata

DECLARED = {
    "eeg": {"fs": (128.0, 1024.0), "n_channels": (16, 256), "units": "uV"},
    "eit": {"fs": (1.0, 100.0), "n_channels": (8, 32), "units": "V"},
}
TIME_TOL_S = 2.0
REQUIRED_META = ["consent", "source"]


def make_tables(n, rng, t0=1.7e9):
    """Clean, correctly paired metadata tables for n records."""
    ids = [f"NS{idx:06d}" for idx in rng.permutation(10 * n)[:n]]
    t = t0 + np.cumsum(rng.uniform(5, 30, size=n))
    eeg = pd.DataFrame({"sample_id": ids, "timestamp": t, "fs": 160.0,
                        "n_channels": 64, "units": "uV", "consent": "yes",
                        "source": "sim-A"})
    eit = pd.DataFrame({"sample_id": ids, "timestamp": t + rng.normal(0, 0.3, n),
                        "fs": 10.0, "n_channels": 16, "units": "V",
                        "consent": "yes", "source": "sim-A"})
    return eeg, eit


def _ok_ids(df):
    return "sample_id" in df.columns


PREDICATE_NAMES = [
    "P1_id_column_present", "P2_no_missing_id", "P3_unique_ids",
    "P4_id_sets_equal", "P5_rate_channels_in_range", "P6_units_declared",
    "P7_timestamps_within_tol", "P8_consent_source_present",
]


def metadata_predicates(eeg, eit):
    """Return ordered dict name -> bool for P1..P8."""
    out = {}
    out["P1_id_column_present"] = _ok_ids(eeg) and _ok_ids(eit)
    if not out["P1_id_column_present"]:
        return {k: False for k in PREDICATE_NAMES}

    def nonmissing(df):
        s = df["sample_id"]
        return bool(s.notna().all() and (s.astype(str).str.strip() != "").all())

    out["P2_no_missing_id"] = nonmissing(eeg) and nonmissing(eit)
    out["P3_unique_ids"] = bool(eeg["sample_id"].is_unique and eit["sample_id"].is_unique)
    out["P4_id_sets_equal"] = set(eeg["sample_id"].dropna()) == set(eit["sample_id"].dropna())

    def in_range(df, mod):
        d = DECLARED[mod]
        return bool(df["fs"].between(*d["fs"]).all() and df["n_channels"].between(*d["n_channels"]).all())

    out["P5_rate_channels_in_range"] = in_range(eeg, "eeg") and in_range(eit, "eit")
    out["P6_units_declared"] = bool((eeg["units"] == DECLARED["eeg"]["units"]).all()
                                    and (eit["units"] == DECLARED["eit"]["units"]).all())
    j = eeg[["sample_id", "timestamp"]].merge(eit[["sample_id", "timestamp"]], on="sample_id",
                                               suffixes=("_e", "_i"))
    out["P7_timestamps_within_tol"] = bool(len(j) > 0 and
                                           (np.abs(j.timestamp_e - j.timestamp_i) <= TIME_TOL_S).all())

    def meta(df):
        return all(c in df.columns and df[c].notna().all() and (df[c].astype(str).str.strip() != "").all()
                   for c in REQUIRED_META)

    out["P8_consent_source_present"] = meta(eeg) and meta(eit)
    return out


ID_ONLY = PREDICATE_NAMES[:4]


@dataclass
class GateConfig:
    use_metadata: list = field(default_factory=lambda: list(PREDICATE_NAMES))
    dependence: str | None = "calib"     # None | "perm" | "calib"
    stat: str = "cca"
    alpha: float = 0.05
    n_perm: int = 199
    reference: Reference | None = None


@dataclass
class GateDecision:
    passed: bool
    reasons: list
    p_value: float | None = None
    statistic: float | None = None


def run_gate(eeg, eit, x_eeg, x_eit, cfg: GateConfig, rng=None):
    """Evaluate the gate. x_eeg/x_eit are payload rows aligned to the rows of
    eeg/eit as delivered (the positional join used by the fusion stage)."""
    reasons = []
    preds = metadata_predicates(eeg, eit)
    for name in cfg.use_metadata:
        if not preds.get(name, False):
            reasons.append(name)
    if reasons:
        return GateDecision(False, reasons)
    p, t = None, None
    if cfg.dependence == "perm":
        t, p = perm_pvalue(x_eeg, x_eit, cfg.stat, cfg.n_perm, rng)
        if p > cfg.alpha:
            reasons.append(f"P9_pairing_dependence_perm_{cfg.stat}")
    elif cfg.dependence == "calib":
        if cfg.reference is None or cfg.reference.n != len(x_eeg):
            raise ValueError("calibrated gate needs a Reference for this batch size")
        t = STATS[cfg.stat](x_eeg, x_eit)
        p = cfg.reference.pvalue(t)
        if t < cfg.reference.threshold(cfg.alpha):
            reasons.append(f"P9_pairing_dependence_calib_{cfg.stat}")
    return GateDecision(len(reasons) == 0, reasons, p, t)


class GateBlocked(RuntimeError):
    pass


def guarded_fuse(decision: GateDecision, fuse_fn, *args, **kw):
    """The only entry point to fusion. Raises GateBlocked if the gate failed."""
    if not decision.passed:
        raise GateBlocked(";".join(decision.reasons))
    return fuse_fn(*args, **kw)
