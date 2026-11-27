"""Figure C: GLRT scan with conformal threshold vs. null-calibrated peak detector
for simulated 2-D difference EIT (pyEIT re-implementation, in-silico only).

* Data are generated on a FINE mesh and reconstructed/scanned on a COARSE mesh
  (~312 nodes / 560 elements, close to the archived EIDORS model) so the
  inverse crime is avoided. Both meshes share the 16 boundary electrodes.
* Adjacent drive / adjacent measurement: m = 208 measurements per frame.
* Noise is defined on the difference signal (draft Eq. 5):
      SNR_dv = 20 log10( ||dv_clean||_2 / (s * sqrt(2m)) )
  For each SNR level s is fixed from the MEDIAN ||dv_clean|| of the random
  anomaly population, so that null (noise-only) frames have a defined s.
* GLRT:  T = max_k (g_k^T dv)^2 / (g_k^T g_k)       (W = I, white noise)
* Peak:  u = (J^T J + lam R)^-1 J^T dv, R = Laplacian prior; score = max(sign*u)
  lam chosen on a separate tuning set (best AUC at 5 dB) to be fair to the baseline.
* Both thresholds are conformal: the ceil((1-alpha)(n+1))-th smallest of n
  null calibration scores; false alarms are then counted on independent nulls.
"""
from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def _meshes(seed=7, h_fine=0.05, h_coarse=0.11):
    from pyeit.mesh import create
    np.random.seed(seed)
    fine = create(16, h0=h_fine)
    np.random.seed(seed + 1)
    coarse = create(16, h0=h_coarse)
    return fine, coarse


def _centroids(mesh):
    return mesh.node[mesh.element][:, :, :2].mean(axis=1)


def _laplacian(mesh):
    """R = L^T L, L = element-difference operator over shared edges."""
    from collections import defaultdict
    edge_owner = defaultdict(list)
    for e, tri in enumerate(mesh.element):
        for a, b in ((0, 1), (1, 2), (0, 2)):
            edge_owner[tuple(sorted((tri[a], tri[b])))].append(e)
    pairs = [v for v in edge_owner.values() if len(v) == 2]
    ne = len(mesh.element)
    L = np.zeros((len(pairs), ne))
    for i, (a, b) in enumerate(pairs):
        L[i, a], L[i, b] = 1.0, -1.0
    return L.T @ L + 1e-6 * np.eye(ne)


def run(seed=11, n_anom=500, n_cal=200, n_null_test=1000, n_tune=100,
        snr_db=(-10, -5, 0, 5, 10, 15, 20), alpha=0.05, roc_snrs=(0, 5, 10)):
    import pyeit.eit.protocol as protocol
    from pyeit.eit.fem import EITForward

    rng = np.random.default_rng(seed)
    fine, coarse = _meshes()
    prot = protocol.create(16, dist_exc=1, step_meas=1, parser_meas="std")
    fwd_f = EITForward(fine, prot)
    fwd_c = EITForward(coarse, prot)
    cf, cc = _centroids(fine), _centroids(coarse)

    v_h = np.real(fwd_f.solve_eit(perm=np.ones(len(fine.element))))
    m = len(v_h)
    J, _ = fwd_c.compute_jac(perm=np.ones(len(coarse.element)))
    J = np.real(J)
    gnorm2 = np.sum(J * J, axis=0)
    R = _laplacian(coarse)
    JtJ = J.T @ J

    def anomaly(n):
        out = []
        for _ in range(n):
            rad = rng.uniform(0.10, 0.25)
            rho = np.sqrt(rng.uniform(0, 1)) * (0.85 - rad)
            th = rng.uniform(0, 2 * np.pi)
            c = np.array([rho * np.cos(th), rho * np.sin(th)])
            contrast = rng.uniform(0.3, 0.7)
            perm = np.ones(len(fine.element))
            inside = np.linalg.norm(cf - c, axis=1) <= rad
            if inside.sum() == 0:
                inside[np.argmin(np.linalg.norm(cf - c, axis=1))] = True
            perm[inside] = contrast
            dv = np.real(fwd_f.solve_eit(perm=perm) - v_h)
            out.append((dv, c, rad, contrast))
        return out

    tune = anomaly(n_tune)
    anoms = anomaly(n_anom)
    dv_clean = np.array([a[0] for a in anoms])
    centres = np.array([a[1] for a in anoms])
    med_norm = float(np.median(np.linalg.norm(dv_clean, axis=1)))

    # sign convention of the reconstruction for a conductivity decrease
    recon0 = np.linalg.solve(JtJ + 1e-2 * np.trace(JtJ) / np.trace(R) * R, J.T @ dv_clean[0])
    sign = 1.0 if recon0[np.argmax(np.abs(recon0))] > 0 else -1.0

    def s_for(snr):
        return med_norm / (math.sqrt(2 * m) * 10 ** (snr / 20))

    def glrt(dv):
        return np.max((dv @ J) ** 2 / gnorm2, axis=1), np.argmax((dv @ J) ** 2 / gnorm2, axis=1)

    def make_peak(lam_rel):
        M = np.linalg.solve(JtJ + lam_rel * np.trace(JtJ) / np.trace(R) * R, J.T)  # (ne, m)

        def peak(dv):
            u = sign * (dv @ M.T)
            return np.max(u, axis=1), np.argmax(u, axis=1)
        return peak

    # ---- tune lambda for the peak baseline on the separate tuning set at 5 dB
    s5 = s_for(5)
    dv_t = np.array([t[0] for t in tune]) + rng.normal(0, s5 * math.sqrt(2), size=(n_tune, m))
    nul_t = rng.normal(0, s5 * math.sqrt(2), size=(n_tune, m))
    lam_scores = {}
    for lam in (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0):
        pk = make_peak(lam)
        lam_scores[lam] = roc_auc_score(np.r_[np.ones(n_tune), np.zeros(n_tune)],
                                        np.r_[pk(dv_t)[0], pk(nul_t)[0]])
    lam_best = max(lam_scores, key=lam_scores.get)
    peak = make_peak(lam_best)

    rank = math.ceil((1 - alpha) * (n_cal + 1))
    levels, rocs = [], {}
    for snr in snr_db:
        s = s_for(snr)
        noise_sd = s * math.sqrt(2)  # difference of two independent noisy frames
        dv = dv_clean + rng.normal(0, noise_sd, size=dv_clean.shape)
        cal = rng.normal(0, noise_sd, size=(n_cal, m))
        nul = rng.normal(0, noise_sd, size=(n_null_test, m))
        realised = 20 * np.log10(np.linalg.norm(dv_clean, axis=1) / (s * math.sqrt(2 * m)))
        row = {"snr_db": snr, "noise_s": s,
               "realised_snr_median": float(np.median(realised)),
               "realised_snr_iqr": [float(np.percentile(realised, 25)), float(np.percentile(realised, 75))]}
        for name, det in (("glrt", glrt), ("peak", peak)):
            sa, ka = det(dv)
            sc, _ = det(cal)
            sn, _ = det(nul)
            thr = np.sort(sc)[rank - 1] if rank <= n_cal else np.inf
            detected = sa > thr
            far = float(np.mean(sn > thr))
            loc_err = np.linalg.norm(cc[ka] - centres, axis=1)
            yv = np.r_[np.ones(len(sa)), np.zeros(len(sn))]
            sv = np.r_[sa, sn]
            row[f"{name}_auc"] = float(roc_auc_score(yv, sv))
            row[f"{name}_detection"] = float(detected.mean())
            row[f"{name}_far"] = far
            row[f"{name}_far_k"] = int(np.sum(sn > thr))
            # marginal FAR: average over 100 independent calibration redraws
            fars = []
            for _ in range(100):
                sc_b, _ = det(rng.normal(0, noise_sd, size=(n_cal, m)))
                fars.append(np.mean(sn > np.sort(sc_b)[rank - 1]))
            row[f"{name}_far_marginal_mean"] = float(np.mean(fars))
            row[f"{name}_far_marginal_p95"] = float(np.percentile(fars, 95))
            row[f"{name}_loc_err_median_detected"] = (float(np.median(loc_err[detected]))
                                                     if detected.any() else None)
            if snr in roc_snrs:
                fpr, tpr, _ = roc_curve(yv, sv)
                rocs[f"{name}_{snr}dB"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(),
                                           "op_far": far, "op_tpr": float(detected.mean())}
        levels.append(row)

    # ---- map the draft's percentage noise levels onto SNR_dv (this pyEIT model)
    ref_perm = np.ones(len(fine.element))
    ref_inside = np.linalg.norm(cf - np.array([0.4, 0.0]), axis=1) <= 0.156
    ref_perm[ref_inside] = 0.5
    dv_ref = np.real(fwd_f.solve_eit(perm=ref_perm) - v_h)
    mapping = []
    for pct in (1, 5, 10):
        s = pct / 100 * float(np.std(v_h))
        mapping.append({"noise_pct_of_std_vh": pct,
                        "snr_dv_reference_anomaly_db": float(20 * np.log10(np.linalg.norm(dv_ref) / (s * math.sqrt(2 * m)))),
                        "snr_dv_population_median_db": float(20 * np.log10(med_norm / (s * math.sqrt(2 * m))))})

    meta = {"fine_mesh": {"nodes": int(len(fine.node)), "elements": int(len(fine.element))},
            "coarse_mesh": {"nodes": int(len(coarse.node)), "elements": int(len(coarse.element))},
            "n_meas": int(m), "n_anomalies": n_anom, "n_cal_null": n_cal, "n_test_null": n_null_test,
            "alpha": alpha, "conformal_rank": rank, "peak_lambda_rel": lam_best,
            "peak_lambda_tuning_auc": {str(k): float(v) for k, v in lam_scores.items()},
            "anomaly_radius": [0.10, 0.25], "anomaly_contrast": [0.3, 0.7],
            "reference_anomaly": {"centre": [0.4, 0.0], "radius": 0.156, "contrast": 0.5,
                                  "fine_elements_inside": int(ref_inside.sum())},
            "noise_mapping": mapping, "solver": "pyEIT (Python); not the EIDORS model"}
    return levels, rocs, meta
