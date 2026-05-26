import numpy as np
def compute_cdf_rms_fit_quality(
    fit_parameters,
    king_pdf,
    gamma_idx=None,
    minimum_counts=100,
):
    alpha = np.asarray(fit_parameters["alpha"])
    beta = np.asarray(fit_parameters["beta"])
    histograms = np.asarray(fit_parameters["histograms"])
    dpsi_bins = np.asarray(fit_parameters["dpsi_bins"])
    event_counts = np.asarray(fit_parameters["event_counts"])

    rms = np.full_like(alpha, np.nan, dtype=float)

    gamma_indices = range(alpha.shape[0]) if gamma_idx is None else [gamma_idx]

    for g in gamma_indices:
        for bin_idx in np.ndindex(*alpha[g].shape):
            param_idx = (g,) + bin_idx

            if event_counts[param_idx] < minimum_counts:
                continue

            bins = dpsi_bins[param_idx]

            # dpsi_bins are padded with zeros; keep only strictly increasing edges
            n_edges = np.sum(np.diff(bins) > 0) + 1
            if n_edges < 3:
                continue

            bins_valid = bins[:n_edges]
            hist_density = histograms[param_idx][: n_edges - 1]

            # Convert stored density back to probability mass
            delta = -2.0 * np.pi * np.diff(np.cos(bins_valid))
            mass = hist_density * delta

            if np.sum(mass) <= 0 or not np.all(np.isfinite(mass)):
                continue

            cdf_hist = np.cumsum(mass)
            cdf_hist = cdf_hist / cdf_hist[-1]

            expected = king_pdf.cdf(
                bins_valid[1:],
                alpha[param_idx],
                beta[param_idx],
            )

            if expected[-1] <= 0 or not np.all(np.isfinite(expected)):
                continue

            expected = expected / expected[-1]

            rms[param_idx] = np.sqrt(np.mean((cdf_hist - expected) ** 2))

    return rms