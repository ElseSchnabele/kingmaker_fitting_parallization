import numpy as np
def cdf_rms(
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

def pdf_rms(
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
            hist = histograms[param_idx]
            bin_alpha = alpha[param_idx]
            bin_beta = beta[param_idx]

            mask = hist > 0
            bin_centers = 0.5 * (bins[:-1] + bins[1:])

            bin_centers = bin_centers[mask]
            hist_valid = hist[mask]

            pdf_fit = king_pdf.pdf(bin_centers, bin_alpha, bin_beta)

            if np.nanmax(pdf_fit) > 0:
                pdf_fit = pdf_fit * np.nanmax(hist_valid) / np.nanmax(pdf_fit)

            rms[param_idx] = np.sqrt(np.mean((hist_valid - pdf_fit) ** 2))

    return rms

#TODO without normalization (weights vs. rms)
def weighted_cdf_rms(
    fit_parameters,
    king_pdf,
    gamma_idx=None,
    minimum_counts=100,
):
    """
    Compute weighted Root Mean squared error of CDF. Weighting goes with number of events in fitted bin to punish more bins where fit is bad even though
    there is a lot of data.
    """
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

            rms[param_idx] = np.sqrt(np.mean(( cdf_hist - expected) ** 2)) * np.sqrt(event_counts[param_idx])

    return rms

#TODO without normalization (weights vs. rms)
def weighted_pdf_rms(
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
            hist = histograms[param_idx]
            bin_alpha = alpha[param_idx]
            bin_beta = beta[param_idx]

            mask = hist > 0
            bin_centers = 0.5 * (bins[:-1] + bins[1:])

            bin_centers = bin_centers[mask]
            hist_valid = hist[mask]

            pdf_fit = king_pdf.pdf(bin_centers, bin_alpha, bin_beta)

            if np.nanmax(pdf_fit) > 0:
                pdf_fit = pdf_fit * np.nanmax(hist_valid) / np.nanmax(pdf_fit)

            rms[param_idx] = np.sqrt(np.mean((hist_valid - pdf_fit) ** 2))* np.sqrt(event_counts[param_idx])

    return rms



def density_weighted_cdf_rms( 
    fit_parameters,
    king_pdf,
    gamma_idx=None,
    minimum_counts=100,
):
    """
    Compute weighted Root Mean squared error of CDF. Weighting goes with number of events in fitted bin to punish more bins where fit is bad even though
    there is a lot of data.
    """
    alpha = np.asarray(fit_parameters["alpha"])
    beta = np.asarray(fit_parameters["beta"])
    histograms = np.asarray(fit_parameters["histograms"])
    dpsi_bins = np.asarray(fit_parameters["dpsi_bins"])
    event_counts = np.asarray(fit_parameters["event_counts"])
    parametrization_bins = fit_parameters['parametrization_bins']
    if isinstance(parametrization_bins, np.ndarray):
        parametrization_bins = parametrization_bins.item()
    rms = np.full_like(alpha, np.nan, dtype=float)
    weights = np.full_like(alpha, np.nan, dtype=float)

    gamma_indices = range(alpha.shape[0]) if gamma_idx is None else [gamma_idx]

    for g in gamma_indices:
        shape = alpha[g].shape
        volumes = np.zeros(shape)
        bin_names = list(parametrization_bins.keys())
        for bin_idx in np.ndindex(*shape):
            volume = 1.0

            for dim, name in enumerate(bin_names):
                bins = np.asarray(parametrization_bins[name])
                i = bin_idx[dim]

                if name == "dec":
                    width = np.sin(bins[i + 1]) - np.sin(bins[i])
                    full_width = np.sin(bins[-1]) - np.sin(bins[0])
                else:
                    width = bins[i + 1] - bins[i]
                    full_width = bins[-1] - bins[0]

                volume *= width / full_width

            volumes[bin_idx] = volume

        
        for bin_idx in np.ndindex(*shape):
            param_idx = (g,) + bin_idx
            volume = volumes[bin_idx]
            if event_counts[param_idx] < minimum_counts:
                continue
            elif not np.isfinite(volume) or volume <= 0:
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

            norm = np.nanmax(expected)

            if norm <= 0 or not np.isfinite(norm):
                continue

            expected = expected / norm
            weights[param_idx] =  np.sqrt(event_counts[param_idx]/(volume * np.max(event_counts)))
            rms[param_idx] = np.sqrt(np.mean(( cdf_hist - expected) ** 2)) 
            
    rms /= np.max(rms[~np.isnan(rms)])
    weights /= np.max(weights[~np.isnan(weights)])
    
    total_score = rms * weights
    total_score /= np.max(total_score[~np.isnan(total_score)])

    return total_score

#TODO without normalization (weights vs. rms)
def density_weighted_pdf_rms(
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

    parametrization_bins = fit_parameters["parametrization_bins"]
    if isinstance(parametrization_bins, np.ndarray):
        parametrization_bins = parametrization_bins.item()

    rms = np.full_like(alpha, np.nan, dtype=float)
    weights = np.full_like(alpha, np.nan, dtype=float)

    gamma_indices = range(alpha.shape[0]) if gamma_idx is None else [gamma_idx]

    for g in gamma_indices:
        shape = alpha[g].shape
        volumes = np.zeros(shape, dtype=float)
        bin_names = list(parametrization_bins.keys())

        for bin_idx in np.ndindex(*shape):
            volume = 1.0

            for dim, name in enumerate(bin_names):
                param_bins = np.asarray(parametrization_bins[name])
                i = bin_idx[dim]

                if name == "dec":
                    width = np.sin(param_bins[i + 1]) - np.sin(param_bins[i])
                    full_width = np.sin(param_bins[-1]) - np.sin(param_bins[0])
                else:
                    width = param_bins[i + 1] - param_bins[i]
                    full_width = param_bins[-1] - param_bins[0]

                volume *= width / full_width

            volumes[bin_idx] = volume

        for bin_idx in np.ndindex(*shape):
            param_idx = (g,) + bin_idx
            volume = volumes[bin_idx]

            if event_counts[param_idx] < minimum_counts:
                continue
            if not np.isfinite(volume) or volume <= 0:
                continue

            bins = dpsi_bins[param_idx]
            hist = histograms[param_idx]

            bin_alpha = alpha[param_idx]
            bin_beta = beta[param_idx]

            n_edges = np.sum(np.diff(bins) > 0) + 1
            if n_edges < 3:
                continue

            bins_valid = bins[:n_edges]
            hist_valid = hist[: n_edges - 1]

            mask = hist_valid > 0
            if np.sum(mask) == 0:
                continue

            bin_centers = 0.5 * (bins_valid[:-1] + bins_valid[1:])
            bin_centers = bin_centers[mask]
            hist_valid = hist_valid[mask]

            pdf_fit = king_pdf.pdf(
                bin_centers,
                bin_alpha,
                bin_beta,
            )

            if not np.all(np.isfinite(pdf_fit)):
                continue

            if np.nanmax(pdf_fit) > 0:
                pdf_fit = pdf_fit * np.nanmax(hist_valid) / np.nanmax(pdf_fit)
            else:
                continue
            weights[param_idx] = np.sqrt(event_counts[param_idx]/(volume * np.max(event_counts)))
            rms[param_idx] = np.sqrt(np.mean((hist_valid - pdf_fit) ** 2)) * weights[param_idx]
            

    return rms



def cdf_root_median_squared(
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

            rms[param_idx] = np.sqrt(np.median((cdf_hist - expected) ** 2))

    return rms

def pdf_root_median_squared(
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
            hist = histograms[param_idx]
            bin_alpha = alpha[param_idx]
            bin_beta = beta[param_idx]

            mask = hist > 0
            bin_centers = 0.5 * (bins[:-1] + bins[1:])

            bin_centers = bin_centers[mask]
            hist_valid = hist[mask]

            pdf_fit = king_pdf.pdf(bin_centers, bin_alpha, bin_beta)

            if np.nanmax(pdf_fit) > 0:
                pdf_fit = pdf_fit * np.nanmax(hist_valid) / np.nanmax(pdf_fit)

            rms[param_idx] = np.sqrt(np.median((hist_valid - pdf_fit) ** 2))

    return rms