from typing import Dict, Any, Callable
import numpy as np

#from kingmaker_fork.likelihood_analysis.analysis_helpers.rms_fit_quality import compute_cdf_rms_fit_quality

""" def fitting_score(
    fit_parameters: Dict[str, Any],
    king_pdf,
    min_counts: int,
    alpha: float = 0.3,
    beta: float = 0.1,
    mu: float = 0.3,
    chi2_failure_threshold: float = 1e4,
    verbose: bool = True,
) -> float:

    #Computes one scalar fitting score averaged over all gammas.

    #Score per gamma:
       # median(RMS) + alpha * p90(RMS)
        #+ beta * frac_skipped
        #+ mu * frac_failed

    #Lower is better.

    rms = compute_cdf_rms_fit_quality(
        fit_parameters=fit_parameters,
        king_pdf=king_pdf,
        minimum_counts=min_counts,
    )

    fit_quality = np.asarray(fit_parameters["fit_quality"])
    event_counts = np.asarray(fit_parameters["event_counts"])

    gamma_scores = []

    diagnostics = []

    for g in range(rms.shape[0]):
        rms_g = rms[g]
        chi2_g = fit_quality[g]
        counts_g = event_counts[g]

        total_bins = rms_g.size

        valid_rms_mask = np.isfinite(rms_g) & (rms_g > 0)

        if np.sum(valid_rms_mask) == 0:
            gamma_scores.append(np.inf)
            diagnostics.append((g, np.inf, np.nan, np.nan, 1.0, 1.0))
            continue

        valid_rms = rms_g[valid_rms_mask]

        median_rmse = np.median(valid_rms)
        rmse_90_perc = np.percentile(valid_rms, 90)

        # skipped = too few events or no valid RMS
        skipped_mask = counts_g < min_counts
        frac_skipped = np.sum(skipped_mask) / total_bins

        # failed = enough events, but invalid RMS or pathological chi2
        failed_mask = (
            (counts_g >= min_counts)
            & (
                ~valid_rms_mask
                | (chi2_g >= chi2_failure_threshold)
                | ~np.isfinite(chi2_g)
            )
        )
        frac_failed = np.sum(failed_mask) / total_bins

        score_g = (
            median_rmse
            + alpha * rmse_90_perc
            + beta * frac_skipped
            + mu * frac_failed
        )

        gamma_scores.append(score_g)

        diagnostics.append(
            (
                g,
                score_g,
                median_rmse,
                rmse_90_perc,
                frac_skipped,
                frac_failed,
            )
        )

    score = float(np.mean(gamma_scores))

    if verbose:
        print("\n========== FITTING SCORE ==========")
        print(f"min_counts       : {min_counts}")
        print(f"alpha            : {alpha}")
        print(f"beta             : {beta}")
        print(f"mu               : {mu}")
        print("-----------------------------------")

        for (
            g,
            score_g,
            median_rmse,
            rmse_90_perc,
            frac_skipped,
            frac_failed,
        ) in diagnostics:
            print(f"gamma index {g}:")
            print(f"  score          : {score_g:.6g}")
            print(f"  median_rmse    : {median_rmse:.6g}")
            print(f"  rms_90_percent : {rmse_90_perc:.6g}")
            print(f"  frac_skipped   : {frac_skipped:.6g}")
            print(f"  frac_failed    : {frac_failed:.6g}")

        print("-----------------------------------")
        print(f"mean score       : {score:.6g}")
        print("===================================\n")

    return score"""
    
def fitting_score_components(
    fit_parameters,
    rms,
    min_counts,
    alpha=1.0,
    beta=0.1,
    mu=2.0,
    median_rms_weight = 0.3,
    chi2_failure_threshold=1e4,
):
    fit_quality = np.asarray(fit_parameters["fit_quality"])
    event_counts = np.asarray(fit_parameters["event_counts"])

    gamma_scores = []

    valid_rms = np.isfinite(rms) & (rms > 0)
    skipped = event_counts < min_counts
    failed = (
        (event_counts >= min_counts)
        & (
            ~valid_rms
            | ~np.isfinite(fit_quality)
            | (fit_quality >= chi2_failure_threshold)
        )
    )

    for g in range(rms.shape[0]):
        rms_g = rms[g]
        valid_g = valid_rms[g]

        if np.sum(valid_g) == 0:
            gamma_scores.append(np.inf)
            continue

        median_rms = np.median(rms_g[valid_g])
        p90_rms = np.percentile(rms_g[valid_g], 90)
        frac_skipped = np.mean(skipped[g])
        frac_failed = np.mean(failed[g])

        score_g = (
            median_rms_weight * median_rms
            + alpha * p90_rms
            + beta * frac_skipped
            + mu * frac_failed
        )

        gamma_scores.append(score_g)

    return {
        "score": float(np.mean(gamma_scores)),
        "score_by_gamma": [float(x) for x in gamma_scores],
        "median_rms": float(np.nanmedian(rms[valid_rms])),
        "p90_rms": float(np.nanpercentile(rms[valid_rms], 90)),
        "frac_skipped": float(np.mean(skipped)),
        "frac_failed": float(np.mean(failed)),
    }
    
def total_score(king_flux: np.ndarray, rayleigh_flux: np.ndarray)-> float:
    """
    Calculate total score by averaging sensitivity flux ratio of king to rayleigh benchmark. 
    """
    score = np.mean(king_flux / rayleigh_flux)
    
    return score