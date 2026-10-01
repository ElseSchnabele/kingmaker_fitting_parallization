from typing import Optional, Tuple, Dict, Any, Literal, Callable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import histlite as hl

from kingmaker_fork.likelihood_analysis.analysis_helpers.rms_fit_quality import cdf_rms
from kingmaker_fork.likelihood_analysis.analysis_helpers.rms_fit_quality import cdf_root_median_squared
from kingmaker_fork.likelihood_analysis.analysis_helpers.rms_fit_quality import weighted_cdf_rms
from kingmaker_fork.likelihood_analysis.analysis_helpers.rms_fit_quality import density_weighted_cdf_rms
from kingmaker_fork.likelihood_analysis.analysis_helpers.rms_fit_quality import pdf_rms
from kingmaker_fork.likelihood_analysis.analysis_helpers.rms_fit_quality import pdf_root_median_squared
from kingmaker_fork.likelihood_analysis.analysis_helpers.rms_fit_quality import weighted_pdf_rms
from kingmaker_fork.likelihood_analysis.analysis_helpers.rms_fit_quality import density_weighted_pdf_rms


def _get_quality_array(fit_parameters, gamma_idx=0, quality_key="fit_quality"):
    quality = np.asarray(fit_parameters[quality_key][gamma_idx]).flatten()
    label = r"$\chi^2$ value" if quality_key == "fit_quality" else "CDF RMS residual"
    return quality, label


def plot_fit_quality_hist(
    fit_parameters: Dict[str, Any],
    gamma_idx: int = 0,
    range_max: float = 500,
    quality_key: str = "fit_quality",
):
    fit_quality, xlabel = _get_quality_array(fit_parameters, gamma_idx, quality_key)
    plot_fit_quality = fit_quality[np.isfinite(fit_quality) & (fit_quality > 0)]

    plt.figure(figsize=(8, 4))
    plt.grid(zorder=1)

    plt.hist(plot_fit_quality, bins=50, zorder=2, range=(0, range_max))

    plt.axvline(
        np.median(plot_fit_quality),
        label=f"Median = {np.median(plot_fit_quality):.3g}",
        color="red",
        linestyle="dashed",
        zorder=3,
    )

    plt.ylabel("Frequency")
    plt.xlabel(xlabel)
    plt.xlim(0, range_max)
    plt.legend()
    plt.show()


def plot_fit_quality_vs_normalized_bin_volume(
    fit_parameters,
    gamma_idx=0,
    n_x_bins=30,
    figsize=(8, 5),
    quality_key="fit_quality",
):
    parametrization_bins = fit_parameters["parametrization_bins"]
    if isinstance(parametrization_bins, np.ndarray):
        parametrization_bins = parametrization_bins.item()

    fit_quality, ylabel_base = _get_quality_array(fit_parameters, gamma_idx, quality_key)
    event_counts = np.asarray(fit_parameters["event_counts"][gamma_idx]).flatten()

    bin_names = list(parametrization_bins.keys())
    shape = fit_parameters[quality_key][gamma_idx].shape

    volumes = np.zeros(shape)

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

    volumes = volumes.flatten()

    mask = (
        np.isfinite(fit_quality)
        & np.isfinite(volumes)
        & np.isfinite(event_counts)
        & (fit_quality > 0)
        & (volumes > 0)
        & (event_counts > 0)
    )

    fit_quality = fit_quality[mask]
    volumes = volumes[mask]

    x_bins = np.logspace(np.log10(volumes.min()), np.log10(volumes.max()), n_x_bins)

    bin_centers = []
    median_fit_quality = []

    for low, high in zip(x_bins[:-1], x_bins[1:]):
        m = (volumes >= low) & (volumes < high)
        if np.sum(m) == 0:
            continue

        bin_centers.append(np.sqrt(low * high))
        median_fit_quality.append(np.median(fit_quality[m]))

    plt.figure(figsize=figsize)
    plt.plot(bin_centers, median_fit_quality, marker="o")

    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel("Normalized parameter-space bin volume")
    plt.ylabel(f"Median {ylabel_base}")

    plt.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.show()


def plot_fit_quality_vs_counts(
    fit_parameters,
    gamma_idx=0,
    n_x_bins=30,
    figsize=(8, 5),
    quality_key="fit_quality",
):
    fit_quality, ylabel_base = _get_quality_array(fit_parameters, gamma_idx, quality_key)
    event_counts = np.asarray(fit_parameters["event_counts"][gamma_idx]).flatten()

    mask = (
        np.isfinite(fit_quality)
        & np.isfinite(event_counts)
        & (fit_quality > 0)
        & (event_counts > 0)
    )

    fit_quality = fit_quality[mask]
    event_counts = event_counts[mask]

    x_bins = np.logspace(
        np.log10(event_counts.min()),
        np.log10(event_counts.max()),
        n_x_bins,
    )

    bin_centers = []
    median_fit_quality = []

    for low, high in zip(x_bins[:-1], x_bins[1:]):
        m = (event_counts >= low) & (event_counts < high)
        if np.sum(m) == 0:
            continue

        bin_centers.append(np.sqrt(low * high))
        median_fit_quality.append(np.median(fit_quality[m]))

    plt.figure(figsize=figsize)
    plt.plot(bin_centers, median_fit_quality, marker="o")

    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel("Number of events in fit bin")
    plt.ylabel(f"Median {ylabel_base}")

    plt.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.show()
    
def plot_quality_heatmap_counts_vs_volume(
    fit_parameters,
    gamma_idx=0,
    quality_key="fit_rms",  # "fit_rms" or "fit_quality"
    n_count_bins=30,
    n_volume_bins=30,
    figsize=(8, 6),
):
    parametrization_bins = fit_parameters["parametrization_bins"]
    if isinstance(parametrization_bins, np.ndarray):
        parametrization_bins = parametrization_bins.item()

    quality = np.asarray(fit_parameters[quality_key][gamma_idx]).flatten()
    event_counts = np.asarray(fit_parameters["event_counts"][gamma_idx]).flatten()

    bin_names = list(parametrization_bins.keys())
    shape = fit_parameters[quality_key][gamma_idx].shape

    volumes = np.zeros(shape)

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

    volumes = volumes.flatten()

    mask = (
        np.isfinite(quality)
        & np.isfinite(event_counts)
        & np.isfinite(volumes)
        & (quality > 0)
        & (event_counts > 0)
        & (volumes > 0)
    )

    quality = quality[mask]
    event_counts = event_counts[mask]
    volumes = volumes[mask]

    count_bins = np.logspace(
        np.log10(event_counts.min()),
        np.log10(event_counts.max()),
        n_count_bins + 1,
    )

    volume_bins = np.logspace(
        np.log10(volumes.min()),
        np.log10(volumes.max()),
        n_volume_bins + 1,
    )

    heatmap = np.full((n_volume_bins, n_count_bins), np.nan)

    for i in range(n_volume_bins):
        for j in range(n_count_bins):
            m = (
                (volumes >= volume_bins[i])
                & (volumes < volume_bins[i + 1])
                & (event_counts >= count_bins[j])
                & (event_counts < count_bins[j + 1])
            )

            if np.any(m):
                heatmap[i, j] = np.median(quality[m])

    fig, ax = plt.subplots(figsize=figsize)

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")

    pcm = ax.pcolormesh(
        count_bins,
        volume_bins,
        np.ma.masked_invalid(np.log10(heatmap)),
        shading="auto",
        cmap=cmap,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel("Number of events in fit bin")
    ax.set_ylabel("Normalized parameter-space bin volume")

    label = r"Median $\chi^2$" if quality_key == "fit_quality" else "Median CDF RMS residual"
    cbar = plt.colorbar(pcm, ax=ax)
    cbar.set_label(f"log10({label})")

    ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.show()

    return ax

def plot_grid_fit_2d(
    fit_parameters: Dict[str, Any],
    x_name: str,
    y_name: str,
    values: np.ndarray,
    title: str,
    cbar_label: str,
    cmap: str = "viridis",
    gamma_index: int = 0,
    y_to_degrees: bool = False,
    value_to_degrees: bool = False,
    log_values: bool = False,
    log_y_scale: bool = False,
    mark_skipped: bool = True,
    mark_failed: bool = True,
    hidden_slices: Optional[dict] = None,
    minimum_counts: int = 100,
    figsize: Tuple[float, float] = (10, 6),
    
):
    parametrization_bins = fit_parameters["parametrization_bins"]

    if isinstance(parametrization_bins, np.ndarray):
        parametrization_bins = parametrization_bins.item()

    bin_names = list(parametrization_bins.keys())

    if x_name not in bin_names:
        raise ValueError(f"{x_name=} not found in bin_names={bin_names}")

    if y_name not in bin_names:
        raise ValueError(f"{y_name=} not found in bin_names={bin_names}")

    values_plot = values.astype(float).copy()

    if value_to_degrees:
        values_plot = np.degrees(values_plot)

    if log_values:
        values_plot[values_plot <= 0] = np.nan
        values_plot = np.log10(values_plot)

    x_dim = bin_names.index(x_name)
    y_dim = bin_names.index(y_name)

    if hidden_slices is None:
        hidden_slices = {}

    slicer = []
    slice_label_parts = []

    for name in bin_names:
        if name in {x_name, y_name}:
            slicer.append(slice(None))
        else:
            bins = parametrization_bins[name]
            idx = hidden_slices.get(name, len(bins) // 2 - 1)

            if idx < 0 or idx >= len(bins) - 1:
                raise ValueError(
                    f"Invalid slice index {idx} for {name}. "
                    f"Allowed range: 0 to {len(bins) - 2}"
                )

            slicer.append(idx)

            low = bins[idx]
            high = bins[idx + 1]

            if name in {"zen", "dec", "angErr"}:
                low_label = np.degrees(low)
                high_label = np.degrees(high)
                unit = "°"
            else:
                low_label = low
                high_label = high
                unit = ""

            slice_label_parts.append(
                f"{name}=[{low_label:.3g}, {high_label:.3g}]{unit}"
            )

    values_2d = values_plot[tuple(slicer)]

    current_remaining_dims = [
        dim for dim, name in enumerate(bin_names)
        if name in {x_name, y_name}
    ]

    if current_remaining_dims == [y_dim, x_dim]:
        values_2d = values_2d.T

    x_bins = parametrization_bins[x_name]
    y_bins = parametrization_bins[y_name]
    y_edges = np.degrees(y_bins) if y_to_degrees else y_bins
    
    # Mask skipped/failed bins so they appear white and do not affect color scaling
    event_counts = fit_parameters["event_counts"][gamma_index]
    fit_quality = fit_parameters["fit_quality"][gamma_index]

    invalid_mask = (
        (event_counts < minimum_counts)
        | ((event_counts >= minimum_counts) & (fit_quality <= 0))
    )

    invalid_2d = invalid_mask[tuple(slicer)]

    if current_remaining_dims == [y_dim, x_dim]:
        invalid_2d = invalid_2d.T

    values_2d = np.ma.masked_where(invalid_2d, values_2d)

    # make masked regions white
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="white")

    fig, ax = plt.subplots(figsize=figsize)

    pcm = ax.pcolormesh(
        x_bins,
        y_edges,
        values_2d.T,
        shading="auto",
        cmap=cmap_obj,
    )

    plt.colorbar(pcm, ax=ax, label=cbar_label)

    if mark_skipped or mark_failed:
        event_counts = fit_parameters["event_counts"][gamma_index]
        fit_quality = fit_parameters["fit_quality"][gamma_index]

        skipped_mask = event_counts < minimum_counts
        failed_mask = (event_counts >= minimum_counts) & (fit_quality <= 0)

        skipped_2d = skipped_mask[tuple(slicer)]
        failed_2d = failed_mask[tuple(slicer)]

        if current_remaining_dims == [y_dim, x_dim]:
            skipped_2d = skipped_2d.T
            failed_2d = failed_2d.T

        def mark_bins(mask_2d, color, hatch, label):
            label_added = False

            for i in range(mask_2d.shape[0]):
                for j in range(mask_2d.shape[1]):
                    if not mask_2d[i, j]:
                        continue

                    rect = patches.Rectangle(
                        (x_bins[i], y_edges[j]),
                        x_bins[i + 1] - x_bins[i],
                        y_edges[j + 1] - y_edges[j],
                        linewidth=1.5,
                        edgecolor=color,
                        facecolor="none",
                        hatch=hatch,
                        label=label if not label_added else None,
                    )
                    ax.add_patch(rect)
                    label_added = True

        if mark_skipped:
            mark_bins(skipped_2d, color="red", hatch="//", label="Skipped")

        if mark_failed:
            mark_bins(failed_2d, color="cyan", hatch="\\\\", label="Failed fit")

    ax.set_xlabel(x_name)
    ax.set_ylabel(f"{y_name} (°)" if y_to_degrees else y_name)

    if log_y_scale:
        ax.set_yscale("log")

    if slice_label_parts:
        ax.set_title(title + "\nSlice: " + ", ".join(slice_label_parts))
    else:
        ax.set_title(title)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right", fontsize=9)

    ax.margins(0)
    plt.tight_layout()
    plt.show()

    return ax


def plot_TS_signal_ray_vs_king(
                            src_sin_dec: float,
                            ray_bkg, 
                            king_bkg, 
                            ray_sig = None, 
                            king_sig = None, 
                            n_sig_ray:int = 0, 
                            n_sig_king:int = 0,
                            figsize: Tuple[float, float] = (8, 6),
                               ):
        fig, ax = plt.subplots(figsize = figsize)
        #king bkg
        h_king = king_bkg.get_hist(bins= 50).normalize()
        hl.plot1d(
            ax,
            h_king,
            color = 'blue',
            crosses=True,
            label=f"Background TS King"
        )

        x_king = h_king.centers[0]
        norm_king = h_king.integrate().values
        ax.grid(which = 'both', alpha = 0.5)
        ax.semilogy(
            x_king,
            norm_king * king_bkg.pdf(x_king),
            lw=1,
            ls="--",
            color = 'black',
        )
        ax.axvline(king_bkg.median(), color = 'red', linestyle = 'dotted')
        
        # ray_bkg
        h_ray = ray_bkg.get_hist(bins= 50).normalize()
        hl.plot1d(
            ax,
            h_ray,
            color = 'red',
            crosses=True,
            label=f"Background TS Rayleigh"
        )

        x_ray = h_ray.centers[0]
        norm_ray = h_ray.integrate().values
        ax.grid(which = 'both', alpha = 0.5)
        ax.semilogy(
            x_ray,
            norm_ray * ray_bkg.pdf(x_ray),
            lw=1,
            ls="--",
            color = 'black',
        )
        ax.axvline(ray_bkg.median(), color = 'red', linestyle = 'dotted')
        
        #king signal injected
        if king_sig is not None:
            h_king_sig = king_sig.get_hist(bins= 50).normalize()
            hl.plot1d(
                ax,
                h_king_sig,
                color = 'C0',
                crosses=True,
                label=r"Signal TS King $n_{inj} = $" + str(n_sig_king)
            )

            x_king_sig = h_king_sig.centers[0]
            norm_king_sig = h_king_sig.integrate().values
            ax.grid(which = 'both', alpha = 0.5)
            ax.semilogy(
                x_king_sig,
                norm_king_sig * king_sig.pdf(x_king_sig),
                lw=1,
                ls="--",
                color = 'black',
            )       
            ax.axvline(king_sig.median(), color = 'C0', linestyle = 'dotted')  
            
        if ray_sig is not None:
            h_ray_sig = ray_sig.get_hist(bins= 50).normalize()
            hl.plot1d(
                ax,
                h_ray_sig,
                color = 'C1',
                crosses=True,
                label=r"Signal TS Rayleigh $n_{inj} = $" + str(n_sig_king)
            )

            x_ray_sig = h_ray_sig.centers[0]
            norm_ray_sig = h_ray_sig.integrate().values
            ax.grid(which = 'both', alpha = 0.5)
            ax.semilogy(
                x_ray_sig,
                norm_ray_sig * ray_sig.pdf(x_ray_sig),
                lw=1,
                ls="--",
                color = 'black',
            )       
            ax.axvline(ray_sig.median(), color = 'C1', linestyle = 'dotted')  
        
        
        
        
        ax.set_xlabel("TS")
        ax.set_ylabel("PDF")
        ax.set_title("$\sin{\delta}$" + f" = {src_sin_dec:.2f}")
        ax.legend()
        plt.tight_layout()
        plt.show()

def plot_selected_fit_bins(
    fit_parameters: Dict[str, Any],
    king_pdf: Callable,
    mode: Literal["best", "worst", "random", "grid"] = "best",
    metric: Literal[
        "CDF_chi2", "CDF_RMS", "CDF_root_median_squared",
        "CDF_weighted_rms", "CDF_density_weighted_rms",
        "PDF_RMS", "PDF_root_median_squared",
        "PDF_weighted_rms", "PDF_density_weighted_rms",
    ] = "CDF_chi2",
    gamma_idx: int = 0,
    n_plots: int = 6,
    minimum_counts: int = 100,
    random_seed: Optional[int] = None,
    x_log: bool = True,
    y_log: bool = True,
    normalize_fit_to_hist: bool = True,
    show_rayleigh_ref: bool = False,
    plot_kind: Literal["density", "radial"] = "density",
    curve_kind: Literal["pdf", "cdf", "both"] = "pdf",
    figsize: Tuple[float, float] = (15, 10),

    # Rayleigh reference options
    signal_events: Optional[Any] = None,
    rayleigh_sigma_name: str = "sigma",
    rayleigh_weighted: bool = False,
    weight_field: Optional[str] = None,
    true_energy_name: str = "true_energy",
    show_data: bool = True,
    selected_bin_idx: Optional[Tuple[int, ...]] = None,
    x_lim: Optional[float] = 3.0,
):
    """
    Plot selected stored King fits.

    Parameters
    ----------
    curve_kind : {"pdf", "cdf", "both"}
        "pdf"
            Plot histogram/density and fitted King PDF.

        "cdf"
            Plot empirical MC CDF and fitted King CDF.

        "both"
            Plot PDF and CDF next to each other for every selected
            parametrization bin.

    plot_kind : {"density", "radial"}
        Only affects PDF plots.

        "density"
            Plot density per steradian.

        "radial"
            Plot radial PDF = 2*pi*sin(psi) * density.

        CDF plots are independent of plot_kind.
    """

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    if curve_kind not in {"pdf", "cdf", "both"}:
        raise ValueError(
            "curve_kind must be one of: 'pdf', 'cdf', 'both'"
        )

    if plot_kind not in {"density", "radial"}:
        raise ValueError(
            "plot_kind must be either 'density' or 'radial'"
        )

    # ---------------------------------------------------------
    # Rayleigh helpers
    # ---------------------------------------------------------

    def rayleigh_density(dpsi, sigma):
        return (
            np.exp(-0.5 * (dpsi / sigma) ** 2)
            / (2 * np.pi * sigma**2)
        )

    def rayleigh_radial(dpsi, sigma):
        return (
            (dpsi / sigma**2)
            * np.exp(-0.5 * (dpsi / sigma) ** 2)
        )

    def rayleigh_cdf(dpsi, sigma):
        return 1.0 - np.exp(
            -0.5 * (dpsi / sigma) ** 2
        )

    # ---------------------------------------------------------
    # Event helpers
    # ---------------------------------------------------------

    def get_event_values(events, name):
        if hasattr(events, name):
            return getattr(events, name)
        return events[name]

    def get_bin_event_mask(
        events,
        parametrization_bins,
        bin_names,
        bin_idx,
    ):
        mask = np.ones(len(events), dtype=bool)

        for dim, name in enumerate(bin_names):
            edges = np.asarray(parametrization_bins[name])
            idx = bin_idx[dim]

            vals = np.asarray(
                get_event_values(events, name)
            )

            if idx == len(edges) - 2:
                mask &= vals >= edges[idx]
                mask &= vals <= edges[idx + 1]
            else:
                mask &= vals >= edges[idx]
                mask &= vals < edges[idx + 1]

        return mask

    if show_rayleigh_ref and signal_events is None:
        raise ValueError(
            "show_rayleigh_ref=True requires signal_events, "
            "i.e. the same MC events used to build the King "
            "fit histograms."
        )

    # ---------------------------------------------------------
    # Stored fit data
    # ---------------------------------------------------------

    parametrization_bins = fit_parameters[
        "parametrization_bins"
    ]

    if isinstance(parametrization_bins, np.ndarray):
        parametrization_bins = parametrization_bins.item()

    bin_names = list(parametrization_bins.keys())

    fit_alpha = fit_parameters["alpha"]
    fit_beta = fit_parameters["beta"]
    histograms = fit_parameters["histograms"]
    uncertainties = fit_parameters["uncertainties"]
    dpsi_bins = fit_parameters["dpsi_bins"]
    fit_quality = fit_parameters["fit_quality"]
    event_counts = fit_parameters["event_counts"]

    # ---------------------------------------------------------
    # Metric helper
    # ---------------------------------------------------------

    def get_metric_function(metric_name):

        if metric_name == "CDF_RMS":
            return cdf_rms

        elif metric_name == "CDF_root_median_squared":
            return cdf_root_median_squared

        elif metric_name == "CDF_weighted_rms":
            return weighted_cdf_rms

        elif metric_name == "CDF_density_weighted_rms":
            return density_weighted_cdf_rms

        elif metric_name == "PDF_RMS":
            return pdf_rms

        elif metric_name == "PDF_root_median_squared":
            return pdf_root_median_squared

        elif metric_name == "PDF_weighted_rms":
            return weighted_pdf_rms

        elif metric_name == "PDF_density_weighted_rms":
            return density_weighted_pdf_rms

        else:
            raise ValueError(
                f"Unknown metric: {metric_name}"
            )

    # ---------------------------------------------------------
    # Select bins
    # ---------------------------------------------------------

    good_bins = []

    if selected_bin_idx is not None:

        selected_bin_idx = tuple(selected_bin_idx)

        if len(selected_bin_idx) != len(bin_names):
            raise ValueError(
                f"selected_bin_idx must have length "
                f"{len(bin_names)} for dimensions {bin_names}, "
                f"got {len(selected_bin_idx)}."
            )

        param_idx = (
            gamma_idx,
        ) + selected_bin_idx

        if not np.all(
            np.asarray(selected_bin_idx) >= 0
        ):
            raise ValueError(
                f"selected_bin_idx must be non-negative, "
                f"got {selected_bin_idx}."
            )

        if any(
            i >= s
            for i, s in zip(
                selected_bin_idx,
                fit_alpha[gamma_idx].shape,
            )
        ):
            raise ValueError(
                f"selected_bin_idx {selected_bin_idx} "
                f"out of range for shape "
                f"{fit_alpha[gamma_idx].shape}."
            )

        if event_counts[param_idx] < minimum_counts:
            raise RuntimeError(
                f"Selected bin {selected_bin_idx} has only "
                f"{event_counts[param_idx]} events "
                f"(minimum_counts={minimum_counts})."
            )

        if not np.any(
            histograms[param_idx] > 0
        ):
            raise RuntimeError(
                f"Selected bin {selected_bin_idx} "
                "has no histogram entries."
            )

        if metric == "CDF_chi2":
            selected_quality = fit_quality[param_idx]

        else:
            func = get_metric_function(metric)

            selected_quality = func(
                fit_parameters=fit_parameters,
                gamma_idx=gamma_idx,
                minimum_counts=minimum_counts,
                king_pdf=king_pdf,
            )[param_idx]

        selected_bins = [
            (
                selected_bin_idx,
                selected_quality,
            )
        ]

    # ---------------------------------------------------------
    # Search all valid bins
    # ---------------------------------------------------------

    for bin_idx in np.ndindex(
        *fit_alpha[gamma_idx].shape
    ):

        param_idx = (
            gamma_idx,
        ) + bin_idx

        if (
            event_counts[param_idx] >= minimum_counts
            and np.any(histograms[param_idx] > 0)
        ):

            if metric == "CDF_chi2":
                quality_metric = fit_quality[param_idx]

            else:
                func = get_metric_function(metric)

                quality_metric = func(
                    fit_parameters=fit_parameters,
                    gamma_idx=gamma_idx,
                    minimum_counts=minimum_counts,
                    king_pdf=king_pdf,
                )[param_idx]

            if (
                np.isfinite(quality_metric)
                and quality_metric > 0
            ):
                good_bins.append(
                    (
                        bin_idx,
                        quality_metric,
                    )
                )

    if selected_bin_idx is None:

        if len(good_bins) == 0:
            raise RuntimeError(
                "No valid fitted bins found."
            )

        if mode == "best":

            selected_bins = sorted(
                good_bins,
                key=lambda x: x[1],
            )[:n_plots]

        elif mode == "worst":

            selected_bins = sorted(
                good_bins,
                key=lambda x: x[1],
                reverse=True,
            )[:n_plots]

        elif mode == "random":

            rng = np.random.default_rng(
                random_seed
            )

            indices = rng.choice(
                len(good_bins),
                size=min(
                    n_plots,
                    len(good_bins),
                ),
                replace=False,
            )

            selected_bins = [
                good_bins[i]
                for i in indices
            ]

        elif mode == "grid":

            good_bin_indices = np.asarray(
                [b[0] for b in good_bins]
            )

            good_quality = np.asarray(
                [b[1] for b in good_bins]
            )

            n_select = min(
                n_plots,
                len(good_bins),
            )

            shape = np.asarray(
                fit_alpha[gamma_idx].shape
            )

            denom = np.maximum(
                shape - 1,
                1,
            )

            good_pos = (
                good_bin_indices / denom
            )

            targets_1d = np.linspace(
                0,
                1,
                n_select,
            )

            target_pos = np.repeat(
                targets_1d[:, None],
                good_pos.shape[1],
                axis=1,
            )

            selected_bins = []
            used = set()

            for target in target_pos:

                distances = np.linalg.norm(
                    good_pos - target,
                    axis=1,
                )

                for idx in np.argsort(
                    distances
                ):

                    bin_tuple = tuple(
                        good_bin_indices[idx]
                    )

                    if bin_tuple not in used:

                        selected_bins.append(
                            (
                                bin_tuple,
                                good_quality[idx],
                            )
                        )

                        used.add(bin_tuple)
                        break

        else:
            raise ValueError(
                "mode must be one of: "
                "'best', 'worst', 'random', 'grid'"
            )

    # ---------------------------------------------------------
    # Figure layout
    # ---------------------------------------------------------

    n_selected = len(selected_bins)

    if curve_kind == "both":

        # One selected bin per row:
        #
        #   PDF | CDF
        #
        n_rows = n_selected
        n_cols = 2

        if figsize == (15, 10):
            figsize_use = (
                14,
                max(
                    4,
                    4 * n_selected,
                ),
            )
        else:
            figsize_use = figsize

    else:

        n_cols = (
            1
            if selected_bin_idx is not None
            else min(
                3,
                n_selected,
            )
        )

        n_rows = int(
            np.ceil(
                n_selected / n_cols
            )
        )

        figsize_use = figsize

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize_use,
        squeeze=False,
    )

    # ---------------------------------------------------------
    # Console output
    # ---------------------------------------------------------

    print(
        f"Found {len(good_bins)} bins "
        "with stored fits"
    )

    print(
        f"Showing {mode} bins:"
    )

    for i, (
        bin_idx,
        quality,
    ) in enumerate(selected_bins):

        print(
            f"{i + 1}: "
            f"bin={bin_idx}, "
            f"{metric}={quality:.3f}"
        )

    # ---------------------------------------------------------
    # Plotting helper
    # ---------------------------------------------------------

    def plot_single_curve(
        ax,
        curve_type,
        bin_idx,
        quality,
    ):

        param_idx = (
            gamma_idx,
        ) + bin_idx

        hist = np.asarray(
            histograms[param_idx],
            dtype=float,
        )

        uncertainty = np.asarray(
            uncertainties[param_idx],
            dtype=float,
        )

        bins = np.asarray(
            dpsi_bins[param_idx],
            dtype=float,
        )

        mask = hist > 0

        if np.sum(mask) == 0:

            ax.text(
                0.5,
                0.5,
                f"Bin {bin_idx}\nNo histogram",
                ha="center",
                va="center",
            )

            ax.set_axis_off()
            return

        bin_centers_all = (
            bins[:-1] + bins[1:]
        ) / 2

        bin_centers = (
            bin_centers_all[mask]
        )

        alpha = fit_alpha[param_idx]
        beta = fit_beta[param_idx]

        # Use the last actually occupied bin.
        valid_indices = np.flatnonzero(mask)

        last_valid_idx = valid_indices[-1]

        dpsi_max = bins[
            last_valid_idx + 1
        ]

        eps = 1e-6

        dpsi_fine = np.geomspace(
            eps,
            dpsi_max,
            1000,
        )

        dpsi_fine = np.insert(
            dpsi_fine,
            0,
            0.0,
        )

        # =====================================================
        # PDF
        # =====================================================

        if curve_type == "pdf":

            if plot_kind == "radial":

                jac_hist = (
                    2
                    * np.pi
                    * np.sin(
                        bin_centers
                    )
                )

                hist_plot = (
                    hist[mask]
                    * jac_hist
                )

                uncertainty_plot = (
                    uncertainty[mask]
                    * jac_hist
                )

                fit_curve = np.asarray(
                    king_pdf.pdf(
                        dpsi_fine,
                        alpha,
                        beta,
                    ),
                    dtype=float,
                )

                fit_curve = (
                    2
                    * np.pi
                    * np.sin(
                        dpsi_fine
                    )
                    * fit_curve
                )

            else:

                hist_plot = hist[mask]

                uncertainty_plot = (
                    uncertainty[mask]
                )

                fit_curve = np.asarray(
                    king_pdf.pdf(
                        dpsi_fine,
                        alpha,
                        beta,
                    ),
                    dtype=float,
                )

            if normalize_fit_to_hist:

                max_fit = np.nanmax(
                    fit_curve
                )

                max_hist = np.nanmax(
                    hist_plot
                )

                if (
                    np.isfinite(max_fit)
                    and max_fit > 0
                    and np.isfinite(max_hist)
                ):
                    fit_curve = (
                        fit_curve
                        * max_hist
                        / max_fit
                    )

            x_data = bin_centers

        # =====================================================
        # CDF
        # =====================================================

        elif curve_type == "cdf":

            # Work only with occupied histogram bins.
            bin_low = bins[:-1][mask]
            bin_high = bins[1:][mask]

            # Exact solid angle of spherical annulus:
            #
            # ΔΩ = 2π [cos(theta_low)-cos(theta_high)]
            #
            solid_angle = (
                2
                * np.pi
                * (
                    np.cos(bin_low)
                    - np.cos(bin_high)
                )
            )

            probability_per_bin = (
                hist[mask]
                * solid_angle
            )

            total_probability = np.sum(
                probability_per_bin
            )

            if (
                not np.isfinite(
                    total_probability
                )
                or total_probability <= 0
            ):
                raise RuntimeError(
                    f"Cannot construct CDF for "
                    f"bin {bin_idx}: histogram "
                    "integral is not positive."
                )

            hist_plot = (
                np.cumsum(
                    probability_per_bin
                )
                / total_probability
            )

            probability_uncertainty = (
                uncertainty[mask]
                * solid_angle
            )

            uncertainty_plot = (
                np.sqrt(
                    np.cumsum(
                        probability_uncertainty**2
                    )
                )
                / total_probability
            )

            # CDF point corresponds to cumulative
            # probability through upper edge.
            x_data = bin_high

            fit_curve = np.asarray(
                king_pdf.cdf(
                    dpsi_fine,
                    alpha,
                    beta,
                ),
                dtype=float,
            )

        else:
            raise ValueError(
                f"Unknown curve type: "
                f"{curve_type}"
            )

        # -----------------------------------------------------
        # MC data
        # -----------------------------------------------------

        ax.errorbar(
            np.degrees(x_data),
            hist_plot,
            yerr=uncertainty_plot,
            fmt="o",
            label="MC Events",
            color="black",
            markersize=4,
        )

        # -----------------------------------------------------
        # King fit
        # -----------------------------------------------------

        ax.plot(
            np.degrees(
                dpsi_fine
            ),
            fit_curve,
            "-",
            linewidth=2,
            label=(
                f"King "
                f"{curve_type.upper()}"
            ),
            color="blue",
        )

        # -----------------------------------------------------
        # Rayleigh reference
        # -----------------------------------------------------

        if show_rayleigh_ref:

            event_mask = (
                get_bin_event_mask(
                    signal_events,
                    parametrization_bins,
                    bin_names,
                    bin_idx,
                )
            )

            sigmas = np.asarray(
                get_event_values(
                    signal_events,
                    rayleigh_sigma_name,
                )[event_mask],
                dtype=float,
            )

            valid_sigma = (
                np.isfinite(sigmas)
                & (sigmas > 0)
            )

            sigmas = sigmas[
                valid_sigma
            ]

            if len(sigmas) > 0:

                if curve_type == "cdf":

                    rayleigh_curves = (
                        rayleigh_cdf(
                            dpsi_fine[
                                :,
                                None,
                            ],
                            sigmas[
                                None,
                                :,
                            ],
                        )
                    )

                else:

                    if plot_kind == "radial":

                        rayleigh_curves = (
                            rayleigh_radial(
                                dpsi_fine[
                                    :,
                                    None,
                                ],
                                sigmas[
                                    None,
                                    :,
                                ],
                            )
                        )

                    else:

                        rayleigh_curves = (
                            rayleigh_density(
                                dpsi_fine[
                                    :,
                                    None,
                                ],
                                sigmas[
                                    None,
                                    :,
                                ],
                            )
                        )

                # ---------------------------------------------
                # Weighted Rayleigh average
                # ---------------------------------------------

                if rayleigh_weighted:

                    if weight_field is None:

                        raise ValueError(
                            "rayleigh_weighted=True "
                            "requires weight_field, "
                            "for example "
                            "weight_field='ow' "
                            "or 'oneweight'."
                        )

                    gamma = (
                        fit_parameters.get(
                            "spectral_indices",
                            None,
                        )
                    )

                    if gamma is not None:

                        gamma = np.asarray(
                            gamma
                        )[gamma_idx]

                    else:

                        raise ValueError(
                            "rayleigh_weighted=True "
                            "requires "
                            "fit_parameters"
                            "['spectral_indices']."
                        )

                    weights = np.asarray(
                        get_event_values(
                            signal_events,
                            weight_field,
                        )[event_mask],
                        dtype=float,
                    )

                    true_energy = np.asarray(
                        get_event_values(
                            signal_events,
                            true_energy_name,
                        )[event_mask],
                        dtype=float,
                    )

                    weights = weights[
                        valid_sigma
                    ]

                    true_energy = (
                        true_energy[
                            valid_sigma
                        ]
                    )

                    weights = (
                        weights
                        * true_energy
                        ** (-gamma)
                    )

                    weights = np.asarray(
                        weights,
                        dtype=float,
                    )

                    valid_w = (
                        np.isfinite(weights)
                        & (weights > 0)
                    )

                    if np.any(valid_w):

                        rayleigh_curve = (
                            np.average(
                                rayleigh_curves[
                                    :,
                                    valid_w,
                                ],
                                axis=1,
                                weights=weights[
                                    valid_w
                                ],
                            )
                        )

                    else:

                        rayleigh_curve = (
                            np.nanmean(
                                rayleigh_curves,
                                axis=1,
                            )
                        )

                else:

                    rayleigh_curve = (
                        np.nanmean(
                            rayleigh_curves,
                            axis=1,
                        )
                    )

                # Only PDF curves get peak normalization.
                if (
                    curve_type == "pdf"
                    and normalize_fit_to_hist
                ):

                    max_rayleigh = np.nanmax(
                        rayleigh_curve
                    )

                    max_hist = np.nanmax(
                        hist_plot
                    )

                    if (
                        np.isfinite(
                            max_rayleigh
                        )
                        and max_rayleigh > 0
                        and np.isfinite(
                            max_hist
                        )
                    ):

                        rayleigh_curve = (
                            rayleigh_curve
                            * max_hist
                            / max_rayleigh
                        )

                ax.plot(
                    np.degrees(
                        dpsi_fine
                    ),
                    rayleigh_curve,
                    "--",
                    linewidth=2,
                    label=(
                        f"Rayleigh "
                        f"{curve_type.upper()}"
                    ),
                    color="red",
                )

        # -----------------------------------------------------
        # Title
        # -----------------------------------------------------

        title = (
            f"α={np.degrees(alpha):.3f}°, "
            f"β={beta:.2f}\n"
        )

        for dim, name in enumerate(
            bin_names
        ):

            idx = bin_idx[dim]

            edges = (
                parametrization_bins[
                    name
                ]
            )

            low = edges[idx]
            high = edges[idx + 1]

            if name in {
                "dec",
                "zen",
                "angErr",
                "sigma",
            }:

                low = np.degrees(low)
                high = np.degrees(high)

                title += (
                    f"{name}="
                    f"[{low:.2g}, "
                    f"{high:.2g}]° "
                )

            else:

                title += (
                    f"{name}="
                    f"[{low:.2g}, "
                    f"{high:.2g}] "
                )

        if show_data:
            ax.set_title(
                title,
                fontsize=9,
            )

        # -----------------------------------------------------
        # Axes
        # -----------------------------------------------------

        ax.set_xlabel(
            "Angular Error (degrees)"
        )

        if curve_type == "cdf":

            ax.set_ylabel("CDF")

            # CDFs are naturally linear in y.
            ax.set_yscale("linear")

            ax.set_ylim(
                0,
                1.05,
            )

        else:

            if plot_kind == "radial":
                ax.set_ylabel(
                    "Radial PDF"
                )

            else:
                ax.set_ylabel(
                    "Density"
                )

            if y_log:
                ax.set_yscale(
                    "log"
                )

        if x_log:
            ax.set_xscale(
                "log"
            )

        ax.grid(
            alpha=0.3
        )

        if x_lim is not None:
            ax.set_xlim(
                0,
                x_lim,
            )

        # -----------------------------------------------------
        # Legend
        # -----------------------------------------------------

        handles, labels = (
            ax.get_legend_handles_labels()
        )

        if show_data:

            handles.extend(
                [
                    Line2D(
                        [],
                        [],
                        linestyle="none",
                        label=(
                            f"{metric}: "
                            f"{quality:.4f}"
                        ),
                    ),
                    Line2D(
                        [],
                        [],
                        linestyle="none",
                        label=(
                            "#Events: "
                            f"{event_counts[param_idx]}"
                        ),
                    ),
                ]
            )

        ax.legend(
            handles=handles,
            fontsize=8,
            framealpha=0.9,
        )

    # ---------------------------------------------------------
    # Draw selected bins
    # ---------------------------------------------------------

    if curve_kind == "both":

        for row, (
            bin_idx,
            quality,
        ) in enumerate(
            selected_bins
        ):

            plot_single_curve(
                axes[row, 0],
                "pdf",
                bin_idx,
                quality,
            )

            plot_single_curve(
                axes[row, 1],
                "cdf",
                bin_idx,
                quality,
            )

    else:

        flat_axes = axes.flatten()

        for ax, (
            bin_idx,
            quality,
        ) in zip(
            flat_axes,
            selected_bins,
        ):

            plot_single_curve(
                ax,
                curve_kind,
                bin_idx,
                quality,
            )

        for ax in flat_axes[
            len(selected_bins):
        ]:
            ax.set_axis_off()

    plt.tight_layout()
    plt.show()

    return selected_bins



def plot_event_count_distributions(
    fit_parameters: Dict[str, Any],
    gamma_idx: int = 0,
    minimum_counts: int = 100,
    nbins: int = 40,
    log_x: bool = False,
    log_y: bool = True,
    figsize: Tuple[float, float] = (10, 6),
):
    """
    Plot event-count distributions for skipped, failed, and fitted bins.

    Parameters
    ----------
    fit_parameters:
        Output dictionary from fit_all_bins().
    gamma_idx:
        Spectral index to inspect.
    minimum_counts:
        Minimum counts threshold used during fitting.
    nbins:
        Number of histogram bins.
    log_x:
        Use logarithmic x-axis.
    log_y:
        Use logarithmic y-axis.
    """

    event_counts = fit_parameters["event_counts"][gamma_idx]
    fit_quality = fit_parameters["fit_quality"][gamma_idx]

    # Flatten everything
    counts_flat = event_counts.flatten()
    fitq_flat = fit_quality.flatten()

    # Masks
    skipped_mask = counts_flat < minimum_counts
    failed_mask = (counts_flat >= minimum_counts) & (fitq_flat <= 0)
    fitted_mask = (counts_flat >= minimum_counts) & (fitq_flat > 0)

    # Extract distributions
    counts_skipped = counts_flat[skipped_mask]
    counts_failed = counts_flat[failed_mask]
    counts_fitted = counts_flat[fitted_mask]

    # Remove zeros for log binning
    counts_skipped = counts_skipped[counts_skipped > 0]
    counts_failed = counts_failed[counts_failed > 0]
    counts_fitted = counts_fitted[counts_fitted > 0]

    # Combine all for shared binning
    all_counts = np.concatenate(
        [counts_skipped, counts_failed, counts_fitted]
    )

    if len(all_counts) == 0:
        raise RuntimeError("No positive event counts found.")

    if log_x:
        bins = np.logspace(
            np.log10(all_counts.min()),
            np.log10(all_counts.max()),
            nbins,
        )
    else:
        bins = np.linspace(
            all_counts.min(),
            all_counts.max(),
            nbins,
        )

    # Plot
    plt.figure(figsize=figsize)

    plt.hist(
        counts_skipped,
        bins=bins,
        histtype="step",
        linewidth=2,
        label="Skipped",
        color="red",
    )

    plt.hist(
        counts_failed,
        bins=bins,
        histtype="step",
        linewidth=2,
        label="Failed",
        color="cyan",
    )

    plt.hist(
        counts_fitted,
        bins=bins,
        histtype="step",
        linewidth=2,
        label="Fitted",
        color="lime",
    )

    if log_x:
        plt.xscale("log")

    if log_y:
        plt.yscale("log")

    plt.xlabel("Events per bin")
    plt.ylabel("Number of bins")

    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Debug prints
    n_total = len(counts_flat)

    n_skipped = np.sum(skipped_mask)
    n_failed = np.sum(failed_mask)
    n_fitted = np.sum(fitted_mask)

    print(
        f"Skipped bins: {n_skipped} "
        f"({100 * n_skipped / n_total:.1f}%)"
    )

    print(
        f"Failed bins: {n_failed} "
        f"({100 * n_failed / n_total:.1f}%)"
    )

    print(
        f"Fitted bins: {n_fitted} "
        f"({100 * n_fitted / n_total:.1f}%)"
    )

    return {
        "counts_skipped": counts_skipped,
        "counts_failed": counts_failed,
        "counts_fitted": counts_fitted,
    }
    
def plot_fit_metric_statistics(
    fit_parameters: Dict[str, Any],
    king_pdf: Callable,
    metrics: Optional[list[str]] = None,
    gamma_idx: Optional[int] = None,
    minimum_counts: int = 100,
    bins: int = 50,
    density: bool = False,
    log_x: bool = False,
    log_y: bool = False,
    figsize: Tuple[float, float] = (10, 6),
    show_statistics: bool = True,
):
    """
    Calculate and plot the distributions of fit-quality metrics over all
    valid parametrization bins.

    Parameters
    ----------
    fit_parameters : dict
        Dictionary containing the stored King-fit results.

    king_pdf : object
        King PDF object used by the metric functions.

    metrics : sequence of str, optional
        Metrics to calculate and plot.

        Available:
            "CDF_chi2"
            "CDF_RMS"
            "CDF_root_median_squared"
            "CDF_weighted_rms"
            "CDF_density_weighted_rms"
            "PDF_RMS"
            "PDF_root_median_squared"
            "PDF_weighted_rms"
            "PDF_density_weighted_rms"

        If None, all metrics are calculated and plotted.

    gamma_idx : int or None
        If an integer is given, only that spectral-index slice is included
        in the histogram.

        If None, all gamma indices are included.

        Importantly, the returned arrays always retain the full original
        shape.

    minimum_counts : int
        Minimum number of events required for a bin to be considered valid.

    bins : int
        Number of histogram bins.

    density : bool
        If True, plot normalized histogram densities rather than counts.

    log_x : bool
        Use logarithmic x-axis.

    log_y : bool
        Use logarithmic y-axis.

    figsize : tuple
        Matplotlib figure size.

    show_statistics : bool
        Add median and number of valid bins to the legend.

    Returns
    -------
    metric_values : dict[str, np.ndarray]
        Dictionary containing one full-dimensional array for each requested
        metric.

        Every array has the same shape as fit_parameters["alpha"].

        Invalid bins are represented by np.nan.

        Example
        -------
        values["CDF_RMS"][gamma_idx, i, j, k]

        gives the CDF RMS for exactly the same parametrization bin as

        fit_parameters["alpha"][gamma_idx, i, j, k].
    """

    # ---------------------------------------------------------
    # Available metrics
    # ---------------------------------------------------------

    metric_functions = {
        "CDF_RMS": cdf_rms,
        "CDF_root_median_squared": cdf_root_median_squared,
        "CDF_weighted_rms": weighted_cdf_rms,
        "CDF_density_weighted_rms": density_weighted_cdf_rms,
        "PDF_RMS": pdf_rms,
        "PDF_root_median_squared": pdf_root_median_squared,
        "PDF_weighted_rms": weighted_pdf_rms,
        "PDF_density_weighted_rms": density_weighted_pdf_rms,
    }

    available_metrics = [
        "CDF_chi2",
        *metric_functions.keys(),
    ]

    if metrics is None:
        metrics = available_metrics

    metrics = list(metrics)

    unknown_metrics = [
        metric
        for metric in metrics
        if metric not in available_metrics
    ]

    if unknown_metrics:
        raise ValueError(
            f"Unknown metric(s): {unknown_metrics}\n"
            f"Available metrics are: {available_metrics}"
        )

    # ---------------------------------------------------------
    # Basic stored quantities
    # ---------------------------------------------------------

    alpha = np.asarray(
        fit_parameters["alpha"],
        dtype=float,
    )

    beta = np.asarray(
        fit_parameters["beta"],
        dtype=float,
    )

    event_counts = np.asarray(
        fit_parameters["event_counts"]
    )

    histograms = np.asarray(
        fit_parameters["histograms"]
    )

    # ---------------------------------------------------------
    # Define which fitted bins are valid
    # ---------------------------------------------------------

    #
    # histogram dimensions:
    #
    #   alpha.shape = (gamma, p1, p2, ...)
    #
    # while histogram usually has one extra dpsi dimension:
    #
    #   histograms.shape = (gamma, p1, p2, ..., dpsi)
    #

    histogram_nonempty = np.any(
        histograms > 0,
        axis=-1,
    )

    valid_fit = (
        (event_counts >= minimum_counts)
        & histogram_nonempty
        & np.isfinite(alpha)
        & np.isfinite(beta)
        & (alpha > 0)
        & (beta > 1)
    )

    # ---------------------------------------------------------
    # Calculate metrics
    # ---------------------------------------------------------

    metric_values = {}

    for metric in metrics:

        if metric == "CDF_chi2":

            values = np.asarray(
                fit_parameters["fit_quality"],
                dtype=float,
            ).copy()

        else:

            func = metric_functions[metric]

            values = np.asarray(
                func(
                    fit_parameters=fit_parameters,
                    gamma_idx=gamma_idx,
                    minimum_counts=minimum_counts,
                    king_pdf=king_pdf,
                ),
                dtype=float,
            )

            # -------------------------------------------------
            # Some of your metric functions may return only
            # one gamma slice when gamma_idx is supplied.
            # Restore it into a full-shape array if necessary.
            # -------------------------------------------------

            if values.shape != alpha.shape:

                if (
                    gamma_idx is not None
                    and values.shape == alpha[gamma_idx].shape
                ):

                    full_values = np.full(
                        alpha.shape,
                        np.nan,
                        dtype=float,
                    )

                    full_values[gamma_idx] = values
                    values = full_values

                else:

                    raise ValueError(
                        f"{metric} returned shape {values.shape}, "
                        f"but expected {alpha.shape}."
                    )

        # -----------------------------------------------------
        # Reject invalid values/bins
        # -----------------------------------------------------

        values = values.copy()

        values[
            ~np.isfinite(values)
        ] = np.nan

        values[
            ~valid_fit
        ] = np.nan

        # If a particular gamma slice was requested, keep all
        # other slices as NaN in the returned object as well.
        if gamma_idx is not None:

            gamma_mask = np.zeros(
                alpha.shape,
                dtype=bool,
            )

            gamma_mask[gamma_idx] = True

            values[
                ~gamma_mask
            ] = np.nan

        metric_values[metric] = values

    # ---------------------------------------------------------
    # Plot
    # ---------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=figsize
    )

    for metric in metrics:

        values = metric_values[metric]

        finite_values = values[
            np.isfinite(values)
        ]

        if len(finite_values) == 0:
            print(
                f"Warning: no valid values found for {metric}"
            )
            continue

        if show_statistics:

            median = np.median(
                finite_values
            )

            label = (
                f"{metric} "
                f"(N={len(finite_values)}, "
                f"median={median:.3g})"
            )

        else:

            label = metric

        ax.hist(
            finite_values,
            bins=bins,
            density=density,
            histtype="step",
            linewidth=2,
            label=label,
        )

    # ---------------------------------------------------------
    # Axes
    # ---------------------------------------------------------

    ax.set_xlabel(
        "Metric value"
    )

    ax.set_ylabel(
        "Density"
        if density
        else "Number of fitted bins"
    )

    if log_x:
        ax.set_xscale("log")

    if log_y:
        ax.set_yscale("log")

    if gamma_idx is None:
        title = (
            "Distribution of King fit-quality metrics"
        )
    else:
        title = (
            "Distribution of King fit-quality metrics "
            f"(gamma_idx={gamma_idx})"
        )

    ax.set_title(title)

    ax.grid(
        alpha=0.3
    )

    ax.legend(
        fontsize=8,
        framealpha=0.9,
    )

    plt.tight_layout()
    plt.show()

    return metric_values