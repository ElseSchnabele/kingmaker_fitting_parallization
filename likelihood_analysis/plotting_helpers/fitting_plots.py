from typing import Optional, Tuple, Dict, Any, Literal
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_fit_quality_hist(fit_parameters: Dict[str, Any], gamma_idx: int = 0):
    fit_quality = fit_parameters["fit_quality"][gamma_idx]
    plot_fit_quality = fit_quality[fit_quality > 0]

    plt.figure(figsize=(8, 4))

    plt.grid(zorder=1)
    plt.hist(plot_fit_quality, bins=50, zorder=2)
    plt.axvline(
        np.median(plot_fit_quality),
        label=f"Median = {np.median(plot_fit_quality):.2f}",
        color="red",
        linestyle="dashed",
        zorder=3,
    )

    plt.ylabel("Frequency")
    plt.xlabel(r"$\chi^2$ value")
    plt.xlim(0, 600)
    plt.legend()
    plt.show()

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

    fig, ax = plt.subplots(figsize=figsize)

    pcm = ax.pcolormesh(
        x_bins,
        y_edges,
        values_2d.T,
        shading="auto",
        cmap=cmap,
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



def plot_selected_fit_bins(
    fit_parameters: Dict[str, Any],
    mode: Literal["best", "worst", "random"] = "best",
    gamma_idx: int = 0,
    n_plots: int = 6,
    minimum_counts: int = 100,
    random_seed: Optional[int] = None,
    figsize: Tuple[float, float] = (15, 10),
):
    """
    Plot selected stored King fits from fit_all_bins output.

    Parameters
    ----------
    fit_parameters:
        Output dictionary from fit_all_bins().
    mode:
        "best", "worst", or "random".
    gamma_idx:
        Spectral index index.
    n_plots:
        Number of bins to plot.
    minimum_counts:
        Minimum event count used to classify fitted bins.
    random_seed:
        Seed for random mode.
    """

    parametrization_bins = fit_parameters["parametrization_bins"]
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

    # Collect bins with valid stored fits
    good_bins = []

    for bin_idx in np.ndindex(*fit_alpha[gamma_idx].shape):
        param_idx = (gamma_idx,) + bin_idx

        if (
            event_counts[param_idx] >= minimum_counts
            and np.any(histograms[param_idx] > 0)
        ):
            chi2 = fit_quality[param_idx]

            if np.isfinite(chi2) and chi2 > 0:
                good_bins.append((bin_idx, chi2))

    if len(good_bins) == 0:
        raise RuntimeError("No valid fitted bins found.")

    if mode == "best":
        selected_bins = sorted(good_bins, key=lambda x: x[1])[:n_plots]
    elif mode == "worst":
        selected_bins = sorted(good_bins, key=lambda x: x[1], reverse=True)[:n_plots]
    elif mode == "random":
        rng = np.random.default_rng(random_seed)
        indices = rng.choice(len(good_bins), size=min(n_plots, len(good_bins)), replace=False)
        selected_bins = [good_bins[i] for i in indices]
    else:
        raise ValueError("mode must be one of: 'best', 'worst', 'random'")

    n_cols = min(3, n_plots)
    n_rows = int(np.ceil(len(selected_bins) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_1d(axes).flatten()

    print(f"Found {len(good_bins)} bins with stored fits")
    print(f"Showing {mode} bins:")

    for i, (bin_idx, chi2) in enumerate(selected_bins):
        print(f"{i + 1}: bin={bin_idx}, chi2={chi2:.3f}")

    for ax, (bin_idx, chi2) in zip(axes, selected_bins):
        param_idx = (gamma_idx,) + bin_idx

        hist = histograms[param_idx]
        uncertainty = uncertainties[param_idx]
        bins = dpsi_bins[param_idx]

        mask = hist > 0

        if np.sum(mask) == 0:
            ax.text(0.5, 0.5, f"Bin {bin_idx}\nNo histogram", ha="center", va="center")
            ax.set_axis_off()
            continue

        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_centers = bin_centers[mask]

        ax.errorbar(
            np.degrees(bin_centers),
            hist[mask],
            yerr=uncertainty[mask],
            fmt="o",
            label="MC Events",
            color="black",
            markersize=4,
        )

        alpha = fit_alpha[param_idx]
        beta = fit_beta[param_idx]

        dpsi_fine = np.linspace(0, min(8 * alpha, np.pi), 1000)

        # Same King PDF shape as your KingPDF implementation should give:
        pdf_fit = (1 - 1 / beta) / (2 * np.pi * alpha**2) * (
            1 + dpsi_fine**2 / (2 * beta * alpha**2)
        ) ** (-beta)

        if np.nanmax(pdf_fit) > 0:
            pdf_fit *= np.nanmax(hist[mask]) / np.nanmax(pdf_fit)

        ax.plot(
            np.degrees(dpsi_fine),
            pdf_fit,
            "-",
            linewidth=2,
            label="King Fit",
            color="blue",
        )

        title = f"α={np.degrees(alpha):.3f}°, β={beta:.2f}\n"

        for dim, name in enumerate(bin_names):
            idx = bin_idx[dim]
            edges = parametrization_bins[name]
            low = edges[idx]
            high = edges[idx + 1]

            if name in {"dec", "zen", "angErr"}:
                low = np.degrees(low)
                high = np.degrees(high)
                title += f"{name}=[{low:.2g}, {high:.2g}]° "
            else:
                title += f"{name}=[{low:.2g}, {high:.2g}] "

        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Angular Error (degrees)")
        ax.set_ylabel("Normalized Density")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

        ax.text(
            0.05,
            0.95,
            f"chi2={chi2:.2f}",
            transform=ax.transAxes,
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
        )

    for ax in axes[len(selected_bins):]:
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