from typing import Any

import histlite as hl
import matplotlib.pyplot as plt
import numpy as np


def plot_bias(
    data_1: dict[str, Any],
    data_2: dict[str, Any],
    sindec: float,
    label_1: str,
    label_2: str,
    ns_bins: np.ndarray,
    expect_gamma: float,
):
    """
    Compare reconstructed n_s and gamma values for two trial datasets.

    Parameters
    ----------
    data_1, data_2
        Trial dictionaries containing the fields:
        ``"ntrue"``, ``"ns"``, and ``"gamma"``.

    sindec
        Source sin(declination), used in the figure title.

    label_1, label_2
        Labels used for the two datasets in the legends.

    ns_bins
        Bin edges for the injected signal count.

    expect_gamma
        Expected injected spectral index.

    Returns
    -------
    fig, axs
        Matplotlib figure and axes.
    """
    fig, axs = plt.subplots(
        2,
        2,
        figsize=(7, 5),
        sharex="col",
        gridspec_kw={"height_ratios": [3, 1]},
    )

    # ------------------------------------------------------------------
    # Reconstructed n_s
    # ------------------------------------------------------------------
    ax = axs[0, 0]
    rax = axs[1, 0]

    h_ns_1 = hl.hist(
        (data_1["ntrue"], data_1["ns"]),
        bins=(ns_bins, 100),
    ).contain_project(1)

    h_ns_2 = hl.hist(
        (data_2["ntrue"], data_2["ns"]),
        bins=(ns_bins, 100),
    ).contain_project(1)

    hl.plot1d(
        ax,
        h_ns_1,
        errorbands=True,
        drawstyle="default",
        label=label_1,
    )
    hl.plot1d(
        ax,
        h_ns_2,
        errorbands=True,
        drawstyle="default",
        label=label_2,
    )

    x_ns = h_ns_1.centers[0]

    # Deviation from the expectation n_s = n_inj
    ns_deviation_1 = h_ns_1.values - x_ns
    ns_deviation_2 = h_ns_2.values - x_ns

    ns_mean_abs_deviation_1 = np.mean(np.abs(ns_deviation_1))
    ns_mean_abs_deviation_2 = np.mean(np.abs(ns_deviation_2))

    rax.axhline(0, color="k", lw=1)

    rax.plot(
        x_ns,
        ns_deviation_1,
        drawstyle="steps-mid",
        label=(
            rf"{label_1}: "
            rf"$\langle |\Delta| \rangle={ns_mean_abs_deviation_1:.2f}$"
        ),
    )
    rax.plot(
        x_ns,
        ns_deviation_2,
        drawstyle="steps-mid",
        label=(
            rf"{label_2}: "
            rf"$\langle |\Delta| \rangle={ns_mean_abs_deviation_2:.2f}$"
        ),
    )

    limits = np.asarray(ns_bins)[[0, -1]]

    ax.plot(
        limits,
        limits,
        color="grey",
        ls="--",
        lw=1,
        zorder=-10,
    )

    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_aspect("equal")

    ax.set_ylabel(r"$n_s$")
    rax.set_ylabel(r"$n_s-n_\mathrm{inj}$")
    rax.set_xlabel(r"$n_\mathrm{inj}$")

    # ------------------------------------------------------------------
    # Reconstructed gamma
    # ------------------------------------------------------------------
    ax = axs[0, 1]
    rax = axs[1, 1]

    h_gamma_1 = hl.hist(
        (data_1["ntrue"], data_1["gamma"]),
        bins=(ns_bins, 100),
    ).contain_project(1)

    h_gamma_2 = hl.hist(
        (data_2["ntrue"], data_2["gamma"]),
        bins=(ns_bins, 100),
    ).contain_project(1)

    hl.plot1d(
        ax,
        h_gamma_1,
        errorbands=True,
        drawstyle="default",
        label=label_1,
    )
    hl.plot1d(
        ax,
        h_gamma_2,
        errorbands=True,
        drawstyle="default",
        label=label_2,
    )

    x_gamma = h_gamma_1.centers[0]

    gamma_deviation_1 = h_gamma_1.values - expect_gamma
    gamma_deviation_2 = h_gamma_2.values - expect_gamma

    gamma_mean_abs_deviation_1 = np.mean(np.abs(gamma_deviation_1))
    gamma_mean_abs_deviation_2 = np.mean(np.abs(gamma_deviation_2))

    rax.axhline(0, color="k", lw=1)

    rax.plot(
        x_gamma,
        gamma_deviation_1,
        drawstyle="steps-mid",
        label=(
            rf"{label_1}: "
            rf"$\langle |\Delta| \rangle={gamma_mean_abs_deviation_1:.2f}$"
        ),
    )
    rax.plot(
        x_gamma,
        gamma_deviation_2,
        drawstyle="steps-mid",
        label=(
            rf"{label_2}: "
            rf"$\langle |\Delta| \rangle={gamma_mean_abs_deviation_2:.2f}$"
        ),
    )

    ax.axhline(
        expect_gamma,
        color="grey",
        ls="--",
        lw=1,
        zorder=-10,
    )

    ax.set_xlim(axs[0, 0].get_xlim())

    ax.set_ylabel(r"$\gamma$")
    rax.set_ylabel(r"$\gamma-\gamma_\mathrm{exp}$")
    rax.set_xlabel(r"$n_\mathrm{inj}$")

    # ------------------------------------------------------------------
    # General formatting
    # ------------------------------------------------------------------
    for current_ax in axs.flat:
        current_ax.grid()

    axs[0, 0].legend()
    axs[1, 0].legend()
    axs[0, 1].legend()
    axs[1, 1].legend()

    fig.suptitle(rf"$\sin(\delta)={sindec:g}$")
    fig.tight_layout()

    return fig, axs