
import os
import pickle
import numpy as np

import csky as cy
from csky.utils import Arrays
from csky.trial import TrialRunner
from kingmaker_fork.kingmaker.wrapper import KingSpatialLikelihood



def save_king_trial_runner_config(
    path,
    *,
    out_dir,
    file_identifier,
    src_sin_dec,
    mp_cpus,
    gamma,
    spectral_indices,
    parametrization_bins,
    weight_field,
    angular_cutoff_deg,
    dpsi_nbins,
    minimum_counts,
    features,
    fits,
):
    cfg = {
        "out_dir": str(out_dir),
        "file_identifier": file_identifier,
        "src_sin_dec": float(src_sin_dec),
        "mp_cpus": int(mp_cpus),
        "gamma": float(gamma),
        "spectral_indices": np.asarray(spectral_indices),
        "parametrization_bins": {
            k: np.asarray(v) for k, v in parametrization_bins.items()
        },
        "weight_field": weight_field,
        "angular_cutoff_deg": float(angular_cutoff_deg),
        "dpsi_nbins": int(dpsi_nbins),
        "minimum_counts": int(minimum_counts),
        "features": features,
        "fits": fits,
        "fit_cache_filename": f"king_fit_custom_gfu_{file_identifier}.npz",
    }

    with open(path, "wb") as f:
        pickle.dump(cfg, f)

    return path


def restore_king_trial_runner(config_path, ana):
    with open(config_path, "rb") as f:
        cfg = pickle.load(f)

    out_dir = cfg["out_dir"]
    gamma = cfg["gamma"]
    src_sin_dec = cfg["src_sin_dec"]

    dec = np.arcsin(src_sin_dec)
    srcs = cy.sources(0, dec)

    fit_cache_path = os.path.join(out_dir, cfg["fit_cache_filename"])

    king_wrapper = KingSpatialLikelihood(
        signal_events=ana[0].sig.as_array,
        parametrization_bins=cfg["parametrization_bins"],
        spectral_indices=cfg["spectral_indices"],
        cache_name=fit_cache_path,
        dpsi_nbins=cfg["dpsi_nbins"],
        minimum_counts=cfg["minimum_counts"],
        weight_field=cfg["weight_field"],
        true_ra_name="true_ra",
        true_dec_name="true_dec",
        true_energy_name="true_energy",
        angular_cutoff=np.radians(cfg["angular_cutoff_deg"]),
    )

    def king_func(ra, dec, sigma, energy, src, pdf_bg, gamma=gamma, **kwargs):
        ev = Arrays({
            "ra": ra,
            "dec": dec,
            "sigma": sigma,
            "energy": energy,
            "log10energy": np.log10(energy),
            "sindec": np.sin(dec),
            "trueRa": ra,
            "trueDec": dec,
            "trueE": energy,
        })

        king_wrapper.set_events(
            events=ev,
            source_ras=src.ra,
            source_decs=src.dec,
        )

        out = king_wrapper.evaluate_pdf(events=ev, gamma=gamma)
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        out[out < 0] = 0.0
        out = out / pdf_bg

        return out

    tr: TrialRunner = cy.get_trial_runner(
        ana=ana,
        src=srcs,
        mp_cpus=cfg["mp_cpus"],
        use_bdt=True,
        flux=cy.hyp.PowerLawFlux(gamma),
        use_all_ev=True,
        use_pdf_bg=True,
        space="generic",
        func=king_func,
        features=cfg["features"],
        fits=cfg["fits"],
        extra_keep=["dec", "ra", "sigma", "sindec", "event", "energy"],
        cut_n_sigma=np.inf,
    )

    return tr, king_wrapper, king_func, cfg