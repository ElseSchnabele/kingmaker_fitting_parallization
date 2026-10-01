
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

    king_params = dict(angular_cutoff = np.radians(cfg['angular_cutoff_deg']),
                   spectral_indicies = cfg["spectral_indices"],
                   parametrization_bins = king_wrapper.parametrization_bins,
                   cache_dir = out_dir)
    
    tr: cy.trial.TrialRunner = cy.get_trial_runner(ana = ana,
                         src = srcs,
                         flux = cy.hyp.PowerLawFlux(gamma),
                         space = 'king',
                         king_params = king_params,
                         window_dist = king_params.get("angular_cutoff", np.pi),
                         circle_cut = False,
                         use_bdt = False
                         )

    return tr, king_wrapper, cfg