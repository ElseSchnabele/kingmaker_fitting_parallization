#!/home/fkrafft/.venv/bin/python

"""
Smoke-test version of king background trial generation.

Runs a small number of trials, prints diagnostics, and writes a test CSV.

call with (example smoke test config):
python king_bkg_trial_generation_test.py \
  --sin_dec 0.0 \
  --N_trials 20 \
  --seed 0 \
  --mp_cpus 1 \
  --gammas 2.0 2.7 \
  --out_dir /tmp/king_debug \
  --ana_dir /data/user/fkrafft/csky_gfu_tests \
  --file_identifier smoke_parallel
"""

import sys
import json
import os
import argparse
import inspect
import traceback

print("Python executable:", sys.executable)
print("Python path:", sys.path)

sys.path.append(os.path.abspath("../../../"))

import numpy as np

from kingmaker_fork.kingmaker.wrapper import KingSpatialLikelihood
from csky_gfu_tests.custom_gfu_specs import GFUDataSpecs as custom_gfu
from kingmaker_fork.likelihood_analysis.utils.trial_runner_config_loading import save_king_trial_runner_config
from kingmaker_fork.likelihood_analysis.utils.parallel_king_trials import get_many_fits_from_trials
from kingmaker_fork.likelihood_analysis.analysis_helpers.to_jsonable import to_jsonable

import csky as cy
from csky.utils import Arrays

timer = cy.timing.Timer()
time = timer.time


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--mp_cpus", type=int, default=1)
parser.add_argument(
    "--gammas",
    type=float,
    nargs="+",
    default=[2.0, 2.5],
    help="List of spectral indices used for King interpolation/fits",
)
parser.add_argument("--N_trials", type=int, default=3)
parser.add_argument("--out_dir", type=str, required=True)
parser.add_argument("--ana_dir", type=str, required=True)
parser.add_argument("--file_identifier", type=str, default="test")
parser.add_argument("--sin_dec", type=float, default=0.0)
parser.add_argument("--dry_run", action="store_true",
                    help="Only initialize analysis/trial runner, do not run fits.")
args = parser.parse_args()


def main():
    print("\n========== TEST CONFIG ==========")
    print(args)
    print("=================================\n")

    if not -1.0 <= args.sin_dec <= 1.0:
        raise ValueError(f"--sin_dec must be in [-1, 1], got {args.sin_dec}")

    ana_dir = cy.utils.ensure_dir(args.ana_dir)
    out_dir = cy.utils.ensure_dir(args.out_dir)

    log_dir = cy.utils.ensure_dir(os.path.join(out_dir, "log"))
    print("ana_dir:", ana_dir)
    print("out_dir:", out_dir)
    print("log_dir:", log_dir)

    seed = args.seed
    mp_cpus = args.mp_cpus
    gamma = 2.0  # default injection gamma
    spectral_indices = np.array(args.gammas)
    N_trials = args.N_trials
    file_identifier = args.file_identifier
    src_sin_dec = args.sin_dec

    weight_field = "oneweight"
    angular_cutoff_deg =15.
    dpsi_nbins = 101
    minimum_counts = 200

    parametrization_bins = {
        "log10energy": np.array([2, 3, 4, 5, 6]),
        "dec": np.arcsin(np.linspace(-1, 1, 8)),
        "sigma": 10,
    }

    repo_cache = cy.utils.ensure_dir("/data/user/fkrafft/csky_repo")
    repo = cy.selections.Repository(
        local_root=repo_cache,
        remote_root="/data/ana/PointSource/GFU/online_v001-p10",
    )

    print("\n========== REPO DEBUG ==========")
    print("Repository signature:", inspect.signature(cy.selections.Repository))
    print("repo:", repo)
    for name in ["root", "local_root", "remote_root", "base_dir", "dir"]:
        print(name, "=", getattr(repo, name, None))
    print("================================\n")

    with time("ana setup"):
        ana = cy.get_analysis(
            repo,
            "version-001-p09",
            custom_gfu.gfu_11yr,
            dir=ana_dir,
        )

    print("\n========== ANA DEBUG ==========")
    print("ana:", ana)
    print("len(ana):", len(ana))
    print("ana[0]:", ana[0])
    print("sig array dtype names:", ana[0].sig.as_array.dtype.names)
    print("n sig events:", len(ana[0].sig.as_array))
    print("===============================\n")

    fit_cache_filename = f"TEST_king_fit_custom_gfu_{file_identifier}.npz"
    fit_cache_path = os.path.join(out_dir, fit_cache_filename)

    print("King fit cache path:", fit_cache_path)

    king_wrapper = KingSpatialLikelihood(
        signal_events=ana[0].sig.as_array,
        parametrization_bins=parametrization_bins,
        spectral_indices=spectral_indices,
        cache_name=fit_cache_path,
        dpsi_nbins=dpsi_nbins,
        minimum_counts=minimum_counts,
        weight_field=weight_field,
        true_ra_name = "true_ra",
        true_dec_name = "true_dec",
        true_energy_name = "true_energy",
        angular_cutoff=np.radians(angular_cutoff_deg),
    )
    print("\n========== KING WRAPPER DEBUG ==========")

    print("type(bin_centers):", type(king_wrapper.bin_centers))
    print("type(parametrization_bins):", type(king_wrapper.parametrization_bins))

    print("bin_centers:")
    for i, bc in enumerate(king_wrapper.bin_centers):
        print(i, type(bc), np.shape(bc), bc[:5] if len(bc) > 5 else bc)

    print("parametrization_bins keys:", parametrization_bins.keys())

    print("========================================\n")
    #NOTE inserted one line in GenericPDFRatioEvaluator in csky 
    def king_func(ra, dec, sigma, energy, src, pdf_bg, gamma=2.0, **kwargs):
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
        
    features={
    "ra": "ra",
    "dec": "dec",
    "sigma": "sigma",
    "energy": "energy",
    }
    fits = {
        "gamma": tuple(args.gammas),
    }
    ra = 0.0
    dec = np.arcsin(src_sin_dec)
    srcs = cy.sources(ra, dec)

    print("\n========== SRC DEBUG ==========")
    print("ra:", ra)
    print("sin_dec:", src_sin_dec)
    print("dec:", dec)
    print("srcs:", srcs)
    print("==============================\n")

    cy.CONF["mp_cpus"] = mp_cpus

    with time("trial runner setup"):
        tr:cy.trial.TrialRunner = cy.get_trial_runner(
            ana=ana,
            src=srcs,
            mp_cpus=mp_cpus,
            use_bdt=True,
            flux=cy.hyp.PowerLawFlux(gamma),
            use_all_ev=True,
            #use_pdf_bg=True,
            use_pdf_bg=True,
            space="generic",
            func=king_func,
            features=features,
            fits=fits,
            extra_keep=[
                "dec",
                "ra",
                "sigma",
                "sindec",
                "event",
                "energy",
            ],
            cut_n_sigma=np.inf,
        )

    print("\n========== TRIAL RUNNER DEBUG ==========")
    print("tr:", tr)
    print("dry_run:", args.dry_run)
    print("========================================\n")

    if args.dry_run:
        print("Dry run requested. Stopping before trial generation.")
        print(timer)
        return

    with time("run test trials"):
        bg_trials = get_many_fits_from_trials(
            tr=tr,
            n_trials=N_trials,
            n_sig=0,
            logging=True,
            mp_cpus=mp_cpus,
            seed=seed,
        )
    print("bg_trials:", bg_trials)
    print("bg_trials keys:", bg_trials.keys())
    dtype = [("gamma", "<f8"), ("ns", "<f8"), ("ts", "<f8")]
    new_bg_array = np.zeros(len(bg_trials), dtype=dtype)

    new_bg_array["gamma"] = bg_trials["gamma"]
    new_bg_array["ns"] = bg_trials["ns"]
    new_bg_array["ts"] = bg_trials["ts"]


    out_file = os.path.join(
        out_dir,
        f"TEST_king_bkg_trials_sindec_{np.round(np.sin(dec), 3)}"
        f"_N_{N_trials}_{file_identifier}.npy",
    )

    np.save(
        out_file,
        new_bg_array,
    )

    print("Wrote test output:")
    print(out_file)
    print("\nTimer:")
    print(timer)
    
    #fit trials with chi2
    bg_chi2 = cy.dists.Chi2TSD(bg_trials)
    
    #estimate sensitivity
    with time('ps sensitivity'):
        print(f'Estimating sensitivity for sin(dec) = {src_sin_dec}')
        sens: dict = tr.find_n_sig(
            # ts, threshold
            bg_chi2.median(), # p = 50%
            # beta, fraction of trials which should exceed the threshold
            0.9, # beta = 90%
            # n_inj step size for initial scan
            n_sig_step=1,
            # this many trials at a time
            batch_size=500,
            # tolerance, as estimated relative error
            tol=.05,
            mp_cpus = mp_cpus
            )
    #saving sensitivity   
    json_sens_dict = to_jsonable(sens)
    with open(os.path.join(out_dir, f"TEST_king_sens_sindec_{np.round(np.sin(dec), 3)}"
        f"_N_{N_trials}_{file_identifier}.json"), "w") as f:
        json.dump(json_sens_dict, f)
        
    #estimate discovery potential
    with time('ps discovery potential'):
        print(f'Estimating discovery potential for sin(dec) = {src_sin_dec}')
        disc: dict = tr.find_n_sig(
            bg_chi2.isf_nsigma(5), # p = 5 sigma
            0.5, # beta = 50%
            n_sig_step=5, 
            batch_size=500, 
            tol=.05,
            mp_cpus= mp_cpus
            )
    
    #saving discovery potential   
    json_sens_dict = to_jsonable(disc)
    with open(os.path.join(out_dir, f"TEST_king_disc_sindec_{np.round(np.sin(dec), 3)}"
        f"_N_{N_trials}_{file_identifier}.json"), "w") as f:
        json.dump(json_sens_dict, f)
    
    
    


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n========== ERROR ==========")
        traceback.print_exc()
        print("===========================\n")
        sys.exit(1)