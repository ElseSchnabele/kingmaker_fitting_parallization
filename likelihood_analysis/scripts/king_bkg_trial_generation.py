#!/home/fkrafft/venvs/icecube-py3v4.4.0/bin/python
"""
Compute background trials for time-integrated equivalent analysis
Basically it just does everything that bg_validation.py does, but without the time-dependent stuff
For use in constructing validation plots
Arguments:
    --seed: sets the seed for the trial runner (default 0)
    --mp_cpus: sets the number of cpus for multiprocessing (default 2)
    --gamma: sets the spectral index (default 2.0, technically doesn't imapct background trials but I added it in anyway)
    --N_trials: sets the number of trials to perform for this seed (default 10000, though this may be overkill for code review)
"""

import sys
import os
import json
import numpy as np
import argparse
import inspect


print("Python executable:", sys.executable)
print("Python path:", sys.path)
sys.path.insert(0, "/home/fkrafft")


from kingmaker_fork.likelihood_analysis.utils.trial_runner_config_loading import save_king_trial_runner_config
from kingmaker_fork.likelihood_analysis.utils.parallel_king_trials import get_many_fits_from_trials
from kingmaker_fork.likelihood_analysis.analysis_helpers.to_jsonable import to_jsonable
from kingmaker_fork.kingmaker.wrapper import KingSpatialLikelihood

from  csky_gfu_tests.custom_gfu_specs import GFUDataSpecs as custom_gfu
import csky as cy
from csky.utils import Arrays

timer = cy.timing.Timer()
time = timer.time

# Cache arguments
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type = int, default = 0, help = 'trial seed')
parser.add_argument("--mp_cpus", type = int, default = 2, help = 'number of CPUs to assign for multiprocessing')
parser.add_argument("--calc_sens", type = int, default = 1, help = 'Flag for running sensitivity and discovery potential estimation')
parser.add_argument(
    "--gammas",
    type=float,
    nargs="+",
    default=[2.0, 2.5],
    help="List of spectral indices used for King interpolation/fits",
)
parser.add_argument("--N_trials", type = int, default = 1000, help = 'number of trials to run')
# king fits will cached and loaded automatically if simulation already exists 
parser.add_argument("--out_dir", type = str )
parser.add_argument("--ana_dir", type = str )
parser.add_argument("--file_identifier", type = str )
parser.add_argument("--sin_dec", type = float)


args = parser.parse_args()

ana_dir = cy.utils.ensure_dir(args.ana_dir)
seed = args.seed
mp_cpus = args.mp_cpus
calc_sens = bool(args.calc_sens)
N_trials = args.N_trials
out_dir = args.out_dir
spectral_indices = np.array(args.gammas)
file_identifier = args.file_identifier
src_sin_dec = args.sin_dec 


################### NOTE: some fixed parameters: ###################
weight_field: str = "oneweight"
angular_cutoff_deg: float = 15 
dpsi_nbins = 101
gamma = 2.0   # default fit/injection gamma
minimum_counts = 300
# fixed bin edges in energy and declination but equal-p in sigma
parametrization_bins = {
    'log10energy':  np.array([2, 2.75, 3.5, 4.25, 5., 6.0]), #  energy bins from 100 GeV to 1 PeV
    'dec': np.arcsin(np.linspace(-1, 1, 10)),  #  10 equal bins in sin_dec
    'sigma': 12
    
}
#####################################################################

# Load analysis
#define repo
repo_cache = cy.utils.ensure_dir('/data/user/fkrafft/csky_repo')
repo = cy.selections.Repository(
    local_root = repo_cache,
    remote_root= '/data/ana/PointSource/GFU/online_v001-p10')


print("Repository signature:", inspect.signature(cy.selections.Repository))
print("repo =", repo)
for name in ["root", "local_root", "remote_root", "base_dir", "dir"]:
    print(name, "=", getattr(repo, name, None))


with time('ana setup (from cache-to-disk)'):
    ana = cy.get_analysis(repo, 'version-001-p09' , custom_gfu.gfu_11yr, dir = ana_dir)

fit_cache_filename = f"king_fit_custom_gfu_{file_identifier}.npz"



king_wrapper = KingSpatialLikelihood(
                                    signal_events = ana[0].sig.as_array,
                                    parametrization_bins = parametrization_bins,
                                    spectral_indices= spectral_indices,
                                    cache_name = os.path.join(out_dir, fit_cache_filename),
                                    dpsi_nbins=dpsi_nbins,
                                    minimum_counts=minimum_counts,
                                    weight_field = weight_field,
                                    true_ra_name = "true_ra",
                                    true_dec_name = "true_dec",
                                    true_energy_name = "true_energy",
                                    angular_cutoff = np.radians(angular_cutoff_deg),
                                       )

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

features = {
    "ra": "ra",
    "dec": "dec",
    "sigma": "sigma",
    "energy": "energy",
}

fits = {
    "gamma": tuple(args.gammas),
}

dtype = [('gamma', '<f8'), ('ns', '<f8'), ('ts', '<f8')]

#src position
ra = 0
dec = np.arcsin(src_sin_dec)


# Configure trial runner
cy.CONF['mp_cpus'] = mp_cpus
mask_deg = 15.

srcs = cy.sources(ra, dec)

# run trial runner passing king function evaluation
tr: cy.trial.TrialRunner = cy.get_trial_runner(ana = ana,
                        src = srcs,
                        mp_cpus = mp_cpus,
                        use_bdt = True,
                        flux = cy.hyp.PowerLawFlux(gamma),
                        use_all_ev = True,
                        use_pdf_bg = True,
                        space = 'generic',
                        func = king_func,
                        features = features,
                        fits = fits,
                        extra_keep = ["dec", "ra", "sigma", "sindec", "event", "energy"],
                        cut_n_sigma = np.inf,
                        )

config_path = os.path.join(
    out_dir,
    f"king_trial_runner_config_sindec_{src_sin_dec}_{file_identifier}.pkl",
)

save_king_trial_runner_config(
    config_path,
    out_dir=out_dir,
    file_identifier=file_identifier,
    src_sin_dec=src_sin_dec,
    mp_cpus=mp_cpus,
    gamma=gamma,
    spectral_indices=spectral_indices,
    parametrization_bins=parametrization_bins,
    weight_field=weight_field,
    angular_cutoff_deg=angular_cutoff_deg,
    dpsi_nbins=dpsi_nbins,
    minimum_counts=minimum_counts,
    features=features,
    fits=fits,
)

# Run and save trials per declination
bg_trials = []



with time("run test trials"):
    bg_trials = get_many_fits_from_trials(
        tr=tr,
        n_trials=N_trials,
        n_sig=0,
        logging=True,
        mp_cpus=mp_cpus,
        seed=seed,
    )
    new_bg_array = np.zeros(len(bg_trials), dtype=dtype)

    new_bg_array["gamma"] = bg_trials["gamma"]
    new_bg_array["ns"] = bg_trials["ns"]
    new_bg_array["ts"] = bg_trials["ts"]

    bg_dir = cy.utils.ensure_dir(out_dir)

    np.save(
        f'{bg_dir}/king_bkg_trials_sindec_{np.round(np.sin(dec), 3)}_N_{N_trials}_{file_identifier}.npy',
        new_bg_array,
    )
    
if calc_sens:
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
    
    
        
### Diagnostics Block ###
print("\n" + "=" * 60)
print("KING BACKGROUND TRIAL CONFIGURATION")
print("=" * 60)

print("\n--- Runtime ---")
print(f"Python executable     : {sys.executable}")
print(f"Working directory     : {os.getcwd()}")

print("\n--- Analysis ---")
print(f"ana_dir               : {ana_dir}")
print(f"repo_cache            : {repo_cache}")
print(f"repo.remote_root      : {repo.remote_root}")

print("\n--- Trial Settings ---")
print(f"N_trials              : {N_trials}")
print(f"seed                  : {seed}")
print(f"mp_cpus               : {mp_cpus}")
print(f"sin_dec               : {src_sin_dec}")
print(f"dec (rad)             : {dec}")
print(f"dec (deg)             : {np.degrees(dec)}")

print("\n--- Spectral Settings ---")
print(f"injection gamma       : {gamma}")
print(f"spectral_indices      : {spectral_indices}")
print(f"fits                  : {fits}")

print("\n--- King Settings ---")
print(f"weight_field          : {weight_field}")
print(f"angular_cutoff_deg    : {angular_cutoff_deg}")
print(f"dpsi_nbins            : {dpsi_nbins}")
print(f"minimum_counts        : {minimum_counts}")

print("\n--- Parameterization Bins ---")
for key, val in parametrization_bins.items():
    print(f"\n{key}:")
    print(f"  n_bins              : {len(val)-1}")
    print(f"  min                 : {np.min(val)}")
    print(f"  max                 : {np.max(val)}")
    print(f"  first 5             : {val[:5]}")
    print(f"  last 5              : {val[-5:]}")

print("\n--- Trial Runner Settings ---")
print(f"use_bdt               : True")
print(f"use_all_ev            : True")
print(f"use_pdf_bg            : True")
print(f"space                 : generic")
print(f"cut_n_sigma           : {np.inf}")
print(f"extra_keep            : {['dec', 'ra', 'sigma', 'sindec', 'event', 'energy']}")

print("\n--- Output ---")
print(f"out_dir               : {out_dir}")
print(f"file_identifier       : {file_identifier}")
print(f"fit_cache_filename    : {fit_cache_filename}")
print(f"fit_cache_path        : {os.path.join(out_dir, fit_cache_filename)}")

print("=" * 60 + "\n")


print(timer)
