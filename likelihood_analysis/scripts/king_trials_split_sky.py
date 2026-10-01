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
import pickle
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
parser.add_argument("--sky", type = str, help = 'sky chosen (sky split analysis)')
parser.add_argument("--mp_cpus", type = int, default = 2, help = 'number of CPUs to assign for multiprocessing')
parser.add_argument("--calc_sens", type = int, default = 1, help = 'Flag for running sensitivity and discovery potential estimation')

parser.add_argument("--N_trials", type = int, default = 1000, help = 'number of trials to run')
# king fits will cached and loaded automatically if simulation already exists 
parser.add_argument("--out_dir", type = str )
parser.add_argument("--config_dir", type = str )
parser.add_argument("--ana_dir", type = str )
parser.add_argument("--candidate_id", type = str )
parser.add_argument("--sin_dec", type = float)


args = parser.parse_args()

ana_dir = cy.utils.ensure_dir(args.ana_dir)
seed = args.seed
mp_cpus = args.mp_cpus
calc_sens = bool(args.calc_sens)
N_trials = args.N_trials
out_dir = args.out_dir
os.makedirs(os.path.join(out_dir, "log"), exist_ok=True)

candidate_id = args.candidate_id
src_sin_dec = args.sin_dec 
config_dir = args.config_dir
sky = args.sky


file_identifier = f"SINDEC_{src_sin_dec}_NTRIALS_{N_trials}_{candidate_id}"

############check if everything exists already first:#################
#src position
ra = 0
dec = np.arcsin(src_sin_dec)
bg_dir = cy.utils.ensure_dir(out_dir)
bg_file = (
    f"{bg_dir}/king_bkg_trials_sindec_{np.round(np.sin(dec), 3)}"
    f"_N_{N_trials}_{file_identifier}.npy"
)
sens_file = os.path.join(out_dir, f"TEST_king_sens_sindec_{src_sin_dec}_N_{N_trials}_{file_identifier}.json")
disc_3sig_file = os.path.join(out_dir, f"TEST_king_3sig_disc_sindec_{src_sin_dec}_N_{N_trials}_{file_identifier}.json")
bias_file = os.path.join(out_dir, f"king_bias_{src_sin_dec}_N_{N_trials}_{file_identifier}.json")

########################################################################
with open(config_dir, "r") as f:
            config= json.load(f)

################### NOTE: Config parameters: ###################
for c in config["candidates"]:
        if c["id"] == candidate_id:
            candidate = c
            break
        
bins_cfg = candidate["parametrization_bins"]
parametrization_bins = {}

#build parametrization bins
for key, value in bins_cfg.items():
    if key == "dec_sindec_edges":
        if isinstance(value, int):
            parametrization_bins["dec"] = value
        else:
            parametrization_bins["dec"] = np.arcsin(np.asarray(value, dtype=float))

    elif key == "log10energy": 
        if isinstance(value, int):
            parametrization_bins["log10energy"] = value
        else:
            parametrization_bins["log10energy"] = np.asarray(value, dtype=float)

    elif key == "sigma":
        parametrization_bins["sigma"] = int(value)

    else:
        parametrization_bins[key] = value

# load spectral indices
spectral_indices= np.asarray(config["spectral_indices"])

#default gamma for signal injection


fixed = config["fixed"]
signal_injection_gamma = fixed["signal_injection_gamma"]
dpsi_nbins=fixed["dpsi_nbins"]
minimum_counts=candidate["minimum_counts"]
weight_field=fixed["weight_field"]
true_ra_name=fixed["true_ra_name"]
true_dec_name=fixed["true_dec_name"]
true_energy_name=fixed["true_energy_name"]
angular_cutoff_deg = fixed["angular_cutoff_deg"]

#####################################################################

if all(os.path.exists(f) for f in [bg_file, sens_file, disc_3sig_file, bias_file]):
    print("All outputs already exist. Nothing to do.")
    sys.exit(0)
# Load analysis
#define repo
repo = cy.selections.Repository(
    local_root = '/data/user/fkrafft/gfu_data_sky_split',
    remote_root= '/data/user/fkrafft/gfu_data_sky_split')


print("Repository signature:", inspect.signature(cy.selections.Repository))
print("repo =", repo)
for name in ["root", "local_root", "remote_root", "base_dir", "dir"]:
    print(name, "=", getattr(repo, name, None))

if sky == 'north':
    gfu_sky_spcs = custom_gfu.north_gfu_19_to_23
elif sky == 'south':
    gfu_sky_spcs = custom_gfu.south_gfu_19_to_23
else:
    raise ValueError(f'Unknown sky split (not `south` nor `north`): {sky}')
with time('ana setup (from cache-to-disk)'):
    ana = cy.get_analysis(repo, 'version-001-p09' , gfu_sky_spcs, dir = ana_dir)


fit_cache_filename = f"king_fit_custom_gfu_{file_identifier}.npz"



king_wrapper = KingSpatialLikelihood(
                                    signal_events = ana[0].sig.as_array,
                                    parametrization_bins = parametrization_bins,
                                    spectral_indices= spectral_indices,
                                    cache_name = os.path.join(out_dir, fit_cache_filename),
                                    dpsi_nbins=dpsi_nbins,
                                    minimum_counts=minimum_counts,
                                    weight_field = weight_field,
                                    true_ra_name = true_ra_name,
                                    true_dec_name = true_dec_name,
                                    true_energy_name = true_energy_name,
                                    angular_cutoff = np.radians(angular_cutoff_deg),
                                       )

features = {
    "ra": "ra",
    "dec": "dec",
    "sigma": "sigma",
    "energy": "energy",
}

fits = {
    "gamma": spectral_indices,
}

dtype = [('gamma', '<f8'), ('ns', '<f8'), ('ts', '<f8')]



# Configure trial runner
cy.CONF['mp_cpus'] = mp_cpus

srcs = cy.sources(ra, dec)

# run trial runner passing king function evaluation
king_params = dict(angular_cutoff = np.radians(angular_cutoff_deg),
                   spectral_indicies = spectral_indices,
                   parametrization_bins = king_wrapper.parametrization_bins,
                   cache_parameters=False)

tr: cy.trial.TrialRunner = cy.get_trial_runner(ana = ana,
                         src = srcs,
                         flux = cy.hyp.PowerLawFlux(signal_injection_gamma),
                         space = 'king',
                         king_params = king_params,
                         window_dist = king_params.get("angular_cutoff", np.pi),
                         circle_cut = False,
                         use_bdt = False
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
    gamma=signal_injection_gamma,
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





if os.path.exists(bg_file):
    print(f"Loading existing background trials from:\n  {bg_file}")
    bg_trials = np.load(bg_file)
else:
    with time("run test trials"):
        raw_bg_trials = tr.get_many_fits(
            n_trials=N_trials,
            n_sig=0,
            logging=True,
            mp_cpus=mp_cpus,
            seed=seed,
        )

        bg_trials = np.empty(len(raw_bg_trials), dtype=dtype)
        bg_trials["gamma"] = raw_bg_trials["gamma"]
        bg_trials["ns"] = raw_bg_trials["ns"]
        bg_trials["ts"] = raw_bg_trials["ts"]

        np.save(bg_file, bg_trials)

        del raw_bg_trials

if calc_sens:
    bg_chi2 = cy.dists.Chi2TSD(bg_trials["ts"])
    
    
    if not os.path.exists(sens_file):
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
        
        #calculating flux
        n_sig_sens = sens['n_sig']
        n_sig_err_sens = sens['n_sig_error']
        
        sens_flux = tr.to_E2dNdE(n_sig_sens, E0=100, unit=1e3)
        sens_flux_err = (
            tr.to_E2dNdE(n_sig_sens + n_sig_err_sens, E0=100, unit=1e3)
            - sens_flux
        )
        sens['flux'] = sens_flux
        sens['flux_err'] = sens_flux_err
        #saving sensitivity   
        json_sens_dict = to_jsonable(sens)
        with open(sens_file, "w") as f:
            json.dump(json_sens_dict, f)
            
        del sens
  
    #estimate 3 sigma discovery potential
    
    
    if not os.path.exists(disc_3sig_file):
        with time('ps 3 sigma discovery potential'):
            print(f'Estimating 3 sigma discovery potential for sin(dec) = {src_sin_dec}')
            disc_3sig: dict = tr.find_n_sig(
                bg_chi2.isf_nsigma(3), # p =35 sigma
                0.5, # beta = 50%
                n_sig_step=5, 
                batch_size=500, 
                tol=.05,
                mp_cpus= mp_cpus
                )
        #calculating flux
        n_sig_3sig_disc = disc_3sig['n_sig']
        n_sig_err_3sig_disc = disc_3sig['n_sig_error']
        
        disc3_flux = tr.to_E2dNdE(n_sig_3sig_disc, E0=100, unit=1e3)
        disc3_flux_err = (
            tr.to_E2dNdE(n_sig_3sig_disc + n_sig_err_3sig_disc, E0=100, unit=1e3)
            - disc3_flux
        )
        disc_3sig['flux'] = disc3_flux
        disc_3sig['flux_err'] = disc3_flux_err
        #saving discovery potential   
        json_3sig_dic_dict = to_jsonable(disc_3sig)
        with open(disc_3sig_file, "w") as f:
            json.dump(json_3sig_dic_dict, f) 
            
        del disc_3sig

    #test for bias
    
    if not os.path.exists(bias_file):
        n_sigs = np.r_[:31:3]
        trials = [tr.get_many_fits(100, n_sig=n_sig, logging=False, seed=n_sig) for n_sig in n_sigs]
            
        #We add the true number of events injected for bookkeeping convenience:
        for (n_sig, t) in zip(n_sigs, trials):
            t['ntrue'] = np.repeat(n_sig, len(t))

        #Concatenate the trial batches:
        allt = cy.utils.Arrays.concatenate(trials)
        with open(bias_file, "wb") as f:
            pickle.dump(allt, f)
        
        del allt
    
        
### Diagnostics Block ###
print("\n" + "=" * 60)
print("KING BACKGROUND TRIAL CONFIGURATION")
print("=" * 60)

print("\n--- Runtime ---")
print(f"Python executable     : {sys.executable}")
print(f"Working directory     : {os.getcwd()}")

print("\n--- Analysis ---")
print(f"ana_dir               : {ana_dir}")
print(f"repo.remote_root      : {repo.remote_root}")

print("\n--- Trial Settings ---")
print(f"N_trials              : {N_trials}")
print(f"seed                  : {seed}")
print(f"mp_cpus               : {mp_cpus}")
print(f"sin_dec               : {src_sin_dec}")
print(f"dec (rad)             : {dec}")
print(f"dec (deg)             : {np.degrees(dec)}")

print("\n--- Spectral Settings ---")
print(f"injection gamma       : {signal_injection_gamma}")
print(f"spectral_indices      : {spectral_indices}")
print(f"fits                  : {fits}")

print("\n--- King Settings ---")
print(f"weight_field          : {weight_field}")
print(f"angular_cutoff_deg    : {angular_cutoff_deg}")
print(f"dpsi_nbins            : {dpsi_nbins}")
print(f"minimum_counts        : {minimum_counts}")

print("\n--- Parameterization Bins ---")
for key, val in parametrization_bins.items():
    if isinstance(val, int):
        print(f"eq.-p bins            : {str(val)}")
    else:
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
