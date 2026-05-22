#!/usr/bin/env python

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

# Imports
import numpy as np
import os
import argparse
import sys

sys.path.append('/data/user/bbrinson/lrd/kingmaker_env/lib/python3.12/site-packages/')
sys.path.append('/data/user/bbrinson/lrd/csky')
sys.path.append('/data/user/bbrinson/lrd/kingmaker')

import csky as cy

timer = cy.timing.Timer()
time = timer.time

# Cache arguments
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type = int, default = 0, help = 'trial seed')
parser.add_argument("--mp_cpus", type = int, default = 2, help = 'number of CPUs to assign for multiprocessing')
parser.add_argument("--gamma", type = float, default = 2.0, help = 'spectral index for signal injection')
parser.add_argument("--N_trials", type = int, default = 1000, help = 'number of trials to run')

args = parser.parse_args()

seed = args.seed
mp_cpus = args.mp_cpus
gamma = args.gamma
N_trials = args.N_trials

from kingmaker.wrapper import KingSpatialLikelihood

parametrization_bins = {
        'sindec': 10,
        'log10energy': 10,
        'sigma': 10,
        }


# Load analysis
ana_dir = cy.utils.ensure_dir('/data/user/fkrafft/csky_king_tests')
ana = cy.get_analysis(cy.selections.repo, 'version-005-p04', cy.selections.NTDataSpecs.NTv5p4, dir = ana_dir, min_sigma = np.radians(0.2))

king_space_eval_79 = KingSpatialLikelihood(signal_events = ana[0].sig.as_array,
                                           parametrization_bins = parametrization_bins,
                                           cache_name = "/data/user/bbrinson/lrd/king_caches/NTv5p4_IC79_cache.npz",
                                           weight_field = "oneweight",
                                           true_ra_name = "true_ra",
                                           true_dec_name = "true_dec",
                                           true_energy_name = "true_energy",
                                           angular_cutoff = np.radians(15),
                                           )

king_space_eval_86 = KingSpatialLikelihood(signal_events = ana[1].sig.as_array,
                                           parametrization_bins = parametrization_bins,
                                           cache_name = "/data/user/bbrinson/lrd/king_caches/NTv5p4_IC86_cache.npz",
                                           weight_field = "oneweight",
                                           true_ra_name = "true_ra",
                                           true_dec_name = "true_dec",
                                           true_energy_name = "true_energy",
                                           angular_cutoff = np.radians(15),
                                           )

def king_func_79(ev, pdf_bg, src, gamma, **kwargs):
    king_space_eval_79.set_events(events = ev, source_ras = src.ra, source_decs = src.dec)
    pdf_ratio = king_space_eval_79.evaluate_pdf(events = ev, gamma = gamma)
    pdf_ratio /= pdf_bg
    return pdf_ratio

def king_func_86(ev, pdf_bg, src, gamma, **kwargs):
    king_space_eval_86.set_events(events = ev, source_ras = src.ra, source_decs = src.dec)
    pdf_ratio = king_space_eval_86.evaluate_pdf(events = ev, gamma = gamma)
    pdf_ratio /= pdf_bg
    return pdf_ratio

dtype = [('gamma', '<f8'), ('ns', '<f8'), ('ts', '<f8')]

ras_deg = [40.67]
decs_deg = [-0.01]

ras = np.radians(ras_deg)
decs = np.radians(decs_deg)

# Configure trial runner
cy.CONF['mp_cpus'] = mp_cpus
mask_deg = 15.

for i, dec in enumerate(decs):
    ra = ras[i]
    srcs = cy.sources(ra, dec)

    tr = cy.get_trial_runner(ana = ana,
                             src = srcs,
                             use_bdt = True,
                             flux = cy.hyp.PowerLawFlux(gamma),
                             use_all_ev = True,
                             use_pdf_bg = True,
                             space = 'generic',
                             func_array = [king_func_79, king_func_86],
                             extra_keep = ["dec", "ra", "sigma", "sigma_bdt", "sindec", "event", "energy"],
                             cut_n_sigma = np.inf,
                             )

    # Run and save trials per declination
    bg_trials = []

    with time('run trials'):
        for j in range(N_trials):
            temp_seed = seed * N_trials + j
            trial = tr.get_one_trial(n_sig = 0, seed = temp_seed)
            #masks = [dist_mask(evs[0], srcs, mask_deg) for evs in trial.evss]
            #fit = tr.get_one_fit_from_trial(trial, _cut_deg = mask_deg)
            fit = tr.get_one_fit_from_trial(trial)
            bg_trials.append(fit)

    bg_trials = np.array(bg_trials)

    new_bg_array = np.zeros(len(bg_trials), dtype = dtype)
    new_bg_array['gamma'] = bg_trials[:, 2]
    new_bg_array['ns'] = bg_trials[:, 1]
    new_bg_array['ts'] = bg_trials[:, 0]

    print(new_bg_array)

    bg_dir = cy.utils.ensure_dir('/data/user/fkrafft/csky_king_tests/bg')
    np.save('{}/N_{}_seed_{}.npy'.format(bg_dir, N_trials, seed), new_bg_array)

print(timer)
