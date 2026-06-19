#!/home/fkrafft/venvs/icecube-py3v4.4.0/bin/python

import argparse
import sys
import numpy as np
sys.path.insert(0, "/home/fkrafft")

import csky as cy
from csky_gfu_tests.custom_gfu_specs import GFUDataSpecs as custom_gfu
from kingmaker_fork.likelihood_analysis.scripts.binning_optimization.scan_scheduler import (
    KingFittingScanScheduler,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan_config", required=True)
    parser.add_argument("--candidate_id", required=True)
    parser.add_argument("--ana_dir", required=True)
    #parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    repo = cy.selections.Repository(
        local_root="/data/user/fkrafft/csky_repo",
        remote_root="/data/ana/PointSource/GFU/online_v001-p10",
    )

    ana = cy.get_analysis(
        repo,
        "version-001-p09",
        custom_gfu.gfu_19_to_23,
        dir=args.ana_dir,
    )
    

    scheduler = KingFittingScanScheduler(
        
        config_path=args.scan_config,
        signal_events=ana[0].sig.as_array,
    )


    candidate = None
    for c in scheduler.config["candidates"]:
        if c["id"] == args.candidate_id:
            candidate = c
            break

    if candidate is None:
        raise ValueError(f"Candidate not found: {args.candidate_id}")

    scheduler.run_candidate(candidate, overwrite= True)


if __name__ == "__main__":
    main()