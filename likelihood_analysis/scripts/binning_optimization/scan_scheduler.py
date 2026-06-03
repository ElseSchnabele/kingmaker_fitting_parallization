import os
import json
import time
import traceback
from typing import Any, Dict

import numpy as np

from kingmaker_fork.kingmaker.fitting import KingPSFFitter
from kingmaker_fork.kingmaker.pdf import KingPDF
from kingmaker_fork.likelihood_analysis.analysis_helpers.rms_fit_quality import compute_cdf_rms_fit_quality
from kingmaker_fork.likelihood_analysis.utils.performance_evaluation import fitting_score_components


class KingFittingScanScheduler:
    def __init__(self, config_path: str, signal_events):
        self.config_path = config_path
        self.signal_events = signal_events

        with open(config_path, "r") as f:
            self.config = json.load(f)

        self.out_dir = self.config["out_dir"]
        os.makedirs(self.out_dir, exist_ok=True)

        self.results_path = os.path.join(self.out_dir, "fitting_scan_results.json")

    def _build_parametrization_bins(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        bins_cfg = candidate["parametrization_bins"]
        bins = {}

        for key, value in bins_cfg.items():
            if key == "dec_sindec_edges":
                bins["dec"] = np.arcsin(np.asarray(value, dtype=float))
            elif isinstance(value, list):
                bins[key] = np.asarray(value, dtype=float)
            else:
                bins[key] = value

        return bins

    def _candidate_paths(self, candidate_id: str) -> Dict[str, str]:
        candidate_dir = os.path.join(self.out_dir, candidate_id)
        os.makedirs(candidate_dir, exist_ok=True)

        return {
            "dir": candidate_dir,
            "fit_npz": os.path.join(candidate_dir, "fit_parameters.npz"),
            "summary_json": os.path.join(candidate_dir, "summary.json"),
            "error_txt": os.path.join(candidate_dir, "error.txt"),
        }

    def run_candidate(self, candidate: Dict[str, Any], overwrite: bool = False) -> Dict[str, Any]:
        candidate_id = candidate["id"]
        paths = self._candidate_paths(candidate_id)

        if os.path.exists(paths["summary_json"]) and not overwrite:
            with open(paths["summary_json"], "r") as f:
                return json.load(f)

        t0 = time.time()

        try:
            fixed = self.config["fixed"]
            parametrization_bins = self._build_parametrization_bins(candidate)

            fitter = KingPSFFitter(
                signal_events=self.signal_events,
                parametrization_bins=parametrization_bins,
                dpsi_nbins=fixed["dpsi_nbins"],
                minimum_counts=candidate["minimum_counts"],
                remove_weight_outliers=fixed["remove_weight_outliers"],
                weight_outlier_percentiles=fixed["weight_outlier_percentiles"],
                weight_field=fixed["weight_field"],
                true_ra_name=fixed["true_ra_name"],
                true_dec_name=fixed["true_dec_name"],
                true_energy_name=fixed["true_energy_name"],
                spectral_indices=np.asarray(self.config["spectral_indices"]),
                angular_cutoff=np.pi,
            )

            fit_parameters = fitter.fit_all_bins(verbose=True)

            king_pdf = KingPDF(angular_cutoff=np.radians(fixed["angular_cutoff_deg"]))

            rms = compute_cdf_rms_fit_quality(
                fit_parameters=fit_parameters,
                king_pdf=king_pdf,
                minimum_counts=candidate["minimum_counts"],
            )

            score_info = fitting_score_components(
                fit_parameters=fit_parameters,
                rms=rms,
                min_counts=candidate["minimum_counts"],
            )

            np.savez_compressed(
                paths["fit_npz"],
                **fit_parameters,
                fit_rms=rms,
            )

            summary = {
                "id": candidate_id,
                "status": "done",
                "runtime_sec": time.time() - t0,
                "minimum_counts": candidate["minimum_counts"],
                "parametrization_bins": candidate["parametrization_bins"],
                "fit_npz": paths["fit_npz"],
                **score_info,
            }

            with open(paths["summary_json"], "w") as f:
                json.dump(summary, f, indent=2)

            return summary

        except Exception:
            err = traceback.format_exc()
            with open(paths["error_txt"], "w") as f:
                f.write(err)

            summary = {
                "id": candidate_id,
                "status": "failed",
                "runtime_sec": time.time() - t0,
                "error_txt": paths["error_txt"],
            }

            with open(paths["summary_json"], "w") as f:
                json.dump(summary, f, indent=2)

            return summary

    def run_all(self, overwrite: bool = False):
        summaries = []

        for candidate in self.config["candidates"]:
            print(f"\n=== Running candidate {candidate['id']} ===")
            summary = self.run_candidate(candidate, overwrite=overwrite)
            summaries.append(summary)

        with open(self.results_path, "w") as f:
            json.dump(summaries, f, indent=2)

        return summaries