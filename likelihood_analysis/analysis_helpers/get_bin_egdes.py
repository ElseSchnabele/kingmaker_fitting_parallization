import numpy as np 

def get_bin_info(parametrization_bins, bin_idx):
    if isinstance(parametrization_bins, np.ndarray):
        parametrization_bins = parametrization_bins.item()

    bin_idx = tuple(bin_idx)

    if len(bin_idx) != len(parametrization_bins):
        raise ValueError(
            f"Expected {len(parametrization_bins)} indices for "
            f"{list(parametrization_bins.keys())}, got {len(bin_idx)}."
        )

    info = {}

    for name, idx in zip(parametrization_bins.keys(), bin_idx):
        edges = np.asarray(parametrization_bins[name])

        if idx < 0 or idx >= len(edges) - 1:
            raise IndexError(
                f"Index {idx} out of range for '{name}' "
                f"(valid: 0-{len(edges) - 2})."
            )

        info[name] = {
            "index": int(idx),
            "low": float(edges[idx]),
            "high": float(edges[idx + 1]),
        }

    return info