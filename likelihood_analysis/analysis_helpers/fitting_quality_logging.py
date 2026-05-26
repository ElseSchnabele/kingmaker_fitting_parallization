from typing import Dict, Any

def log_fit_quality_metrics(out_path: str, fitted_parameters:Dict[str, Any])-> None:
    """
    Calculate a series of quality performance metrics and save the to a row in a csv file (for all ).
    Namely
    - percentage of fitted/failed/skipped bins
    - median fit quality value
    """
    pass