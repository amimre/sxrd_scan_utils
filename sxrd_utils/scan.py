import numpy as np
from nexusformat import nexus
from pathlib import Path

from sxrd_utils.fitaid_structure_factors import FitaidOutput


class SXRDScan:
    """Every scan has a sequential number, acquisition time and (h, k) index."""

    def __init__(self, nxs_path):
        # ensure file exists
        self.nxs_path = nxs_path
        if not self.nxs_path.is_file():
            raise RuntimeError(f"File {self.nxs_path} does not exist.")
        # get id number
        self.id = int(self.nxs_path.name[4:-4])

        # get nexus file
        self.nx_file = nexus.nxload(self.nxs_path)

        self.automated_scan = (
            True if "scan_request" in self.nx_file.entry.diamond_scan.keys() else False
        )

        # get number of datapoints
        self.n_datapoints = self.nx_file.entry.diamond_scan.scan_shape

        # read out h,k,l values
        self.h_values = np.array(self.nx_file.entry.instrument.hkl.h)
        self.k_values = np.array(self.nx_file.entry.instrument.hkl.k)
        self.l_values = np.array(self.nx_file.entry.instrument.hkl.l)

        if self.automated_scan:
            scan_request = str(self.nx_file.entry.diamond_scan.scan_request)
            scan_request = scan_request.replace("true", "True").replace(
                "false", "False"
            )
            self.scan_request = eval(scan_request)
            self.continuous = self.scan_request["compoundModel"]["models"][0][
                "continuous"
            ]
        else:
            self.scan_request = None
            self.continuous = None  # could be either in this case


def _collapse_value(arr, eps, integer_only):
    digits = int(-np.log10(eps))
    value = round(np.mean(arr), digits) + 0.0  # +.0 forces -0.0 to 0.0

    if integer_only:
        if np.any(abs(np.round(arr, digits) - arr) > eps):
            raise ValueError(
                "integer_only=True was specified, " f"but values are non-integer."
            )
        value = int(value)
    return value


class RockingCurve(SXRDScan):
    # *should* have a single L value, but moving H,K (==omega)
    def __init__(self, nxs_path, integer_only=True, eps=1e-1):
        super().__init__(nxs_path)
        if eps <= 0:
            raise ValueError("eps must be >= 0")

        # check that curve has a single L value
        if np.std(self.l_values) > eps:
            raise ValueError(
                f"Rocking scan {self.id} has non-constant "
                f"L value(s): {self.l_values}"
            )
        self.l = _collapse_value(self.l_values, eps, integer_only)
        self.h = _collapse_value(self.h_values, eps, integer_only)
        self.k = _collapse_value(self.k_values, eps, integer_only)
        self.hk = (self.h, self.k)


class LScan(SXRDScan):
    # *should* have a single H,K value
    def __init__(self, nxs_path, integer_only=True, eps=1e-1):
        super().__init__(nxs_path)
        if eps <= 0:
            raise ValueError("eps must be >= 0")

        # check that L scan has a single h,k value
        if np.std(self.h_values) > eps or np.std(self.k_values) > eps:
            raise ValueError(f"L scan {self.id} has non-constant H,K value(s).")

        # if the above check passes, set a fixed h,k
        self.h = _collapse_value(self.h_values, eps, integer_only=integer_only)
        self.k = _collapse_value(self.k_values, eps, integer_only=integer_only)
        self.hk = (self.h, self.k)
