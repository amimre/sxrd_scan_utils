import numpy as np
from nexusformat import nexus
from pathlib import Path


class SXRDScan:
    """Every scan has a sequential number, acquisition time and (h, k) index.
    """
    def __init__(self, nxs_path):
        # ensure file exists
        self.nxs_path  = nxs_path
        if not self.nxs_path.is_file():
            raise RuntimeError(f"File {self.nxs_path} does not exist.")
        # get id number
        self.id = int(self.nxs_path.name[4:-4])

        # get nexus file
        self.nx_file = nexus.nxload(self.nxs_path)

        self.automated_scan = (True if 'scan_request'
                               in self.nx_file.entry.diamond_scan.keys() else False)

        # get number of datapoints
        self.n_datapoints = self.nx_file.entry.diamond_scan.scan_shape

        # read out h,k,l values
        self.h_values = np.array(self.nx_file.entry.instrument.hkl.h)
        self.k_values = np.array(self.nx_file.entry.instrument.hkl.k)
        self.l_values = np.array(self.nx_file.entry.instrument.hkl.l)

        if self.automated_scan:
            scan_request = str(self.nx_file.entry.diamond_scan.scan_request)
            scan_request = scan_request.replace('true', 'True').replace('false', 'False')
            self.scan_request = eval(scan_request)
            self.continous = self.scan_request['compoundModel']['models'][0]['continuous']
        else:
            self.scan_request = None
            self.continous = None  # could be either in this case

def _collapse_value(arr, eps, integer_only):
    digits = int(-np.log10(eps))
    value = round(np.mean(arr),digits) +.0  # +.0 forces -0.0 to 0.0

    if integer_only:
        if np.any(abs(np.round(arr, digits) - arr) > eps):
            raise ValueError("integer_only=True was specified, "
                             f"but values are non-integer.")
        value = int(value)
    return value


class RockingCurve(SXRDScan):
    # *should* have a single L value, but moving H,K (==omega)
    def __init__(self, nxs_path, integer_only=True, eps=1e-3):
        super().__init__(nxs_path)
        if eps <= 0:
            raise ValueError("eps must be >= 0")

        # check that curve has a single L value
        if np.std(self.l_values) > eps:
            raise ValueError(f"Rocking scan {self.id} has non-constant "
                             f"L value(s): {self.l_values}")
        self.l = _collapse_value(self.l_values, eps, integer_only)
        self.h = _collapse_value(self.h_values, eps, integer_only)
        self.k = _collapse_value(self.k_values, eps, integer_only)


class LScan(SXRDScan):
    # *should* have a single H,K value
    def __init__(self, nxs_path, integer_only=True, eps=1e-3):
        super().__init__(nxs_path)
        if eps <= 0:
            raise ValueError("eps must be >= 0")

        # check that L scan has a single h,k value
        if np.std(self.h_values) > eps or np.std(self.k_values) > eps:
            raise ValueError(f"L scan {self.id} has non-constant H,K value(s).")

        # if the above check passes, set a fixed h,k
        self.h = _collapse_value(self.h_values, eps, integer_only=integer_only)
        self.k = _collapse_value(self.k_values, eps, integer_only=integer_only)


class SXRDExperiment:
    """Container class for an SXRD characterization experiment at I07

    Every experiment should correspond to *one* sample and preparation.
    It may contain measurements with multiple characterization
    techniques (such as L-scans or rocking curves) as long as they are measured
    on the same unaltered sample.
    """

    def __init__(self, base_path):
        self.base_path = base_path

        # create lists of stored scans
        self.l_scans = set()
        self.rocking_scans = set()
        self.scan_groups = (self.l_scans, self.rocking_scans)

    @property
    def all_scans(self):
        return set.union(*self.scan_groups)

    def register_scan(self, scan):
        if isinstance(scan, LScan):
            self.l_scans.add(scan)
        elif isinstance(scan, RockingCurve):
            self.rocking_scans.add(scan)
        else:
            raise ValueError("Not a valid scan type.")

    def _assign_scans(self, scans, groups):
        assigned_scans = {hk:set() for hk in groups}
        for scan in scans:
            assigned_scans[(scan.h, scan.k)].add(scan)
        return _sort_dict_by_hk(assigned_scans)

    @property
    def assigned_scans(self):
        return self._assign_scans(self.all_scans, self.hk_groups)

    @property
    def assigned_lscans(self):
        return self._assign_scans(self.l_scans, self.l_scan_hk_groups)

    @property
    def assigned_lscan_numbers(self):
        return self._assign_numbers(self.l_scans, self.l_scan_hk_groups)

    @property
    def assigned_scan_numbers(self):
        return self._assign_numbers(self.all_scans, self.hk_groups)

    def _assign_numbers(self, scans, groups):
        hk_assigned_scan_numbers = {hk:set() for hk in groups}
        for scan in scans:
            hk_assigned_scan_numbers[(scan.h, scan.k)].add(scan.id)
        for hk in groups:
            hk_assigned_scan_numbers[hk] = (
                tuple(sorted(hk_assigned_scan_numbers[hk]))
            )
        return _sort_dict_by_hk(hk_assigned_scan_numbers)

    @property
    def l_scan_hk_groups(self):
        return set(((s.h,s.k) for s in self.l_scans))

    @property
    def hk_groups(self):
        return set(((s.h,s.k) for s in self.all_scans))


def _sort_dict_by_hk(hk_indexed_dict):
    return {k: v for k, v in sorted(hk_indexed_dict.items(),
                                    key=lambda item: item[0])}


def grab_scan_nr_list(scan_nr_file):
    with open(scan_nr_file) as file:
        lines = file.readlines()
        scan_numbers = [int(l) for l in lines]
    return scan_numbers


def sorted_output_for_processing(assigned_scan_numbers):
    sorted_scans = ''
    for _, scan_numbers in assigned_scan_numbers.items():
        sorted_scans += ' '.join(str(nr) for nr in scan_numbers) + '\n'
    sorted_scans = sorted_scans.strip()  # remove trailing \n
    return sorted_scans
