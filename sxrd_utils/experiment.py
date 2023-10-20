import numpy as np

from SXRD.sxrd_utils.scan import RockingCurve, LScan


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
        self.raw_fitaid_data = set()
        self.scan_groups = (self.l_scans, self.rocking_scans)

        # keep track of all known scan numbers
        self.scan_numbers = set()

    @property
    def fit_per_scan(self):
        fit_per_scan = {id: set() for id in self.all_scan_numbers}
        for fit in self.raw_fitaid_data:
            fit_per_scan[fit.scan_nr].add(fit)
        return fit_per_scan

    @property
    def fit_per_hk(self):
        fit_per_hk = {hk: set() for hk in self.hk_groups}
        for scan_nr, fits in self.fit_per_scan.items():
            fit_per_hk[self.scan_number_hk[scan_nr]].update(fits)
        return _sort_dict_by_hk(fit_per_hk)

    def filtered_fit_per_hk(self, filter_type):
        filtered = {hk: None for hk in self.hk_groups}
        for hk, fits in self.fit_per_hk.items():
            filtered[hk] = set((fit for fit in fits if fit.type == filter_type))
        return _sort_dict_by_hk(
            {hk: fits for hk, fits in filtered.items() if len(fits) > 0}
        )

    @property
    def all_scan_numbers(self):
        return set(scan.id for scan in self.all_scans)

    @property
    def all_scans(self):
        return set.union(*self.scan_groups)

    def register_scan(self, scan):
        if isinstance(scan, LScan):
            if scan.id in self.scan_numbers:
                raise RuntimeError(f"Scan {scan.id} already registered.")
            self.l_scans.add(scan)
            self.scan_numbers.add(scan.id)
        elif isinstance(scan, RockingCurve):
            if scan.id in self.scan_numbers:
                raise RuntimeError(f"Scan {scan.id} already registered.")
            self.rocking_scans.add(scan)
            self.scan_numbers.add(scan.id)
        elif isinstance(scan, FitaidOutput):
            self.raw_fitaid_data.add(scan)
        else:
            raise ValueError("Not a valid scan type.")

    def _assign_scans(self, scans, groups):
        assigned_scans = {hk: set() for hk in groups}
        for scan in scans:
            assigned_scans[(scan.h, scan.k)].add(scan)
        return _sort_dict_by_hk(assigned_scans)

    @property
    def scan_number_hk(self):
        scan_number_hk = {}
        for hk, numbers in self.assigned_scan_numbers.items():
            for nr in numbers:
                scan_number_hk[nr] = hk
        return scan_number_hk

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
        hk_assigned_scan_numbers = {hk: set() for hk in groups}
        for scan in scans:
            hk_assigned_scan_numbers[scan.hk].add(scan.id)
        for hk in groups:
            hk_assigned_scan_numbers[hk] = tuple(sorted(hk_assigned_scan_numbers[hk]))
        return _sort_dict_by_hk(hk_assigned_scan_numbers)

    @property
    def l_scan_hk_groups(self):
        return set((s.hk for s in self.l_scans))

    @property
    def hk_groups(self):
        return set((s.hk for s in self.all_scans))


def _sort_dict_by_hk(hk_indexed_dict):
    return {k: v for k, v in sorted(hk_indexed_dict.items(), key=lambda item: item[0])}


def grab_scan_nr_list(scan_nr_file):
    with open(scan_nr_file) as file:
        lines = file.readlines()
        scan_numbers = [int(l) for l in lines]
    return scan_numbers


def sorted_output_for_processing(assigned_scan_numbers):
    sorted_scans = ""
    for _, scan_numbers in assigned_scan_numbers.items():
        sorted_scans += " ".join(str(nr) for nr in scan_numbers) + "\n"
    sorted_scans = sorted_scans.strip()  # remove trailing \n
    return sorted_scans
