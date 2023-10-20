from collections import defaultdict

import numpy as np

from sxrd_utils.ctr import CTR
from sxrd_utils.scan import SXRDScan


class SXRDExperiment:
    """Container class for an SXRD characterization experiment at I07

    Every experiment should correspond to *one* sample and preparation.
    It may contain measurements with multiple characterization
    techniques (such as L-scans or rocking curves) as long as they are measured
    on the same unaltered sample.
    """

    def __init__(self, base_path):
        self.base_path = base_path
        self.ctrs = {}  #TODO: probably this should be a tuple/set and there should be an additional property ctrs_per_hk which is a dict...

    @property
    def all_scans(self):
        return set.union(*(ctr.scans for ctr in self.ctrs.values()))

    @property
    def l_scans(self):
        return set.union(*(ctr.l_scans for ctr in self.ctrs.values()))

    @property
    def rocking_curves(self):
        return set.union(*(ctr.rocking_scans for ctr in self.ctrs.values()))

    @property
    def all_scan_numbers(self):
        return set.union(*(ctr.scan_numbers for ctr in self.ctrs.values()))

    @property
    def all_fits(self):
        return set.union(*(ctr.fits for ctr in self.ctrs.values()))

    @property
    def filtered_fits(self, filter_type):
        return set.union(*(ctr.filtered_fits(filter_type) for ctr in self.ctrs))

    @property
    def hk_per_scan_number(self):
        """Dict with (h, k) for every scan number.

        Inverse of scan_number_per_hk"""
        scan_number_hk = {}
        for hk, numbers in self.assigned_scan_numbers.items():
            for nr in numbers:
                scan_number_hk[nr] = hk
        return scan_number_hk

    @property
    def assigned_scan_numbers(self):
        """Dict with scan numbers for every (h, k).

        Inverse of hk_per_scan_number"""
        return self._assign_numbers(self.all_scans)

    def _assign_numbers(self, scans):
        hk_assigned_scan_numbers = {hk: set() for hk in self.hk_groups}
        for scan in scans:
            hk_assigned_scan_numbers[scan.hk].add(scan.id)
        for hk, unsorted_scans in hk_assigned_scan_numbers.items():
            hk_assigned_scan_numbers[hk] = tuple(sorted(unsorted_scans))
        return _sort_dict_by_hk(hk_assigned_scan_numbers)

    @property
    def hk_groups(self):
        return tuple(sorted(self.ctrs.keys()))

    def register_scan(self, scan):
        if not isinstance(scan, SXRDScan):
            raise ValueError("Must be an SXRD scan to register.")
        hk = scan.hk
        if not isinstance(hk, tuple) or len(hk) != 2:
            raise ValueError("Cannot create CTR object for h,k " f"values {hk}.")
        if hk not in self.ctrs.keys():
            self.ctrs[hk] = CTR(h=hk[0], k = hk[1])
        self.ctrs[hk].register_scan(scan)

    def register_fit(self, fit):
        """Registers a BINoculars fitaid fit to the experiment.

        The fit does by default not know which (h, k) CTR it belongs to, so we 
        need to figure it out based on the filename and scan number."""
        fit_hk = self.hk_per_scan_number[fit.scan_nr]
        self.ctrs[fit_hk].register_fit(fit)

    @property
    def max_hk(self):
        return max(ctr.hk[0] for ctr in self.ctrs), max(ctr.hk[1] for ctr in self.ctrs)


def _sort_dict_by_hk(hk_indexed_dict):
    return {k: v for k, v in sorted(hk_indexed_dict.items(), key=lambda item: item[0])}


def grab_scan_nr_list(scan_nr_file):
    with open(scan_nr_file, encoding="utf-8") as file:
        lines = file.readlines()
        scan_numbers = [int(l) for l in lines]
    return scan_numbers


def sorted_output_for_processing(assigned_scan_numbers):
    sorted_scans = ""
    for _, scan_numbers in assigned_scan_numbers.items():
        sorted_scans += " ".join(str(nr) for nr in scan_numbers) + "\n"
    sorted_scans = sorted_scans.strip()  # remove trailing \n
    return sorted_scans
