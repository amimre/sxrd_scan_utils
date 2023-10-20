from collections import defaultdict

import numpy as np

from sxrd_utils.scan import SXRDScan, RockingCurve, LScan
from sxrd_utils.ctr import CTR


class SXRDExperiment:
    """Container class for an SXRD characterization experiment at I07

    Every experiment should correspond to *one* sample and preparation.
    It may contain measurements with multiple characterization
    techniques (such as L-scans or rocking curves) as long as they are measured
    on the same unaltered sample.
    """

    def __init__(self, base_path):
        self.base_path = base_path
        self.ctrs = defaultdict(_ctr_default_factory)

    @property
    def all_scans(self):
        return set.union(ctr.scans for ctr in self.ctrs)

    @property
    def l_scans(self):
        return set.union(ctr.l_scans for ctr in self.ctrs)

    @property
    def rocking_curves(self):
        return set.union(ctr.rocking_scans for ctr in self.ctrs)

    @property
    def all_scan_numbers(self):
        return set.union(ctr.scan_numbers for ctr in self.ctrs)

    @property
    def all_fits(self):
        return set.union(ctr.fits for ctr in self.ctrs)

    @property
    def filtered_fits(self, filter_type):
        return set.union(ctr.filtered_fits(filter_type) for ctr in self.ctrs)

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
        return tuple(sorted((ctr.hk for ctr in self.ctrs)))


def _ctr_default_factory(hk):
    if not isinstance(hk, tuple) or len(hk) != 2:
        raise ValueError("Cannot create CTR object for h,k "
                         f"values {hk}.")
    return CTR(h=hk[0], k=hk[1])

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
