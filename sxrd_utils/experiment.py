from collections import defaultdict
import json

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
        self.l_limits = (None, None)
        self.fit_threshold = 1
        self.mask_outside_scan = True
        self.ctrs = (
            {}
        )  # TODO: probably this should be a tuple/set and there should be an additional property ctrs_per_hk which is a dict...

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
            self.ctrs[hk] = CTR(h=hk[0], k=hk[1])
        self.ctrs[hk].register_scan(scan)

    def register_fit(self, fit):
        """Registers a BINoculars fitaid fit to the experiment.

        The fit does by default not know which (h, k) CTR it belongs to, so we
        need to figure it out based on the filename and scan number."""
        fit_hk = self.hk_per_scan_number[fit.scan_nr]
        self.ctrs[fit_hk].register_fit(fit)

    def register_reciprocal_space_map(self, file,
                                      associated_scan_nr, hkl_resolution):
        """Register a reciprocal space map with the experiment.
        
        Similarly to the fits, the RSM does not know which (h, k) CTR it
        belongs to, so we need to figure it out based on the filename and scan
        number."""
        rsm_hk = self.hk_per_scan_number[associated_scan_nr]
        self.ctrs[rsm_hk].register_reciprocal_space_map(file, hkl_resolution)

    @property
    def max_hk(self):
        return max(ctr.hk[0] for ctr in self.ctrs.values()), max(
            ctr.hk[1] for ctr in self.ctrs.values()
        )

    def write_experiment_metadata(self, metadata_file):
        """Write metadata, such as masked CTR regions, to a JSON file."""
        metadata = {"l_limits": self.l_limits,
                    "mask_outside_scans": self.mask_outside_scan,}
        for ctr in self.ctrs.values():
            metadata[str(ctr.hk)] = {
                "masks": ctr.masks,
                "hk_center": ctr.rsm_center_and_background[0],
                "hk_background": ctr.rsm_center_and_background[1],
            }
        with open(metadata_file, "w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=4)

    def read_experiment_metadata(self, metadata_file, integer_only=True):
        with open(metadata_file, "r", encoding="utf-8") as file:
            metadata = json.load(file)
        # read global experiment settings that apply to all CTRs
        self.l_limits = _decode_limits(metadata)
        self.mask_outside_scan = _decode_mask_outside(metadata)
        file_hk_groups = (key for key in metadata.keys() if key.startswith("("))
        # verify that we have the correct CTRs
        for hk_str in file_hk_groups:
            if not integer_only:
                raise NotImplementedError
            # read out CTR specific settings
            h_str, k_str = hk_str[1:-1].split(",")
            hk = (int(h_str), int(k_str))
            if hk not in self.ctrs.keys():
                raise ValueError(
                    f"CTR with (h, k) = {hk} is not present in the experiment."
                )
            # read out limits, masked regions, and RSM center and background
            self.ctrs[hk].masks = _decode_masks(metadata[hk_str])
            rsm_hk_center = _decode_hk(metadata[hk_str], 'hk_center')
            rsm_hk_background = _decode_hk(metadata[hk_str], 'hk_background')
            if self.ctrs[hk].rsm is not None:
                self.ctrs[hk].set_rsm_center_and_background(
                    rsm_hk_center, rsm_hk_background)


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


def _decode_limits(dct):
    if "l_limits" in dct:
        l_limits = dct["l_limits"]
        return tuple(float(limit) if limit is not None else None for limit in l_limits)
    return (None, None)


def _decode_masks(dct):
    if "masks" in dct:
        read_masks = dct["masks"]
        if read_masks is None:
            return None
        if not isinstance(read_masks, list):
            raise ValueError("Masks must be a list of tuples.")
        return [
            (float(mask_start), float(mask_stop))
            for (mask_start, mask_stop) in read_masks
        ]
    return None


def _decode_mask_outside(dct):
    if "mask_outside_scans" in dct:
        return bool(dct["mask_outside_scans"])
    return True


def _decode_hk(dct, hk_name):
    if hk_name in dct:
        if dct[hk_name] is None:
            return (None, None)
        return tuple(float(hk) for hk in dct[hk_name])
    return (None, None)
