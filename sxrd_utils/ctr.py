import numpy as np

from sxrd_utils.scan import RockingCurve, LScan
from sxrd_utils.fitaid_structure_factors import FitaidOutput


class CTR:
    """Class for connecting all information related to one crystal truncation rod."""

    def __init__(self, h, k):
        self.h, self.k = h, k
        self.hk = (h, k)

        self.l_limits = (None, None)
        self.masks = []
        self.fits = set()

        # scans
        self.l_scans, self.rocking_scans = set(), set()

    def mask_region(self, from_l, to_l):
        # This could be made nicer with the python-intervals package
        self.masks.append([from_l, to_l])

    def clear_masks(self):
        self.masks = []

    @property
    def scans(self):
        return set.union(self.l_scans, self.rocking_scans)

    @property
    def scan_numbers(self):
        return set(scan.id for scan in self.scans)

    def register_scan(self, scan):
        if isinstance(scan, LScan):
            self.l_scans.add(scan)
        elif isinstance(scan, RockingCurve):
            self.rocking_scans.add(scan)
        else:
            raise ValueError("Not a valid scan type.")

    def register_fit(self, fit):
        if not isinstance(fit, FitaidOutput):
            raise ValueError("Not a valid Fitaid fit.")
        self.fits.add(fit)

    def filtered_fits(self, filter_type):
        return set(fit for fit in self.fits if fit.type == filter_type)
