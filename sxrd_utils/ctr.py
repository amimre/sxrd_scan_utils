from copy import deepcopy
import numpy as np

from sxrd_utils.scan import RockingCurve, LScan
from sxrd_utils.fitaid_structure_factors import FitaidOutput


class CTR:
    """Class for connecting all information related to one crystal truncation rod."""

    def __init__(self, h, k):
        self.h, self.k = h, k
        self.hk = (h, k)

        self.masks = None
        self.fits = set()

        # scans
        self.l_scans, self.rocking_scans = set(), set()

    def mask_region(self, from_l, to_l):
        if self.masks is None:
            self.masks = []
        if to_l < from_l:
            raise ValueError("from_l must be <= to_l.")
        # This could be made nicer with the python-intervals package
        self.masks.append([from_l, to_l])

    def clear_masks(self):
        self.masks = None

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

    def masked_fits(
        self, l_limits=(None, None), mask_below=1e-3, filter_type=None
    ):  # TODO: come up with better name
        if filter_type:
            fits = self.filtered_fits(filter_type)
        else:
            fits = self.fits
        # concatenate all fits into one array for easier handling
        all_l = np.concatenate([fit.l_values for fit in fits])
        all_sf = np.concatenate([fit.values for fit in fits])
        masked_nan = np.ma.masked_array(all_sf)  # masked array

        _masks = deepcopy(self.masks) if self.masks else []
        if l_limits[0] is not None:
            _masks.append((-np.inf, l_limits[0]))
        if l_limits[1] is not None:
            _masks.append((l_limits[1], np.inf))

        # apply masked l_values
        if _masks:
            masked_l_values = [
                np.logical_and(all_l > mask[0], all_l < mask[1]) for mask in _masks
            ]
            masked_l_values = np.logical_or.reduce(masked_l_values)
        else:
            masked_l_values = np.full_like(all_l, False)
        # mask any values below mask_below
        masked_below_threshold = all_sf <= mask_below
        # mask any NaNs
        masked_nan = np.isnan(all_sf)

        # combine masks
        combined_mask = np.logical_or.reduce(
            (masked_l_values, masked_below_threshold, masked_nan)
        )

        # drop masked values
        masked_l, masked_sf = all_l[~combined_mask], all_sf[~combined_mask]

        # finally sort by L values
        sorting_indices = np.argsort(masked_l)
        masked_l, masked_sf = masked_l[sorting_indices], masked_sf[sorting_indices]

        return masked_l, masked_sf
