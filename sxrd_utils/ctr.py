from copy import deepcopy

import numpy as np

from sxrd_utils.scan import RockingCurve, LScan
from sxrd_utils.fitaid_structure_factors import FitaidOutput
from sxrd_utils.structure_factor_extraction import ReciprocalSpaceMap


class CTR:
    """Class for connecting all information related to one crystal truncation rod."""

    def __init__(self, h, k):
        self.h, self.k = h, k
        self.hk = (h, k)

        self.masks = None
        self.fits = set()

        # scans
        self.l_scans, self.rocking_scans = set(), set()

        self._rsm = None

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

    def register_reciprocal_space_map(self, file, hkl_resolution):
        """Register a reciprocal space map with the experiment."""
        self._rsm = ReciprocalSpaceMap(file, hkl_resolution)

    @property
    def rsm(self):
        return self._rsm

    @property
    def rsm_valid_center_and_background_set(self):
        if self.rsm is None:
            return False
        return not (None in self.rsm.central_hk
                    or None in self.rsm.background_hk)

    @property
    def rsm_center_and_background(self):
        if not self.rsm_valid_center_and_background_set:
            return None, None
        return self.rsm.central_hk, self.rsm.background_hk

    def set_rsm_center_and_background(self, hk_center, hk_background):
        if self.rsm is None:
            raise ValueError("No reciprocal space map registered.")
        self.rsm.central_hk = hk_center
        self.rsm.background_hk = hk_background


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
        self,
        l_limits=(None, None),
        sf_threshold=1e-3,
        filter_type=None,
        mask_outside_scans=True,  # mask any L region outside what is captured in self.scans
        fill_nan=True,
    ):  # TODO: come up with better name
        if filter_type:
            fits = self.filtered_fits(filter_type)
        else:
            fits = self.fits
        # concatenate all fits into one array for easier handling
        all_l = np.concatenate([fit.l_values for fit in fits])
        all_sf = np.concatenate([fit.values for fit in fits])
        masked_nan = np.ma.masked_array(all_sf)  # masked array

        # mask l limits
        _masks = deepcopy(self.masks) if self.masks else []
        if l_limits[0] is not None:
            _masks.append((-np.inf, l_limits[0]))
        if l_limits[1] is not None:
            _masks.append((l_limits[1], np.inf))

        # mask areas outside scan
        if mask_outside_scans:
            covered_regions = [(np.min(scan.l_values), np.max(scan.l_values)) for scan in self.scans]
            covered_l_values = [
                np.logical_and(all_l >= region[0], all_l <= region[1]) for region in covered_regions
            ]
            covered_l_values = ~np.logical_or.reduce(covered_l_values)
        else:
            covered_l_values = np.full_like(all_l, False)

        # apply masked l_values
        if _masks:
            masked_l_values = [
                np.logical_and(all_l >= mask[0], all_l <= mask[1]) for mask in _masks
            ]
            masked_l_values = np.logical_or.reduce(masked_l_values)
        else:
            masked_l_values = np.full_like(all_l, False)

        # mask any values below mask_below
        masked_below_threshold = all_sf <= sf_threshold
        # mask any NaNs
        masked_nan = np.isnan(all_sf)

        # combine masks
        combined_mask = np.logical_or.reduce(
            (masked_l_values, masked_below_threshold, masked_nan, covered_l_values)
        )

        if fill_nan:  # replace masked values with NaNs
            masked_l = all_l
            masked_sf = np.ma.filled(
                np.ma.masked_array(all_sf, mask=combined_mask), fill_value=np.nan
            )
        else:  # drop masked values
            masked_l, masked_sf = all_l[~combined_mask], all_sf[~combined_mask]

        # finally sort by L values
        sorting_indices = np.argsort(masked_l)
        masked_l, masked_sf = masked_l[sorting_indices], masked_sf[sorting_indices]

        return masked_l, masked_sf
