import numpy as np
import binoculars

from sxrd_utils.ctr import CTR

class ReciprocalSpaceMap:

    def __init__(self, file, hkl_resolution):
        self.file = file
        self.space = binoculars.load(file)
        # immediately rebin to desired resolution
        # (saves memory)
        self.space = self.space.rebin(hkl_resolution)
        self.voxel_h_values, self.voxel_k_values, _ = self.space.get_grid()
        self.ctr_central_hk = (None, None)

    @property
    def center_l(self):
        # center of l axis (for our purposes any valid l value would do)
        return(self.space.axes[2].min + self.space.axes[2].max) / 2

    def mask_l_axis_cylinder(self, hk_center, radius):
        """Return a cylindical mask for the reciprocal space map along the l axis.

        Return a boolean mask for the binned reciprocal space map, where all voxels
        outside the cylinder with radius `radius` around the l axis are masked.
        The radius is given in reciprocal space units and takes h and k into equally
        into account, i.e. a voxel is inside the cylinder if
        |(h, k, l) - (h_0, k_0, l)| <= radius."""

        # compare distance of each voxel to center with radius
        h_distance_to_center = self.voxel_h_values - hk_center[0]
        k_distance_to_center = self.voxel_k_values - hk_center[1]
        hk_distance_to_center = np.linalg.norm(np.array([h_distance_to_center, k_distance_to_center]), axis=0)
        return hk_distance_to_center >= radius

    def integrate_hk(self, mask):
        """Integrate over the reciprocal space map along the h and k axes.

        Return a 1D array of the integrated intensity along the h and k axes.
        The mask is a boolean array of the same shape as the reciprocal space map
        and is used to exclude voxels from the integration."""

        return np.sum(self.space.photons * mask, axis=(0, 1))

    @property
    def central_voxel_index(self):
        if None in self.ctr_central_hk:
            raise RuntimeError("No central pixel has been assigned yet.")
        return self.space.get_key((self.ctr_central_hk[0], self.ctr_central_hk[1], self.center_l))
