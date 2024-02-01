import numpy as np
import binoculars


class ReciprocalSpaceMap:

    def __init__(self, file, hkl_resolution):
        self.file = file
        self.space = binoculars.load(file)
        # immediately rebin to desired resolution
        # (saves memory)
        self.space = self.space.rebin(hkl_resolution)
        self.voxel_h_values, self.voxel_k_values, _ = self.space.get_grid()
        self._central_hk = (None, None)
        self._background_hk = (None, None)

    @property
    def h_axis_array(self):
        return np.linspace(self.space.axes[0].min,
                           self.space.axes[0].max,
                           len(self.space.axes[0]))

    @property
    def k_axis_array(self):
        return np.linspace(self.space.axes[1].min,
                           self.space.axes[1].max,
                           len(self.space.axes[1]))

    @property
    def l_axis_array(self):
        return np.linspace(self.space.axes[2].min,
                           self.space.axes[2].max,
                           len(self.space.axes[2]))

    @property
    def background_hk(self):
        return self._background_hk

    @background_hk.setter
    def background_hk(self, hk):
        self._background_hk = hk

    @property
    def central_hk(self):
        return self._central_hk

    @central_hk.setter
    def central_hk(self, hk):
        self._central_hk = hk

    @property
    def center_l(self):
        # center of l axis (any valid l value would do)
        return (self.space.axes[2].min + self.space.axes[2].max) / 2

    def mask_l_axis_cylinder(self, hk_center, radius):
        """Return a cylindrical mask for the reciprocal space map along the l axis.

        Return a boolean mask for the binned reciprocal space map, where all voxels
        outside the cylinder with radius `radius` around the l axis are masked.
        The radius is given in reciprocal space units and takes h and k into equally
        into account, i.e. a voxel is inside the cylinder if
        |(h, k, l) - (h_0, k_0, l)| <= radius."""

        # compare distance of each voxel to center with radius
        h_distance_to_center = self.voxel_h_values - hk_center[0]
        k_distance_to_center = self.voxel_k_values - hk_center[1]
        hk_distance_to_center = np.linalg.norm(
            np.array([h_distance_to_center, k_distance_to_center]), axis=0)
        return hk_distance_to_center >= radius

    @property
    def central_pixel_voxel_index(self):
        if None in self.central_hk:
            raise RuntimeError("No central pixel has been assigned yet.")
        return self.space.get_key((self.central_hk[0], self.central_hk[1], self.center_l))

    def l_profile(self, hk, radius):
        """Return the l profile for a given hk value.

        Return the l profile for a given hk value, where the profile is the sum of
        all voxel intensities in a cylinder with radius `radius` around the l axis.
        """

        mask = self.mask_l_axis_cylinder(hk, radius)

        profile = np.sum(np.ma.masked_array(self.space.photons,
                                  mask=self.mask_l_axis_cylinder(hk, radius)),
               axis=(0,1))

        return profile


    def raw_center_profile(self, radius):
        """Return the l profile for the central pixel.

        Return the l profile for the central pixel, where the profile is the sum of
        all voxel intensities in a cylinder with radius `radius` around the l axis.
        """
        return self.l_profile(self.central_hk, radius)

    # TODO: smoothing over background
    def background_profile(self, radius):
        """Return the l profile for the background pixel.

        Return the l profile for the background pixel, where the profile is the sum of
        all voxel intensities in a cylinder with radius `radius` around the l axis.
        """
        return self.l_profile(self.background_hk, radius)

    def background_subtracted_profile(self, radius, bkg_radius=None):
        """Subtract the background profile from the central profile.

        Subtract the background profile from the central profile, where the profile is
        the sum of all voxel intensities in a cylinder with radius `radius` around the
        l axis.
        """
        if bkg_radius is None:
            bkg_radius = radius
        return self.raw_center_profile(radius) - self.background_profile(radius)
