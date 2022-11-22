"""
This module provides tools for interpolating data.
"""

import numpy as np

class FastGridInterpolator:
    """
    Class to perform fast interpolation using gridded data. The interpolator
    simply returns the values of the grid cells in which the queried points
    fall. Grid coordinates are assumed to be the centroids of each grid cell.

    Parameters
    ----------
    bin_x : list or array-like
        The x-coordinates of the grid (ascending).

    bin_y : list or array-like
        The y-coordinates of the grid (decsending).

    values : array of shape (Ny, Nx)
        The values of the gridded data.
    """

    def __init__(self, bin_x, bin_y, values):
        self.bin_x = bin_x[:-1]
        self.bin_y = bin_y[:-1]
        self.values = values

    def __call__(self, positions):
        """
        Evaluate the interpolator at the given positions.

        Parameters
        ----------
        positions : array of shape (Np, 2)
            Array of points to query. The first column contains the x-values,
            the second column contains the y-values.
        """
        x_idx = np.digitize(positions[:, 0], self.bin_x) - 1
        y_idx = np.digitize(positions[:, 1], self.bin_y) - 1
        return self.values[x_idx, y_idx]