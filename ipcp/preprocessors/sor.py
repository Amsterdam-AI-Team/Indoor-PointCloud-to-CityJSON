"""Statistical Outlier Removal Preprocessor"""
import numpy as np
import logging
import open3d as o3d

logger = logging.getLogger(__name__)

class SOR:
    """
    Preprocessor class for the statistical outlier removal of pointcloud data. 
    It computes first the average distance of each point to its neighbors.
    Then it rejects the points that are farther than the average distance plus
    a number of times the standard deviation (the max distance will be:
    average distance + n * standard deviation).

    Parameters
    ----------
    knn : int (default=6)
        The number of neighbours that will be used to compute 
        the 'mean distance to neighbors' for each point.
    n_sigma : float (default=1.0)
        The standard deviation. 
    """

    def __init__(self, knn=6, n_sigma=1.0):
        """ Init variables """
        self.knn = knn
        self.n_sigma = n_sigma

    def process(self, pcd):
        """ 
        Parameters
        ----------   
        points : array of shape (n_points, 3)
            The point cloud <x, y, z>.
        """

        logger.info(f'Removing outliers...')

        refCloud, _ = pcd.remove_statistical_outlier(nb_neighbors=self.knn, std_ratio=self.n_sigma)
        return refCloud
