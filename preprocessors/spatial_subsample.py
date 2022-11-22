"""Spatial Subsample Preprocessor"""
import numpy as np
import logging
import open3d as o3d

logger = logging.getLogger(__name__)

class SpatialSubsample:
    """
    Preprocessor class for the subsampling of pointcloud data. 
    The class reduce the number of points in the pointcloud using spatial.

    Parameters
    ----------
    min_distance : float (default=0.05)
        The minimal space between points.
    """

    def __init__(self, min_distance=0.05):
        """ Init variables """
        self.min_distance = min_distance

    def process(self, pcd):
        """ 
        Parameters
        ----------   
        points : array of shape (n_points, 3)
            The point cloud <x, y, z>.
        """

        logger.info(f'Reducing pointcloud...')

        refCloud = pcd.voxel_down_sample(self.min_distance)
        
        return refCloud
