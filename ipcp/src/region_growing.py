import numpy as np
import open3d as o3d
import copy
import logging
import time
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from scipy.spatial import KDTree

import warnings

warnings.filterwarnings('error')

from scipy import spatial

logger = logging.getLogger(__name__)


class RegionGrowing:
    """
    Region growing implementation based on:
    https://pcl.readthedocs.io/projects/tutorials/en/latest/region_growing_segmentation.html
    """
    def __init__(self):
        """ Init variables. """

    def _ransac_plane_fit(self, pcd):
        plane_model, _ = pcd.segment_plane(distance_threshold=0.01,
                                                ransac_n=5,
                                                num_iterations=100)
        [a, b, c, d] = plane_model
        normal = np.array([a,b,c])
        normal /= np.linalg.norm(normal)
        center, _ = pcd.compute_mean_and_covariance()
        return normal, center

    def _growing_neighbours(self, tree, points, radius=.2):
        return flatten(tree.query_ball_point(points, r=radius))
        
    def _region_growing(self, pcd, regions, exclude=[-1]):
        """
        The work of this region growing algorithm is based on the comparison
        of the angles between the points normals.

        The same can also be performed in Python using scipy.spatial.cKDTree
        with query_ball_tree or query.
        """

        regions_to_grow = np.unique(regions)
        regions_to_grow = regions_to_grow[~np.isin(regions_to_grow, exclude)]
        regions_to_grow = np.sort(regions_to_grow)
        
        unassigned_mask = regions == -1 
        conv = np.where(unassigned_mask)[0]

        P = np.asarray(pcd.points)
        N = np.asarray(pcd.normals)
        tree = KDTree(P[unassigned_mask])

        for region_label in tqdm(regions_to_grow):
            R = np.where(regions==region_label)[0]    # Region
            F = R.tolist()                            # Front
            if len(F) < 5:
                continue

            pcd_ = pcd.select_by_index(np.where(regions==region_label)[0])
            n, c = self._ransac_plane_fit(pcd_) # Normal & Center

            while len(F) > 0:
                try:
                    k_idx = conv[self._growing_neighbours(tree, P[F], .2)]
                    F = k_idx[~np.isin(k_idx, R)] # remove points in R and grown
                    F = F[np.rad2deg(np.arccos(np.abs(np.clip(np.dot(N[F], n),-1.0,1.0)))) < 25] # normal criterium
                    F = F[np.abs(np.dot(P[F] - c, n)) < 0.06] # distance criterium
                    R = np.append(R, F)
                except:
                    F = []
            regions[R] = region_label

        logger.debug(f'Done. Added {np.sum(regions[conv]!=-1)} points.')

        return regions
        

    def _edge_refinement(self, pcd, regions):
        logger.debug('Refine edges ...')
        P = np.asarray(pcd.points)
        start = time.time()
        assigned_mask = regions >= 0
        assigned_tree = KDTree(P[assigned_mask])

        d, idx = assigned_tree.query(P[~assigned_mask], k=1, distance_upper_bound=0.25)
        unassigned_regions = np.full(np.sum(~assigned_mask),-1)

        idx_unassigned = np.where(idx<np.sum(assigned_mask))[0]
        idx = idx[idx_unassigned]
        nearest_regions = regions[assigned_mask][idx]

        for r in np.unique(nearest_regions):
            normal, center = self._ransac_plane_fit(pcd.select_by_index(np.where(regions==r)[0]))
            nearest_points = P[~assigned_mask][idx_unassigned[nearest_regions==r]]
            refine_mask = np.abs(np.dot(nearest_points - center, normal)) < .05
            unassigned_regions[idx_unassigned[nearest_regions==r][refine_mask]] = r

        regions[~assigned_mask] = unassigned_regions
        logger.debug(f"Done. Added {np.sum(unassigned_regions>0)} points. {round(time.time()-start,2)}\n")

        return regions
    
    def process(self, pcd, labels):
        """
        Returns the label mask for the given pointcloud.

        Parameters
        ----------
        points : array of shape (n_points, 3)
            The point cloud <x, y, z>.

        Returns
        -------
        An array of shape (n_points,) with dtype=bool indicating which points
        should be labelled according to this fuser.
        """
        logger.debug('KDTree based Region Growing.')

        grown_labels = np.copy(labels)

        grown_labels = self._region_growing(pcd, grown_labels)
        grown_labels = self._edge_refinement(pcd, grown_labels)

        return grown_labels

def flatten(l):
    return np.unique([j for i in l for j in i])

def detection_prob(n, s, N, k=1):
    return 1 - np.power(1 - np.power((n/N),k),s)
