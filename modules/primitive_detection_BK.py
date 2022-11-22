"""Floor Splitting Module"""
import numpy as np
import logging
import os
import time
import open3d as o3d
import subprocess

from scipy import signal
import matplotlib.pyplot as plt
from src.region_growing import RegionGrowing
import src.utils.math_utils as math_utils

logger = logging.getLogger(__name__)

NOISE = 0
SLANTED = 1
ALMOST_VERTICAL = 2
ALMOST_HORIZONTAL = 3

class PrimitiveDetector:
    """
    PrimitiveDetector class for the geometric primitives in the pointcloud. 
    The class labels primitives as horizontal, vertical or slanted.

    Parameters
    ----------
    min_peak_height : int (default=2500)
        The required height of a peak in the vertical density profile.
    threshold : int (default=250)
        The required threshold of peaks, the vertical distance to its neighboring samples.
    distance : int (default=20)
        The required minimal horizontal distance (>= 1) in samples between neighbouring peaks.
    prominence : int (default=2)
        The required prominence of peaks.
    min_floor_height : float (default=2.1)
        The minimal height a floor must be.
    floor_buffer : float (default=0.1)
        The buffer used to include the whole floor/ceiling.
    """

    def __init__(self, excecutable_path, subsample_size=0.03):
        self.excecutable_path = excecutable_path
        self.subsample_size = subsample_size

    def _detect_planes(self, pcd, labels):
        
        mask = np.where(labels==NOISE)[0]
        pcd_planes = o3d.t.geometry.PointCloud()
        pcd_planes.point['positions'] = o3d.core.Tensor(
            np.asarray(pcd.select_by_index(mask).points), o3d.core.float64)
        pcd_planes.point['plane_index'] = o3d.core.Tensor(np.full((len(mask),1),-1),o3d.core.int32)
        
        tmp_file = 'wrdir/cloud.ply'
        if not os.path.isdir('wrdir'):
            os.mkdir('wrdir')

        # ALMOST_HORIZONTAL
        mask = np.where(labels==ALMOST_HORIZONTAL)[0]
        pcd_ = efficient_ransac(pcd.select_by_index(mask), self.excecutable_path, tmp_file, .12)
        pcd_planes = pcd_planes.append(pcd_)

        # ALMOST_VERTICAL
        mask = np.where(labels==ALMOST_VERTICAL)[0]
        pcd_ = efficient_ransac(pcd.select_by_index(mask), self.excecutable_path, tmp_file, .12)
        pcd_.point['plane_index'][pcd_.point['plane_index']>=0] += pcd_planes.point['plane_index'].max() + 1
        pcd_planes = pcd_planes.append(pcd_)

        # SLANTED
        mask = np.where(labels==SLANTED)[0]
        pcd_ = efficient_ransac(pcd.select_by_index(mask), self.excecutable_path, tmp_file, .12)
        pcd_.point['plane_index'][pcd_.point['plane_index']>=0] += pcd_planes.point['plane_index'].max() + 1
        pcd_planes = pcd_planes.append(pcd_)

        # Remove wrdir
        os.rmdir('wrdir')

        # Convert back
        plane_labels = np.hstack(pcd_planes.point['plane_index'].numpy())
        pcd_planes = pcd_planes.to_legacy()

        return pcd_planes, plane_labels

    def _labels_to_primitives(self, pcd, labels, exclude_labels=[-1], min_points=200):
        planes = {}
        rectangle_labels = [r for r in np.unique(labels) if r not in exclude_labels and np.sum(labels==r) > min_points]
        for i in rectangle_labels:
            region_cloud = pcd.select_by_index(np.where(labels==i)[0])
            region_pts = np.asarray(region_cloud.points)
            pts_surface = len(region_pts)*(self.subsample_size**2)
            
            # TODO: filter outliers out!
            a, b, c, d = region_cloud.segment_plane(distance_threshold=0.01,
                                                    ransac_n=5,
                                                    num_iterations=100)[0]
            normal = np.array([a,b,c])
            normal /= np.linalg.norm(normal)
            center = np.mean(region_pts,axis=0)
            slope = np.rad2deg(np.arccos(np.abs(normal[2] / 1)))

            # Compute bounding box
            if slope < 5:
                plane_type = ALMOST_HORIZONTAL
                min_bbox = math_utils.minimum_bounding_rectangle(region_pts[:,:2])
                if min_bbox[2] < .2:
                    continue
                bbox_points = min_bbox[0]
                bbox_points = np.concatenate((bbox_points, np.full((4,1), center[2])),axis=1) 
            else:
                xaxis = np.cross(normal, [0, 0, 1])
                yaxis = np.cross(normal, xaxis)
                xaxis /= np.linalg.norm(xaxis)
                yaxis /= np.linalg.norm(yaxis)

                new_x = np.dot(region_pts-center, xaxis)    
                new_y = np.dot(region_pts-center, yaxis)

                xmin, ymin, xmax, ymax = math_utils.compute_bounding_box(np.vstack([new_x, new_y]).T)
                bbox_points = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
                bbox_points = center + bbox_points[:,0][:, None]*xaxis + bbox_points[:,1][:, None]*yaxis
                if slope > 80:
                    plane_type = ALMOST_VERTICAL
                else:
                    plane_type = SLANTED

            surface = bbox_area(bbox_points)
            coverage = pts_surface/surface
            if coverage < 0.05: # minimal coverage
                continue

            line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(bbox_points),
                lines=o3d.utility.Vector2iVector([[0, 1],[1, 2],[2, 3],[3, 0]]),
            )

            plane_object = {
                'region': i,
                'bbox': bbox_points,
                'lineset': line_set,
                'surface': surface,
                'coverage': coverage,
                'center': center,
                'normal': normal,
                'D': d,
                'slope': slope,
                'type': plane_type
            }

            planes[i] = plane_object

        return planes

    def process(self, pcd):
        """
        Parameters
        ----------
        points : array of shape (n_points, 3)
            The point cloud <x, y, z>.

        Returns
        -------
        An array of masks, where each mask represents a floor in the pointcloud.
        """

        logger.debug('Detecting primitives in pointcloud...')
        labels = np.full(len(pcd.points), SLANTED, dtype=np.uint8)

        logger.debug('Labelling points...')
        verticality = compute_verticality(pcd, radius=.2)
        labels[verticality > 0.75] = ALMOST_VERTICAL
        labels[verticality < 0.05] = ALMOST_HORIZONTAL
        pcd_ = pcd.select_by_index(np.where(labels != ALMOST_VERTICAL)[0])
        selection_planarity = compute_planarity(pcd_, radius=.15)
        labels[np.where(labels != ALMOST_VERTICAL)[0][selection_planarity < .5]] = NOISE
        pcd_ = pcd.select_by_index(np.where(labels == ALMOST_VERTICAL)[0])
        selection_planarity = compute_planarity(pcd_, radius=.15)
        labels[np.where(labels == ALMOST_VERTICAL)[0][selection_planarity < .3]] = NOISE

        # RANSAC on labels
        logger.debug('RANSAC for point groups...')
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamRadius(radius=.12))
        pcd, plane_labels = self._detect_planes(pcd, labels)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamRadius(radius=.12))
        logger.debug(f'{len(np.unique(plane_labels))} primitives.')

        # Grow planes
        RG = RegionGrowing()
        plane_labels = RG.process(pcd, plane_labels)
        primitives = self._labels_to_primitives(pcd, plane_labels)

        return pcd, primitives, plane_labels

def compute_verticality(pcd, radius):
    '''Bla'''
    logger.debug('Computing verticality..')
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamRadius(radius=radius))
    verticality = 1 - np.abs(np.asarray(pcd.normals)[:,2])
    logger.debug('Done.')
    return verticality

def compute_planarity(pcd, radius):
    '''Bla'''
    logger.debug('Computing planarity..')
    pcd.estimate_covariances(
        search_param=o3d.geometry.KDTreeSearchParamRadius(radius=radius))
    eig_val, _ = np.linalg.eig(np.asarray(pcd.covariances))
    eig_val = np.sort(eig_val, axis=1)
    planarity = (eig_val[:,1]-eig_val[:,0])/eig_val[:,2]
    logger.debug('Done.')
    return planarity
       
def efficient_ransac(pcd, excecutable_path, file_path, normals_radius=.12):
    '''Bla'''
    labels = np.full(len(pcd.points), -1)
    logger.debug(f'Start Efficient RANSAC on cloud of {len(labels)} points ...')
    start = time.time()

    try:
        # Compute normals
        if not pcd.has_normals():
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamRadius(radius=normals_radius))

        # Write point cloud
        o3d.io.write_point_cloud(file_path, pcd)

        # RANSAC
        subprocess.run([excecutable_path, file_path, file_path, '0.005', '200', '0.04', '0.12', '0.5'],
             timeout=20, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Read point cloud
        pcd = o3d.t.io.read_point_cloud(file_path)

    except subprocess.TimeoutExpired:
        logger.error('RANSAC timeout')
    except subprocess.CalledProcessError as CPE:
        logger.error(f'Error in RANSAC with returncode {CPE.returncode}.')
    except Exception as e:
        logger.error(str(e))

    if os.path.isfile(file_path):
        os.remove(file_path)

    logger.debug(f'Done. {np.round(time.time()-start,2)}s')
    
    return pcd

def bbox_area(bbox):
    return np.linalg.norm(bbox[0,:]-bbox[1,:]) * np.linalg.norm(bbox[2,:]-bbox[1,:])
