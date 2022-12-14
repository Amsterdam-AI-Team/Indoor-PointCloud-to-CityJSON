"""Primitive Detection Module"""

import os
import logging
import subprocess
import numpy as np
import open3d as o3d
from scipy.spatial import KDTree

import src.utils.math_utils as math_utils
from src.region_growing import RegionGrowing
from src.utils.pcd_utils import merge_point_clouds

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

    def _plane_normal(self, pcd):
        a,b,c,_ =  pcd.segment_plane(distance_threshold=0.03,
                                    ransac_n=5,
                                    num_iterations=10)[0]
        normal = np.array([a,b,c])
        normal /= np.linalg.norm(normal)
        return normal

    def _merge_parallel(self, pcd, labels):
        un_labels_ = np.unique(labels[labels>-1])
        normals = {}
        for label in un_labels_:
            pcd_label = pcd.select_by_index(np.where(labels==label)[0])
            normal = self._plane_normal(pcd_label)
            normals[label] = normal

        points = np.asarray(pcd.points)
        labels_iter = list(un_labels_)
        merge_count = 0
        pairs = []
        while len(labels_iter) > 0:
            label = labels_iter.pop(0)
            label_mask = np.where(labels==label)[0]
            kd_i = KDTree(points[label_mask])
            label_normal = normals[label]
            
            for i in labels_iter:
                angle = np.rad2deg(angle_between(label_normal[:2], normals[i][:2]))
                if angle < 3:  
                    num_pairs = np.sum(kd_i.query(points[labels==i], k=1, distance_upper_bound=.12)[1]<len(label_mask))
                    if num_pairs > 10:
                        pairs.append((label,i))
                        merge_count += 1
                
        lookup = {}
        for i,j in pairs:
            if j in lookup:
                lookup[i] = lookup[j]
            elif i in lookup:
                lookup[j] = lookup[i]
            else:
                lookup[j] = i 
        
        for k,v in lookup.items():
            labels[labels==k] = v
            
        labels = consecutive_labels(labels)

        logger.debug(f'merged: {merge_count}')
        return labels
    
    def _detect_vertical(self, pcd):
        tmp_file = './tmp_pr/cloud.ply'
        if not os.path.isdir('./tmp_pr'):
            os.mkdir('./tmp_pr')

        # point selection
        labels = np.full(len(pcd.points),-1)
        verticality = compute_verticality(pcd, radius=.2)
        mask = np.where(verticality > 0.75)[0]
        pcd_ = pcd.select_by_index(mask)
        planarity = compute_planarity(pcd_, radius=.15)
        mask = mask[planarity > .3]

        # detect primtives
        pcd_ = pcd.select_by_index(mask)
        pcd_, ransac_lables = efficient_ransac(pcd_, self.excecutable_path, tmp_file)
        labels[-len(mask):] = ransac_lables
        pcd = merge_point_clouds(pcd.select_by_index(mask, invert=True), pcd_)
        
        # region grow
        region_growing = RegionGrowing()
        labels = region_growing.process(pcd, labels)

        # merge primitives
        labels = self._merge_parallel(pcd, labels)

        return pcd, labels
    
    def _detect_non_vertical(self, pcd):
        tmp_file = './tmp_pr/cloud.ply'
        if not os.path.isdir('./tmp_pr'):
            os.mkdir('./tmp_pr')

        # point selection
        labels = np.full(len(pcd.points), -1)
        planarity = compute_planarity(pcd, radius=.15)
        mask = np.where(planarity > .5)[0]

        # detect primitives
        pcd_ = pcd.select_by_index(mask)
        pcd_, ransac_lables = efficient_ransac(pcd_,
                            self.excecutable_path, tmp_file, prob='0.005', 
                            eps='0.06', cluster_thres='0.15')
        labels[-len(mask):] = ransac_lables
        pcd = merge_point_clouds(pcd.select_by_index(mask, invert=True), pcd_)

        # clean verticals
        un_labels_ = np.unique(labels[labels>-1])
        for label in un_labels_:
            pcd_ = pcd.select_by_index(np.where(labels==label)[0])
            normal = self._plane_normal(pcd_)
            angle = np.rad2deg(np.arccos(np.abs(normal[2] / 1)))
            if angle > 85:
                labels[labels==label] = -1

        # region grow
        region_growing = RegionGrowing()
        labels = region_growing.process(pcd, labels)

        return pcd, labels
    
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

        logger.debug('Searching vertical primitives...')
        pcd, labels = self._detect_vertical(pcd)
        logger.debug(f'Done. Found {len(np.unique(labels))-1} primitives')

        logger.debug('Searching other primitives...')
        idx = np.where(labels==-1)[0]
        pcd_ = pcd.select_by_index(idx)
        pcd_, labels_ = self._detect_non_vertical(pcd_)
        logger.debug(f'Done. Found {len(np.unique(labels))-1} primitives')

        # Merge
        pcd = merge_point_clouds(pcd.select_by_index(idx, invert=True), pcd_)
        labels_[labels_>-1] += labels.max()+1
        labels = np.concatenate((labels[labels!=-1], labels_))
        primitives = self._labels_to_primitives(pcd, labels)

        return pcd, primitives, labels

def compute_verticality(pcd, radius):
    '''Bla'''
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamRadius(radius=radius))
    verticality = 1 - np.abs(np.asarray(pcd.normals)[:,2])
    return verticality

def compute_planarity(pcd, radius):
    '''Bla'''
    pcd.estimate_covariances(
        search_param=o3d.geometry.KDTreeSearchParamRadius(radius=radius))
    eig_val, _ = np.linalg.eig(np.asarray(pcd.covariances))
    eig_val = np.sort(eig_val, axis=1)
    planarity = (eig_val[:,1]-eig_val[:,0])/eig_val[:,2]
    return planarity
       
def efficient_ransac(pcd, excecutable_path, file_path, normals_radius=.12, prob='0.001',
                     min_pts='200', eps='0.03', cluster_thres='0.12', normal_thres='0.5'):
    '''Bla'''

    labels = np.full(len(pcd.points),-1)

    try:
        # Compute normals
        if not pcd.has_normals():
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamRadius(radius=normals_radius))

        # Write point cloud
        o3d.io.write_point_cloud(file_path, pcd)

        # RANSAC
        subprocess.run([excecutable_path, file_path, file_path, prob, min_pts, eps, cluster_thres, normal_thres],
             timeout=20, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Read point cloud
        pcd_ = o3d.t.io.read_point_cloud(file_path)
        pcd = pcd_.to_legacy()
        labels = np.hstack(pcd_.point['plane_index'].numpy())

    except subprocess.TimeoutExpired:
        logger.error('RANSAC timeout')
    except subprocess.CalledProcessError as CPE:
        logger.error(f'Error in RANSAC with returncode {CPE.returncode}.')
    except Exception as e:
        logger.error(str(e))

    if os.path.isfile(file_path):
        os.remove(file_path)
    
    return pcd, labels

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    
def consecutive_labels(labels):
    labels[labels>-1] = np.unique(labels[labels>-1], return_inverse=True)[1]
    return labels
    
def bbox_area(bbox):
    return np.linalg.norm(bbox[0,:]-bbox[1,:]) * np.linalg.norm(bbox[2,:]-bbox[1,:])
