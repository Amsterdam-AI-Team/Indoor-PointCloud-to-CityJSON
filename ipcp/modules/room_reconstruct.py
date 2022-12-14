"""room_reconstruct Module"""
import numpy as np
import logging
import os

import subprocess
import pymeshlab
from pathlib import Path
import src.utils.pcd_utils as pcd_utils
import open3d as o3d
import time
import shutil

logger = logging.getLogger(__name__)

ALMOST_HORIZONTAL = 3

class RoomReconstructor:
    """
    PlaneReconstruct class for the reconstruction of a point cloud
    to a mesh.

    Parameters
    ----------
    fitting : float (default=0.25)
        The fitting parameter for PolyFit.
    coverage : float (default=0.45)
        The coverage parameter for PolyFit.
    complexity : float (default=0.60)
        The complexity parameter for PolyFit.
    """

    def __init__(self, exe_ransac_polyfiy, exe_polyfit, fitting=0.5, coverage=0.27,
                 complexity=0.23):
        self.exe_ransac_polyfiy = exe_ransac_polyfiy
        self.exe_polyfit = exe_polyfit
        self.fitting = fitting
        self.coverage = coverage
        self.complexity = complexity

    def _clean_mesh(self, meshset):
        meshset.meshing_remove_duplicate_vertices()
        meshset.meshing_merge_close_vertices()
        meshset.meshing_re_orient_faces_coherentely()
        volume = meshset.get_geometric_measures()['mesh_volume']
        if volume < 0:
            meshset.meshing_invert_face_orientation()
            volume = meshset.get_geometric_measures()['mesh_volume']
            
        return meshset

    def _add_missing(self, points, labels, primitives):
        horizontal_z = [points[labels==l,2].min() for l in np.unique(labels) if l in primitives.keys() and primitives[l]['type'] == ALMOST_HORIZONTAL]
        if len(horizontal_z) == 0 or np.min(horizontal_z) - points[:,2].min() > 0.75:
            logger.info("No floor, create")
            bin_size = 0.15
            x_bins = np.arange(points[:,0].min(),points[:,0].max(),bin_size)
            y_bins = np.arange(points[:,1].min(),points[:,1].max(),bin_size)
            projection, xedges, yedges = np.histogram2d(points[:,0], points[:,1], bins=(x_bins, y_bins))
            coords = np.where(projection>0)
            coords = np.vstack([xedges[coords[0]], yedges[coords[1]], np.full(len(coords[0]), points[:,2].min())]).T
            points = np.append(points, coords, axis=0)
            labels = np.append(labels, np.full(len(coords), labels.max()+1))

        return points, labels

    def _ransac_polyfit(self, pcd):
        
        meshset = None
        try:
            # Write file
            logger.debug(f'Write temp file to {self.ply_infile}')
            pcd = o3d.t.geometry.PointCloud(pcd.point['positions'])
            o3d.t.io.write_point_cloud(self.ply_infile, pcd, write_ascii=True)

            logger.debug(f'Run PolyFit')
            subprocess.run([self.exe_ransac_polyfiy, self.ply_infile, self.dir_path,'0.01','100','0.1','1.5','0.6','0.5','0.27','0.23'],
                    timeout=10, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            meshset = pymeshlab.MeshSet()
            meshset.load_new_mesh(self.mesh_outfile)
            meshset = self._clean_mesh(meshset)

            logger.debug('Succes.')

        except subprocess.TimeoutExpired:
            logger.debug('Ployfit timeout')
        except subprocess.CalledProcessError as CPE:
            logger.debug(f'Error in Ployfit with returncode {CPE.returncode}.')
        except Exception as e:
            logger.debug('Hai', str(e))

        return meshset

    def _user_polyfit(self, pcd):

        meshset = None

        try: 
            # Write file
            logger.debug(f'Write temp file to {self.ply_infile}')
            o3d.t.io.write_point_cloud(self.ply_infile, pcd, write_ascii=True)
            
            logger.debug(f'Run PolyFit')

            subprocess.run([self.exe_polyfit, self.ply_infile, self.dir_path,
                             str(self.fitting), str(self.coverage),
                             str(self.complexity)], timeout=15, 
                             check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            meshset = pymeshlab.MeshSet()
            meshset.load_new_mesh(self.mesh_outfile)
            meshset = self._clean_mesh(meshset)

            logger.debug('Succes.')

        except:
            meshset = self._ransac_polyfit(pcd)

        return meshset

    def process(self, points, primitive_index, primitives):
        """
        Parameters
        ----------
        points : array of shape (n_points, 3)
            The point cloud <x, y, z> of a room to reconsturct.

        Returns
        -------
        mesh : MeshSet
            The volume of the input mesh.
        """
        logger.debug(f'Room reconstruction for pointcloud of {len(points)} points and {len(np.unique(primitive_index))} faces')

        meshset = None

        self.dir_path = './tmp_pr'
        self.ply_infile = self.dir_path + '/polyfit_input.ply'
        self.mesh_outfile = self.dir_path + '/polyfit_result.obj'
        if not os.path.isdir(self.dir_path):
            os.mkdir(self.dir_path)

        # points, primitive_index = self._add_missing(points, primitive_index, primitives)

        pcd = o3d.t.geometry.PointCloud(o3d.core.Tensor(points, o3d.core.float32))
        segment_index = np.unique(primitive_index, return_inverse=True)[1][:,np.newaxis]
        pcd.point['segment_index'] = o3d.core.Tensor(segment_index, o3d.core.int32)

        if len(points) > 20000:
            pcd = pcd.voxel_down_sample(0.1)

        meshset = self._user_polyfit(pcd)

        shutil.rmtree('./tmp_pr')

        return meshset
