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

    def __init__(self, excecutable_path, fitting=0.2, coverage=0.1,
                 complexity=0.7):
        self.excecutable_path = excecutable_path
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

    def process(self, points, primitive_index, primitives, normals=None):
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

        dir_path = './tmp_pr'
        ply_infile = dir_path + '/polyfit_input.ply'
        mesh_outfile = dir_path + '/polyfit_result.obj'
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)

        points, primitive_index = self._add_missing(points, primitive_index, primitives)


        try:
            
            # Create point cloud
            pcd = o3d.t.geometry.PointCloud(o3d.core.Tensor(points, o3d.core.float32))
            segment_index = np.unique(primitive_index, return_inverse=True)[1][:,np.newaxis]
            pcd.point['segment_index'] = o3d.core.Tensor(segment_index, o3d.core.int32)
            if normals is None:
                pcd.estimate_normals(max_nn=16) # parameters or use old
                pcd.point['normals'] = pcd.point['normals'].to(o3d.core.float32)
            else:
                pcd.point['normals'] = o3d.core.Tensor(normals, o3d.core.float32)
            

            # Write file
            logger.debug(f'Write temp file to {ply_infile}')
            o3d.t.io.write_point_cloud(ply_infile, pcd, write_ascii=True)
            
            logger.debug(f'Run PolyFit')

            subprocess.run([self.excecutable_path, ply_infile, dir_path,
                             str(self.fitting), str(self.coverage),
                             str(self.complexity)], timeout=25, 
                             check=True, stdout=subprocess.DEVNULL)
            
            meshset = pymeshlab.MeshSet()
            meshset.load_new_mesh(mesh_outfile)
            meshset = self._clean_mesh(meshset)

            logger.debug('Succes.')

        except subprocess.TimeoutExpired:
            logger.debug('Ployfit timeout')
        except subprocess.CalledProcessError as CPE:
            logger.debug(f'Error in Ployfit with returncode {CPE.returncode}.')
        except Exception as e:
            logger.debug('Hai', str(e))

        # if os.path.exists(ply_infile):
        #     os.remove(ply_infile)
        # if os.path.exists(mesh_outfile):
        #     os.remove(mesh_outfile)
        # if os.path.exists(dir_path):
        #     os.remove(dir_path)

        return meshset
