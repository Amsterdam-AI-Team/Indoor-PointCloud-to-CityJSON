"""Mesh Area Volume Module"""
import numpy as np
import logging

import pymeshlab
import json
import argparse
import os

logger = logging.getLogger(__name__)

class MeshAnalyser:
    """
    MeshAnalyser class for the volume and area computation of a mesh. 
    The class compute both the volume and area for a given mesh.

    Parameters
    ----------
    face_condition : float (default=-0.95)
        The condition for face selection.
    """

    def __init__(self, face_condition=-0.95):
        self.face_condition = face_condition

    def _get_area_volume(self, meshset):
        meshset.meshing_remove_duplicate_vertices()
        meshset.meshing_merge_close_vertices()
        meshset.meshing_re_orient_faces_coherentely()
        
        geom = meshset.get_geometric_measures()
        if 'mesh_volume' in geom.keys():
            volume = geom['mesh_volume']
        else:
            volume = 0
        logger.debug(f'volume is {volume}')
        
        if volume<0:
            meshset.meshing_invert_face_orientation()
            geom = meshset.get_geometric_measures()
            volume = geom['mesh_volume']
            logger.debug(f'after inverting, volume is now {volume}')
           
        statement = ('(fnz < ' + str(self.face_condition) + ')') 
        meshset.compute_selection_by_condition_per_face(condselect=statement)
        meshset.apply_selection_inverse(invfaces = True)
        meshset.meshing_remove_selected_faces()
        
        geom2 = meshset.get_geometric_measures()
        logger.debug(f'floor area is {geom2["surface_area"]}')
        
        floorarea = geom2['surface_area']
        
        return volume, floorarea


    def process(self, mesh):
        """
        Parameters
        ----------
        mesh : pmeshlab.Mesh
            The pymeshlab Meshset.

        Returns
        -------
        volume :  float
            The volume of the input mesh.
        floorarea :  float
            The floorarea of the input mesh.
        """

        logger.debug(f'Analysing mesh...')

        try: 
            volume, floorarea = self._get_area_volume(mesh)
            logger.debug(f'The volume of room is {volume:.2f} and the floorarea is {floorarea:.2f}')
        except:
            volume, floorarea = None, None
            logger.debug(f'metrics cannot be calculated')

        return volume, floorarea
