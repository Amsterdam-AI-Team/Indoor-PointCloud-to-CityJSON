"""Room Reconstruction Pipeline"""

import os
import pathlib
import time
import logging
import gc
import pymeshlab
import numpy as np
import open3d as o3d
from tqdm import tqdm

from .utils import cityjson_utils
from .utils import pcd_utils

logger = logging.getLogger(__name__)


class Pipeline:
    """
    Pipeline for room reconstruction. The class processes a single point cloud
    or a folder of pointclouds by applying the given modules consecutively. 

    Parameters
    ----------
    floor_splitter : FloorSplitter
        The module that splits the point cloud into separate floors
    room_detector : RoomDetector
        The module that detects rooms within a single floor.
    surface_reconstructor : SurfaceReconstructor
        The module that reconstructs the surface to a mesh.
    mesh_volume : MeshVolume
        The module that computes the area and volume of a mesh.
    preprocessors : iterable of type PreProcessor
        The preprocessors to apply, in order.
    """

    FILE_TYPES = ('.LAS', '.las', '.LAZ', '.laz', '.ply')

    def __init__(self, primitive_detector, floor_splitter, room_detector,
                 room_reconstructor, mesh_analyser, preprocessors=[]):
        if len(preprocessors) == 0:
            logger.info('No preprocessors specified.')
        self.preprocessors = preprocessors
        self.primitive_detector = primitive_detector
        self.floor_splitter = floor_splitter
        self.room_detector = room_detector
        self.room_reconstructor = room_reconstructor
        self.mesh_analyser = mesh_analyser

    def _process_cloud(self, pcd):
        """
        Process a single point cloud.

        Parameters
        ----------
        points : array of shape (n_points, 3)
            The point cloud <x, y, z>.

        Returns
        -------
        An cityJSON object representing the indoor of a building.
        """



        # 1. Preprocess pointcloud
        logger.info(f'Preprocessing...')
        for obj in self.preprocessors:
            start = time.time()
            pcd = obj.process(pcd)
            duration = time.time() - start
            logger.info(f'Processor finished in {duration:.2f}s, ' +
                        f'{len(pcd.points)} points.')
            gc.collect() 


        # 2. Primitive Detection
        start = time.time()
        logger.info(f'Detecting primitives...')
        pcd, primitives, primitive_labels = self.primitive_detector.process(pcd)
        duration = time.time() - start
        logger.info(f'Done. Detected {len(primitives.keys())}. {duration:.2f}s')
        gc.collect()

        # 3. Detect floors
        start = time.time()
        logger.info(f'Detecting floors...')
        floors = self.floor_splitter.process(pcd, primitive_labels, primitives)
        duration = time.time() - start
        logger.info(f'Done. Detected {len(floors)} floors. {duration:.2f}s')
        gc.collect()

        # 4. Detect Rooms
        start = time.time()
        logger.info(f'Detecting rooms...')
        rooms = []
        for floor_mask in floors:
            floor_pcd = pcd.select_by_index(np.where(floor_mask)[0])
            floor_labels = primitive_labels[floor_mask]
            floor_rooms, _ = self.room_detector.process(floor_pcd, floor_labels)
            for room_i in range(floor_rooms.shape[1]):
                room_mask = np.zeros(len(pcd.points), dtype=bool)
                room_mask[floor_mask] = floor_rooms[:,room_i]
                rooms.append(room_mask)
        gc.collect()
        duration = time.time() - start
        logger.info(f'Done. Detected {len(rooms)} rooms. {duration:.2f}s')

        # 5. Reconstruct rooms
        start = time.time()
        logger.info(f'Reconstructing rooms into meshes...')
        room_meshes = []
        for room_mask in tqdm(rooms):
            meshset = self.room_reconstructor.process(np.asarray(pcd.points)[room_mask], primitive_labels[room_mask], primitives)
            if meshset is not None:
                room_meshes.append(meshset)
        gc.collect()
        duration = time.time() - start
        logger.info(f'Done. Succesfully reconstructed {len(room_meshes)}/{len(rooms)} rooms. {duration:.2f}s')

        # 6. Convert CityJSON
        cityjson = cityjson_utils.to_cityjson_v1(room_meshes)

        # 7. Compute Area and Volume
        start = time.time()
        logger.info(f'Computing mesh metrics')
        room_stats = []
        for i, room_mesh in enumerate(room_meshes):
            volume, floorarea = self.mesh_analyser.process(room_mesh)
            room_stats.append((volume, floorarea))
            logger.debug(f'volume room {i}: {volume}, floorarea: {floorarea}')
        duration = time.time() - start
        logger.info(f'Done. {duration:.2f}s')

        return cityjson, room_stats
    
    def process_file(self, in_file, out_folder=None, out_prefix=''):
            """
            Process a single LAS file and save the result as .laz file.

            Parameters
            ----------
            in_file : str
                The file to process.
            out_file : str (default: None)
                The name of the output file. If None, the input will be
                overwritten.
            """
            logger.info(f'Processing file {in_file}.')
            start = time.time()
            if not os.path.isfile(in_file):
                logger.error('The input file specified does not exist')
                return None
            elif not in_file.endswith(self.FILE_TYPES):
                logger.error('The input file specified has the wrong format')
                return None

            filename = pathlib.Path(in_file).stem
            outputname = out_prefix + filename
            in_folder = os.path.dirname(in_file)
            if out_folder is None:
                out_folder = in_folder
            else:
                pathlib.Path(out_folder).mkdir(parents=True, exist_ok=True)
            out_path = out_folder + '/' +  outputname + '.city.json'

            pcd = pcd_utils.read_pointcloud(in_file)
            
            citysjon, room_stats = self._process_cloud(pcd)
            cityjson_utils.save_to_file(citysjon, out_path)

            # write stats
            lines = []
            for i, stats in enumerate(room_stats):
                line = 'Room ' + str(i) + ': volume='+str(stats[0])+', surface='+str(stats[1])+'\n'
                lines.append(line)
            stats_path = out_folder + '/' + outputname + '_stats.txt'
            with open(stats_path, 'w') as f:
                f.writelines(lines)

            duration = time.time() - start
            # stats = analysis_tools.get_label_stats(labels)
            # logger.info('STATISTICS\n' + stats)
            logger.info(f'File processed in {duration:.2f}s, ' +
                        f'output written to {out_path}.\n' + '='*20)
