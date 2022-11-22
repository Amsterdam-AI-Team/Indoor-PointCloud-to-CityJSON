import argparse
import os
import sys

from src.pipeline import Pipeline
from preprocessors.sor import SOR
from preprocessors.spatial_subsample import SpatialSubsample
from modules.primitive_detection import PrimitiveDetector
from modules.floor_split import FloorSplitter
from modules.room_detection import RoomDetector
from modules.room_reconstruct import RoomReconstructor
from modules.mesh_stats import MeshAnalyser

ransac_exe = './modules/efficient_ransac'
reconstruct_exe = './modules/polyfit'
reconstruct_ransac_exe = './modules/polyfit_ransac'

def main(in_file, out_folder):
    sor = SOR()
    ss = SpatialSubsample()
    primitive_detector = PrimitiveDetector(ransac_exe)
    floor_splitter = FloorSplitter()
    room_detector = RoomDetector()
    room_reconstructor = RoomReconstructor(reconstruct_ransac_exe, reconstruct_exe)
    mesh_analyser = MeshAnalyser()

    pipeline = Pipeline(primitive_detector, floor_splitter, room_detector, room_reconstructor, mesh_analyser, preprocessors=[ss,sor])

    pipeline.process_file(in_file, out_folder)

if __name__ == "__main__":
    global args

    desc_str = '''This script provides room reconstruction for indoor point clouds.'''
    parser = argparse.ArgumentParser(description=desc_str)

    parser.add_argument('--in_file', metavar='path', action='store',
                        type=str, required=True)
    parser.add_argument('--out_folder', metavar='path', action='store',
                        type=str, required=False)
    args = parser.parse_args()

    main(args.in_file, args.out_folder)