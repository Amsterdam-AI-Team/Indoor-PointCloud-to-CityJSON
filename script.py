from src.pipeline import Pipeline
from preprocessors.sor import SOR
from preprocessors.spatial_subsample import SpatialSubsample
from modules.primitive_detection import PrimitiveDetector
from modules.floor_split import FloorSplitter
from modules.room_detection import RoomDetector
from modules.room_reconstruct import RoomReconstructor
from modules.mesh_stats import MeshAnalyser

in_file = './datasets/stadskwekerij_subsampled.ply'
ransac_exe = './modules/efficient_ransac'
reconstruct_exe = './modules/polyfit'
reconstruct_ransac_exe = './modules/polyfit_ransac'

def main():
    sor = SOR()
    ss = SpatialSubsample()
    primitive_detector = PrimitiveDetector(ransac_exe)
    floor_splitter = FloorSplitter()
    room_detector = RoomDetector()
    room_reconstructor = RoomReconstructor(reconstruct_ransac_exe, reconstruct_exe)
    mesh_analyser = MeshAnalyser()

    pipeline = Pipeline(primitive_detector, floor_splitter, room_detector, room_reconstructor, mesh_analyser, preprocessors=[ss,sor])

    pipeline.process_file(in_file)

if __name__ == "__main__":
    main()