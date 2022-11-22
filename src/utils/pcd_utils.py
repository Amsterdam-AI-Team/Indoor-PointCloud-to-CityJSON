import os
from laspy import read as read_las
import numpy as np
import open3d as o3d

def read_pointcloud(infile):
    """Read a file and return the pointcloud object."""
    filename, file_extension = os.path.splitext(infile)

    if file_extension  == '.ply':
        return o3d.io.read_point_cloud(infile)
    else:
        las = read_las(infile)
        points = np.vstack([las.x,las.y,las.z]).T
        return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))

def write_pointcloud(points, filename):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    o3d.io.write_point_cloud(filename, pcd)
    return True

def merge_point_clouds(pcd_1, pcd_2):
    points = np.concatenate((pcd_1.points, pcd_2.points))
    normals = np.concatenate((pcd_1.normals, pcd_2.normals))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)

    return pcd

