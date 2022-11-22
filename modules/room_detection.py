"""Room Detection Module"""
import os
import logging

import cv2
import open3d as o3d
import numpy as np
import random as rng
import open3d as o3d
from scipy import signal
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import src.utils.clip_utils as clip_utils
import src.utils.math_utils as math_utils
from skspatial.objects import Plane
import networkx as nx
from scipy.stats import binned_statistic_2d
from scipy.ndimage import binary_closing, binary_dilation, binary_erosion, generate_binary_structure, binary_fill_holes
import rasterio
from rasterio import features
import shapely
from shapely.geometry import Point, Polygon
from shapely.ops import transform
from skimage import measure
from src.interpolation import FastGridInterpolator

logger = logging.getLogger(__name__)

# Point labels
NOISE = 0
SLANTED = 1
ALMOST_VERTICAL = 2
ALMOST_HORIZONTAL = 3

# Primitive Connections
WALL_WALL = 1
WALL_CEILING = 2
WALL_FLOOR = 3
CEILING_CEILING = 4
WALL_SLANTEDWALL = 5

# Primitive Classes
WALL = 1
CEILING = 2
FLOOR = 3
SLANTED_WALL = 4
CLUTTER = 5

class RoomDetector:
    """
    RoomDetector class for the detection of rooms in a pointcloud floor. 
    The class .....

    Parameters
    ----------
    min_peak_height : int (default=1500)
        The required height of a peak in the vertical density profile.
    threshold : int (default=250)
        The required threshold of peaks, the vertical distance to its neighboring samples.
    distance : int (default=20)
        The required minimal horizontal distance (>= 1) in samples between neighbouring peaks.
    prominence : int (default=2)
        The required prominence of peaks.
    thickness : int (default=20)
        The description of the parameter.
    """

    def __init__(self, subsample_size=0.03, plot=False):
        self.subsample_size = subsample_size
        self.plot = plot

    def _get_edge_type(self, primitive_i, primitive_j):
        if primitive_i['type'] == ALMOST_VERTICAL and primitive_j['type'] == ALMOST_VERTICAL:
            return WALL_WALL
        elif primitive_i['type'] == ALMOST_VERTICAL:
            if primitive_i['bbox'][:,2].mean() > primitive_j['bbox'][:,2].max():
                return WALL_FLOOR
            else:
                return WALL_CEILING
        elif primitive_j['type'] == ALMOST_VERTICAL:
            if primitive_i['bbox'][:,2].max() < primitive_j['bbox'][:,2].mean():
                return WALL_FLOOR
            else:
                return WALL_CEILING
        else:
            return CEILING_CEILING

    def _find_neighbouring_primitives(self, points, labels, primitives):
        neighbours = set()
        buffer = .5

        for _,r in primitives.items():
            bbox = r['bbox']
            rp = Polygon(bbox[:,:2]).buffer(buffer)
            c_mask = clip_utils.poly_box_clip(points, rp, bbox[:,2].min()-buffer, bbox[:,2].max()+buffer)
            for rn in np.unique(labels[c_mask]):
                if rn != r['region'] and rn in primitives.keys():
                    neighbours.add(tuple(sorted((int(rn),r['region']))))

        logger.debug(f'Found {len(neighbours)} neighbouring planes.')
        return neighbours
    
    def _adjacency_graph(self, points, labels, primitives, neighbours):

        G = nx.Graph()
        G.add_nodes_from([(i,{'type':r['type']}) for i,r in primitives.items()])

        d_adj = .25
        l_intersect = .2

        # Create edges
        valid_pairs = 0
        for (i,j) in neighbours:
            angle_ij = math_utils.vector_angle(primitives[i]['normal'], primitives[j]['normal'])
            angle_ij = 90-abs(90-angle_ij)
            if angle_ij > 5:
                plane_i = Plane(primitives[i]['bbox'].mean(axis=0), primitives[i]['normal'])
                plane_j = Plane(primitives[j]['bbox'].mean(axis=0), primitives[j]['normal'])
                ij_intersect = plane_i.intersect_plane(plane_j)
                dist_i = math_utils.line_dist(points[labels==i], ij_intersect.point, ij_intersect.vector)
                dist_j = math_utils.line_dist(points[labels==j], ij_intersect.point, ij_intersect.vector)
                if np.sum(dist_i<d_adj)>10 and np.sum(dist_j<d_adj) > 10:
                    proj_i = np.dot((points[labels==i][dist_i<d_adj]-ij_intersect.point), ij_intersect.vector)
                    proj_j = np.dot((points[labels==j][dist_j<d_adj]-ij_intersect.point), ij_intersect.vector)
                    overlap = [np.max([proj_i.min(),proj_j.min()]),np.min([proj_i.max(),proj_j.max()])]
                    # snap_points = ij_intersect.point+ij_intersect.vector*np.array(overlap)[:,np.newaxis]
                    if overlap[1]-overlap[0]>l_intersect:
                        edge_type = self._get_edge_type(primitives[i],primitives[j])
                        G.add_edges_from([(i, j, {'type':edge_type})])
                        valid_pairs+=1

        logger.debug(f'Number of valid pairs: {valid_pairs}')
        return G

    def _classify_graph(self, G):
        for V in nx.nodes(G):
            if G.nodes[V]['type'] == ALMOST_VERTICAL:
                if np.sum([G.edges[V,j]['type'] == WALL_CEILING for j in G[V]]) >= 1:
                    G.nodes[V]['label'] = WALL
                elif np.sum([G.edges[V,j]['type'] == WALL_WALL for j in G[V]]) > 0 and \
                    np.sum([G.edges[V,j]['type'] == WALL_SLANTEDWALL for j in G[V]]) > 0:
                    G.nodes[V]['label'] = SLANTED_WALL
                else:
                    G.nodes[V]['label'] = CLUTTER
            elif G.nodes[V]['type'] == ALMOST_HORIZONTAL: # MISSING wall-wall edge count
                if np.sum([G.edges[V,j]['type'] == WALL_CEILING for j in G[V]]) >= 2:
                    G.nodes[V]['label'] = CEILING
                elif np.sum([G.edges[V,j]['type'] == WALL_FLOOR for j in G[V]]) >= 2:
                    G.nodes[V]['label'] = FLOOR
                else:
                    G.nodes[V]['label'] = CLUTTER
            elif G.nodes[V]['type'] == SLANTED: # MISSING wall-wall edge count
                if np.sum([G.edges[V,j]['type'] == WALL_CEILING for j in G[V]]) >= 2:
                    G.nodes[V]['label'] = CEILING
                else:
                    G.nodes[V]['label'] = CLUTTER
            else:
                G.nodes[V]['label'] = CLUTTER

        return G

    def _get_rooms(self, projection, origin, bin_size):
        room_labels = measure.label(~projection, background=0)
        min_surface = 0.6
        min_width = 0.65

        # Removes pillars and inner walls in top down projection
        for i in np.unique(room_labels):
            if i > 0:
                lcc_pts = np.vstack(np.where(room_labels==i)).T
                if len(lcc_pts) < min_surface/(bin_size**2): # min. sqr meters
                    room_labels[np.where(room_labels==i)] = 0
                else:
                    min_dims, max_dims = math_utils.minimum_bounding_rectangle(lcc_pts)[2:4]
                    if min_dims < min_width / bin_size: # min. room dimension
                        room_labels[np.where(room_labels==i)] = 0
                    else:
                        room_labels[binary_closing(room_labels==i, structure=generate_binary_structure(2, 2), iterations=2)] = i

        if self.plot:
            plt.figure(figsize=(6, 6))
            plt.subplot(111)
            plt.imshow(room_labels.T, cmap='nipy_spectral')
            plt.axis('off')
       
       # Convert to polygons
        rooms = []
        for shape, value in features.shapes(room_labels.astype(np.int16), mask=(room_labels>0), transform=rasterio.Affine(1.0, 0, 0, 0, 1.0, 0)):
            room_polygon = shapely.geometry.shape(shape).simplify(0.75)
            room_polygon = transform(lambda x, y, z=None: (y*bin_size+origin[0], x*bin_size+origin[1]), room_polygon)
            shape_proj = binary_fill_holes(room_labels==value)
            rooms.append((room_polygon, shape_proj))     

        return rooms

    def _clean_mid_horizontal(self, pcd, labels, primitives):

        horizontal_labels = [i for i,k in primitives.items() if k['type'] != ALMOST_VERTICAL]
        horizontal_mask = np.isin(labels, horizontal_labels)
        points = np.asarray(pcd.points)[horizontal_mask]

        x_bins = np.arange(points[:,0].min(), points[:,0].max()+.125, 0.25)
        y_bins = np.arange(points[:,1].min(), points[:,1].max()+.125, 0.25)

        stat_min =  binned_statistic_2d(points[:,0], points[:,1], points[:,2], 'min', bins=[x_bins, y_bins])[0]
        min_z = FastGridInterpolator(x_bins, y_bins, stat_min)
        floor_offset = points[:,2] - min_z(points)

        stat_max =  binned_statistic_2d(points[:,0], points[:,1], points[:,2], 'max', bins=[x_bins, y_bins])[0]
        max_z = FastGridInterpolator(x_bins, y_bins, stat_max)
        ceil_offset = max_z(points) - points[:,2]

        for label in horizontal_labels:
            label_mask = labels[horizontal_mask] == label
            if floor_offset[label_mask].mean() > .4 and ceil_offset[label_mask].mean() > .4:
                labels[labels==label] = -1
                if label in primitives:
                    del primitives[label]

    def _labels_to_primitives(self, pcd, labels, exclude_labels=[-1]):
        planes = {}
        # TODO: change to box_area
        rectangle_labels = [r for r in np.unique(labels) if r not in exclude_labels and np.sum(labels==r) > .3/(self.subsample_size**2)]
        for i in rectangle_labels:
            region_cloud = pcd.select_by_index(np.where(labels==i)[0])
            region_pts = np.asarray(region_cloud.points)
            
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
                bbox_points = math_utils.minimum_bounding_rectangle(region_pts[:,:2])[0]
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
            line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(bbox_points),
                lines=o3d.utility.Vector2iVector([[0, 1],[1, 2],[2, 3],[3, 0]]),
            )

            plane_object = {
                'region': i,
                'bbox': bbox_points,
                'lineset': line_set,
                'surface': surface,
                'center': center,
                'normal': normal,
                'D': d,
                'slope': slope,
                'type': plane_type
            }
            planes[i] = plane_object

        return planes

    def process(self, pcd, labels):
        """
        Parameters
        ----------
        points : array of shape (n_points, 3)
            The point cloud <x, y, z> of a floor.

        Returns
        -------
        An array of masks, where each mask represents a floor in the pointcloud.
        """

        points = np.asarray(pcd.points)
        logger.debug(f'detecting rooms in floor of {len(points)} points')

        # Classify primitives
        logger.debug(f'Classify primitives...')
        primitives = self._labels_to_primitives(pcd, labels)

        # Clear middle planes
        self._clean_mid_horizontal(pcd, labels, primitives)

        # Primitive adjacency classification
        neighbours = self._find_neighbouring_primitives(points, labels, primitives)
        primitive_graph = self._adjacency_graph(points, labels, primitives, neighbours)
        primitive_graph = self._classify_graph(primitive_graph)
        ceiling_nodes = [v for v in primitive_graph.nodes() if primitive_graph.nodes[v]['label']==CEILING]
        floor_nodes = [v for v in primitive_graph.nodes() if primitive_graph.nodes[v]['label']==FLOOR]
        wall_nodes = [v for v in primitive_graph.nodes() if primitive_graph.nodes[v]['label']==WALL]
        slantedwall_nodes = [v for v in primitive_graph.nodes() if primitive_graph.nodes[v]['label']==SLANTED_WALL]
        clutter_nodes = [v for v in primitive_graph.nodes() if primitive_graph.nodes[v]['label']==CLUTTER]
        prim_regions = wall_nodes+floor_nodes+ceiling_nodes
        logger.debug(f'{(len(ceiling_nodes),len(floor_nodes),len(wall_nodes),len(slantedwall_nodes),len(clutter_nodes))}')    

        # plot 
        pcd_group_1 = pcd.select_by_index(np.where(np.isin(labels, floor_nodes))[0])
        pcd_group_1 = pcd_group_1.paint_uniform_color([1.0,0.3,0.3])
        pcd_group_2 = pcd.select_by_index(np.where(np.isin(labels, ceiling_nodes))[0])
        pcd_group_2 = pcd_group_2.paint_uniform_color([0.2,0.2,1.0])
        pcd_group_3 = pcd.select_by_index(np.where(np.isin(labels, wall_nodes))[0])
        pcd_group_3 = pcd_group_3.paint_uniform_color([0.2,1.0,0.2])
        pcd_group_4 = pcd.select_by_index(np.where(np.isin(labels, clutter_nodes))[0])
        pcd_group_4 = pcd_group_4.paint_uniform_color([0,0,0])
        pcd_group_5 = pcd.select_by_index(np.where(np.isin(labels, slantedwall_nodes))[0])
        pcd_group_5 = pcd_group_5.paint_uniform_color([0,1,1])
        if self.plot:
            o3d.visualization.draw_geometries([pcd_group_1, pcd_group_2, pcd_group_3, pcd_group_4])

        # Detect Rooms
        logger.debug(f'Detect rooms...')

        # create grid
        bin_size = 0.05
        x_bins = np.arange(points[:,0].min()-bin_size/2,points[:,0].max()+bin_size,bin_size)
        y_bins = np.arange(points[:,1].min()-bin_size/2,points[:,1].max()+bin_size,bin_size)
        origin = [x_bins[0]+bin_size/2,y_bins[0]-bin_size/2]

        # Create projection
        walls = np.zeros((len(x_bins)-1,len(y_bins)-1))
        for i,r in primitives.items():
            if i in wall_nodes:
                length = np.linalg.norm(r['bbox'][0,:2]-r['bbox'][1,:2])
                line = np.linspace(r['bbox'][0,:2],r['bbox'][1,:2], int(length*20))
                walls += np.histogram2d(line[:,0],line[:,1],[x_bins, y_bins])[0]
        walls = walls>0 
        walls = binary_dilation(walls, structure=generate_binary_structure(2, 2), iterations=3, border_value=1)
        walls = binary_erosion(walls, structure=generate_binary_structure(2, 2), iterations=2)
        ceiling_mask = np.isin(labels, ceiling_nodes + floor_nodes)
        projection, xedges, yedges = np.histogram2d(points[ceiling_mask,0], points[ceiling_mask,1], bins=[x_bins,y_bins])
        projection = binary_closing(projection!=0, np.ones((3,3)), iterations=2)
        projection = np.logical_or(~projection,walls)
        if self.plot:
            plt.figure(figsize=(6, 6))
            plt.subplot(111)
            plt.imshow(projection.T)
            plt.axis('off')

        # Get room polygons
        room_polys = self._get_rooms(projection, origin, bin_size)

        room_mask = np.zeros((len(points),len(room_polys)), dtype=bool)
        for i, item in enumerate(room_polys):
            room_poly = item[0]
            clip_mask = clip_utils.poly_clip(points, room_poly.buffer(0.2))

            search = binary_dilation(binary_erosion(item[1]) ^ binary_dilation(item[1]), iterations=4)
            wall_mask = clip_mask & np.isin(labels, wall_nodes)
            walls = np.histogram2d(points[wall_mask,0],points[wall_mask,1],[x_bins, y_bins])[0]>0
            walls_dil = binary_dilation(walls, iterations=6)
            search[walls_dil] = False

            # Add missing walls
            outside_primitives = [i for i in np.unique(labels[clip_mask]) if i in clutter_nodes]
            track = walls.astype(int)
            news = []
            for label in outside_primitives:
                label_mask = (labels==label) & clip_mask
                label_grid = np.histogram2d(points[label_mask,0],points[label_mask,1],[x_bins, y_bins])[0]>0
                if search[label_grid].sum() > 10:
                    track[label_grid] = 2
                    news.append(label)
                    wall_nodes.append(label)
                    clutter_nodes.remove(label)

            # Clean inside walls
            # TODO: delete all points?
            room_inside = binary_erosion(item[1], iterations=6)
            room_primitives = [i for i in np.unique(labels[clip_mask]) if i in wall_nodes]
            track = room_inside.astype(int)
            inside = []
            for label in room_primitives:
                label_mask = (labels==label) & clip_mask
                label_grid = np.histogram2d(points[label_mask,0],points[label_mask,1],[x_bins, y_bins])[0]>0
                if label_grid[~room_inside].sum() == 0:
                    track[label_grid] = 2
                    inside.append(label)
                    clutter_nodes.append(label)
                    wall_nodes.remove(label)
            
            prim_regions = wall_nodes+floor_nodes+ceiling_nodes
            room_mask[:,i] = clip_mask & np.isin(labels, prim_regions)

        return room_mask, labels #, room_polys, (wall_nodes,floor_nodes,ceiling_nodes)

def bbox_area(bbox):
    return np.linalg.norm(bbox[0,:]-bbox[1,:]) * np.linalg.norm(bbox[2,:]-bbox[1,:])
