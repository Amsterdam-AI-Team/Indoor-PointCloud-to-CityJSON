"""Floor Splitting Module"""
import numpy as np
import logging
import time


from scipy.ndimage import label, binary_closing, binary_dilation, binary_erosion
from scipy.interpolate import griddata
from scipy.stats import binned_statistic_2d
from src.interpolation import FastGridInterpolator
from src.utils.clip_utils import poly_box_clip
from rasterio import features, Affine
from shapely import geometry
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union, transform
import matplotlib.pyplot as plt

NOISE = 0
SLANTED = 1
ALMOST_VERTICAL = 2
ALMOST_HORIZONTAL = 3

logger = logging.getLogger(__name__)

class FloorSplitter:
    """
    FloorSplitter class for the division of pointcloud data into floors. 
    The class splits the pointcloud into floors using density analysis along the z-axis.

    Parameters
    ----------
    
    """

    def __init__(self, subsample_size=0.03):
        self.subsample_size = subsample_size
    
    def _func(self, values):
        """
        Converts detected peaks into top and bottom split values

        Parameters
        ----------
        values : array
            The indices of the peaks

        Returns
        -------
        An array of split values for each floor
        """
    
    def _process_poly(self, poly: Polygon) -> Polygon:
        """
        Close polygon holes by limitation to the exterior ring.
        Args:
            poly: Input shapely Polygon
        Example:
            df.geometry.apply(lambda p: close_holes(p))
        """
        if poly.interiors:
            return Polygon(list(poly.exterior.coords)).buffer(3).buffer(-2).simplify(1)
        else:
            return poly.buffer(3).buffer(-2).simplify(1)



    def _create_3d_grid(self, points, bin_size=.1):
        min_x, max_x = min(points[:, 0])-2*bin_size, max(points[:, 0])+2*bin_size
        min_y, max_y = min(points[:, 1])-2*bin_size, max(points[:, 1])+2*bin_size
        min_z, max_z = min(points[:, 2])-2*bin_size, max(points[:, 2])+2*bin_size
        dimx = max_x - min_x
        dimy = max_y - min_y
        dimz = max_z - min_z
        bins = [np.uint(dimx/bin_size), np.uint(dimy/bin_size), np.uint(dimz/bin_size)]
        hist_range = [[min_x, max_x], [min_y, max_y], [min_z, max_z]]
        
        counts, edges = np.histogramdd(points, range=hist_range, bins=bins)
        origin = (hist_range[0][0]+bin_size/2,hist_range[1][0]+bin_size/2, hist_range[2][0]+bin_size/2)
        grid = counts > 0

        return grid, edges, bins, hist_range, origin

    def _check_multistory(self, primitives, labels, min_floor_height=1.8, min_slab_size=4):
        slab_z = []
        for i,r in primitives.items():
            if r['type'] != ALMOST_VERTICAL:
                if np.sum(labels==i) * (self.subsample_size**2) > min_slab_size:
                    slab_z.append(r['bbox'][:,2].max())
        if len(slab_z) > 1:
            slab_z = np.sort(slab_z)
            ceil_slabs = slab_z[slab_z > slab_z[0] + min_floor_height]
            if len(ceil_slabs) > 0:
                if np.sum(ceil_slabs > ceil_slabs[0] + min_floor_height) > 0:
                    return 2
                else:
                    return 1
        return 0

    def _extract_floor(self, points, labels, primitives, bin_size, min_floor_size, min_cluster_size):
            
        # create grid
        grid, edges, bins, hist_range, origin = self._create_3d_grid(points, bin_size)
        
        # Define floor primitives
        floor_primitives = []
        for i,r in primitives.items():
            mask = labels==i
            if r['type'] == ALMOST_HORIZONTAL and np.sum(mask) > min_floor_size/(self.subsample_size**2):
                floor_primitives.append([i,points[mask][:,2].min(), points[mask][:,2].max()])
        if len(floor_primitives) == 0:
            return np.ones(len(points), dtype=bool)
        floor_primitives = np.asarray(floor_primitives)
        floor_idx = np.argmin(floor_primitives[:,1])
        floor_point_mask = labels==floor_primitives[floor_idx,0]

        # floor grid bounds
        l, t = np.digitize(floor_primitives[floor_idx,1:], edges[2])-1
        r = int(t+np.ceil(1.5/bin_size))
        floor_proj = binary_dilation(np.histogram2d(*points[floor_point_mask,:2].T, range=hist_range[:2], bins=bins[:2])[0]>0)
        # plt.imshow(floor_proj)
        # plt.show()

        # Project space above floor
        space_proj = binary_dilation(np.sum(grid[:,:,l:r],axis=2)>0)
        lcc_space_proj = label(space_proj)[0]
        lcc_space_mask = np.where(np.unique(lcc_space_proj, return_counts=True)[1] < min_cluster_size/(bin_size**2))
        lcc_space_proj[np.isin(lcc_space_proj, lcc_space_mask)] = 0 # remove small blobs in space projection
        space_proj = np.isin(lcc_space_proj, [l for l in np.unique(lcc_space_proj[floor_proj]) if l > 0])

        # Merge floor & space
        floor_proj = space_proj | floor_proj
        ceil_proj = np.zeros(floor_proj.shape, dtype=bool)
        ceiling_map = np.full(floor_proj.shape, np.nan)
        cnf_proj = np.copy(floor_proj)
        # plt.imshow(floor_proj)
        # plt.show()

        # list candidate ceilings
        ceiling_candidates = []
        for i,r in primitives.items():
            if r['type'] != ALMOST_VERTICAL and np.sum(labels==i) > min_cluster_size/(self.subsample_size**2):
                if points[labels==i][:,2].max() > floor_primitives[floor_idx,1]+1.8:
                    ceiling_candidates.append((i,points[labels==i][:,2].min(), points[labels==i][:,2].max()))
        ceiling_candidates = np.asarray(ceiling_candidates)
        ceiling_candidates = ceiling_candidates[np.argsort(ceiling_candidates[:,1])]

        # Loop through ceiling candidates
        for l, _, ceil_max_z in ceiling_candidates:
            cand_z_proj = binned_statistic_2d(*points[labels==l].T, statistic='max', range=hist_range[:2], bins=bins[:2])[0]
            cand_proj = binary_closing(cand_z_proj>0)
            
            # print(l, np.sum(labels==l), np.sum(floor_proj[cand_proj]), np.sum(cand_proj))
            floor_overlap = (np.sum(floor_proj[cand_proj]) / np.sum(cand_proj)).round(2)
            ceiling_overlap = (np.sum(ceil_proj[cand_proj]) / np.sum(cand_proj)).round(2)

            # fig, ax = plt.subplots(1, 3, sharey=True)
            # ax[0].imshow(floor_proj)
            # ax[1].imshow(cand_proj)
            # ax[2].imshow(ceil_proj)
            # plt.show()
            # print(l, ceiling_overlap, floor_overlap)
            
            # Check if ceiling is valid
            if floor_overlap > .5 and ceiling_overlap < .1:
                floor_proj[cand_proj] = False
                cnf_proj[cand_proj] = True
                ceiling_map[cand_proj] = cand_z_proj[cand_proj] + .1 # some nan ? interpolation 
                
            # Tracking purposes..
            ceil_proj[cand_proj] = True
            keep_searching = np.any(np.unique(label(binary_erosion(floor_proj))[0], return_counts=True)[1][1:] > 0.8*min_cluster_size/(bin_size**2))
            if not keep_searching:
                break

        # Interpolate ceiling
        ceil_mask = ceiling_map > 0
        ceiling_map[~ceil_mask] = griddata(np.vstack(np.where(ceil_mask)).T, ceiling_map[ceil_mask], np.where(~ceil_mask), method='nearest')

        # Floor polygon
        floor_complete = binary_dilation(cnf_proj).astype(np.uint8)
        generator = features.shapes(floor_complete, mask=floor_complete>0)
        floor_polygons = [self._process_poly(geometry.shape(shape)) for shape, _ in generator]
        floor_polygon = transform(lambda x, y, z=None: (y*bin_size+origin[0], x*bin_size+origin[1]), unary_union(floor_polygons))

        # Create mask
        max_z_interpolator = FastGridInterpolator(bin_x=edges[0], bin_y=edges[1], values=ceiling_map)
        min_z = floor_primitives[floor_idx,1]-.1
        floor_mask = poly_box_clip(points, floor_polygon, bottom=min_z, top=np.max(ceiling_map)+.15)
        height_mask = points[floor_mask, 2] <= max_z_interpolator(points[floor_mask])
        floor_mask[floor_mask] = height_mask

        return floor_mask

        

    def process(self, pcd, labels, primitives):
        """
        Parameters
        ----------
        points : array of shape (n_points, 3)
            The point cloud <x, y, z>.

        Returns
        -------
        An array of masks, where each mask represents a floor in the pointcloud.
        """

        logger.debug('Analysing pointcloud for floors...')
        points = np.asarray(pcd.points)
        un_mask = np.ones(len(points), dtype=bool)
        floors = []
        start = time.time()
        while self._check_multistory(primitives, labels[un_mask]) > 0:
            # TODO: error test
            mask = self._extract_floor(points[un_mask], labels[un_mask], primitives, bin_size=0.1, min_floor_size=5, min_cluster_size=.5)
            floor_mask = np.zeros(len(points), dtype=bool)
            floor_mask[un_mask] = mask
            floors.append(floor_mask)
            un_mask[floor_mask] = False

        logger.debug(f"Done. Number of floor extracted {len(floors)} floors. {round(time.time()-start,2)}s\n")

        return floors
