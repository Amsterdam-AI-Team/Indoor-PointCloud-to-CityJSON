import numpy as np
from numba import jit
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon


def vector_angle(u, v=np.array([0., 0., 1.])):
    """
    Returns the angle in degrees between vectors 'u' and 'v'. If only 'u' is
    provided, the angle between 'u' and the vertical axis is returned.
    """
    # see https://stackoverflow.com/a/2827466/425458
    c = np.dot(u/np.linalg.norm(u), v/np.linalg.norm(v))
    clip = np.minimum(1, np.maximum(c, -1))
    return np.rad2deg(np.arccos(clip))

def lineseg_dist(p, a, b):
    """
    Returns the distance between a set of points and a linesegment. 
    It takes into account the ending of the line.
    """
    # see https://stackoverflow.com/a/56467661

    d = np.divide(b - a, np.linalg.norm(b - a))
    s = np.dot(a - p, d)
    t = np.dot(p - b, d)
    h = np.maximum.reduce([s, t, 0])
    c = np.cross(p - a, d)
    return np.hypot(h, np.linalg.norm(c))

#@jit(nopython=True, cache=True, parallel=True)
def line_dist(p, a, d):
    """
    Returns the distance between a set of points and a linesegment. 
    It takes into account the ending of the line.
    """
    return np.linalg.norm(np.cross(p - a, d), axis=1)

@jit(nopython=True, cache=True, parallel=True)
def compute_bounding_box(points):
    """
    Get the min/max values of a point list.

    Parameters
    ----------
    points : array of shape (n_points, 2)
        The (x, y) coordinates of the points. Any further dimensions will be
        ignored.

    Returns
    -------
    tuple
        (x_min, y_min, x_max, y_max)
    """
    x_min = np.min(points[:, 0])
    x_max = np.max(points[:, 0])
    y_min = np.min(points[:, 1])
    y_max = np.max(points[:, 1])

    return (x_min, y_min, x_max, y_max)


def convex_hull_poly(points):
    """Return convex hull as a shapely Polygon."""
    hull = points[ConvexHull(points).vertices]
    return Polygon(np.vstack((hull, hull[0])))


def minimum_bounding_rectangle(points):
    """
    Find the smallest bounding rectangle for a set of points.
    Returns a set of points representing the corners of the bounding box.

    :param points: an nx2 matrix of coordinates
    :rval: an nx2 matrix of coordinates
    """
    pi2 = np.pi/2.

    # get the convex hull for the points
    hull_points = points[ConvexHull(points).vertices]

    # calculate edge angles
    edges = np.zeros((len(hull_points)-1, 2))
    edges = hull_points[1:] - hull_points[:-1]

    angles = np.zeros((len(edges)))
    angles = np.arctan2(edges[:, 1], edges[:, 0])

    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)

    # find rotation matrices
    rotations = np.vstack([
        np.cos(angles),
        np.cos(angles-pi2),
        np.cos(angles+pi2),
        np.cos(angles)]).T
    rotations = rotations.reshape((-1, 2, 2))

    # apply rotations to the hull
    rot_points = np.dot(rotations, hull_points.T)

    # find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    # find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    # return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]

    # Calculate center point and project onto rotated frame
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    center_point = np.dot([center_x, center_y], r)

    min_bounding_rect = np.zeros((4, 2))
    min_bounding_rect[0] = np.dot([x1, y2], r)
    min_bounding_rect[1] = np.dot([x2, y2], r)
    min_bounding_rect[2] = np.dot([x2, y1], r)
    min_bounding_rect[3] = np.dot([x1, y1], r)

    # Compute the dims of the min bounding rectangle
    dims = [(x1 - x2), (y1 - y2)]

    return min_bounding_rect, hull_points, min(dims), max(dims), center_point
