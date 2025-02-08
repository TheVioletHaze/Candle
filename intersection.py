"""
# Intersection
Provides functions for 
- working with vectors and triangles
- calculating intersections
"""
import warnings
import numpy as np
from numpy import newaxis as na
from opt_einsum import contract

warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
warnings.filterwarnings('ignore', r'invalid value encountered in divide')

RUNS = 100

def normalize_vector(vectors):
    """Return unit vector

    Parameters
    ----------
    vectors : ndarray
        ([m], 3)([vector], coordinate)

    Returns
    -------
    ndarray
        ([m], 3)([vector], coordinate)
    """
    magnitude = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors/magnitude

def normal_from_triangle(triangles):
    """Returns Normals for triangles

    Parameters
    ----------
    triangles : ndarray
        ([m], 3, 3)([triangle], vertex, coordinate)
        ([m], 3, 3)([triangle], vertex, coordinate)

    Returns
    -------
    ndarray
        ([m], 3)([triangle], vector)
        ([m], 3)([triangle], vector)
    """
    a_b = triangles[..., 0, :] - triangles[..., 1, :]
    a_c = triangles[..., 0, :] - triangles[..., 2, :]
    normals = np.cross(a_b, a_c)
    return normals


def intersection_plane_line(triangle_pl_pts, triangle_pl_nml, line_vec, line_pts):
    """Returns scalar for line vector to intersection

    Parameters
    ----------
    triangle_pl_pts : ndarray
        (m, 3)(triangle, any point)
    triangle_pl_nml : ndarray
        (m, 3)(triangle, vector)
    line_vec : ndarray
        ([m], 3)([line], vector)
        ([m], 3)([line], vector)
    line_pts : ndarray
        ([m], 3)([line], point) 
        ([m], 3)([line], point) 

    Returns
    -------
    ndarray
        ([m], n, 1)([line], triangle, scalar)
    """
    po_qo = line_pts[..., na, :] - triangle_pl_pts[na, :, :] # (Strahl, Ebene, Punkt)
    n_po_qo = contract("...jk, jk->...j", po_qo, triangle_pl_nml, optimize='optimal') * -1

    n_p = contract("...k, jk->...j", line_vec, triangle_pl_nml, optimize='optimal')

    t = n_po_qo / n_p
    return t

def inside_out_test(triangles, normals, points):
    """Returns bool array whether point inside triangle

    Parameters
    ----------
    triangles : ndarray
        (m, 3, 3)(triangle, vertex, coordinate)
    normals : ndarray
        (m, 3)(triangle, vector)
    points : ndarray
        ([m], n, 3)([line], triangle, coordinate)
        ([m], n, 3)([line], triangle, coordinate)

    Returns
    -------
    ndarray
        ([m], n, 1)([line], triangle, boolean)
    """

    offset1 = np.roll(triangles, -1, axis=-2) # [b, c, a]
    offset2 = np.roll(triangles, -2, axis=-2) # [c, b, a]

    line_vec = offset1 - offset2 # opposite side
    line_normals = np.cross(line_vec, normals[..., na, :]) # normal of the opposite side

    points_vert = points[..., na, :] - offset1
    vert_opp = triangles - offset1

    point_dotprods = contract("...n, ...n -> ...", points_vert, line_normals)[..., na]
    vert_dotprods = contract("...n, ...n -> ...", vert_opp, line_normals)[..., na]

    points_vert_multi = point_dotprods * vert_dotprods
    side_inter_bool = points_vert_multi > 0
    side_inter_bool_merged = np.all(side_inter_bool, axis=-2)
    return side_inter_bool_merged


def intersection_ray_triangle(line_vec, line_pts, triangles, triangle_normals):
    """Returns scalars for intersections and what triangle is intersected

    Parameters
    ----------
    line_vec : ndarray
        ([m], 3)([line], coordinate)
    line_pts : ndarray
        ([m], 3)([line], coordinate)
    triangles : ndarray
        (n, 3, 3)(triangle, vertex, coordinate)
    triangle_normals : ndarray
        (n, 3)(triangle, coordinate) must match triangles

    Returns
    -------
    tuple(scalar of first intersection([m], n, 1), intersected triangle)

    Raises
    ------
    ValueError
        shapes of the triangles don't match
    """
    vec_shape = line_vec.shape
    pts_shape = line_pts.shape
    if vec_shape != pts_shape:
        raise ValueError(f"shape of line_vec {vec_shape} and line_pts {pts_shape} doesn't match.")

    line_shape = vec_shape[:-1] + (1,)
    min_value = np.full(line_shape, np.nan)
    min_index = np.full(line_shape, 0, dtype="uint")

    for index in range(0, triangles.shape[0]): #triangles.shape[0]
        triangle = triangles[index][na, ...]
        tri_point = triangle[..., 0, :]
        tri_normal = triangle_normals[index][na, ...]
        tri_index = np.full(line_shape, index, dtype="uint")

        tri_inter = intersection_plane_line(tri_point, tri_normal, line_vec, line_pts)
        inter_point = (tri_inter * line_vec) + line_pts
        tri_hit = inside_out_test(triangle, tri_normal, inter_point)
        tri_value = np.where(tri_hit, tri_inter, np.nan)

        all_values = np.concatenate([min_value, tri_value], axis=-1)
        all_index = np.concatenate([min_index, tri_index], axis=-1)

        all_min = np.nanmin(all_values, axis=-1, keepdims=True) #nanargmin throws error when all nan
        all_min_mask = all_values==all_min
        all_min_index = np.argmax(all_min_mask, axis=-1, keepdims=True)

        min_value = np.take_along_axis(all_values, all_min_index, -1)
        min_index = np.take_along_axis(all_index, all_min_index, -1)
    return (min_value, min_index)



def vector_angle(triangle, ray):
    """returns the angle between two vectors

    Parameters
    ----------
    triangle : ndarray
        ([m], 3) vectors 1
    ray : ndarray
        ([m], 3) vectors 2, shape must match

    Returns
    -------
    ndarray
        ([m], 1) 
    """
    triangle_nrm = normalize_vector(triangle)
    ray_nrm = normalize_vector(ray)
    angle = np.einsum("...m, ...m -> ...", triangle_nrm, ray_nrm)[..., na]
    absolute = np.abs(angle)
    return absolute
