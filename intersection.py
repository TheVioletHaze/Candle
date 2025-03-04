"""
# Intersection
Provides functions for 
- working with vectors and triangles
- calculating intersections
"""
import warnings
import numpy as np
from numpy import newaxis as na
from alive_progress import alive_bar
from opt_einsum import contract

warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
warnings.filterwarnings('ignore', r'invalid value encountered in divide')


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
    triangle_indices = np.indices(triangles.shape[:-2]).transpose(1, 0)[..., 0]

    with alive_bar(vec_shape[0] * vec_shape[1]) as pixel:
        for i in range(0, vec_shape[0]):
            for j in range(0, vec_shape[1]):
                vector = line_vec[i, j]
                point = line_pts[i, j]

                filtered_index = triangle_indices
                filtered_triangles = triangles[filtered_index]
                tri_normal = triangle_normals[filtered_index]
                tri_point = filtered_triangles[..., 0, :]

                tri_inter = intersection_plane_line(tri_point, tri_normal, vector, point)[..., na]
                inter_point = (tri_inter * vector) + point
                tri_hit = inside_out_test(filtered_triangles, tri_normal, inter_point)
                values = np.where(tri_hit, tri_inter, np.nan)

                value_min = np.nanmin(values) # argmin error when all nan
                min_mask = values==value_min
                hit_index = np.argmax(min_mask)
                min_index[i, j] = filtered_index[hit_index]
                min_value[i, j] = value_min
                pixel()
    return (min_value, min_index)

def shadow_hit_light(inter_p, light_ray, triangles, triangle_normals, hit_tri_index):
    """casts shadow rays to see if any triangles intersect a line from point to light source.

    Parameters
    ----------
    inter_p : ndarray
        ([m], 3)([point], coordinate)
    light_ray : ndarray
        ([m], 3)([ray], coordinate)
    triangles : ndarray
        (n, 3)(triangle, coordinate)
    triangle_normals : ndarray
        (n, 3)(triangle, coordinate)
    hit_tri_index : ndarray
        ([m], 1)([point], index of hit triangle)

    Returns
    -------
    ndarray
        ([m], 1)([point], scalar) scaral is nan if no triangle is hit.

    Raises
    ------
    ValueError
        shape of light_ray and inter_p doesn't match
    """
    if light_ray.shape != inter_p.shape:
        raise ValueError(f"shape of light_ray {light_ray.shape} \
                         and inter_p {inter_p.shape} doesn't match.")

    line_shape = light_ray.shape[:-1] + (1,)
    min_value = np.full(line_shape, np.nan)
    min_index = np.full(line_shape, 0, dtype="uint")
    hit_tri_index_br = np.broadcast_to(hit_tri_index[..., na, :], line_shape)

    with alive_bar(len(range(0, triangles.shape[0]))) as progress:
        for index in range(0, triangles.shape[0]):
            triangle = triangles[index][na, ...]
            tri_point = triangle[..., 0, :]
            tri_normal = triangle_normals[index][na, ...]
            tri_index = np.full(line_shape, index, dtype="uint")

            tri_inter = intersection_plane_line(tri_point, tri_normal, light_ray, inter_p)

            same_tri_mask = np.where(tri_index == hit_tri_index_br)
            tri_inter[same_tri_mask] = np.nan #intersections with tri hit by light ray filtered
            before_mask = np.where(tri_inter < 0)
            tri_inter[before_mask] = np.nan #inter before ray filtered out
            after_mask = np.where(tri_inter > 1)
            tri_inter[after_mask] = np.nan #inter before ray filtered out

            inter_point = (tri_inter * light_ray) + inter_p
            tri_hit = inside_out_test(triangle, tri_normal, inter_point)
            tri_value = np.where(tri_hit, tri_inter, np.nan)
            all_values = np.concatenate([min_value, tri_value], axis=-1)
            all_index = np.concatenate([min_index, tri_index], axis=-1)

            all_min = np.nanmin(all_values, axis=-1, keepdims=True) #error when all nan
            all_min_mask = all_values==all_min
            all_min_index = np.argmax(all_min_mask, axis=-1, keepdims=True)

            min_value = np.take_along_axis(all_values, all_min_index, -1)
            min_index = np.take_along_axis(all_index, all_min_index, -1)
            progress()
    return min_value

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
    return angle
