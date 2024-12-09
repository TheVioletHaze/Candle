"""Finds at what points rays hit which triangle

Returns
-------
matrix
    _description_ #todo
"""
import warnings
import numpy as np
from numpy import newaxis as na

warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')



def vector_from_points(point_1, point_2):
    """Returns A Vector from point a to b

    Parameters
    ----------
    point_1 : ndarray
        ([m], 3)([points], coordinate)
    point_2 : ndarray
        ([m], 3)([points], coordinate)

    Returns
    -------
    ndarray
        ([m], 3)([line], coordinate)
    """
    return point_1-point_2

def normal_from_triangle(triangles):
    """Returns Normals for triangles

    Parameters
    ----------
    triangles : matrix
        ([m], 3, 3)([triangle], vertex, coordinate)
        ([m], 3, 3)([triangle], vertex, coordinate)

    Returns
    -------
    matrix
        ([m], 3)([triangle], vector)
        ([m], 3)([triangle], vector)
    """
    a_b = triangles[..., 0, :] - triangles[..., 1, :]
    a_c = triangles[..., 0, :] - triangles[..., 2, :]
    return np.cross(a_b, a_c)


def intersection_pln_line(triangle_pl_pts, triangle_pl_nml, line_vec, line_pts):
    """Returns scalar for line vector to intersection

    Parameters
    ----------
    triangle_pl_pts : matrix
        (m, 3)(triangle, any point)
    triangle_pl_nml : matrix
        (m, 3)(triangle, vector)
    line_vec : matrix
        ([m], 3)([line], vector)
        ([m], 3)([line], vector)
    line_pts : matrix
        ([m], 3)([line], point) 
        ([m], 3)([line], point) 

    Returns
    -------
    matrix
        ([m], n, 1)([line], triangle, scalar)
        ([m], n, 1)([line], triangle, scalar)
    """

    po_qo = line_pts[..., na, :] - triangle_pl_pts[na, :, :] # (Strahl, Ebene, Punkt)
    n_po_qo = np.einsum("...jk, jk->...j", po_qo, triangle_pl_nml)[..., na] * -1

    n_p = np.einsum("...k, jk->...j", line_vec, triangle_pl_nml)[..., na]

    t = n_po_qo / n_p
    return t

def inside_out_test(triangles, normals, points):
    """Returns bool array whether point inside triangle

    Parameters
    ----------
    triangles : matrix
        (m, 3, 3)(triangle, vertex, coordinate)
    normals : matrix
        (m, 3)(triangle, vector)
    points : matrix
        ([m], n, 3)([line], triangle, coordinate)
        ([m], n, 3)([line], triangle, coordinate)

    Returns
    -------
    matrix
        ([m], n, 1)([line], triangle, boolean)
        ([m], n, 1)([line], triangle, boolean)
    """
    offset1 = np.roll(triangles, -1, axis=-2) # [b, c, a]
    offset2 = np.roll(triangles, -2, axis=-2) # [c, b, a]

    line_vec = offset1 - offset2 # opposite side
    line_vec = offset1 - offset2 # opposite side
    line_normals = np.cross(line_vec, normals[..., na, :]) # normal of the opposite side


    tri_pts_exp = triangles[na, :, :, na, :] # exp for num of lines and concatenate
    points_exp = points[..., na, na, :] # exp for num of triangles and concatenate
    tri_pts_exp = triangles[na, :, :, na, :] # exp for num of lines and concatenate
    points_exp = points[..., na, na, :] # exp for num of triangles and concatenate

    shape = points_exp.shape[:-4] + tri_pts_exp.shape[-4:]
    triangle_pts_br = np.broadcast_to(tri_pts_exp, shape)
    triangle_pts_br = np.broadcast_to(tri_pts_exp, shape)
    points_br = np.broadcast_to(points_exp, shape)


    merged = np.concatenate([triangle_pts_br, points_br], axis=-2)
        # ([n], m, 3, 2, 3) ([lin], tri, vert, [P_tri, P_lin], xyz)

    side_points = merged - offset1[na, :, :, na, :]

    side_dotprods = np.einsum("...vmn, ...vn -> ...vm", side_points, line_normals)
    side_dotprods_prod = np.prod(side_dotprods, axis=-1)[..., na]
    side_inter_bool = side_dotprods_prod > 0
    side_inter_bool_merged = np.all(side_inter_bool, axis=-2)
    return side_inter_bool_merged

def intersection_ray_triangle(line_vec, line_pts, triangles):
    """Returns the scalar of the first intersection point of rays for given triangles. 
    Triangles not first hit or missed nan.

    Parameters
    ----------
    line_vec : ndarray
        ([m], 3)([vector], coordinate) Can be any given dimension. Shape must match line_pts.
    line_pts : ndarray
        ([m], 3)([point], coordinates) Can be any given dimension. Shape must match line_pts.
    triangles : ndarray
        (m, 3, 3)(triangle, vertex, coordinate) Must match shape.

    Returns
    -------
    ndarray
        ([m], n, 1)([line], triangle, scalar)
    """

    triangle_pl_pts = triangles[:, 0]
    triangle_pl_nml = normal_from_triangle(triangles)

    # Schnittpunkte
    inter_sc = intersection_pln_line(triangle_pl_pts, triangle_pl_nml, line_vec, line_pts)
    inter_points = line_pts[..., na, :] + (line_vec[..., na, :] * inter_sc)

    inter_hits_mask = inside_out_test(triangles, triangle_pl_nml, inter_points)
    inter_sc_hit_mskd = np.where(inter_hits_mask, inter_sc, np.nan) #nan if ray doesn't hit

    inter_sc_min = np.nanmin(inter_sc_hit_mskd, axis=-2, keepdims=True)
    inter_min_mask = inter_sc == inter_sc_min
    inter_sc_min_mskd = np.where(inter_min_mask, inter_sc, np.nan) # also nan if not first hit
    return inter_sc_min_mskd

def main(): #todo remove
    """testing method
    """
    #Strahlen
    line_pts = np.array([
        [1, 2, 3],
        [4, 5, 6],
        ])

    line_vec = np.array([
        [7, 8, 9],
        [10, 11, 12],
    ])

    #Dreiecke
    triangles = np.array([
        [[-10, -10, 0], [100, 0, 0], [0, 100, 0]],  # Triangle 1 (XY Plane)
        [[-10, -10, 0], [0, 0, 0], [0, 100, 100]],  # Triangle 1 (XY Plane)
        [[0, 0, 0], [1, 0, 0], [0, 0, 1]],  # Triangle 2 (XZ Plane)
        # [[0, 0, 0], [0, 1, 0], [0, 0, 1]],  # Triangle 3 (YZ Plane)
        # [[0, 0, 0], [1, 1, 0], [0, 1, 1]],  # Triangle 4 (Diagonal Plane)
    ])
    print(intersection_ray_triangle(line_vec, line_pts, triangles))

if __name__ == "__main__":
    main()
