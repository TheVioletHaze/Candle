import numpy as np
from numpy import newaxis as na


def normal_from_triangle(triangles):
    """Returns Normals for triangles

    Parameters
    ----------
    triangles : matrix
        (m, 3, 3)(triangle, vertex, coordinate)

    Returns
    -------
    matrix
        (m, 3)(triangle, vector)
    """
    a_b = triangles[:, 0, ] - triangles[:, 1, ]
    a_c = triangles[:, 0, ] - triangles[:, 2, ]
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
        (m, 3)(line, vector)
    line_pts : matrix
        (m, 3)(line, point) 

    Returns
    -------
    matrix
        (m, n, 1)(line, triangle, scalar)
    """

    po_qo = line_pts[:, na, :] - triangle_pl_pts[na, :, :] # (Strahl, Ebene, Punkt)
    n_po_qo = np.einsum("ijk, jk->ij", po_qo, triangle_pl_nml)[..., na] * -1

    n_p = np.einsum("ik, jk->ij", line_vec, triangle_pl_nml)[..., na]

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
        (m, n, 3)(line, triangle, coordinate)

    Returns
    -------
    matrix
        (m, n, 1)(line, triangle, boolean)
    """
    offset1 = np.roll(triangles, -1, axis=-2) # [b, c, a]
    offset2 = np.roll(triangles, -2, axis=-2) # [c, b, a]

    line_vec = offset1 - offset2 # for each Point of the triangle the opposite side
    line_normals = np.cross(line_vec, normals[:, na]) # normal of the opposite side


    line_normals_exp = line_normals[na, :, :, na, :]
    tri_points_exp = triangles[na, :, :, na, :]
    points_exp = points[:, :, na, na, :]

    tri_points_br = np.broadcast_to(tri_points_exp, ((points_exp.shape[0],) + tri_points_exp.shape[1:5])) #broadcast to number of intersections
    points_br = np.broadcast_to(points_exp, (points_exp.shape[0:2] + (line_normals_exp.shape[2],) + points_exp.shape[3:5]))

    merged = np.concatenate([tri_points_br, points_br], axis=-2) # (n, m, 3, 2, 3) (tri, lin, vert, [P_tri, P_lin], xyz)

    side_points = merged - offset1[na, :, :, na, :]

    side_dotprods = np.einsum("iklmn, kln -> iklm", side_points, line_normals)
    side_dotprods_prod = np.prod(side_dotprods, axis=-1)[..., na]
    side_inter_bool = side_dotprods_prod > 0
    side_inter_bool_merged = np.all(side_inter_bool, axis=-2)
    return side_inter_bool_merged

def main():
    """Gives first intersection for each ray given rays and triangles
    """
    #Strahlen
    line_pts = np.array([
        [1, 2, 3],
        [4, 5, 6]
        ])

    line_vec = np.array([
        [7, 8, 9],
        [10, 11, 12]
    ])

    #Dreiecke
    triangles = np.array([
        [[-10, -10, 0], [100, 0, 0], [0, 100, 0]],  # Triangle 1 (XY Plane)
        [[0, 0, 0], [1, 0, 0], [0, 0, 1]],  # Triangle 2 (XZ Plane)
        [[0, 0, 0], [0, 1, 0], [0, 0, 1]],  # Triangle 3 (YZ Plane)
        [[0, 0, 0], [1, 1, 0], [0, 1, 1]],  # Triangle 4 (Diagonal Plane)
    ])

    triangle_pl_pts = triangles[:, 0]
    triangle_pl_nml = normal_from_triangle(triangles)
    # Schnittpunkte
    inter_scalars = intersection_pln_line(triangle_pl_pts, triangle_pl_nml, line_vec, line_pts)
    inter_points = line_pts[:, na, :] + (line_vec[:, na, :] * inter_scalars)

    inter_hits = inside_out_test(triangles, triangle_pl_nml, inter_points)
    print(inter_hits)

if __name__ == "__main__":
    main()
