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
    AB = triangles[:, 0, ] - triangles[:, 1, ]
    AC = triangles[:, 0, ] - triangles[:, 2, ]
    return np.cross(AB, AC)


def intersection_plane_line(triangles_plane_points, triangles_plane_normals, line_vectors, line_points):
    """Returns scalar for line vector to intersection

    Parameters
    ----------
    triangles_plane_points : matrix
        (m, 3)(triangle, any point)
    triangles_plane_normals : matrix
        (m, 3)(triangle, vector)
    line_vectors : matrix
        (m, 3)(line, vector)
    line_points : matrix
        (m, 3)(line, point) 

    Returns
    -------
    matrix
        (m, n, 1)(line, triangle, scalar)
    """
    
    po_qo = line_points[:, na, :] - triangles_plane_points[na, :, :] # (Strahl, Ebene, Punkt)
    n_po_qo = np.einsum("ijk, jk->ij", po_qo, triangles_plane_normals)[..., na] * -1

    n_p = np.einsum("ik, jk->ij", line_vectors, triangles_plane_normals)[..., na]
    
    t = n_po_qo / n_p
    return t

def inside_out_test(triangles, normals, points):
    offset1 = np.roll(triangles, -1, axis=0) # [b, c, a]
    offset2 = np.roll(triangles, -2, axis=0) # [c, b, a]
    
    line_vectors = offset1 - offset2 # for each Point of the triangle the opposite side 
    line_normals = np.cross(line_vectors, normals[:, na]) # normal of the opposite side


    line_normals_exp = line_normals[na, :, :, na, :]
    tri_points_exp = triangles[na, :, :, na, :]
    points_exp = points[:, :, na, na, :]

    tri_points_br = np.broadcast_to(tri_points_exp, ((points_exp.shape[0],) + tri_points_exp.shape[1:5])) #broadcast to number of intersections
    points_br = np.broadcast_to(points_exp, (points_exp.shape[0:2] + (line_normals_exp.shape[2],) + points_exp.shape[3:5]))

    merged = np.concatenate([tri_points_br, points_br], axis=-2) # (n, m, 3, 2, 3) (tri, lin, vert, [P_tri, P_lin], xyz)
    print(line_normals.shape)
    print(merged.shape)
    # merged_dot = np.einsum("", merged, line_normals)

if __name__ == "__main__":
    #Strahlen
    line_points = np.array([
        [1, 2, 3], 
        [4, 5, 6]
        ])
    
    line_vectors = np.array([
        [7, 8, 9], 
        [10, 11, 12]
    ])

    #Dreiecke
    triangles = np.array([
        [[0, 0, 0], [1, 0, 0], [0, 1, 0]],  # Triangle 1 (XY Plane)
        [[0, 0, 0], [1, 0, 0], [0, 0, 1]],  # Triangle 2 (XZ Plane)
        [[0, 0, 0], [0, 1, 0], [0, 0, 1]],  # Triangle 3 (YZ Plane)
        [[0, 0, 0], [1, 1, 0], [0, 1, 1]],  # Triangle 4 (Diagonal Plane)
    ])

    triangles_plane_points = triangles[:, 0]
    triangles_plane_normals = normal_from_triangle(triangles)
    # Schnittpunkte 
    inter_scalars = intersection_plane_line(triangles_plane_points, triangles_plane_normals, line_vectors, line_points)
    inter_points = line_points[:, na, :] + (line_vectors[:, na, :] * inter_scalars)
    inside_out_test(triangles, triangles_plane_normals, inter_points)