import numpy as np


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
    
    po_qo = line_points[:, np.newaxis, :] - triangles_plane_points[np.newaxis, :, :] # (Strahl, Ebene, Punkt)
    n_po_qo = np.einsum("ijk, jk->ij", po_qo, triangles_plane_normals)[..., np.newaxis] * -1

    n_p = np.einsum("ik, jk->ij", line_vectors, triangles_plane_normals)[..., np.newaxis]
    
    t = n_po_qo / n_p
    return t

def inside_out_test(triangles, normals, points):
    offset1 = np.roll(triangles, -1, axis=0) # [b, c, a]
    offset2 = np.roll(triangles, -2, axis=0) # [c, b, a]
    
    line_vectors = offset1 - offset2 # for each Point of A triangle the opposite Site (line to P shouldn't cross)
    line_normals = np.cross(line_vectors, normals[:, np.newaxis])

    line_normals_broadcast = np.broadcast_to()
    print(line_normals[np.newaxis, :, :, :].shape)
    print(points[:, :, np.newaxis, :].shape)
    merged = np.concatenate([line_normals[np.newaxis, :, :, :], points[:, :, np.newaxis, :]], axis=0)

    print(merged)

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
    intersections = intersection_plane_line(triangles_plane_points, triangles_plane_normals, line_vectors, line_points)
    print(intersections.shape)
    # inside_out_test(triangles, triangles_plane_normals, intersections)