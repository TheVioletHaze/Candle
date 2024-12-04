import numpy as np

def normal_from_triangle(triangles):
    AB = triangles[:, 0, ] - triangles[:, 1, ]
    AC = triangles[:, 0, ] - triangles[:, 2, ]
    return np.cross(AB, AC)

def intersection_plane_line(triagles_plane_points, triangles_plane_normals, vectors):
    tri_p_n_long = triangles_plane_normals[np.newaxis, :, :]
    po_qo = points[:, np.newaxis, :] - triagles_plane_points[np.newaxis, :, :] # (Strahl, Ebene, Punkt)
    n_po_qo = np.einsum("ijk, ijk->ij", po_qo, tri_p_n_long)[..., np.newaxis] * -1

    n_p = np.einsum("ijk, ijk->ij", vectors[:, np.newaxis, :], tri_p_n_long)[..., np.newaxis]
    
    t = n_po_qo / n_p
    return t

def inside_out(triangles, inter_points):
    print("todo")

if __name__ == "__main__":
    #Strahlen
    points = np.array([
        [1, 2, 3], [4, 5, 6]
        ])
    
    vectors = np.array([
        [7, 8, 9], [10, 11, 12]
    ])

    #Dreiecke
    triangles = np.array([
        [[0, 0, 0], [1, 0, 0], [0, 1, 0]],  # Triangle 1 (XY Plane)
        [[0, 0, 0], [1, 0, 0], [0, 0, 1]],  # Triangle 2 (XZ Plane)
        [[0, 0, 0], [0, 1, 0], [0, 0, 1]],  # Triangle 3 (YZ Plane)
        [[0, 0, 0], [1, 1, 0], [0, 1, 1]],  # Triangle 4 (Diagonal Plane)
    ])

    triagles_plane_points = triangles[:, 0]
    triangles_plane_normals = normal_from_triangle(triangles)
    print(triangles_plane_normals)

    # Schnittpunkte 
    intersections = intersection_plane_line(triagles_plane_points, triangles_plane_normals, vectors)
    inter_points = vectors[:, np.newaxis, :] * intersections + points[:, np.newaxis, :]

    # inside out
