import numpy as np
import intersection as inter

def main():
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
    print(inter.intersection_ray_triangle(line_vec, line_pts, triangles))

if __name__ == "__main__":
    main()