import numpy as np
from PIL import Image
import intersection as inter

def main():
    """testing method
    """
    #Strahlen
    points=np.array([
        [[15, -1, -1], [15, -1, -1], [15, -1, -1]],
        [[15, -1, -1], [15, -1, -1], [15, -1, -1]]
    ])
    vectors = inter.normalize_vector(inter.vector_from_points(points, [0, 0, 0]))

    #Dreiecke
    triangles = np.array([
        [[15, -5, -5], [20, -5, 5], [20, 5, -5]],
        [[15, -5, -5], [20, -5, 5], [20, 5, -5]],
    ])

    origin_br = np.broadcast_to([0,0,0], vectors.shape)
    intersections = inter.intersection_ray_triangle(vectors, origin_br, triangles)
    angle = inter.incidence_angle(triangles, vectors)
    print(angle.shape)
    print(angle)
if __name__ == "__main__":
    main()
