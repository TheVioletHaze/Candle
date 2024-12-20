import numpy as np
from PIL import Image
import intersection as inter

def main():
    """testing method
    """
    #Strahlen
    n=1000
    m=10
    length = m / n
    start_p=[10, -5, -5]
    points = []
    for i in range(0, n):
        points_temp = []
        for j in range(0,n):
            points_temp.append([start_p[0], start_p[1]+(i*length), start_p[2]+(j*length)])
        points.append(points_temp)

    points=np.array(points)
    vectors = inter.normalize_vector(inter.vector_from_points(points, [0, 0, 0]))

    #Dreiecke
    triangles = np.array([
        [[15, -5, -5], [20, -5, 5], [20, 5, -5]],
    ])

    intersections =inter.intersection_ray_triangle(vectors, points, triangles)

    not_nan_mask = ~np.isnan(intersections[..., 0, 0])
    rgb_image = np.zeros((n, n, 3), dtype=np.uint8)
    rgb_image[not_nan_mask] = [0, 0, 255]
    image = Image.fromarray(rgb_image)
    image.show()

if __name__ == "__main__":
    main()
