import numpy as np
from numpy import newaxis as na
from PIL import Image
import intersection as inter

def vector_from_points(point_1, point_2):
    """Returns a Vector from point a to b

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
def main():
    """testing method
    """
    #Strahlen
    n=1000
    m=10
    length = m / n
    start_p=[6, -5, -5]
    points = []
    for i in range(0, n):
        points_temp = []
        for j in range(0,n):
            points_temp.append([start_p[0], start_p[1]+(i*length), start_p[2]+(j*length)])
        points.append(points_temp)

    points=np.array(points)
    vectors = inter.normalize_vector(inter.vector_from_points(points, [0, 0, 0]))

    origin_br = np.broadcast_to([0,0,0], vectors.shape)

    #Dreiecke
    triangles = np.array([
        [[15, -10, -10], [50, -14, 30], [50, 30, -14]],
        [[7,-3,-3], [7,-3.2,-3.5], [7,-3.5,-3.2]],
    ])

    triangles_color = np.array([
        [255, 255, 255],
        [0, 255, 0]
    ])

    intersections = inter.intersection_ray_triangle(vectors, points, triangles)

    color = np.broadcast_to(triangles_color, (intersections.shape[:-1] + (3,)))[..., na, :]
    angle = inter.incidence_angle(triangles, vectors)

    color_shaded = ((color * angle[..., na]) / intersections[..., na])  # diffuse
    color_shaded = np.abs(color_shaded) +20 # ambient
    not_nan_mask = ~np.isnan(intersections)[..., na]
    color_mskd = np.where(not_nan_mask, color_shaded, np.nan)
    color_sum = np.nansum(color_mskd, axis=-3)
    color_abs = np.abs(color_sum)
    color_squeeze= np.squeeze(color_abs)
    print(color_squeeze)
    print(angle)

    rgb_image = color_squeeze.astype(np.uint8)

    image = Image.fromarray(rgb_image)
    image.show()

    # not_nan_mask = ~np.isnan(intersections[..., 0, 0])
    # rgb_image = np.zeros((n, n, 3), dtype=np.uint8)
    # rgb_image[not_nan_mask] = [255, 255, 255]

    # print(rgb_image.shape)
    # image = Image.fromarray(rgb_image)
    # image.show()

if __name__ == "__main__":
    main()
