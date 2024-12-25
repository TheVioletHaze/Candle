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


def calculate_color(vectors, points, triangles, colors):
    """gives the color each pixel should have

    Parameters
    ----------
    vectors : ndarray
        ([m], 3)([lines], coordinates)
    points : ndarray
        ([m], 3)([lines], coordinates), shape must match vectors
    triangles : ndarray
        (n, 3, 3)(triangles, vertecies, coordinates)
    colors : ndarray
        (n, 1)(triangles, color) number of triangles must match

    Returns
    -------
    ndarray
        color that arrives at every point
    """
    intersections = inter.intersection_ray_triangle(vectors, points, triangles)

    color = np.broadcast_to(colors, (intersections.shape[:-1] + (3,)))[..., na, :]
    angle = inter.incidence_angle(triangles, vectors)

    color_shaded = color * angle[..., na]  # diffuse
    color_shaded = np.abs(color_shaded)# ambient
    not_nan_mask = ~np.isnan(intersections)[..., na]
    color_mskd = np.where(not_nan_mask, color_shaded, np.nan)
    color_sum = np.nansum(color_mskd, axis=-3)
    color_abs = np.abs(color_sum)
    color_squeeze= np.squeeze(color_abs)


    rgb_image = color_squeeze.astype(np.uint8)
    return rgb_image

def pixel_grid(point_a, point_b, res_b, point_c, res_c):
    """returns point grid between three points to simulate screen

    Parameters
    ----------
    point_a : ndarray
        (3)(coordinate) top left corner
    point_b : ndarray
        (3)(coordinate) bottom left corner
    res_b : int
        resolution from top to bottom
    point_c : ndarray
        (3)(coordinate) top right corner
    res_c : int
        resolution from left to right


    Returns
    -------
    ndarray
        (height, length)
    """
    vec_b = point_b - point_a
    vec_c = point_c - point_a

    shift_b = vec_b / (res_b-1)
    shift_c = vec_c / (res_c-1)

    points = []
    for i in range(0, res_b):
        row = []
        row_first = point_a + (shift_b * i)
        for j in range(0, res_c):
            row.append(row_first + (shift_c * j))
        
        points.append(row)

    return np.array(points)


def render_image(pov, points, triangles, triangles_color):
    """renders a pillow image for 3d scene

    Parameters
    ----------
    pov : ndarray
        (3)(coodrinate) point of view for camera
    points : ndarray
        (m, n, 3)(height, length, coordinate) grid for pixels of screen
    triangles : ndarray
        (m, 3, 3)(triangles, coordinate)
    triangles_color : ndarray
        (m, 1)(triangles, vertecies, coordinates) number of triangles must match

    Returns
    -------
    pillow image
        pillow image render of scene
    """

    vectors = inter.normalize_vector(vector_from_points(points, pov))

    rgb_image = calculate_color(vectors, points, triangles, triangles_color)
    image = Image.fromarray(rgb_image)
    return image

def main():
    """testing method
    """
    #Strahlen
    a = np.array([5, -5, -5])
    b = np.array([5, 4, -5])
    c = np.array([5, -5, 4])
    m = 500
    points = pixel_grid(a, b, m, c, m)

    origin = np.array([0, 0, 0])

    #Dreiecke
    triangles = np.array([
        [[15, -10, -10], [50, -14, 30], [50, 30, -14]],
        [[7,-3,-3], [7,-3.2,-3.5], [7,-3.5,-3.2]],
    ])

    triangles_color = np.array([
        [255, 255, 255],
        [0, 255, 0]
    ])

    image = render_image(origin, points, triangles, triangles_color)
    image.show()

if __name__ == "__main__":
    main()

