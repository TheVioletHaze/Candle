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


def calculate_color(vectors, points, triangles, scene):
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
    triangles_coord = np.array([tri["xyz"] for tri in triangles])
    triangles_color = np.array([tri["color"] for tri in triangles])
    triangles_diffuse = np.array([tri["diffuse"] for tri in triangles])
    triangles_specular = np.array([tri["specular"] for tri in triangles])


    intersections = inter.intersection_ray_triangle(vectors, points, triangles_coord)

    color = np.broadcast_to(triangles_color, (intersections.shape[:-1] + (3,)))[..., na, :]


    triangles_ambient = np.array([tri["ambient"] for tri in triangles])[..., na]
    tri_ambient_br = np.broadcast_to(triangles_ambient, intersections.shape)

    triangles_diffuse = np.array([tri["diffuse"] for tri in triangles])[..., na]
    tri_diffuse_br = np.broadcast_to(triangles_diffuse, intersections.shape)

    triangles_specular = np.array([tri["specular"] for tri in triangles])[..., na]
    tri_specular_br = np.broadcast_to(triangles_specular, intersections.shape)

    
    angle = inter.incidence_angle(triangles_coord, vectors)

    # shading
    shade_ambient = tri_ambient_br * scene["ambient"]
    shade_diffuse = tri_diffuse_br * scene["diffuse"] * angle

    shade_combine = shade_ambient + shade_diffuse

    color_shaded = color * shade_combine[..., na]

    not_nan_mask = ~np.isnan(intersections)[..., na]
    color_mskd = np.where(not_nan_mask, color_shaded, np.nan)
    color_sum = np.nansum(color_mskd, axis=-3)
    color_abs = np.abs(color_sum)
    color_squeeze= np.squeeze(color_abs)

    np.where(color_squeeze, color_squeeze > 255, 255)
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


def render_image(pov, points, triangles, scene):
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

    rgb_image = calculate_color(vectors, points, triangles, scene)
    image = Image.fromarray(rgb_image)
    return image

def main():
    """testing method
    """
    #Strahlen
    a = np.array([-5, -5, 5])
    b = np.array([5, -5, 5])
    c = np.array([-5, 5, 5])
    m = 500
    points = pixel_grid(a, b, m, c, m)

    origin = np.array([0, 0, 0])

    #Szene
    scene = {
        "ambient": 1,
        "diffuse": 1,
        "specular": 1
    }

    #Dreiecke
    triangles = [
            {
                "xyz": [
                    [-10, -10, 30],
                    [-14, 30, 40],
                    [30, -14, 40]],
                "color": [255, 255, 255],
                "ambient": 0.3,
                "diffuse": 0.4,
                "specular": 0.3
            },
            {
                "xyz": [
                    [-3,-3, 7],
                    [-3.2,-3.5, 7],
                    [-3.5,-3.2, 7]],
                "color": [0, 255, 0],
                "ambient": 0.3,
                "diffuse": 0.4,
                "specular": 0.3
            },
        ]

    image = render_image(origin, points, triangles, scene)
    image.show()

if __name__ == "__main__":
    main()
