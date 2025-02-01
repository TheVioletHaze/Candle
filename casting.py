"""
# Casting
Functionality for:
- casting rays in a scene
- calculating color
"""
import warnings
import numpy as np
from numpy import newaxis as na
from PIL import Image
import intersection as inter
warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
warnings.filterwarnings('ignore', r'invalid value encountered in divide')


def transform_dict(input_dict):
    """allows shortcuts in a dictionary.

    Parameters
    ----------
    input_dict : dictionary
        dictionary with shortcuts

    Returns
    -------
    dictionary
        dictionary where shortcuts are expanded
    """
    output_dict = {}

    for key, value in input_dict.items():
        if key == "material":
            if value == "1":
                output_dict["ambient"] = 0.3
                output_dict["diffuse"] = 0.4
                output_dict["specular"] = 0.3
                output_dict["specular_spread"] = 2
        else:
            output_dict[key] = value
    return output_dict


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
    return point_2 - point_1

def point_distance(point_1, point_2):
    """Returns the distance between point a and b

    Parameters
    ----------
    point_1 : ndarray
        ([m], 3)
    point_2 : ndarray
        ([m], 3)

    Returns
    -------
    ndarray
        ([m], 1)([line], distance)
    """
    vector = vector_from_points(point_1, point_2)
    return np.linalg.norm(vector, axis=-1, keepdims=True)


def calculate_color(vectors, points, scene):
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
    # scene
    sc_ambient = scene["general"]["ambient"]
    origin = scene["general"]["origin"]

    dist_const_0 = scene["general"]["distance_constants"][0]
    dist_const_1 = scene["general"]["distance_constants"][1]
    dist_const_2 = scene["general"]["distance_constants"][2]

    # triangle
    triangles_coord = np.array([tri["xyz"] for tri in scene["triangles"]])
    triangles_color = np.array([tri["color"] for tri in scene["triangles"]])
    triangles_diffuse = np.array([tri["diffuse"] for tri in scene["triangles"]])
    triangles_specular = np.array([tri["specular"] for tri in scene["triangles"]])

    # intersections
    intersections = inter.intersection_ray_triangle(vectors, points, triangles_coord)
    points_br = np.broadcast_to(points[..., na, :], (intersections.shape[:-1] + (3,)))
    vectors_br = np.broadcast_to(vectors[..., na, :], (intersections.shape[:-1] + (3,)))
    inter_p =  points_br + (intersections * vectors_br)

    color = np.broadcast_to(triangles_color, (intersections.shape[:-1] + (3,)))[..., na, :]
    # light
    lights_coord = np.array([light["xyz"] for light in scene["lights"]])
    light_coord_br = np.broadcast_to(lights_coord[..., na, :],
                        (inter_p.shape[:-1] + lights_coord.shape[:-1] + (inter_p.shape[-1],)))

    # other
    inter_p_br = np.broadcast_to(inter_p[..., na, :], light_coord_br.shape)
    light_ray = inter.normalize_vector(inter_p_br - light_coord_br)
    triangles_nml = inter.normalize_vector(inter.normal_from_triangle(triangles_coord))
    triangles_nml_br = np.broadcast_to(triangles_nml[..., na, :], light_coord_br.shape)
    angle = inter.vector_angle(triangles_nml_br, light_ray)

    # lights i
    lights_diffuse = np.array([light["diffuse"] for light in scene["lights"]])[..., na]
    lights_diffuse_br = np.broadcast_to(lights_diffuse[..., na, :], angle.shape)

    lights_specular = np.array([light["specular"] for light in scene["lights"]])[..., na]
    lights_specular_br = np.broadcast_to(lights_specular[..., na, :], angle.shape)

    # triangle k
    triangles_ambient = np.array([tri["ambient"] for tri in scene["triangles"]])[..., na]
    tri_ambient_br = np.broadcast_to(triangles_ambient, intersections.shape)
    tri_amb_max = np.max(tri_ambient_br, -2)[..., na]
    triangles_diffuse = np.array([tri["diffuse"] for tri in scene["triangles"]])[..., na]
    tri_diffuse_br = np.broadcast_to(triangles_diffuse[..., na, :], angle.shape)

    triangles_specular = np.array([tri["specular"] for tri in scene["triangles"]])[..., na]
    tri_specular_br = np.broadcast_to(triangles_specular[..., na, :], angle.shape)
    tri_spec_spr = np.array([tri["specular_spread"] for tri in scene["triangles"]])[..., na]
    tri_spec_spr_br = np.broadcast_to(tri_spec_spr[..., na, :], angle.shape)

    # shading
    distance = point_distance(inter_p_br, light_coord_br)
    phong_distance = dist_const_0 + (distance * dist_const_1) + (np.square(distance) * dist_const_2)
    phong_dist_div = 1 / phong_distance
    # ambient
    shade_ambient =  sc_ambient * tri_amb_max

    # diffuse
    shade_diffuse_br = lights_diffuse_br * tri_diffuse_br * angle * phong_dist_div
    shade_diffuse = np.nansum(shade_diffuse_br, axis=-3)

    # specular
    light_nml_prj = triangles_nml_br * (np.einsum("...jk, ...jk->...j",
                                                  triangles_nml_br, light_ray)[..., na])
    light_ref = 2 * light_nml_prj - light_ray

    ray_inter_orig = inter.normalize_vector(inter_p_br - origin)
    light_ref_ang = inter.vector_angle(light_ref, ray_inter_orig)
    ang_power = np.power(light_ref_ang, tri_spec_spr_br)
    shade_specular_br = lights_specular_br * tri_specular_br * ang_power * phong_dist_div
    shade_specular = np.nansum(shade_specular_br, axis=-3)

    # combine
    shade_combine = shade_ambient + shade_specular + shade_diffuse
    color_shaded = color * shade_combine[..., na]
    not_nan_mask = ~np.isnan(intersections)[..., na]
    color_mskd = np.where(not_nan_mask, color_shaded, np.nan)
    color_sum = np.nanmin(color_mskd, axis=-3)
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


def render_image(pov, points, scene):
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
    vectors = inter.normalize_vector(vector_from_points(pov, points))
    rgb_image = calculate_color(vectors, points, scene)
    image = Image.fromarray(rgb_image)
    return image

def main():
    """testing method
    """
    #Strahlen
    a = np.array([-20, 5, 20])
    b = np.array([-20, 5, -20])
    c = np.array([20, 5, 20])
    m = 2000
    points = pixel_grid(a, b, m, c, m)

    origin = np.array([0, -40, 0])

    #Szene
    general = {
        "ambient": 1,
        "origin": origin,
        "distance_constants": (1, 0.02, 0.01),
    }

    #Dreiecke
    A_1 = [8.21262, 18.85506, 3.09627]
    B_1 = [-10, 20, -5.87594]
    C_1 = [-16.063217043248358, 33.422827240347814, 8.144619015757515]
    D_1 = [2.149402956751641, 32.277887240347816, 17.116829015757514]
    E_1 = [14.924463187542646, 34.0875183841308, -8.58424110840877]
    F_1 = [-3.2881568124573537, 35.2324583841308, -17.55645110840877]
    G_1 = [-9.351373855705711, 48.655285624478616, -3.5358920926512583]
    H_1 = [8.861246144294288, 47.51034562447862, 5.436317907348743]
    triangles = [
            {
                "xyz": [A_1, B_1, C_1],
                "color": [0, 255, 255],
                "material": "1",

            },
            {
                "xyz": [A_1, C_1, D_1],
                "color": [0, 255, 255],
                "material": "1",

            },
            {
                "xyz": [A_1, B_1, E_1],
                "color": [255, 255, 0],
                "material": "1",

            },
            {
                "xyz": [F_1, B_1, E_1],
                "color": [255, 255, 0],
                "material": "1",

            },
            {
                "xyz": [A_1, D_1, H_1],
                "color": [255, 0, 255],
                "material": "1",

            },
            {
                "xyz": [A_1, H_1, E_1],
                "color": [255, 0, 255],
                "material": "1",

            },
        ]
    lights = [
        {
            "specular": 1,
            "diffuse": 1,
            "xyz": [5, 5, -10], 
        }
    ]

    triangles = [transform_dict(tri) for tri in triangles]
    scene = {
        "general": general,
        "triangles": triangles,
        "lights": lights
    }

    image = render_image(origin, points, scene)
    image.save("temp.png")
    image.show()

if __name__ == "__main__":
    main()
