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
            if value == "0":
                output_dict["ambient"] = 0.2
                output_dict["diffuse"] = 0.4
                output_dict["specular"] = 0.3
                output_dict["specular_spread"] = 2
            elif value == "1":
                output_dict["ambient"] = 0.2
                output_dict["diffuse"] = 0.4
                output_dict["specular"] = 0.4
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


def calculate_color(vectors, scene):
    """Returns the color that arrives at every point

    Parameters
    ----------
    vectors : ndarray
        ([m], 3)(point, coordinate)
    points : ndarray
        ([m], 3)(point, coordinate)
    scene : dict
        dict containing triangles, lightsources and general values

    Returns
    -------
    ndarray
        ([m], 3)(point, color)
    """
    # scene
    sc_ambient = scene["general"]["ambient"]
    origin = scene["general"]["origin"]

    dist_const_0 = scene["general"]["distance_constants"][0]
    dist_const_1 = scene["general"]["distance_constants"][1]
    dist_const_2 = scene["general"]["distance_constants"][2]

    # triangle
    triangles_coord = np.array([tri["xyz"] for tri in scene["triangles"]])
    triangle_nmls = inter.normalize_vector(np.array([tri["normal"] for tri in scene["triangles"]]))

    triangles_color = np.array([tri["color"] for tri in scene["triangles"]])

    triangles_ambient = np.array([tri["ambient"] for tri in scene["triangles"]])
    triangles_diffuse = np.array([tri["diffuse"] for tri in scene["triangles"]])
    triangles_specular = np.array([tri["specular"] for tri in scene["triangles"]])[..., na]
    tri_spec_spr = np.array([tri["specular_spread"] for tri in scene["triangles"]])

    # light
    lights_coord = np.array([light["xyz"] for light in scene["lights"]])
    lights_diffuse = np.array([light["diffuse"] for light in scene["lights"]])[..., na]
    lights_specular = np.array([light["specular"] for light in scene["lights"]])[..., na]

    # intersections
    intersection, inter_index  = \
        inter.intersection_ray_triangle(vectors, scene["points"], triangles_coord, triangle_nmls)
    inter_p =  scene["points"] + (intersection * vectors)
    lights_coord_br = np.broadcast_to(lights_coord, inter_p.shape[:-1] + lights_coord.shape)
    inter_p_br = np.broadcast_to(inter_p[..., na, :], lights_coord_br.shape)


    # shadow
    light_ray =  lights_coord - inter_p[..., na, :]
    prior_hits = \
        inter.shadow_hit_light(inter_p_br, light_ray, triangles_coord, triangle_nmls, inter_index)
    shadow_mask = np.where(~np.isnan(prior_hits))
    light_ray[shadow_mask] = np.nan
    # shading
    light_ray_norm = inter.normalize_vector(light_ray)
    angle = inter.vector_angle(light_ray_norm, triangle_nmls[inter_index])
    distance = point_distance(inter_p_br, lights_coord_br)
    phong_distance = dist_const_0 + (distance * dist_const_1) + (np.square(distance) * dist_const_2)
    phong_dist_div = 1 / phong_distance

    # ambient
    shade_ambient =  sc_ambient * triangles_ambient

    # diffuse
    shade_diffuse_br = \
        lights_diffuse * triangles_diffuse[inter_index][..., na] * angle * phong_dist_div
    shade_diffuse = np.nansum(shade_diffuse_br, axis=-2)

    # specular
    light_nml_prj = triangle_nmls[inter_index] * \
         np.einsum("...j, ...j -> ...", triangle_nmls[inter_index], light_ray_norm)[..., na]
    light_ref = 2 * light_nml_prj - light_ray_norm

    ray_inter_orig = inter.normalize_vector(inter_p - origin)
    light_ref_ang = inter.vector_angle(light_ref, ray_inter_orig[..., na, :])
    ang_power = np.power(light_ref_ang, tri_spec_spr[inter_index][..., na, :])

    tri_spec_br = triangles_specular[inter_index]
    lights_spec_br = np.broadcast_to(lights_specular, \
                                     tri_spec_br.shape[:-2] + lights_specular.shape)
    shade_specular_br = lights_spec_br * tri_spec_br * ang_power * phong_dist_div
    shade_specular = np.nansum(shade_specular_br, axis=-2)

    # combine
    shade_combine = shade_ambient[inter_index] + shade_specular + shade_diffuse
    color_shaded = triangles_color[inter_index] * shade_combine[..., na]
    not_nan_mask = ~np.isnan(intersection)[..., na]
    color_mskd = np.where(not_nan_mask, color_shaded, np.nan)
    color_sum = np.nanmin(color_mskd, axis=-2)
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
    vectors = np.array([shift_b, shift_c])
    index_array = np.indices((res_b, res_c)).transpose(1, 2, 0)
    array = point_a + np.sum((index_array[..., na] * vectors), axis=-2)

    return np.array(array)


def render_image(scene):
    """renders a pillow image for 3d scene

    Parameters
    ----------
    pov : ndarray
        (3)(coodrinate) point of view for camera
    points : ndarray
        (m, n, 3)(height, length, coordinate) grid for pixels of screen
    scene : dict
        dict containing triangles, lightsources and general values

    Returns
    -------
    pillow image
        pillow image render of scene
    """
    vectors = inter.normalize_vector(vector_from_points(scene["origin"], scene["points"]))
    rgb_image = calculate_color(vectors, scene)
    image = Image.fromarray(rgb_image)
    return image
