"""loads & renders a scene with an stl file
"""
import warnings
import sys
import numpy as np
from stl import mesh
import casting as cast
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
                output_dict["specular_spread"] = 3
            elif value == "2":
                output_dict["ambient"] = 0.5
                output_dict["diffuse"] = 0.4
                output_dict["specular"] = 0.4
                output_dict["specular_spread"] = 2
        else:
            output_dict[key] = value
    return output_dict

def main():
    """takes a resolution and an stl file given as CLI arguments, 
    calculates color for every pixel and shows image
    """
    #Strahlen
    a = np.array([-20, -5, 20])
    b = np.array([-20, -5, -20])
    c = np.array([20, -5, 20])
    m = int(sys.argv[1]) # resolution
    points = cast.pixel_grid(a, b, m, c, m)
    origin = np.array([0, -80, 0])

    #Szene
    general = {
        "ambient": 1,
        "origin": origin,
        "distance_constants": (1, 0.02, 0.01),
        "color_ambient": np.array([255, 255, 255]), 
    }

    #Dreiecke
    stl_file = sys.argv[2]
    object_mesh = mesh.Mesh.from_file(stl_file)
    triangles = []
    for triangle in object_mesh.data:
        triangle_dict = {
            "normal": triangle[0],
            "xyz": triangle[1],
            "color_ambient": np.array([196, 88, 36]),
            "color_diffuse": np.array([196, 88, 36]),
            "color_specular": np.array([255, 255, 255]), #often white
            "material": "1"
        }
        triangles.append(triangle_dict)
    triangles = [transform_dict(tri) for tri in triangles]

    lights = [
        {
            "specular": 2,
            "diffuse": 2,
            "color_diffuse": np.array([255, 255, 255]),
            "color_specular": np.array([255, 255, 255]),
            "xyz": [-13.493, -33.255, 6.1512], 
        },
        # {
        #     "specular": 3,
        #     "diffuse": 1,
        #     "color_diffuse": np.array([255, 255, 255]),
        #     "color_specular": np.array([255, 255, 255]),
        #     "xyz": [-13.493, -33.255, 6.1512], 
        # },
    ]

    scene = {
        "general": general,
        "triangles": triangles,
        "lights": lights,
        "points": points
    }

    image = cast.render_image(scene)
    image.save("temp.png")
    image.show()

if __name__ == "__main__":
    main()
