"""loads & renders a scene with an stl file
"""
import warnings
import sys
import numpy as np
from stl import mesh
import casting as cast
warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
warnings.filterwarnings('ignore', r'invalid value encountered in divide')


def main():
    """testing method
    """
    #Strahlen
    a = np.array([-20, -5, 20])
    b = np.array([-20, -5, -20])
    c = np.array([20, -5, 20])
    m = 500 # resolution
    points = cast.pixel_grid(a, b, m, c, m)

    origin = np.array([0, -80, 0])

    #Szene
    general = {
        "ambient": 1,
        "origin": origin,
        "distance_constants": (1, 0.02, 0.01),
    }

    #Dreiecke


    stl_file = sys.argv[1]
    object_mesh = mesh.Mesh.from_file(stl_file)

    triangles = []
    for triangle in object_mesh.data:
        triangle_dict = {
            "normal": triangle[0],
            "xyz": triangle[1],
            "color": [255, 255, 255],
            "material": "1"
        }
        triangles.append(triangle_dict)
    triangles = [cast.transform_dict(tri) for tri in triangles]


    lights = [
        {
            "specular": 1,
            "diffuse": 1,
            "xyz": [-24.968, -11.2209, 5.66921], 
        }
    ]

    scene = {
        "general": general,
        "triangles": triangles,
        "lights": lights
    }

    image = cast.render_image(origin, points, scene)
    image.save("temp.png")
    image.show()

if __name__ == "__main__":
    main()
