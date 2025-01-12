"""
# Animation
Functionality for:
- animating multiple frames in a scene
- merging frames into a video
"""
import shutil
from pathlib import Path
import os
import numpy as np

import casting as cast

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

def main(framepath):
    """testing method
    """
    frame_num = 2
    fps = 10
    decimals = len(str(frame_num)) +1

    if framepath.exists() and framepath.is_dir():
        shutil.rmtree(framepath)

    os.makedirs(framepath)

    #Strahlen
    a = np.array([-5, -5, 5])
    b = np.array([5, -5, 5])
    c = np.array([-5, 5, 5])
    m = 500
    points = cast.pixel_grid(a, b, m, c, m)

    origin = np.array([0, 0, 0])

    #Szene
    general = {
        "ambient": 1,
        "origin": origin
    }

    #Dreiecke
    a = 3
    for i in range(0, frame_num):
        triangles = [
            {
                "xyz": [
                    [-10, -10, 30 + (i*a)],
                    [-14, 30, 40 + (i*a)],
                    [30, -14, 40 + (i*a)]],
                "color": [255, 255, 255],
                "material": "1",

            },
            {
                "xyz": [
                    [-3,-3, 7],
                    [-3.2,-3.5, 7],
                    [-3.5,-3.2, 7]],
                "color": [0, 255, 0],
                "material": "1",

            },
        ]
        lights = [
            {
                "xyz": [5, 5, -10],
                "diffuse": 1,
                "specular": 1,
            },

        ]

        triangles = [transform_dict(tri) for tri in triangles]
        scene = {
            "general": general,
            "triangles": triangles,
            "lights": lights
        }



        image = cast.render_image(origin, points, scene)
        image.save(f"./frames/{i:0{decimals}d}.bmp")

    os.system(
        f"ffmpeg -r {fps}/1 -start_number 1 -i \
        {framepath}/%{decimals}d.bmp -c:v libx264 \
          -r 30 -pix_fmt yuv420p out.mp4 -y -hide_banner -loglevel warning")

if __name__ == "__main__":
    FRAMEPATH  = Path("./frames") # files in folder will be deleted!!!
    main(FRAMEPATH)
