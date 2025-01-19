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


def main(framepath):
    """testing method
    """
    total_distance = 0
    total_diffuse = 0
    total_import = 0
    total_intersection = 0
    total_specular = 0

    frame_num = 5
    fps = 30
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
        "origin": origin,
        "distance_constants": (1, 0.02, 0.01),
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

        triangles = [cast.transform_dict(tri) for tri in triangles]
        scene = {
            "general": general,
            "triangles": triangles,
            "lights": lights,
        }


        image = cast.render_image(origin, points, scene)

        total_distance = total_distance + image[1][0]
        total_diffuse = total_diffuse + image[1][1]
        total_import = total_import + image[1][2]
        total_intersection = total_intersection + image[1][3]
        total_specular = total_specular + image[1][4]
        image[0].save(f"./frames/{i:0{decimals}d}.bmp")
    
    print("import:          ", total_import)
    print("intersection:    ", total_intersection)
    print("distance:        ", total_distance)
    print("diffuse:         ", total_diffuse)
    print("specular:        ", total_specular)

    os.system(
        f"ffmpeg -r {fps}/1 -start_number 1 -i \
        {framepath}/%{decimals}d.bmp -c:v libx264 \
          -r 30 -pix_fmt yuv420p out.mp4 -y -hide_banner -loglevel warning")

if __name__ == "__main__":
    FRAMEPATH  = Path("./frames") # files in folder will be deleted!!!
    main(FRAMEPATH)
