import shutil
from pathlib import Path
import os
import numpy as np
from PIL import Image

import casting as cast

def main(framepath):
    """testing method
    """
    frame_num = 30
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
    scene = {
        "ambient": 1,
        "diffuse": 1,
        "specular": 1,
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



        image = cast.render_image(origin, points, triangles, scene)
        image.save(f"./frames/{i:0{decimals}d}.bmp")

    os.system(
        f"ffmpeg -r 10/1 -start_number 1 -i \
        {framepath}/%{decimals}d.bmp -c:v libx264 -r 30 -pix_fmt yuv420p out.mp4 -y")

if __name__ == "__main__":
    FRAMEPATH  = Path("./frames") # files in folder will be deleted!!!
    main(FRAMEPATH)
