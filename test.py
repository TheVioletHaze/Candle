import numpy as np
from PIL import Image
import intersection as inter

def main():
    """testing method
    """
    #Strahlen
    points=np.array([
        [[0, 0, 0], [5, 0, 0], [5, 3, 4]],
        [[2, 7, 8], [7, 6, 11], [3, 6, 1]]
    ])
    vectors = inter.vector_from_points(points, [0, 0, 0])

    print(inter.normalize_vector(vectors))

if __name__ == "__main__":
    main()
