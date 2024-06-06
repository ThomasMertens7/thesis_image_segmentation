"""
This Python package is written in the context of the Master's thesis of Robbe Ramon, KU Leuven.

Inspired from the following references:

    [1] Y. Rathi, A. Tannenbaum and O. Michailovich, "Segmenting Images on the Tensor Manifold," 2007 IEEE Conference on
    Computer Vision and Pattern Recognition, Minneapolis, MN, USA, 2007, pp. 1-8, doi: 10.1109/CVPR.2007.383010.

    [2] C. Li, C. Xu, C. Gui and M. D. Fox, "Distance Regularized Level Set Evolution and Its Application to Image
    Segmentation," in IEEE Transactions on Image Processing, vol. 19, no. 12, pp. 3243-3254, Dec. 2010,
    doi: 10.1109/TIP.2010.2069690.

    [3] Bini, D.A., Iannazzo, B. A note on computing matrix geometric means. Advances in Computational Mathematics 35,
    175–192 (2011). https://doi.org/10.1007/s10444-010-9165-0

Author: Robbe Ramon
Released under MIT license
"""

import numpy as np
from scipy.linalg import eig, inv


def geodesic_distance(matrix1: np.ndarray, matrix2: np.ndarray):
    m1_inv = inv(matrix1)
    x = m1_inv @ matrix2
    eigenv = eig(x)[0]

    distance = np.sqrt(np.sum(np.square(np.log(eigenv))))

    return distance
