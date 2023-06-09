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

import random
import numpy as np
from scipy.linalg import logm, expm, pinv, fractional_matrix_power


# Implemented as described by Rathi et al. [1]
def gradient_descent_mean(*args):
    n = len(args)
    p = args[0].shape[0]

    # Set initial parameters and learning rate
    mu_t = random.choice(args)
    d = 0.01 # learning rate

    # Set threshold for change in objective function
    threshold = 1e-1

    max_iter = 10000

    t = 1

    while t < max_iter:

        mu_sqrt = fractional_matrix_power(mu_t, 0.5)

        sum_logs = np.sum(logm(pinv(mu_sqrt) @ args[i] @ pinv(mu_sqrt)) for i in range(n))

        mu_t_1 = mu_sqrt @ expm((1 / n) * sum_logs) @ mu_sqrt
        delta = np.abs(np.subtract(mu_t_1, mu_t))
        t += 1

        mu_t = mu_t_1

        if any(i.any() < threshold for i in delta.flatten()):
            return mu_t_1

    return mu_t_1
