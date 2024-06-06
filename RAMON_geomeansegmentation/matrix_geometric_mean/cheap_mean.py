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
from scipy.linalg import logm, expm, pinv, det


# Implemented as described by Bini et al. [3]
def cheap_mean(*args):
    n = args[0].shape[0]
    p = len(args)
    tol = 1e-12
    maxiter = 1000

    A = [args[k] for k in range(p)]
    ct = 0

    while True:

        ct += 1

        A1 = [np.zeros((n, n)) for k in range(p)]
        for k in range(p):

            B = pinv(A[k])
            S = np.zeros((n, n))
            for h in range(p):

                if h != k:
                    X = B @ A[h]
                    X_logm = np.zeros((n, n)) \
                        if det(X) == 0 \
                        else logm(X)
                    S = np.add(S, X_logm)

            A1[k] = A[k] @ expm(1 / p * S)

        ni = np.linalg.norm(A1[0] - A[0])
        if ni < tol:
            iter = ct
            break
        if ct == maxiter:
            print('CHEAP: Max number of iterations reached')
            iter = ct
            break
        A = A1.copy()

    C = A1[0]
    for k in range(1, p):
        C = np.add(C, A1[k])
    C = C / p

    return C, iter
