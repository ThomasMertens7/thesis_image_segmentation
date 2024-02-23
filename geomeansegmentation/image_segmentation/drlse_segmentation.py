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
from enum import Enum
from geomeansegmentation.matrix_geometric_mean.geodesic_distance import geodesic_distance
from geomeansegmentation.matrix_geometric_mean.euclidean_distance import euclidean_distance
from geomeansegmentation.matrix_geometric_mean.cheap_mean import cheap_mean
from scipy.ndimage import laplace
from scipy.ndimage import gaussian_filter
from scipy.linalg import eigh
import cv2


class PotentialFunction(Enum):
    SINGLE_WELL = 0
    DOUBLE_WELL = 1


class EdgeIndicator(Enum):
    GEODESIC_DISTANCE = 0
    SCALAR_DIFFERENCE = 1
    EUCLIDEAN_DISTANCE = 2


def perform_segmentation(
        image: np.ndarray,
        initial_contours_coordinates: [tuple[int, int, int, int]],
        iter_inner: int = 10,
        iter_outer: int = 30,
        lmbda: float = 5,
        alfa: float = -3,
        epsilon: float = 1.5,
        sigma: float = 0.8,
        potential_function: PotentialFunction = PotentialFunction.DOUBLE_WELL,
        edge_indicator: EdgeIndicator = EdgeIndicator.GEODESIC_DISTANCE,
        amount_of_points: int = 10
):
    """Perform image segmentation using the one of three edge indicators en level set evolution

    Parameters:
    image (np.ndarray): The image to segment
    initial_contours_coordinates ([tuple[int, int, int, int]]): The x and y coordinates of all initial contours
    iter_inner (int): amount of inner iterations during level set evolution
    iter_outer (int): amount of outer iterations during level set evolution
    lmbda (float): weight of the weighted length term [2]
    alfa (float): weight of the weighted area term [2]
    epsilon (float): width of Dirac Delta function
    potential_function (EdgeIndicator): choice of potential function in distance regularization term [2]
    amount_of_points (int): the amount of points utilised for calculating
        the matrix geometric mean in the geodesic distance edge indicator

    Yields:
    phi:contour update after evolution step

   """

    # Construct g
    if len(image.shape) == 3:
        g = construct_g(image, edge_indicator, sigma, amount_of_points)
    else:
        g = _construct_g_greyscale(image, edge_indicator, sigma, amount_of_points)

    # Initialize LSF as binary step function
    phi = _initialize_lsf(initial_contours_coordinates, image)
    yield phi

    timestep = 1
    mu = 0.2 / timestep  # coefficient of the distance regularization term R(phi) [2]

    for n in range(iter_outer):
        phi = _update_lsf(phi, g, lmbda, mu, alfa, epsilon, timestep, iter_inner, potential_function)
        yield phi

    # Refine the zero level contour by further level set evolution with alfa=0 [2]
    alfa = 0
    iter_refine = 10
    phi = _update_lsf(phi, g, lmbda, mu, alfa, epsilon, timestep, iter_refine, potential_function)
    yield phi


def construct_g(image: np.ndarray, edge_detector: EdgeIndicator, sigma: float, amount_of_points: int):

    if edge_detector == EdgeIndicator.GEODESIC_DISTANCE:

        img_smooth = gaussian_filter(image, sigma if len(image.shape) < 3 else [sigma, sigma, 0])
        tensor_manifold = _construct_tensor_manifold(img_smooth)
        return _manifold_to_g(tensor_manifold, amount_of_points)
    if edge_detector == EdgeIndicator.SCALAR_DIFFERENCE:

        if len(image.shape) < 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            img_smooth = gaussian_filter(image, sigma)
            [Iy, Ix] = np.gradient(img_smooth)
            f = np.square(Ix) + np.square(Iy)
        else:
            img_smooth = gaussian_filter(image, [sigma, sigma, 0])
            [Iy_1, Ix_1] = np.gradient(img_smooth[:, :, 0])
            [Iy_2, Ix_2] = np.gradient(img_smooth[:, :, 1])
            [Iy_3, Ix_3] = np.gradient(img_smooth[:, :, 2])
            f = np.square(Ix_1) + np.square(Iy_1) + np.square(Ix_2) + np.square(Iy_2) + np.square(Ix_3) + np.square(Iy_3)

        return 1 / (1 + f)
    if edge_detector == EdgeIndicator.EUCLIDEAN_DISTANCE:
        img_smooth = gaussian_filter(image, sigma if len(image.shape) < 3 else [sigma, sigma, 0])
        tensor_manifold = _construct_tensor_manifold(img_smooth)
        return _manifold_to_g_euclidean_mean(tensor_manifold)


def _construct_g_greyscale(image: np.ndarray, edge_detector: EdgeIndicator, sigma: float, amount_of_points: int):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    if edge_detector == EdgeIndicator.GEODESIC_DISTANCE:
        tensor_manifold = _construct_tensor_manifold_greyscale(image)
        return _manifold_to_g(tensor_manifold, amount_of_points)
    if edge_detector == EdgeIndicator.SCALAR_DIFFERENCE:
        img_smooth = gaussian_filter(image, sigma)
        [Iy, Ix] = np.gradient(img_smooth)
        f = np.square(Ix) + np.square(Iy)
        return 1 / (1 + f)


def _manifold_to_g(tensor_manifold: np.ndarray, amount_of_points: int):

    # Calculating the geometric mean
    h, w = tensor_manifold[:2]
    arr = np.zeros(tensor_manifold.shape[:2])
    n_points = amount_of_points
    indices = np.linspace(0, arr.size - 1, n_points, dtype=int)
    x_coords, y_coords = np.unravel_index(indices, arr.shape)
    selected_matrices = tensor_manifold[x_coords, y_coords].reshape(-1, 2, 2)
    selected_matrices = [m for m in selected_matrices if not (m.shape == (2, 2) and np.all(m == 0))]
    mu = cheap_mean(*selected_matrices)[0]

    # Calculating geodesic distances
    distance_matrix = np.zeros(tensor_manifold.shape[:2])
    for idx, x in enumerate(tensor_manifold):
        for idy, y in enumerate(x):

            if np.any(np.iscomplex(mu)):
                mu = np.real(mu)

            distance_matrix[idx, idy] = geodesic_distance(
                tensor_manifold[idx, idy], mu)

    # Calculating the g value
    calculate_g_value = lambda x: (1 / (1 + x ** 8))
    g = np.vectorize(calculate_g_value)(distance_matrix)

    return g


def _manifold_to_g_euclidean_mean(tensor_manifold: np.ndarray):

    # Calculating the Euclidean mean
    selected_matrices = tensor_manifold.reshape(-1, 2, 2)
    selected_matrices = [m for m in selected_matrices if not (m.shape == (2, 2) and np.all(m == 0))]
    mu = np.mean(selected_matrices, axis=0)

    # Calculating Euclidean distances
    distance_matrix = np.zeros(tensor_manifold.shape[:2])
    for idx, x in enumerate(tensor_manifold):
        for idy, y in enumerate(x):

            if np.any(np.iscomplex(mu)):
                mu = np.real(mu)

            distance_matrix[idx, idy] = euclidean_distance(
                tensor_manifold[idx, idy], mu)

    # Calculating the g value
    calculate_g_value = lambda x: (1 / (1 + x ** 8))
    g = np.vectorize(calculate_g_value)(distance_matrix)

    return g


def _construct_tensor_manifold(image: np.ndarray):
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]

    d_x_r, d_y_r = np.gradient(r)
    d_x_g, d_y_g = np.gradient(g)
    d_x_b, d_y_b = np.gradient(b)

    d_x_x_r = np.gradient(d_x_r)[0]
    d_y_y_r = np.gradient(d_y_r)[1]
    d_x_y_r = np.gradient(d_x_r)[1]

    d_x_x_g = np.gradient(d_x_g)[0]
    d_y_y_g = np.gradient(d_y_g)[1]
    d_x_y_g = np.gradient(d_x_g)[1]

    d_x_x_b = np.gradient(d_x_b)[0]
    d_y_y_b = np.gradient(d_y_b)[1]
    d_x_y_b = np.gradient(d_x_b)[1]

    tensor_manifold = np.empty((d_x_r.shape[0], d_y_r.shape[1]) + (2, 2))

    for idx, x in enumerate(tensor_manifold):
        for idy, y in enumerate(tensor_manifold[idx]):
            tensor_manifold[idx][idy][0][0] = d_x_x_r[idx][idy] + d_x_x_g[idx][idy] + d_x_x_b[idx][idy]
            tensor_manifold[idx][idy][0][1] = d_x_y_r[idx][idy] + d_x_y_g[idx][idy] + d_x_y_b[idx][idy]
            tensor_manifold[idx][idy][1][0] = d_x_y_r[idx][idy] + d_x_y_g[idx][idy] + d_x_y_b[idx][idy]
            tensor_manifold[idx][idy][1][1] = d_y_y_r[idx][idy] + d_y_y_g[idx][idy] + d_y_y_b[idx][idy]

            eigenvalues, eigenvectors = eigh(tensor_manifold[idx][idy])
            if np.any(eigenvalues < 0) & np.all(eigenvalues > 1e-8):
                x = eigenvectors @ np.diag(np.abs(eigenvalues)) @ np.transpose(eigenvectors)
                tensor_manifold[idx][idy] = x
            elif np.any(eigenvalues < 1e-8):
                eigenvalues = np.where(np.abs(eigenvalues) < 1e-8, 0.01, eigenvalues)
                x = eigenvectors @ np.diag(np.abs(eigenvalues)) @ np.transpose(eigenvectors)
                tensor_manifold[idx][idy] = x

    return tensor_manifold


def _construct_tensor_manifold_greyscale(image: np.ndarray):
    d_x, d_y = np.gradient(image)
    d_x_x = np.gradient(d_x)[0]
    d_y_y = np.gradient(d_y)[1]
    d_x_y = np.gradient(d_x)[1]

    tensor_manifold = np.empty((d_x_x.shape[0], d_y_y.shape[1]) + (2, 2))

    for idx, x in enumerate(tensor_manifold):
        for idy, y in enumerate(tensor_manifold[idx]):
            tensor_manifold[idx][idy][0][0] = d_x_x[idx][idy]
            tensor_manifold[idx][idy][0][1] = d_x_y[idx][idy]
            tensor_manifold[idx][idy][1][0] = d_x_y[idx][idy]
            tensor_manifold[idx][idy][1][1] = d_y_y[idx][idy]

            eigenvalues, eigenvectors = eigh(tensor_manifold[idx][idy])
            if np.any(eigenvalues < 0) & np.all(eigenvalues > 1e-8):
                x = eigenvectors @ np.diag(np.abs(eigenvalues)) @ np.transpose(eigenvectors)
                tensor_manifold[idx][idy] = x
            elif np.any(eigenvalues < 1e-8):
                eigenvalues = np.where(np.abs(eigenvalues) < 1e-8, 0.01, eigenvalues)
                x = eigenvectors @ np.diag(np.abs(eigenvalues)) @ np.transpose(eigenvectors)
                tensor_manifold[idx][idy] = x

    return tensor_manifold


def _initialize_lsf(initial_contours_coordinates: [tuple[int, int, int, int]],
                    image: np.ndarray):
    c0 = 2
    print("Shape: " + str(image.shape[:2]))
    initial_lsf = c0 * np.ones(image.shape[:2])

    for contour in initial_contours_coordinates:
        initial_lsf[
            contour[0]:contour[1],
            contour[2]:contour[3]
        ] = -c0

    return initial_lsf


def _update_lsf(phi_ref: np.ndarray,
                g: np.ndarray,
                lmbda: int,
                mu: float,
                alfa: int,
                epsilon: float,
                timestep: int,
                iterations: int,
                potential_function: PotentialFunction):
    phi = phi_ref.copy()
    [dy, dx] = np.gradient(g)
    for k in range(iterations):
        phi = _neumann_bound_cond(phi)
        [phi_y, phi_x] = np.gradient(phi)
        s = np.sqrt(np.square(phi_x) + np.square(phi_y))
        delta = 1e-10
        n_x = phi_x / (s + delta)  # add a small positive number to avoid division by zero
        n_y = phi_y / (s + delta)
        curvature = _div(n_x, n_y)

        if potential_function == PotentialFunction.SINGLE_WELL:
            dist_reg_term = laplace(phi,
                                    mode='nearest') - curvature  # compute distance regularization term in Equation 13 [2] with the single-well potential p1 [2]
        elif potential_function == PotentialFunction.DOUBLE_WELL:
            dist_reg_term = _dist_reg_p2(
                phi)  # compute the distance regularization term in Equation 13 [2] with the double-well potential p2
        else:
            raise Exception('Error: Wrong choice of potential function. Please input the string "single-well" or '
                            '"double-well" in the drlse_edge function.')

        dirac_phi = _dirac(phi, epsilon)
        area_term = dirac_phi * g  # balloon/pressure force [2]
        edge_term = dirac_phi * (dx * n_x + dy * n_y) + dirac_phi * g * curvature
        phi += timestep * (mu * dist_reg_term + lmbda * edge_term + alfa * area_term)

    return phi


def _dist_reg_p2(phi):
    [phi_y, phi_x] = np.gradient(phi)
    s = np.sqrt(np.square(phi_x) + np.square(phi_y))
    a = (s >= 0) & (s <= 1)
    b = (s > 1)
    ps = a * np.sin(2 * np.pi * s) / (2 * np.pi) + b * (
                s - 1)  # compute first order derivative of the double-well potential p2 in Equation 16 in [2]
    dps = ((ps != 0) * ps + (ps == 0)) / ((s != 0) * s + (
                s == 0))  # compute d_p(s)=p'(s)/s in Equation 10 [2]
    return _div(dps * phi_x - phi_x, dps * phi_y - phi_y) + laplace(phi, mode='nearest')


def _div(nx: np.ndarray, ny: np.ndarray):
    [_, nxx] = np.gradient(nx)
    [nyy, _] = np.gradient(ny)
    return nxx + nyy


def _dirac(x: np.ndarray, sigma: np.ndarray):
    f = (1 / 2 / sigma) * (1 + np.cos(np.pi * x / sigma))
    b = (x <= sigma) & (x >= -sigma)
    return f * b


def _neumann_bound_cond(f):
    g = f.copy()
    g[np.ix_([0, -1], [0, -1])] = g[np.ix_([2, -3], [2, -3])]
    g[np.ix_([0, -1]), 1:-1] = g[np.ix_([2, -3]), 1:-1]
    g[1:-1, np.ix_([0, -1])] = g[1:-1, np.ix_([2, -3])]
    return g
