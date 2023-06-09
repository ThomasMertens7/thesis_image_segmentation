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
import matplotlib.pyplot as plt
import numpy as np
from skimage import measure

plt.ion()
fig1 = plt.figure(1)
fig2 = plt.figure(2)


def show_lsf(phi: np.ndarray):
    fig1.clf()
    ax1 = fig1.add_subplot(111, projection='3d')
    y, x = phi.shape
    x = np.arange(0, x, 1)
    y = np.arange(0, y, 1)
    X, Y = np.meshgrid(x, y)
    ax1.view_init(30, -65)
    ax1.plot_surface(X, Y, np.flip(np.flip(phi), 1),
                     rstride=2, cstride=2, color='r',
                     linewidth=0, alpha=0.6, antialiased=True)
    ax1.contour(X, Y, np.flip(np.flip(phi), 1), 0, colors='g', linewidths=2)


def show_contour(phi: np.ndarray, img: np.ndarray):
    fig2.clf()
    contours = measure.find_contours(phi, 0)
    ax2 = fig2.add_subplot(111)
    ax2.imshow(img, interpolation='nearest')
    for n, contour in enumerate(contours):
        ax2.plot(contour[:, 1], contour[:, 0], linewidth=2)


def show_fig2_grey(phi: np.ndarray, img: np.ndarray):
    fig2.clf()
    contours = measure.find_contours(phi, 0)
    ax2 = fig2.add_subplot(111)
    ax2.imshow(img, interpolation='nearest', cmap=plt.get_cmap('gray'))
    for n, contour in enumerate(contours):
        ax2.plot(contour[:, 1], contour[:, 0], linewidth=2)


def show_all(phi: np.ndarray, img: np.ndarray, pause=0.03):
    show_contour(phi, img)
    show_lsf(phi)
    plt.pause(pause)


def draw_all_grey(phi: np.ndarray, img: np.ndarray, pause=0.03):
    show_fig2_grey(phi, img)
    show_lsf(phi)
    plt.pause(pause)
