from PIL import Image

import matplotlib.pyplot as plt
import numpy as np

data = np.zeros((4,1,4))

#print(data)
data2 = np.transpose(data, (1,0,2))
#print(data2)

def plot_planes(ax, arrays, step, cmap):
    """For a given list of 3d *array* plot a plane with *fixed_coord*"""
    y_offset = 0
    for p in arrays:
        nx = p.shape[0]
        ny = p.shape[1]
        nz = p.shape[2]

        plane_data = p[(slice(None), ny // 2, slice(None))]

        min_val = p.min()
        max_val = p.max()

        cmap = plt.get_cmap(cmap)

        facecolors = cmap((plane_data - min_val) / (max_val - min_val))
        X, Z = np.mgrid[0:nx, 0:nz]
        Y = ny * 10 * np.ones_like(X) + y_offset
        y_offset += step
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=facecolors, shade=False)





def figure_3D_array_slices(arrays, cmap=None):
    """Plot a 3d array using a list of planes of the same shape"""
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.set_box_aspect([arrays[0].shape[0], 50,arrays[0].shape[2]])
    p = plot_planes(ax, arrays, 100, cmap=cmap)

    fig.colorbar(p, ax=ax)

    return fig, ax

def plot_hypo_time(data):
    # nx, ny, nz = 70, 100, 50
    # r_square = [((np.mgrid[-1:1:1j * nx, -1:1:1j * ny, -1:1:1j * nz] ** 2).sum(0)) for _ in range(8)]
    # figure_3D_array_slices(r_square, cmap='viridis_r')

    result = np.split(data, data.shape[1], axis=1)
    figure_3D_array_slices(result, cmap='viridis_r')

    plt.grid(False)
    plt.axis('off')
    plt.savefig("hypo_plot")