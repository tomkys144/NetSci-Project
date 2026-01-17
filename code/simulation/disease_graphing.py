import matplotlib.pyplot as plt
from networkx.algorithms.bipartite import color
import numpy as np

from plot_style import plt_style, THEME_CREAM

data = np.zeros((4, 1, 4))

# print(data)
data2 = np.transpose(data, (1, 0, 2))


# print(data2)

def _plot_planes(ax, arrays, step, cmap):
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

        LR_idx, DV_idx = np.mgrid[0:nx, 0:nz]
        X = LR_idx * 1000
        Z = DV_idx * 1000
        Y = nx * 10 * np.ones_like(X) + y_offset
        y_offset += step
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=facecolors, shade=False)


def plot_hypo_time(data, output=''):
    # nx, ny, nz = 70, 100, 50
    # r_square = [((np.mgrid[-1:1:1j * nx, -1:1:1j * ny, -1:1:1j * nz] ** 2).sum(0)) for _ in range(8)]
    # figure_3D_array_slices(r_square, cmap='viridis_r')
    plt.style.use(plt_style)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    data = np.moveaxis(data, [0, 1, 2], [1, 2, 0])

    masked_data = np.ma.masked_where(data == -1, data)
    v_min, v_max = masked_data.min(), masked_data.max()

    result = np.split(data, data.shape[1], axis=1)
    ax.set_box_aspect([11400, 13200 * 2.5, 8000])

    cmap = plt.get_cmap('tug')
    y_offset = 0
    step = 1320

    for p in result:
        nx, ny, nz = p.shape[0], p.shape[1], p.shape[2]
        plane_data = p[:, ny // 2, :]

        norm_data = (plane_data - v_min) / (v_max - v_min)
        facecolors = cmap(norm_data)

        # Make -1 values completely transparent
        facecolors[plane_data == -1, -1] = 0

        LR_idx, DV_idx = np.mgrid[0:nx, 0:nz]
        X = LR_idx * 1000
        Z = DV_idx * 1000
        Y = (ny * 10 * np.ones_like(X)) + y_offset
        y_offset += step

        ax.plot_surface(X, Y, Z, facecolors=facecolors, shade=False, antialiased=True)

    sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=v_min, vmax=v_max), cmap=cmap)
    fig.colorbar(sm, ax=ax, shrink=0.5, aspect=10, pad=0.1)

    ax.set_ylabel(r"Anterior <-> Posterior ($\mu m$)", fontsize=10, labelpad=15)
    ax.set_zlabel(r"Dorsal <-> Ventral ($\mu m$)", fontsize=10, labelpad=15)
    ax.set_xlabel(r"Left <-> Right ($\mu m$)", fontsize=10, labelpad=15)

    ax.set_xticks(np.arange(0, 11400, 3000))
    ax.set_zticks(np.arange(0, 8000, 2000))

    plt.tight_layout()
    plt.savefig(output, facecolor=THEME_CREAM, dpi=300)


def plot_cbf(data1, data2=[], output=''):
    fig = plt.figure()
    plt.style.use(plt_style)

    max_len1 = max(len(arr) for arr in data1)
    data1_pad = np.array([np.pad(arr, (0, max_len1 - len(arr)), constant_values=0) for arr in data1])

    med1 = np.median(data1_pad, axis=0)
    scale1 = med1[0]
    med1 = med1/scale1
    upper1 = np.nanpercentile(data1_pad, 95, axis=0)/scale1
    lower1 = np.nanpercentile(data1_pad, 5, axis=0)/scale1

    iqr1l = np.nanpercentile(data1_pad, 25, axis=0)/scale1
    iqr1u = np.nanpercentile(data1_pad, 75, axis=0)/scale1

    time1 = np.arange(0, len(med1), 1)

    if data2:
        plt.plot(time1, med1, lw=2, color='#e4154b', label='CBF without anastomosis')
    else:
        plt.plot(time1, med1, lw=2, color='#e4154b', label='CBF')

    plt.fill_between(time1, lower1, upper1, color='#e4154b', alpha=0.2, label='90% Range')
    plt.fill_between(time1, iqr1l, iqr1u, color='#e4154b', alpha=0.4, label='IQR')

    if data2:
        max_len2 = max(len(arr) for arr in data2)
        data2_pad = np.array([np.pad(arr, (0, max_len2 - len(arr)), constant_values=0) for arr in data2])

        med2 = np.median(data2_pad, axis=0)
        scale2 = med2[0]
        med2 = med2 / scale2
        upper2 = np.nanpercentile(data2_pad, 95, axis=0) / scale2
        lower2 = np.nanpercentile(data2_pad, 5, axis=0) / scale2

        iqr2l = np.nanpercentile(data2_pad, 25, axis=0) / scale2
        iqr2u = np.nanpercentile(data2_pad, 75, axis=0)/ scale2

        time2 = np.arange(0, len(med2), 1)

        plt.plot(time2, med2, lw=2, color='#9e9e9e', label='CBF with anastomosis')

        plt.fill_between(time2, lower2, upper2, color="#9e9e9e", alpha=0.2, label='90% Range')
        plt.fill_between(time2, iqr2l, iqr2u, color="#9e9e9e", alpha=0.4, label='IQR')

    plt.ylabel('Flow (Normalized)')
    plt.xlabel('Time/Steps')
    plt.grid()
    plt.legend()

    if output:
        plt.savefig(output, facecolor=THEME_CREAM, dpi=300)
    else:
        plt.show()
