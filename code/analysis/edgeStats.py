import logging
import warnings

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

from brainNet import BrainNet
from plot_style import plt_style

logger = logging.getLogger("ThrombosisAnalysis.edgeStats")


def calc(brainNet: BrainNet, bins, ax=None):
    w = np.array(list(nx.get_edge_attributes(brainNet.graph, 'avgRadiusAvg').values()))
    y, x = np.histogram(w, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    best_dist = []

    target_dists = ['norm', 'lognorm', 'exponnorm', 'skewnorm', 't', 'logistic', 'truncnorm', 'foldnorm', 'gennorm']

    for ii, dist in enumerate(target_dists):

        logger.info("{:>3} / {:<3}: {}".format(ii + 1, len(target_dists), dist))

        distribution = getattr(stats, dist)

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")

                params = distribution.fit(w)

                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))

                best_dist.append((distribution, params, sse))
        except Exception:
            pass
    return sorted(best_dist, key=lambda x: x[2]), w


def print_pdf(dist, params, data: np.ndarray, sz=10000, output=''):
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    start = data.min()
    end = data.max()

    x = np.linspace(start, end, sz)
    y = dist.pdf(x, loc=loc, scale=scale, *arg)
    pdf = pd.Series(y, x)

    dataSeries = pd.Series(data)
    plt.figure()
    plt.style.use(plt_style)
    dataSeries.plot(kind='hist', bins=50, density=True, alpha=0.5,
                    color=list(matplotlib.rcParams['axes.prop_cycle'])[1]['color'], label='Edge Avg Radius',
                    legend=True)
    pdf.plot(lw=2, label='PDF', legend=True)

    if output:
        plt.savefig(output)
    else:
        plt.show()


def report(dist, params, sse):
    txt = "-- Edge Statistics --\n"
    txt += f"Distribution: {dist.name}\n"
    txt += f"Args: {params[:-2]} | Loc: {params[-2]} | Scale: {params[-1]}\n"
    txt += f"SSE: {sse}\n"

    print(txt)
    with open("log.txt", "a") as f:
        f.write(txt)
        f.close()


if __name__ == "__main__":
    brainNet = BrainNet("synthetic_graph_1")
    dists, w = calc(brainNet, 100)
    bestDist = dists[0]
    report(bestDist[0], bestDist[1], bestDist[2])
    print_pdf(bestDist[0], bestDist[1], w)
