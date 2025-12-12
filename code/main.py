from argparse import ArgumentParser
from datetime import datetime

import brainNet
import comunities
from brainNet import BrainNet
from centralities import compute_centralities
import numpy as np
from aenum import Enum, auto

from analysis import communities, centralities, edgeStats
from brainNet import BrainNet


class Task(Enum):
    ALL = auto()
    LOAD = auto()
    COMMUNITIES = auto()
    CENTRALITIES = auto()
    EDGE = auto()
    DRAW = auto()


def main(tasks, dataset: str = "synthetic_graph_1"):
    if len(tasks) == 0:
        tasks = [Task.ALL]

    with open("log.txt", "a") as log:
        tm = datetime.now()
        log.write(f"\n--------------\nDataset: {dataset}\nTime: {tm.isoformat()}\n--------------\n")

        log.close()

    # Load data
    if (Task.ALL in tasks) or (Task.LOAD in tasks):
        brainNet = load(dataset, True)
    else:
        brainNet = load(dataset)

    if (Task.ALL in tasks) or (Task.DRAW in tasks):
        draw(dataset, brainNet)


    # Compute and visualize centralities
    compute_centralities(graph=brainNet.graph,methods=None,weighted=True)


    # Find communities
    if (Task.ALL in tasks) or (Task.COMMUNITIES in tasks):
        communities_task(dataset, brainNet)

    # Find centralities
    if (Task.ALL in tasks) or (Task.CENTRALITIES in tasks):
        centralities_task(dataset, brainNet)

    if (Task.ALL in tasks) or (Task.EDGE in tasks):
        edges_task(dataset, brainNet)

    print("Done")


def load(dataset: str, load_gt=False):
    brainNet = BrainNet(dataset)

    if load_gt:
        brainNet.get_gt()

    return brainNet


def draw(dataset: str, brainNet: BrainNet):
    print("Printing raw graph...")
    brainNet.draw_gt(f"../results/graph-{dataset}-xy.png", coords=(0, 1))
    brainNet.draw_gt(f"../results/graph-{dataset}-xz.png", coords=(0, 2))
    brainNet.draw_gt(f"../results/graph-{dataset}-yz.png", coords=(1, 2))


def communities_task(dataset: str, brainNet: BrainNet):
    print("Community detection...")
    sbmState = communities.sbm(brainNet, nmcmc=100)

    print("Drawing CD results...")
    communities.draw_sbm(sbmState, f"../results/sbm-{dataset}.png")
    communities.draw_sbm_shape(sbmState, brainNet, f"../results/sbm-{dataset}-xy.png", coords=(0, 1), layer=1)
    communities.draw_sbm_shape(sbmState, brainNet, f"../results/sbm-{dataset}-xz.png", coords=(0, 2), layer=1)
    communities.draw_sbm_shape(sbmState, brainNet, f"../results/sbm-{dataset}-yz.png", coords=(1, 2), layer=1)


def centralities_task(dataset: str, brainNet: BrainNet):
    cent = centralities.calc(brainNet)
    centralities.report(cent)

    centralities.draw_hist(np.array(cent['degree']), xlabel='Degree',
                           output=f'../results/hist-degree-{dataset}.png')
    centralities.draw_hist(np.array(cent['eigenvector']), xlabel='Eigenvector Centrality',
                           output=f'../results/hist-eigenvector-{dataset}.png')
    centralities.draw_hist(np.array(cent['pagerank']), xlabel='PageRank Centrality',
                           output=f'../results/hist-pagerank-{dataset}.png')
    centralities.draw_hist(np.array(cent['betweenness']), xlabel='Betweenness Centrality',
                           output=f'../results/hist-betweenness-{dataset}.png')
    centralities.draw_hist(np.array(cent['closeness']), xlabel='Closeness Centrality',
                           output=f'../results/hist-closeness-{dataset}.png')

    centralities.draw_cdf(np.array(cent['degree']), xlabel='Degree',
                          output=f'../results/cdf-degree-{dataset}.png')
    centralities.draw_cdf(np.array(cent['eigenvector']), xlabel='Eigenvector Centrality',
                          output=f'../results/cdf-eigenvector-{dataset}.png')
    centralities.draw_cdf(np.array(cent['pagerank']), xlabel='PageRank Centrality',
                          output=f'../results/cdf-pagerank-{dataset}.png')
    centralities.draw_cdf(np.array(cent['betweenness']), xlabel='Betweenness Centrality',
                          output=f'../results/cdf-betweenness-{dataset}.png')
    centralities.draw_cdf(np.array(cent['closeness']), xlabel='Closeness Centrality',
                          output=f'../results/cdf-closeness-{dataset}.png')

def edges_task(dataset: str, brainNet: BrainNet):
    dists, w =edgeStats.calc(brainNet, 100)

    bestDist = dists[0]

    edgeStats.report(bestDist[0], bestDist[1], bestDist[2])
    edgeStats.print_pdf(bestDist[0],bestDist[1], w, output=f"../results/pdf-edges-{dataset}.png")

if __name__ == "__main__":
    parser = ArgumentParser(
        prog="ThrombosisAnalysis",
        description="Net Sci Project",
        epilog="If no task is selected, all tasks are run.",
    )

    parser.add_argument("-d", "--dataset", default="synthetic_graph_1", help="Dataset to use")
    parser.add_argument("--load", action="store_true", help="Flag to load graph-tool graph")
    parser.add_argument("--communities", action="store_true", help="Flag to run community detection")
    parser.add_argument("--centralities", action="store_true", help="Flag to run centralities analysis")
    parser.add_argument("--edges", action="store_true", help="Flag to run edge statistics")
    parser.add_argument("--draw", action="store_true", help="Flag to draw raw graph")

    args, unknown = parser.parse_known_args()

    tasks = []
    if args.load:
        tasks.append(Task.LOAD)
    if args.communities:
        tasks.append(Task.COMMUNITIES)
    if args.centralities:
        tasks.append(Task.CENTRALITIES)
    if args.draw:
        tasks.append(Task.DRAW)
    if args.edges:
        tasks.append(Task.EDGES)

    main(tasks=tasks, dataset=args.dataset)
