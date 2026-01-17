import logging
from argparse import ArgumentParser
from datetime import datetime

import numpy as np
from aenum import Enum, auto
from klepto.archives import dir_archive

from analysis import communities, centralities, edgeStats, clustering
from brainNet import BrainNet
from simulation import disease, disease_graphing


class Task(Enum):
    ALL = auto()
    LOAD = auto()
    COMMUNITIES = auto()
    CENTRALITIES = auto()
    EDGE = auto()
    CLUSTERING = auto()
    SIMULATION = auto()


def main(tasks, dataset: str = "synthetic_graph_1", imgs: bool = True, cache: bool = True):
    if len(tasks) == 0:
        tasks = [Task.ALL]

    with open("log.txt", "a") as log:
        tm = datetime.now()
        log.write(f"\n--------------\nDataset: {dataset}\nTime: {tm.isoformat()}\n--------------\n")
        logger.info(f"Dataset: {dataset}")

        log.close()

    if cache:
        db = dir_archive('cache/' + dataset, {}, cached=False, compression=5, protocol=-1)
        db.load()
    else:
        db = None

    # Load data
    if (Task.ALL in tasks) or (Task.LOAD in tasks):
        brainNet = load(dataset, True, cache=db)
    else:
        brainNet = load(dataset, cache=db)

    if imgs:
        draw(dataset, brainNet)

    # Find communities
    if (Task.ALL in tasks) or (Task.COMMUNITIES in tasks):
        communities_task(dataset, brainNet, imgs=imgs, cache=db)

    # Find centralities
    if (Task.ALL in tasks) or (Task.CENTRALITIES in tasks):
        centralities_task(dataset, brainNet, imgs=imgs, cache=db)

    # Find statistics on edges
    if (Task.ALL in tasks) or (Task.EDGE in tasks):
        edges_task(dataset, brainNet, imgs=imgs, cache=db)

    # Calculate clusters
    if (Task.ALL in tasks) or (Task.CLUSTERING in tasks):
        clustering_task(dataset, brainNet, imgs=imgs, cache=db)

    # Run simulation
    if (Task.ALL in tasks) or (Task.SIMULATION in tasks):
        simulation_task(dataset, brainNet, imgs=imgs, cache=db)

    if db:
        db.dump()

    logging.info("Done")


def load(dataset: str, load_gt=False, cache=None):
    if cache and 'brainNet' in cache.keys():
        brainNet = cache['brainNet']
    else:
        brainNet = BrainNet(dataset, v_norm=[], e_norm=[])

    if load_gt:
        brainNet.get_gt()

    if cache:
        cache['brainNet'] = brainNet

    return brainNet


def draw(dataset: str, brainNet: BrainNet):
    logger.info("Printing raw graph...")
    brainNet.draw_gt(f"results/graph-{dataset}-xy.pdf", coords=(0, 1))
    brainNet.draw_gt(f"results/graph-{dataset}-xz.pdf", coords=(0, 2))
    brainNet.draw_gt(f"results/graph-{dataset}-yz.pdf", coords=(1, 2))


def communities_task(dataset: str, brainNet: BrainNet, imgs: bool = True, cache=None):
    logger.info("Community detection...")
    if cache and 'sbmState' in cache.keys():
        sbmState = cache['sbmState']
    else:
        sbmState = communities.sbm(brainNet, nmcmc=1000)

        if cache:
            cache['sbmState'] = sbmState

    if imgs:
        logger.info("Drawing CD results...")

        communities.draw_sbm(sbmState, f"results/sbm-{dataset}.pdf")
        communities.draw_sbm_shape(sbmState, brainNet, f"results/sbm-{dataset}-xy.pdf", coords=(0, 1), layer=0)
        communities.draw_sbm_shape(sbmState, brainNet, f"results/sbm-{dataset}-xz.pdf", coords=(0, 2), layer=0)
        communities.draw_sbm_shape(sbmState, brainNet, f"results/sbm-{dataset}-yz.pdf", coords=(1, 2), layer=0)

        communities.draw_sbm_shape(sbmState, brainNet, f"results/sbm-{dataset}-1-xy.pdf", coords=(0, 1), layer=1)
        communities.draw_sbm_shape(sbmState, brainNet, f"results/sbm-{dataset}-1-xz.pdf", coords=(0, 2), layer=1)
        communities.draw_sbm_shape(sbmState, brainNet, f"results/sbm-{dataset}-1-yz.pdf", coords=(1, 2), layer=1)

        communities.draw_sbm_shape(sbmState, brainNet, f"results/sbm-{dataset}-4-xy.pdf", coords=(0, 1), layer=4)
        communities.draw_sbm_shape(sbmState, brainNet, f"results/sbm-{dataset}-4-xz.pdf", coords=(0, 2), layer=4)
        communities.draw_sbm_shape(sbmState, brainNet, f"results/sbm-{dataset}-4-yz.pdf", coords=(1, 2), layer=4)

        communities.draw_sbm_shape(sbmState, brainNet, f"results/sbm-{dataset}-5-xy.pdf", coords=(0, 1), layer=5)
        communities.draw_sbm_shape(sbmState, brainNet, f"results/sbm-{dataset}-5-xz.pdf", coords=(0, 2), layer=5)
        communities.draw_sbm_shape(sbmState, brainNet, f"results/sbm-{dataset}-5-yz.pdf", coords=(1, 2), layer=5)


def centralities_task(dataset: str, brainNet: BrainNet, imgs: bool = True, cache=None):
    if cache and 'centralities' in cache.keys():
        cent = cache['centralities']
    else:
        cent = centralities.calc(brainNet)

        if cache:
            cache['centralities'] = cent

    centralities.report(cent)

    if imgs:
        centralities.draw_hist(np.array(cent['degree']), xlabel='Degree',
                               output=f'results/hist-degree-{dataset}.pdf')
        centralities.draw_hist(np.array(cent['eigenvector']), xlabel='Eigenvector Centrality',
                               output=f'results/hist-eigenvector-{dataset}.pdf')
        centralities.draw_hist(np.array(cent['pagerank']), xlabel='PageRank Centrality',
                               output=f'results/hist-pagerank-{dataset}.pdf')
        # centralities.draw_hist(np.array(cent['betweenness']), xlabel='Betweenness Centrality',
        #                        output=f'../results/hist-betweenness-{dataset}.pdf')
        # centralities.draw_hist(np.array(cent['closeness']), xlabel='Closeness Centrality',
        #                        output=f'../results/hist-closeness-{dataset}.pdf')

        centralities.draw_cdf(np.array(cent['degree']), xlabel='Degree',
                              output=f'results/cdf-degree-{dataset}.pdf')
        centralities.draw_cdf(np.array(cent['eigenvector']), xlabel='Eigenvector Centrality',
                              output=f'results/cdf-eigenvector-{dataset}.pdf')
        centralities.draw_cdf(np.array(cent['pagerank']), xlabel='PageRank Centrality',
                              output=f'results/cdf-pagerank-{dataset}.pdf')
        # centralities.draw_cdf(np.array(cent['betweenness']), xlabel='Betweenness Centrality',
        #                       output=f'../results/cdf-betweenness-{dataset}.pdf')
        # centralities.draw_cdf(np.array(cent['closeness']), xlabel='Closeness Centrality',
        #                       output=f'../results/cdf-closeness-{dataset}.pdf')


def edges_task(dataset: str, brainNet: BrainNet, imgs: bool = True, cache=None):
    if cache and 'edgesDist' in cache.keys():
        bestDist = cache['edgesDist']
        w = cache['edgesDistW']
    else:
        dists, w = edgeStats.calc(brainNet, 100)
        bestDist = dists[0]

        if cache:
            cache['edgesDist'] = bestDist
            cache['edgesDistW'] = w

    edgeStats.report(bestDist[0], bestDist[1], bestDist[2])

    if imgs:
        edgeStats.print_pdf(bestDist[0], bestDist[1], w, output=f"results/pdf-edges-{dataset}.pdf")


def clustering_task(dataset: str, brainNet: BrainNet, imgs: bool = True, cache=None):
    if cache and 'clustering' in cache.keys():
        local_clust = cache['clustering']
    else:
        local_clust = clustering.compute_clustering(brainNet)

        if cache:
            cache['clustering'] = local_clust

    if imgs:
        clustering.plot(local_clust, f"results/clustering-local-{dataset}.pdf")
        clustering.plot(local_clust, f"results/clustering-local-{dataset}-log.pdf", log=True)


def simulation_task(dataset: str, brainNet: BrainNet, imgs: bool = True, cache=None, runs=2):
    if cache and 'simulation' in cache.keys() and 'simulation-a' in cache.keys():
        stats = cache['simulation']
        stats_a = cache['simulation-a']
    else:
        stats = []
        for run in range(runs):
            logger.info(f"Running iteration {run} of simulation")
            stats.append(
                disease.disease_simulation(brainNet, maxIter=1e9, random_selection=False, step_len=20, hypo_thr=0.4)
            )
        stats_a = []
        for run in range(runs):
            logger.info(f"Running iteration {run} of simulation with anastomosis")
            stats_a.append(
                disease.disease_simulation(brainNet, maxIter=1e9, random_selection=False, step_len=20, hypo_thr=0.4, anastomosis_thr=0.6)
            )

        if cache:
            cache['simulation'] = stats
            cache['simulation-a'] = stats_a

    if imgs:
        cbf = []
        hypo = []
        cbf_a = []
        hypo_a = []
        for run in stats:
            cbf.append(run['CBF'])
            hypo.append(run['hypo_time'])
        for run in stats_a:
            cbf_a.append(run['CBF'])
            hypo_a.append(run['hypo_time'])

        hypostack = np.stack(hypo, axis=0)
        hypo_med = np.median(hypostack, axis=0)

        hypostack_a = np.stack(hypo_a, axis=0)
        hypo_med_a = np.median(hypostack_a, axis=0)

        disease_graphing.plot_cbf(cbf, output=f"results/cbf-{dataset}.pdf")
        disease_graphing.plot_cbf(cbf, cbf_a, output=f"results/cbf-a-{dataset}.pdf")

        disease_graphing.plot_hypo_time(hypo_med, output=f"results/hypo-time-{dataset}.pdf")
        disease_graphing.plot_hypo_time(hypo_med_a, output=f"results/hypo-time-a-{dataset}.pdf")




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
    parser.add_argument("--clustering", action="store_true", help="Flag to run clustering analysis")
    parser.add_argument("--sim", action="store_true", help="Flag to run simulation")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose mode")
    parser.add_argument("-l", "--log", help="Log file (default is stdout)")
    parser.add_argument("-c", "--cache", action="store_true", help="Cache file")
    parser.add_argument("-i", "--image", action="store_true", help="Draw images")

    args, unknown = parser.parse_known_args()
    logger = logging.getLogger('ThrombosisAnalysis')
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    if args.log:
        fh = logging.FileHandler(args.log)
        if args.verbose:
            fh.setLevel(logging.INFO)
        else:
            fh.setLevel(logging.WARNING)

        ch = logging.StreamHandler()
        ch.setLevel(logging.ERROR)

        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)
    else:
        ch = logging.StreamHandler()
        if args.verbose:
            ch.setLevel(logging.INFO)
        else:
            ch.setLevel(logging.WARNING)

        ch.setFormatter(formatter)

        logger.addHandler(ch)

    tasks = []
    if args.load:
        tasks.append(Task.LOAD)
    if args.communities:
        tasks.append(Task.COMMUNITIES)
    if args.centralities:
        tasks.append(Task.CENTRALITIES)
    if args.edges:
        tasks.append(Task.EDGE)
    if args.clustering:
        tasks.append(Task.CLUSTERING)
    if args.sim:
        tasks.append(Task.SIMULATION)

    main(tasks=tasks, dataset=args.dataset, imgs=args.image, cache=args.cache)
