import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import logging

from brainNet import BrainNet

logger = logging.getLogger("ThrombosisAnalysis.centralities")

_centrality_methods = [
    'degree',
    'eigenvector',
    'pagerank',
]

_unused_methods = [
    'betweenness',
    'closeness',
]


def calc(brainNet: BrainNet, methods=None):
    if methods is None:
        methods = _centrality_methods[:]

    results = {}

    edge_weight = 'avgRadiusAvg'

    if "degree" in methods:
        logger.info(f"Computing degree centrality...")
        try:
            results['degree'] = dict(brainNet.graph.degree(weight=edge_weight))
        except Exception as e:
            logger.error("!! degree centrality failed:", e)

    if "betweenness" in methods:
        logger.info(f"Computing betweenness centrality...")
        try:
            results['betweenness'] = nx.betweenness_centrality(
                brainNet.graph, weight=edge_weight, normalized=True, k=1000)
        except Exception as e:
            logger.error("!! betweenness failed:", e)

    if "closeness" in methods:
        logger.info(f"Computing closeness centrality...")
        try:
            results['closeness'] = nx.closeness_centrality(
                brainNet.graph,
                distance=edge_weight,
                wf_improved=True
            )
        except Exception as e:
            logger.error("!! closeness failed:", e)

    if "eigenvector" in methods:
        logger.info(f"Computing eigenvector centrality (max_iter=2000)...")
        try:
            results['eigenvector'] = nx.eigenvector_centrality(
                brainNet.graph, max_iter=2000, weight=edge_weight
            )
        except nx.PowerIterationFailedConvergence:
            logger.warning("\n!! Eigenvector centrality did not converge.")
            logger.warning("   Falling back to largest connected component...\n")

            # If error this means the graph is not connected, so we need to extracr the largest connected component
            H = brainNet.graph.subgraph(max(nx.connected_components(brainNet.graph), key=len))
            # Compute now only in the largest connected component only
            ev = nx.eigenvector_centrality(H, max_iter=2000, weight=edge_weight)
            results['eigenvector'] = {n: ev.get(n, 0.0) for n in brainNet.graph.nodes()}

    if "pagerank" in methods:
        logger.info(f"Computing pagerank....")
        try:
            results['pagerank'] = nx.pagerank(brainNet.graph, weight=edge_weight)
        except Exception as e:
            logger.error("!! pagerank failed:", e)

    logger.info("Making dataframe...")
    df = pd.DataFrame(results)
    df.index.name = 'node_id'

    return df


def report(df: pd.DataFrame):
    txt = "-- Centralities --\n"

    desc = df.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    txt += f"Summary:\n{desc.to_string()}\n"

    txt += "\nUnique (rounded to 8dp):"
    for col in df.columns:
        txt += f"\nTop 10 by {col}\n"
        txt += df[col].nlargest(10).to_string()

    try:
        txt += "\nPearson correlation:\n"
        txt += df.corr(method='pearson').to_string()
    except Exception as e:
        logger.error("pearson correlation failed:", e)

    try:
        txt += "\nSpearman correlation:\n"
        txt += df.corr(method='spearman').to_string()
    except Exception as e:
        logger.error("Spearman correlation failed:", e)

    txt += "\n"
    print(txt)
    with open("log.txt", "a") as log:
        log.write(txt)
        log.close()

    return


def draw_hist(arr: np.ndarray, output='', xlabel: str = ''):
    plt.figure()
    plt.hist(arr, bins=30)
    plt.xlabel(xlabel)
    plt.ylabel('Count')
    plt.grid()

    if output:
        plt.savefig(output)
    else:
        plt.show()

    plt.close()


def draw_cdf(arr: np.ndarray, output: str = '', xlabel: str = ''):
    x, counts = np.unique(arr, return_counts=True)
    cdf = np.cumsum(counts)

    x = np.insert(x, 0, x[0])

    cdf = cdf / cdf[-1]
    cdf = np.insert(cdf, 0, 0.)

    plt.figure()
    plt.plot(x, cdf, drawstyle='steps-post')

    plt.xlabel(xlabel)
    plt.ylabel(f'F({xlabel})')
    plt.grid()
    if output:
        plt.savefig(output)
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":
    brainNet = BrainNet("synthetic_graph_1")
    centralities = calc(brainNet)

    report(centralities)

    draw_hist(np.array(centralities['degree']), xlabel='Degree')
    draw_hist(np.array(centralities['eigenvector']), xlabel='Eigenvector Centrality')
    draw_hist(np.array(centralities['pagerank']), xlabel='PageRank Centrality')
    draw_hist(np.array(centralities['betweenness']), xlabel='Betweenness Centrality')
    draw_hist(np.array(centralities['closeness']), xlabel='Closeness Centrality')

    draw_cdf(np.array(centralities['degree']), xlabel='Degree')
    draw_cdf(np.array(centralities['eigenvector']), xlabel='Eigenvector Centrality')
    draw_cdf(np.array(centralities['pagerank']), xlabel='PageRank Centrality')
    draw_cdf(np.array(centralities['betweenness']), xlabel='Betweenness Centrality')
    draw_cdf(np.array(centralities['closeness']), xlabel='Closeness Centrality')
