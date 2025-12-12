import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import alpha

from brainNet import BrainNet


def calc(brainNet: BrainNet):
    deg = dict(brainNet.graph.degree)

    print("Calculating eigenvector centrality...")
    eig = nx.eigenvector_centrality_numpy(brainNet.graph)

    print("Calculating pagerank...")
    pr = nx.pagerank(brainNet.graph)

    print("Calculating betweenness centrality...")
    btw = nx.betweenness_centrality(brainNet.graph)

    print("Calculating closeness centrality...")
    clo = nx.closeness_centrality(brainNet.graph)

    print("Making dataframe...")
    df = pd.DataFrame()
    df["node"] = list(deg.keys())
    df["degree"] = df["node"].map(deg)
    df["eigenvector"] = df["node"].map(eig)
    df["pagerank"] = df["node"].map(pr)
    df["betweenness"] = df["node"].map(btw)
    df["closeness"] = df["node"].map(clo)
    df = df.set_index("node")

    return df

def report(centralities: pd.DataFrame):
    txt = "-- Centralities --\n"

    maxdf = centralities.eq(centralities.max(axis=0), axis=1)
    mindf = centralities.eq(centralities.min(axis=0), axis=1)

    avgDeg = centralities['degree'].mean()
    avgEig = centralities['eigenvector'].mean()
    avgPr = centralities['pagerank'].mean()
    avgBet = centralities['betweenness'].mean()
    avgClo = centralities['closeness'].mean()

    maxDeg = centralities['degree'].max()
    maxEig = centralities['eigenvector'].max()
    maxPr = centralities['pagerank'].max()
    maxBet = centralities['betweenness'].max()
    maxClo = centralities['closeness'].max()

    minDeg = centralities['degree'].min()
    minEig = centralities['eigenvector'].min()
    minPr = centralities['pagerank'].min()
    minBet = centralities['betweenness'].min()
    minClo = centralities['closeness'].min()

    txt += f"Stat      | Avg   | Max   | numMax    | Min   | nimMin\n"
    txt += f"Degree    | {avgDeg}  | {maxDeg}  | {maxdf['degree'].sum()}   | {minDeg}  | {mindf['degree'].sum()}\n"
    txt += f"EigVec    | {avgEig}  | {maxEig}  | {maxdf['eigenvector'].sum()}   | {minEig}  | {mindf['eigenvector'].sum()}\n"
    txt += f"PageRank  | {avgPr}  | {maxPr}  | {maxdf['pagerank'].sum()}   | {minPr}  | {mindf['pagerank'].sum()}\n"
    txt += f"Between   | {avgBet}  | {maxBet}  | {maxdf['betweenness'].sum()}   | {minBet}  | {mindf['betweenness'].sum()}\n"
    txt += f"Closeness | {avgClo}  | {maxClo}  | {maxdf['closeness'].sum()}   | {minClo}  | {mindf['closeness'].sum()}\n"

    print(txt)

    with open("log.txt", "a") as log:
        log.write(txt)
        log.close()

    return

def draw_hist(arr: np.ndarray, output='', xlabel: str = ''):
    plt.hist(arr, bins=30)
    plt.xlabel(xlabel)
    plt.ylabel('Count')
    plt.grid()

    if output:
        plt.savefig(output)
    else:
        plt.show()

def draw_cdf(arr: np.ndarray, output: str = '', xlabel: str = ''):
    x, counts = np.unique(arr, return_counts=True)
    cdf = np.cumsum(counts)

    x = np.insert(x, 0, x[0])

    cdf = cdf / cdf[-1]
    cdf = np.insert(cdf, 0, 0.)

    plt.plot(x, cdf, drawstyle='steps-post')

    plt.xlabel(xlabel)
    plt.ylabel(f'F({xlabel})')
    plt.grid()
    if output:
        plt.savefig(output)
    else:
        plt.show()

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
