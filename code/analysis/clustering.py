import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from brainNet import BrainNet
import logging


logger = logging.getLogger("ThrombosisAnalysis.clustering")


def compute_clustering(brainNet=None, dataset: str = "synthetic_graph_1"):
    if brainNet is None:
        brainNet = BrainNet(dataset)
    G = brainNet.graph

    txt = "-- Clustering --\n"

    logger.info("Computing local clustering coefficients...")
    local_clust = nx.clustering(G)

    logger.info("Computing global clustering coefficient...")
    global_clust = nx.transitivity(G)

    txt += f"Global clustering coefficient: {global_clust:.4f}\n"
    logger.info(f"Number of Nodes (N): {G.number_of_nodes()}")
    logger.info(f"Number of Edges (E): {G.number_of_edges()}")
    # denstity formula from https://www.sciencedirect.com/topics/computer-science/network-density#:~:text=Hence%2C%20we%20could%20say%20that,edges%20in%20the%20network%2C%20respectively.
    density = (2 * G.number_of_edges()) / (G.number_of_nodes() * (G.number_of_nodes() - 1))
    txt += f"Network Density (D): {density:.6f}\n"

    # Generate random graph with same degree sequence

    # First created a multi graph and then removed self-loops
    # So that the degree sequence is preserved but the structure is random
    # Got this idea after reading this: https://networksciencebook.com/chapter/3#degree-distribution
    #print("Generating random configuration-model graph....")
    #degree_sequence = [d for _, d in G.degree()]

    #RG = nx.configuration_model(degree_sequence)
    #RG = nx.Graph(RG)  
    #RG.remove_edges_from(nx.selfloop_edges(RG))

    #print(f"Random graph nodes: {RG.number_of_nodes()}, edges: {RG.number_of_edges()}") # to check if it is same as the original
    #Not a good idea as it leads to disconnected graphs, zero clustering coefficient

    logger.info("Generating Erdős-Rényi random graph....")

    n = G.number_of_nodes()
    #m = G.number_of_edges()

    #RG = nx.gnm_random_graph(n, m) doesnt account for the dencity

    RG = nx.erdos_renyi_graph(n, density) # Erdős-Rényi random graph with desicty as probability, but does not have same number of edges
    logger.info(f"Random graph nodes: {RG.number_of_nodes()}, edges: {RG.number_of_edges()}") # to check if it is same as the original

    random_global_clust = nx.transitivity(RG)

    txt += f"Original graph global clustering: {global_clust}\n"

    if random_global_clust == 0:
        logger.info("Random graph has zero global clustering coefficient, cannot compute clustering ratio.")
        random_global_clut_appox= nx.average_clustering(RG)
        txt += f" Aproxximate Random graph global clustering: {random_global_clut_appox}\n"
        ratio_aporxx = global_clust / random_global_clut_appox
        txt += f"Clustering ratio (original / random): {ratio_aporxx:.2f}\n"  # If higher than 1 original is more clustered than random
    else:
        ratio = global_clust / random_global_clust
        txt += f"Random graph global clustering: {random_global_clust}\n"
        txt += f"Clustering ratio (original / random): {ratio:.2f}\n"  # If higher than 1 original is more clustered than random


    print(txt)
    with open('log.txt', 'a') as log:
        log.write(txt)
        log.close()

    return local_clust, global_clust

def plot(local_clust, global_clust):
    values = list(local_clust.values())
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    
    axs[0].hist(values, bins=20, color='skyblue', edgecolor='black')
    axs[0].set_title("Local Clustering Coefficient Distribution (Linear Scale)")
    axs[0].set_xlabel("Clustering Coefficient") 
    axs[0].set_ylabel("Number of Nodes")   
    axs[0].grid(True, linestyle='--', alpha=0.5)

    

    axs[1].hist(values, bins=20, color='lightcoral', edgecolor='black', log=True)
    axs[1].set_title("Local Clustering Coefficient Distribution (Log Scale)")
    axs[1].set_xlabel("Clustering Coefficient")  
    axs[1].set_ylabel("Number of Nodes (log scale)") 
    axs[1].grid(True, linestyle='--', alpha=0.5)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    brainNet = BrainNet("synthetic_graph_1")
    compute_clustering(brainNet=brainNet)


