from brainNet import BrainNet
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data_set  = "CD1-E_no2"   # can be chanded to thr CD1-E_no2 dataset but it will take hours to compute

# Used the 5 centrality methods (These methods were the ones mentione in the recording of class in Teach Center)

centrality_methods = [
    'degree',
    'betweenness',
    'closeness',
    'eigenvector',
    'pagerank',
]
# Function to compute centralities
def compute_centralities(dataset=None, graph=None, directed=False, methods=None,
                         use_cache=True, out_put_csv=None, weighted=True):

    # If no methods specified, compute all
    if methods is None:
        methods = centrality_methods[:]
    # Loading graoh from BrainNet
    print("\n Loading Graph.....\n")
    if graph is None:
        bn = BrainNet(dataset, directed, use_cache)
        G = bn.graph
    else:
        G = graph


    # Check if all nodes are presnet (chech if graph is loaded correctly)
    node_ids_graph = set(G.nodes())
    print(f"Graph contains {len(node_ids_graph)} nodes.")


    print("\nComputing centralities.....\n")
    results = {}

    edge_weight = 'avgRadiusAvg'
    weight_msg  = "(weighted)"

    # calculate degreee
    # Th
    if "degree" in methods:
        print(f"Computing degree centrality {weight_msg}...")
        try:     
            results['degree'] = dict(G.degree(weight=edge_weight))
        except Exception as e:
            print("!! degree centrality failed:", e)


    # betweenness 
    if "betweenness" in methods:
        print(f"Computing betweenness centrality {weight_msg}...")
        try:
            results['betweenness'] = nx.betweenness_centrality(
                G, weight=edge_weight, normalized=True)
        except Exception as e:
            print("!! betweenness failed:", e)

    # closeness
    if "closeness" in methods:
        print(f"Computing closeness centrality {weight_msg}...")
        try:
            results['closeness'] = nx.closeness_centrality(
                G,
                distance=edge_weight,
                wf_improved=True
            )
        except Exception as e:
            print("!! closeness failed:", e)

    # eigenvector
    # used https://www.geeksforgeeks.org/data-science/eigenvector-centrality-centrality-measure/ as a reference 
    if "eigenvector" in methods:
        print(f"Computing eigenvector centrality {weight_msg} (max_iter=2000)...")
        try:
            results['eigenvector'] = nx.eigenvector_centrality(
                G, max_iter=2000, weight=edge_weight
            )
        except nx.PowerIterationFailedConvergence:
            print("\n!! Eigenvector centrality did not converge.")
            print("   Falling back to largest connected component...\n")

            # If error this means the graph is not connected, so we need to extracr the largest connected component 
            H = G.subgraph(max(nx.connected_components(G), key=len))
            # Compute now only in the largest connected component only 
            ev = nx.eigenvector_centrality(H, max_iter=2000, weight=edge_weight)
            results['eigenvector'] = {n: ev.get(n, 0.0) for n in G.nodes()}

    #  Pagerank 
    if "pagerank" in methods:
        print(f"Computing pagerank {weight_msg}....")
        try:
            results['pagerank'] = nx.pagerank(G, weight=edge_weight)
        except Exception as e:
            print("!! pagerank failed:", e)

    # Create dataframe
    df = pd.DataFrame(results)
    df.index.name = 'node_id'

    print("\n---Summary Statistics ---\n")
    print(df.describe().T)

    print("\nPercentiles:")
    print(df.quantile([0.01,0.05,0.25,0.5,0.75,0.95,0.99]).T)


    # The number of unique values for each centrality (rounded to 6 decimal places)
    print("\nUnique (rounded to 8dp):")
    for col in df.columns:
        uniq_vals = len(set(np.round(df[col].values, 8)))
        print(f"  {col}: {uniq_vals}")
    # The top 10 nodes for each centrality
    print("\n=== Top 10 nodes for each centrality ===")
    for col in df.columns:
        print(f"\nTop 10 by {col}:")
        print(df[col].nlargest(10))


    # Pearson correlation
    try:
        print("\nPearson correlation:")
        print(df.corr(method='pearson'))
    except Exception as e:
        print("pearson correlation failed:", e)


    # Spearman correlation
    try:
        print("\nSpearman correlation:")
        print(df.corr(method='spearman'))
    except Exception as e:
        print("Spearman correlation failed:", e)
    

    # histograms
    eps = 1e-15
    for col in df.columns:
        vals = df[col].values
        if len(vals) == 0:
            continue

        fig, axs = plt.subplots(1, 2, figsize=(10, 4))

        axs[0].hist(vals, bins=50)
        axs[0].set_title(f'{col} (linear)')
        axs[0].set_xlabel('value')

        vals_log = np.log10(np.clip(vals, eps, None))
        axs[1].hist(vals_log, bins=50)
        axs[1].set_title(f'{col} (log10)')
        axs[1].set_xlabel('log10(value)')

        fig.tight_layout()

        plt.show()

    # OUTPUT CSV
    if out_put_csv:
        df.to_csv(out_put_csv)
        print(f"\nSaved centrality table to: {out_put_csv}")

    return df


if __name__ == "__main__":
    df = compute_centralities(data_set, directed=False, methods=None, use_cache=False,
                              out_put_csv=None, weighted=True)
    print(df)