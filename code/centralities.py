from brainNet import BrainNet
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import graph_tool.all as gt



data_set  = "synthetic_graph_1"   

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
                         use_cache=True, out_put_csv=None, weighted=True, k_betweenness=1000):

    # If no methods specified, compute all
    if methods is None:
        methods = centrality_methods[:]


    # Loading graoh from BrainNet
    print("\n Loading Graph.....\n")

    bn = BrainNet(dataset, directed, use_cache)
    bn.get_gt()
    gtG = bn.gtGraph
    N = gtG.num_vertices()

    print(f"Graph contains {N} nodes.\n")

    weight = gtG.ep["avgRadiusAvg"] if weighted else None



    print("\nComputing centralities.....\n")
    results = {}

    # calculate degreee
    # Th
    if "degree" in methods:
        print(f"Computing degree centrality...")
        try:     
            deg = np.array([v.out_degree() + v.in_degree() for v in gtG.vertices()], dtype=np.float32)
            results["degree"] = deg
        except Exception as e:
            print("!! degree centrality failed:", e)

        
    #Betweeness
    start = time.time()

    if "betweenness" in methods:
        print(f"Computing betweenness centrality...")
        try:
            between, _ = gt.betweenness(gtG, weight=weight)

            betw_array = np.array([between[v] for v in gtG.vertices()], dtype=np.float32)
            results["betweenness"] = betw_array

            print(f"Time for betweenness: {time.time() - start:.2f} sec")
        except Exception as e:
            print("!! betweenness failed:", e)

    # closeness
    if "closeness" in methods:
        print(f"Computing closeness centrality...")
        try:
            closeness = gt.closeness(gtG, weight=weight)
            close = np.array([closeness[v] for v in gtG.vertices()], dtype=np.float32)
            results["closeness"] = close

        except Exception as e:
            print("!! closeness failed:", e)

    # eigenvector
    if "eigenvector" in methods:
        print(f"Computing eigenvector centrality (max_iter=2000)...")
        try:
            eigen = gt.eigenvector(gtG, weight=weight)[1]
            eig = np.array([eigen[v] for v in gtG.vertices()], dtype=np.float32)
            results["eigenvector"] = eig
        except Exception as e:
            print("!! eigenvector failed:", e)

    #  Pagerank 
    if "pagerank" in methods:
        print(f"Computing pagerank....")
        try:
            pager = gt.pagerank(gtG, weight=weight)
            pr = np.array([pager[v] for v in gtG.vertices()], dtype=np.float32)
            results["pagerank"] = pr
        except Exception as e:
            print("!! pagerank failed:", e)

    # convert results to DataFrame
    df = pd.DataFrame(results)
    df.index.name = "vertex_index"   

    print("\n Dicribtive Statistics ---\n")
    print(df.describe().T)

    print("\nPearson correlation:")
    print(df.corr(method="pearson"))

    print("\nSpearman correlation:")
    print(df.corr(method="spearman"))


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
    Centralities = compute_centralities(data_set, directed=False, methods=None, use_cache=False,
                              out_put_csv=None, weighted=True)
    print(Centralities)