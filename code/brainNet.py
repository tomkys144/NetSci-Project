import os.path as op
import os
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import pickle


def _normalize_df_by_key(df: pd.DataFrame, key: str) -> pd.DataFrame:
    minValue = df[key].min()
    maxValue = df[key].max()

    df[key] = (df[key] - minValue) / (maxValue - minValue)

    return df


class BrainNet:
    graph: nx.Graph | nx.DiGraph

    def __init__(self, dataset: str, directed: bool = False, useCache: bool = True):
        root = op.dirname(__file__)
        if op.basename(root) != "code":
            raise Exception("Unexpected file location")
        csvEdgesPath = op.join(root, "datasets", dataset, "edges.csv")
        csvNodesPath = op.join(root, "datasets", dataset, "nodes.csv")

        print("Loadin nodes...")
        nodes = pd.read_csv(
            csvNodesPath,
            header=0,
            delimiter=";",
            index_col="id",
            dtype={
                "pos_x": np.float64,
                "pos_y": np.float64,
                "pos_z": np.float64,
                "degree": np.int32,
                "isAtSampleBorder": np.bool,
            },
        )

        print("Loadin edges...")
        edges = pd.read_csv(
            csvEdgesPath,
            header=0,
            delimiter=";",
            index_col="id",
            usecols=("id", "node1id", "node2id", "avgRadiusAvg"),
            dtype={
                "node1id": np.int32,
                "node2id": np.int32,
                "avgRadiusAvg": np.float64,
            },
        )

        print("Normalizing nodes...")
        nodesKeysToNormalize = ["pos_x", "pos_y", "pos_z"]
        for key in nodesKeysToNormalize:
            nodes = _normalize_df_by_key(nodes, key)

        print("Normalizing edges...")
        edgesKeysToNormalize = ["avgRadiusAvg"]
        for key in edgesKeysToNormalize:
            edges = _normalize_df_by_key(edges, key)

        print("Creating graph...")
        if useCache:
            if directed:
                cachePath = op.join(root, "cache", f"{dataset}_dir.pickle")
            else:
                cachePath = op.join(root, "cache", f"{dataset}_udir.pickle")

            try:
                self.graph = pickle.load(open(cachePath, "rb"))
            except FileNotFoundError:
                if directed:
                    self.graph = nx.from_pandas_edgelist(
                        df=edges,
                        source="node1id",
                        target="node2id",
                        edge_attr="avgRadiusAvg",
                        create_using=nx.DiGraph,
                    )
                else:
                    self.graph = nx.from_pandas_edgelist(
                        df=edges,
                        source="node1id",
                        target="node2id",
                        edge_attr="avgRadiusAvg",
                        create_using=nx.Graph,
                    )
                nx.set_node_attributes(self.graph, nodes.to_dict("index"))
                if not op.exists(op.join(root, "cache")):
                    os.makedirs(op.join(root, "cache"))
                pickle.dump(self.graph, open(cachePath, "wb"))
        else:
            if directed:
                self.graph = nx.from_pandas_edgelist(
                    df=edges,
                    source="node1id",
                    target="node2id",
                    edge_attr="avgRadiusAvg",
                    create_using=nx.DiGraph,
                )
            else:
                self.graph = nx.from_pandas_edgelist(
                    df=edges,
                    source="node1id",
                    target="node2id",
                    edge_attr="avgRadiusAvg",
                    create_using=nx.Graph,
                )
            nx.set_node_attributes(self.graph, nodes.to_dict("index"))

    def visualize(self, outputFile: str = "", show: bool = True):
        fig, ax = plt.subplots()
        ax = fig.add_subplot(projection="3d")

        print("Graphing...")
        progress = 0
        for source, target, attr in self.graph.edges(data=True):
            X = [self.graph.nodes[source]["pos_x"], self.graph.nodes[target]["pos_x"]]
            Y = [self.graph.nodes[source]["pos_y"], self.graph.nodes[target]["pos_y"]]
            Z = [self.graph.nodes[source]["pos_z"], self.graph.nodes[target]["pos_z"]]
            ax.plot(X, Y, Z, linewidth=attr["avgRadiusAvg"] * 5, color="red", alpha=0.4)
            if progress % 5000 == 0:
                print(
                    str(100 * progress / self.graph.number_of_edges()) + "%         \r"
                )
            progress += 1

        print("Done!")

        if outputFile:
            plt.savefig(outputFile)

        if show:
            plt.show()


if __name__ == "__main__":
    brainNet = BrainNet("synthetic_graph_1", directed=False, useCache=False)
    brainNet.visualize()
