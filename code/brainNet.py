import os
import os.path as op
import pickle

import graph_tool.all as gt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def _normalize_df_by_key(df: pd.DataFrame, key: str) -> pd.DataFrame:
    minValue = df[key].min()
    maxValue = df[key].max()

    df[key] = (df[key] - minValue) / (maxValue - minValue)

    return df


def _get_prop_type(value, key=None):
    """
    Performs typing and value conversion for the graph_tool PropertyMap class.
    If a key is provided, it also ensures the key is in a format that can be
    used with the PropertyMap. Returns a tuple, (type name, value, key)
    """
    # Deal with the value
    if isinstance(value, bool):
        tname = 'bool'

    elif isinstance(value, int):
        tname = 'float'
        value = float(value)

    elif isinstance(value, float):
        tname = 'float'

    elif isinstance(value, dict):
        tname = 'object'

    else:
        tname = 'string'
        value = str(value)

    return tname, value, key


class BrainNet:
    graph: nx.Graph | nx.DiGraph
    gtGraph: gt.Graph

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

    def get_gt(self):
        print("Converting NX graph to GT...")
        self.gtGraph = gt.Graph(directed=self.graph.is_directed())

        # 1. Graph Properties
        for key, value in self.graph.graph.items():
            tname, value, key = _get_prop_type(value, key)
            self.gtGraph.graph_properties[key] = self.gtGraph.new_graph_property(tname)
            self.gtGraph.graph_properties[key] = value

        # 2. Register Vertex Properties
        nprops = set()
        for node, data in self.graph.nodes(data=True):
            for key, val in data.items():
                if key in nprops: continue
                tname, _, key = _get_prop_type(val, key)
                self.gtGraph.vertex_properties[key] = self.gtGraph.new_vertex_property(tname)
                nprops.add(key)

        # Add 'id' property explicitly
        self.gtGraph.vertex_properties['id'] = self.gtGraph.new_vertex_property('string')

        # 3. Register Edge Properties
        eprops = set()
        for src, dst, data in self.graph.edges(data=True):
            for key, val in data.items():
                if key in eprops: continue
                tname, _, key = _get_prop_type(val, key)
                self.gtGraph.edge_properties[key] = self.gtGraph.new_edge_property(tname)
                eprops.add(key)

        # 4. Add Vertices and Data
        vertices = {}
        for node, data in self.graph.nodes(data=True):
            v = self.gtGraph.add_vertex()
            vertices[node] = v

            # Manually set ID
            self.gtGraph.vp['id'][v] = str(node)

            for key, value in data.items():
                # Ensure we cast the value exactly as we determined the type
                _, casted_val, _ = _get_prop_type(value, key)
                self.gtGraph.vp[key][v] = casted_val

        # 5. Add Edges and Data
        for src, dst, data in self.graph.edges(data=True):
            e = self.gtGraph.add_edge(vertices[src], vertices[dst])
            for key, value in data.items():
                _, casted_val, _ = _get_prop_type(value, key)
                self.gtGraph.ep[key][e] = casted_val

        # 6. Create Position Vector for Visualization
        if "pos" not in self.gtGraph.vp and "pos_x" in self.gtGraph.vp:
            pos = self.gtGraph.new_vertex_property("vector<double>")

            # .a returns the numpy array view of the property
            px = self.gtGraph.vp["pos_x"].a
            py = self.gtGraph.vp["pos_y"].a
            pz = self.gtGraph.vp["pos_z"].a

            for v in self.gtGraph.vertices():
                idx = int(v)
                pos[v] = [px[idx], py[idx], pz[idx]]

            self.gtGraph.vp["pos"] = pos

    def draw_gt(self, outputFile: str = ""):
        if outputFile:
            gt.graph_draw(self.gtGraph,
                          pos=self.gtGraph.vp["pos"],
                          output_size=(8000, 8000),
                          vertex_size=5,
                          output=outputFile)

        else:
            gt.graph_draw(
                self.gtGraph,
                pos=self.gtGraph.vp["pos"],
                edge_pen_width=gt.prop_to_size(self.gtGraph.ep["avgRadiusAvg"], mi=0, ma=5)
            )


if __name__ == "__main__":
    brainNet = BrainNet("CD1-E_no2", directed=False, useCache=False)
    brainNet.get_gt()
    brainNet.draw_gt(outputFile="graph.png")
    print("done")
