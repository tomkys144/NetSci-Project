import logging
import os.path as op

import graph_tool.all as gt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

logger = logging.getLogger("ThrombosisAnalysis.BrainNet")

def _fit_to_atlas(df: pd.DataFrame) -> pd.DataFrame:
    logger.info(f"Transforming coordinates to Allen Mouse Brain Atlas space...")

    atlas_shape = (13200, 800, 11400) #size in microns
    cube_size = 800 #size in microns of subsections

    axis_map = {
        "pos_y": 0,  # AP
        "pos_z": 1,  # DV
        "pos_x": 2  # ML
    }

    new_axis = {}

    for col, axis_idx in axis_map.items():
        src_min = df[col].min()
        src_max = df[col].max()

        tgt_max = atlas_shape[axis_idx]

        scale = (tgt_max - 1) / (src_max - src_min)
        new_axis[axis_idx] = (df[col] - src_min) * scale

    df['pos_x'] = new_axis[0]
    df['pos_y'] = new_axis[1]
    df['pos_z'] = new_axis[2]
    df['cube_x'] = df['pos_x']//cube_size
    df['cube_y'] = df['pos_y']//cube_size
    df['cube_z'] = df['pos_z']//cube_size

    return df

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

    elif isinstance(value, (int, np.integer)):
        tname = 'float'
        value = float(value)

    elif isinstance(value, (float, np.floating)):
        tname = 'float'

    elif isinstance(value, dict):
        tname = 'object'

    else:
        tname = 'string'
        value = str(value)

    return tname, value, key


class BrainNet:
    graph: nx.Graph
    gtGraph: gt.Graph
    diGraph: nx.DiGraph
    gtDiGraph: gt.Graph

    def __init__(self, dataset: str, v_norm=[],
                 e_norm=[]):
        root = op.dirname(__file__)
        if op.basename(root) != "code":
            raise Exception("Unexpected file location")
        csvEdgesPath = op.join(root, "datasets", dataset, "edges.csv")
        csvNodesPath = op.join(root, "datasets", dataset, "nodes.csv")

        logger.info("Loadin nodes...")
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

        logger.info("Loadin edges...")
        edges = pd.read_csv(
            csvEdgesPath,
            header=0,
            delimiter=";",
            index_col="id",
            usecols=("id", "node1id", "node2id", "avgRadiusAvg", "length"),
            dtype={
                "node1id": np.int32,
                "node2id": np.int32,
                "avgRadiusAvg": np.float32,
                "length": np.float32
            },
        )

        edges['capacity'] = (edges['avgRadiusAvg'].astype(np.float64) ** 4) / edges['length'].astype(np.float64)
        edges.loc[edges["length"] <= 0, "capacity"] = 0.0

        logger.info("Normalizing nodes...")
        for key in v_norm:
            nodes = _normalize_df_by_key(nodes, key)

        nodes = _fit_to_atlas(df=nodes)

        logger.info("Normalizing edges...")
        for key in e_norm:
            edges = _normalize_df_by_key(edges, key)

        logger.info("Creating graph...")

        self.graph = nx.from_pandas_edgelist(
            df=edges,
            source="node1id",
            target="node2id",
            edge_attr=["avgRadiusAvg", "length", "capacity"],
            create_using=nx.Graph,
        )
        nx.set_node_attributes(self.graph, nodes.to_dict("index"))

    def visualize(self, outputFile: str = "", show: bool = True):
        fig, ax = plt.subplots()
        ax = fig.add_subplot(projection="3d")

        logger.info("Graphing...")
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

        if outputFile:
            plt.savefig(outputFile)

        if show:
            plt.show()

    def get_gt(self):
        logger.info("Converting NX graph to GT...")
        self.gtGraph = gt.Graph(directed=self.graph.is_directed())

        reserved_keys = {"pos"}

        # --- 1. Register Vertex Properties ---
        nprops = set()
        # Scan nodes to register properties
        for node, data in self.graph.nodes(data=True):
            for key, val in data.items():
                if key in nprops or key in reserved_keys:
                    continue
                tname, _, _ = _get_prop_type(val, key)
                self.gtGraph.vertex_properties[key] = self.gtGraph.new_vertex_property(tname)
                nprops.add(key)

        # Store ID explicitly
        self.gtGraph.vertex_properties["id"] = self.gtGraph.new_vertex_property("string")

        # --- 2. Register Edge Properties ---
        eprops = set()
        # Scan edges to register properties (like avgRadiusAvg)
        if self.graph.number_of_edges() > 0:
            # We look at the first edge to infer types (assuming homogeneous attributes)
            # If your graph has sparse attributes, iterate over all edges instead of breaking.
            for u, v, data in self.graph.edges(data=True):
                for key, val in data.items():
                    if key in eprops:
                        continue
                    tname, _, _ = _get_prop_type(val, key)
                    self.gtGraph.edge_properties[key] = self.gtGraph.new_edge_property(tname)
                    eprops.add(key)
                break

                # --- 3. Add Vertices ---
        vertices = {}  # Mapping from NX ID to GT Vertex object
        for node, data in self.graph.nodes(data=True):
            v = self.gtGraph.add_vertex()
            vertices[node] = v
            self.gtGraph.vp["id"][v] = str(node)

            for key, value in data.items():
                if key in reserved_keys:
                    continue
                # Safely cast value
                _, casted_val, _ = _get_prop_type(value, key)
                # Assign to property map
                self.gtGraph.vp[key][v] = casted_val

        # --- 4. Add Edges ---
        for u, v, data in self.graph.edges(data=True):
            if u in vertices and v in vertices:
                e = self.gtGraph.add_edge(vertices[u], vertices[v])

                # Copy edge attributes
                for key, value in data.items():
                    if key in eprops:
                        _, casted_val, _ = _get_prop_type(value, key)
                        self.gtGraph.ep[key][e] = casted_val

        # --- 5. Build Position Property ---
        pos = self.gtGraph.new_vertex_property("vector<double>")
        for v in self.gtGraph.vertices():
            pos[v] = [
                float(self.gtGraph.vp["pos_x"][v]),
                float(self.gtGraph.vp["pos_y"][v]),
                float(self.gtGraph.vp["pos_z"][v]),
            ]
        self.gtGraph.vp["pos"] = pos

    def draw_gt(self, outputFile: str = "", coords=(0, 1)):
        pos3d = self.gtGraph.vp.pos.get_2d_array(pos=[0, 1, 2])
        pos2d = pos3d[list(coords), :]

        pos = self.gtGraph.new_vertex_property("vector<double>")
        pos.set_2d_array(pos2d)

        if outputFile:
            gt.graph_draw(self.gtGraph,
                          pos=pos,
                          vertex_size=5,
                          edge_pen_width=gt.prop_to_size(self.gtGraph.ep["avgRadiusAvg"], mi=0, ma=5),
                          output=outputFile)

        else:
            gt.graph_draw(
                self.gtGraph,
                pos=pos,
                edge_pen_width=gt.prop_to_size(self.gtGraph.ep["avgRadiusAvg"], mi=0, ma=5)
            )

    def generate_digraph(self, flow=None):
        self.diGraph = self.graph.to_directed()

        if hasattr(self, 'gtGraph') and self.gtGraph:
            g_undirected = self.gtGraph
            g_directed = gt.Graph(directed=True)
            g_directed.add_vertex(g_undirected.num_vertices())

            for key, prop in g_undirected.vertex_properties.items():
                g_directed.vertex_properties[key] = g_directed.new_vertex_property(prop.value_type())

                val_type = prop.value_type()

                if prop.a is not None:
                    g_directed.vp[key].a = prop.a.copy()
                elif val_type.startswith("vector"):
                    if key == "pos":
                        pos_data = prop.get_2d_array(pos=[0, 1, 2])
                        g_directed.vp[key].set_2d_array(pos_data)
                    else:
                        # Fallback for other vectors (slower but safe)
                        for i in range(g_undirected.num_vertices()):
                            g_directed.vp[key][g_directed.vertex(i)] = prop[g_undirected.vertex(i)]
                else:
                    for i in range(g_undirected.num_vertices()):
                        # Get vertex handle by index
                        v_src = g_undirected.vertex(i)
                        v_dst = g_directed.vertex(i)
                        # Copy value
                        g_directed.vp[key][v_dst] = prop[v_src]

            edge_data = g_undirected.get_edges([
                g_undirected.ep["avgRadiusAvg"],
                g_undirected.ep["capacity"],
                g_undirected.ep["length"]
            ])

            u = edge_data[:, 0].astype(np.int32)
            v = edge_data[:, 1].astype(np.int32)
            radii = edge_data[:, 2]
            caps = edge_data[:, 3]
            lens = edge_data[:, 4]

            sources_bi = np.concatenate((u, v))
            targets_bi = np.concatenate((v, u))
            radii_bi = np.concatenate((radii, radii))
            caps_bi = np.concatenate((caps, caps))
            lens_bi = np.concatenate((lens, lens))

            g_directed.add_edge_list(np.transpose([sources_bi, targets_bi]))

            g_directed.ep["avgRadiusAvg"] = g_directed.new_edge_property("float")
            g_directed.ep["capacity"] = g_directed.new_edge_property("double")  # double for precision
            g_directed.ep["length"] = g_directed.new_edge_property("float")

            g_directed.ep["avgRadiusAvg"].get_array()[:] = radii_bi
            g_directed.ep["capacity"].get_array()[:] = caps_bi
            g_directed.ep["length"].get_array()[:] = lens_bi

            self.gtDiGraph = g_directed
            logger.info(f"Generated DiGraph with {g_directed.num_edges()} edges")


if __name__ == "__main__":
    brainNet = BrainNet("CD1_E_no2")
    brainNet.get_gt()
    brainNet.generate_digraph()
    print("done")
