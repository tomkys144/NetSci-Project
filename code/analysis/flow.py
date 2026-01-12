import logging

import networkx as nx
import numpy as np
import pandas as pd
import pyamg
from klepto.archives import dir_archive
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans

from brainNet import BrainNet

logger = logging.getLogger("ThrombosisAnalysis.flow")


def _get_largest_component(G):
    if not G:
        return G
    largest_cc_nodes = max(nx.weakly_connected_components(G), key=len)
    return G.subgraph(largest_cc_nodes).copy()


def _find_inlet_outlets(G: nx.Graph, radius_percentile=99.0):
    G_clean = _get_largest_component(G)
    nodes = pd.DataFrame.from_dict(dict(G_clean.nodes(data=True)), orient='index')
    edges = nx.to_pandas_edgelist(G_clean)

    min_radius = edges['avgRadiusAvg'].quantile(radius_percentile / 100)
    large_vessels = edges[edges['avgRadiusAvg'] >= min_radius]

    suspect_ids = np.unique(np.concatenate([large_vessels['source'].values, large_vessels['target'].values]))

    candidates = nodes.loc[nodes.index.isin(suspect_ids) & (nodes['degree'] == 1)]

    if candidates.empty:
        # Fallback if specific border logic fails
        candidates = nodes.loc[nodes.index.isin(suspect_ids)]

    coords = candidates[['pos_x', 'pos_y', 'pos_z']].values

    kmeans = KMeans(n_clusters=2, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(coords)

    candidates = candidates.copy()
    candidates['cluster'] = labels

    mean_y_0 = candidates[candidates['cluster'] == 0]['pos_y'].mean()
    mean_y_1 = candidates[candidates['cluster'] == 1]['pos_y'].mean()

    if mean_y_0 < mean_y_1:
        # Cluster 0 is lower (Ventral) -> Inlets
        inlets = candidates[candidates['cluster'] == 0].index.values
        outlets = candidates[candidates['cluster'] == 1].index.values
    else:
        # Cluster 1 is lower (Ventral) -> Inlets
        inlets = candidates[candidates['cluster'] == 1].index.values
        outlets = candidates[candidates['cluster'] == 0].index.values

    return inlets, outlets

def calculate_flow_physics(brainNet: BrainNet, P_in=100.0, P_out=0.0):
    G = brainNet.graph.copy()
    inlets, outlets = _find_inlet_outlets(G)

    nodes = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient='index')
    nodes['idx'] = np.arange(len(nodes))
    node2idx = nodes['idx'].to_dict()
    N = len(nodes)

    logger.info("Building sparse matrix")
    edges = nx.to_pandas_edgelist(G)

    edges['u'] = edges['source'].map(node2idx)
    edges['v'] = edges['target'].map(node2idx)

    edges['capacity'] = edges['capacity'].fillna(1e-12)
    edges.loc[edges['capacity'] <= 0, 'capacity'] = 1e-12

    u = edges['u'].values
    v = edges['v'].values
    data = edges['capacity'].values

    rows = np.concatenate((u, v))
    cols = np.concatenate((v, u))
    vals = np.concatenate((-data, -data))

    diag_vals = np.bincount(rows, weights=-vals)
    diag_rows = np.arange(len(diag_vals))
    diag_cols = np.arange(len(diag_vals))

    all_rows = np.concatenate((rows, diag_rows))
    all_cols = np.concatenate((cols, diag_cols))
    all_vals = np.concatenate((vals, diag_vals))

    A = csr_matrix((all_vals, (all_rows, all_cols)), shape=(N, N))

    penalty = 1e10 * diag_vals.mean()
    rhs = np.zeros(N)

    inlet_idx = nodes.loc[nodes.index.isin(inlets), 'idx'].values
    A[inlet_idx, inlet_idx] += penalty
    rhs[inlet_idx] += penalty * P_in

    outlet_idx = nodes.loc[nodes.index.isin(outlets), 'idx'].values
    A[outlet_idx, outlet_idx] += penalty
    rhs[outlet_idx] += penalty * P_out

    logger.info(f"Solving system for {N} nodes")

    ml = pyamg.smoothed_aggregation_solver(A)
    pressure = ml.solve(rhs, tol=1e-5, accel="cg")

    logger.info("Mapping results back to graph...")
    pressure_dict = dict(zip(nodes.index, pressure))
    nx.set_node_attributes(G, pressure_dict, "pressure")

    p_u = edges['u'].map(lambda x: pressure[x])
    p_v = edges['v'].map(lambda x: pressure[x])

    flows = edges['capacity'] * (p_u - p_v)

    flow_map = dict(zip(zip(edges['source'], edges['target']), flows))
    nx.set_edge_attributes(G, flow_map, "flow")

    return G


if __name__ == "__main__":
    dataset = "CD1_E_no2"
    db = dir_archive('../cache/' + dataset, {}, cached=False, compression=5, protocol=-1)
    db.load()
    if not 'brainNet' in db.keys():
        brainNet = BrainNet(dataset)
        brainNet.get_gt()

        # db['brainNet'] = brainNet
    else:
        brainNet = db['brainNet']

    calculate_flow_physics(brainNet)
