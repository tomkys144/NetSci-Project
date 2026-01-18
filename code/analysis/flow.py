import logging

import networkx as nx
import numpy as np
import pandas as pd
import pyamg
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans

from brainNet import BrainNet

logger = logging.getLogger("ThrombosisAnalysis.flow")


def _get_largest_component(G):
    if not G:
        return G
    largest_cc_nodes = max(nx.weakly_connected_components(G.to_directed()), key=len)
    return G.subgraph(largest_cc_nodes).copy()


def find_inlet_outlets(G: nx.Graph, radius_percentile=99.5):
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


def calculate_flow_physics(edges: pd.DataFrame, nodes: pd.DataFrame, inlets, outlets, P_in=100.0, P_out=0.0):
    nodes['idx'] = np.arange(len(nodes))
    node2idx = nodes['idx'].to_dict()
    N = len(nodes)

    logger.debug("Building sparse matrix")

    edges['u'] = edges['source'].map(node2idx)
    edges['v'] = edges['target'].map(node2idx)

    edges['capacity'] = edges['capacity'].fillna(1e-9)
    edges.loc[edges['capacity'] <= 0, 'capacity'] = 1e-9

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

    penalty = 1e4 * diag_vals.mean()
    rhs = np.zeros(N)

    inlet_idx = nodes.loc[nodes.index.isin(inlets), 'idx'].values
    A[inlet_idx, inlet_idx] += penalty
    rhs[inlet_idx] += penalty * P_in

    outlet_idx = nodes.loc[nodes.index.isin(outlets), 'idx'].values
    A[outlet_idx, outlet_idx] += penalty
    rhs[outlet_idx] += penalty * P_out

    logger.debug(f"Solving system for {N} nodes")

    try:
        ml = pyamg.smoothed_aggregation_solver(A)
        pressure = ml.solve(rhs, tol=1e-5, accel="cg")
    except Exception as e:
        logger.error(f"Solver failed (likely singular matrix): {e}")
        # Return None to signal failure to the simulation loop
        return None, None

    logger.debug("Mapping results back to graph...")
    nodes['pressure'] = pressure

    p_u = edges['u'].map(lambda x: pressure[x])
    p_v = edges['v'].map(lambda x: pressure[x])

    edges['flow'] = edges['capacity'] * (p_u - p_v)

    return edges, nodes


def calculate_stats(edges: pd.DataFrame, inlets, outlets, thresholds):
    is_outlet_edge = edges['source'].isin(outlets) | edges['target'].isin(outlets)

    CBF = edges.loc[is_outlet_edge, 'flow'].abs().sum()

    # Hypoperfusion
    abs = edges['flow'].abs()

    hypo_mask = abs <  thresholds

    return {
        "flow": CBF,
        "hypoperfused_vessel": hypo_mask,
        "hypoperfused_vessel_fraction": hypo_mask.mean(),
    }


if __name__ == "__main__":
    dataset = "CD1_E_no2"
    brainNet = BrainNet(dataset)
