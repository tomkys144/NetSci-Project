import logging

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree

from analysis import centralities
import analysis.flow as flow
from brainNet import BrainNet
from graphing import plot_hypo_time

logger = logging.getLogger("ThrombosisAnalysis.disease")


def disease_simulation(brainNet: BrainNet, maxIter=1e9, random_selection=False, step_len=20, hypo_thr=0.4,
                       anastomosis_thr=-1.0):
    G = brainNet.graph

    nodes = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient='index')
    edges = nx.to_pandas_edgelist(G)

    inlets, outlets = flow.find_inlet_outlets(G)
    edges, nodes = flow.calculate_flow_physics(edges, nodes, inlets, outlets, 100, 0)

    edges = edges.merge(nodes[['cube_x', 'cube_y', 'cube_z']], left_on='source', right_index=True) \
        .merge(nodes[['cube_x', 'cube_y', 'cube_z']], left_on='target', right_index=True, suffixes=('_src', '_tgt'))

    edges['edge_cube_x'] = edges[['cube_x_src', 'cube_x_tgt']].min(axis=1)
    edges['edge_cube_y'] = edges[['cube_y_src', 'cube_y_tgt']].min(axis=1)
    edges['edge_cube_z'] = edges[['cube_z_src', 'cube_z_tgt']].min(axis=1)

    edges.drop(columns=['cube_x_src', 'cube_x_tgt', 'cube_y_src', 'cube_y_tgt', 'cube_z_src', 'cube_z_tgt'])

    cube_sz = np.array([nodes['cube_x'].max(), nodes['cube_y'].max(), nodes['cube_z'].max()]).astype(int) + 1

    sum_grid = np.zeros(cube_sz)
    count_grid = np.zeros(cube_sz)

    ex = edges['edge_cube_x'].values.astype(int)
    ey = edges['edge_cube_y'].values.astype(int)
    ez = edges['edge_cube_z'].values.astype(int)

    thresholds = edges['flow'].abs() * 0.4

    if "sediment_state" not in edges.columns:
        edges["sediment_state"] = 0

    stats_history = {
        "CBF": [],
        "CBF_drop": [],
        "hypo_frac": [],
        "flow_reversal_frac": [],
        "cubes_state": [],
        "hypo_time": np.full(cube_sz, -1)
    }

    edges_pre = edges

    dead = 0

    for iters in range(int(maxIter)):
        edges_post = edges_pre.copy()
        active = disease_step(edges_post, random_selection)
        if not active:
            break

        if iters % step_len == 0:
            edges_post, nodes = flow.calculate_flow_physics(edges_post, nodes, inlets, outlets, 100, 0)
            stats = flow.calculate_stats(edges_pre, edges_post, inlets, outlets, thresholds)
            if iters == 0:
                stats_history["CBF"].append(stats['baseline_flow'])

            stats_history["CBF"].append(stats['post_obstruction_flow'])
            stats_history["CBF_drop"].append(stats['global_cbf_drop'])
            stats_history["hypo_frac"].append(stats['hypoperfused_vessel_fraction'])
            stats_history["flow_reversal_frac"].append(stats['flow_reversal_fraction'])

            sum_grid.fill(0)
            count_grid.fill(0)

            np.add.at(sum_grid, (ex, ey, ez), stats['hypoperfused_vessel'].values.astype(float))
            np.add.at(count_grid, (ex, ey, ez), 1.0)

            cube_percentages = np.divide(sum_grid, count_grid, out=np.zeros_like(sum_grid), where=count_grid != 0)

            hit_mask = (cube_percentages >= hypo_thr) & (stats_history["hypo_time"] == -1)
            stats_history["hypo_time"][hit_mask] = iters

            if anastomosis_thr > 0:
                if stats['post_obstruction_flow'] < stats_history['CBF'][0] * anastomosis_thr:
                    anastomosis_thr = -1.0
                    new_edges = perform_anastomosis(brainNet, nodes, edges, stats_history['hypo_time'])

                    if new_edges is not None:
                        edges_post, nodes = flow.calculate_flow_physics(new_edges, nodes, inlets, outlets, 100, 0)
                        dead = 0

            if stats['post_obstruction_flow'] < stats_history["CBF"][0] * 0.001:
                dead += 1
                if dead >= 10: break  # No need to continue, brain is dead for at least 1000 steps
        else:
            is_inlet_edge = edges['source'].isin(inlets) | edges['target'].isin(inlets)
            CBF = edges_post.loc[is_inlet_edge, 'flow'].abs().sum()
            stats_history["CBF"].append(CBF)
            stats_history["CBF_drop"].append((stats_history["CBF"][-2] - CBF) / stats_history["CBF"][-2])

        edges_pre = edges_post

    logger.info(f"Simulation complete.")
    plot_hypo_time(stats_history["hypo_time"])
    return stats_history


def disease_step(df: pd.DataFrame, random_selection=False, constriction_factor=0.5):
    candidate_mask = (df['sediment_state'] < 2) & (df['flow'] > 0)

    if not candidate_mask.any():
        candidate_mask = (df['sediment_state'] < 2)

    if not candidate_mask.any():
        return False

    candidate_indices = df.index[candidate_mask]

    if random_selection:
        target_idx = np.random.choice(candidate_indices)
    elif 'flow' in df.columns:
        flows = df.loc[candidate_indices, 'flow'].abs()
        weights = 1.0 / (flows + 1e-12) ** 2
        probabilities = weights / weights.sum()
        target_idx = np.random.choice(candidate_indices, p=probabilities)
    else:
        target_idx = np.random.choice(candidate_indices)

    current_state = df.at[target_idx, 'sediment_state']

    if current_state == 0:
        df.at[target_idx, 'sediment_state'] = 1

        old_radius = df.at[target_idx, 'avgRadiusAvg']
        new_radius = old_radius * constriction_factor
        df.at[target_idx, 'avgRadiusAvg'] = new_radius

        length = df.at[target_idx, 'length']
        if length > 0:
            new_capacity = (new_radius ** 4) / length
            df.at[target_idx, 'capacity'] = new_capacity

    elif current_state == 1:
        df.at[target_idx, 'sediment_state'] = 2
        new_capacity = 1e-12
        df.at[target_idx, 'capacity'] = new_capacity

    return True


def perform_anastomosis(brainNet:BrainNet,nodes: pd.DataFrame, edges: pd.DataFrame,hypo_times: np.ndarray, r_scale:float = 1.1 ):
    # 0. Calculate radius
    radius = edges['avgRadiusAvg'].max() * r_scale

    # 1. Calculate PageRank Centrality
    pr_cent = centralities.calc(brainNet, ['pagerank'])

    # 2. Filter for high-value donor candidates (Top 5%)
    pr_series = pd.Series(pr_cent['pagerank'])
    donor_candidates = pr_series[pr_series >= pr_series.quantile(0.95)].index.tolist()

    # 3. Build a KD-Tree of these donor candidates for spatial querying
    donor_coords = nodes.loc[donor_candidates, ['pos_x', 'pos_y', 'pos_z']].values
    donor_tree = KDTree(donor_coords)

    # 4. Identify the "Surgical Target"
    flat_times = hypo_times[hypo_times > 0]
    if len(flat_times) == 0:
        return None

    worst_time = np.min(flat_times)
    coords = np.where(hypo_times == worst_time)
    cx, cy, cz = coords[0], coords[1], coords[2]
    targets = nodes[
        (nodes['cube_x'].isin(cx)) &
        (nodes['cube_y'].isin(cy)) &
        (nodes['cube_z'].isin(cz))
    ]

    min_pressure = targets['pressure'].min()
    tgt_indices = targets.index[targets['pressure'] == min_pressure]
    target_coords = targets.loc[tgt_indices, ['pos_x', 'pos_y', 'pos_z']].values

    # 5. Find the closest robust donor to that target
    distances, indices = donor_tree.query(target_coords)

    distances = distances.flatten()
    indices = indices.flatten()

    best_idx_in_subset = np.argmin(distances)

    donor_idx = indices[best_idx_in_subset]
    best_donor_node = donor_candidates[donor_idx]

    tgt_idx = tgt_indices[best_idx_in_subset]

    dist = distances[best_idx_in_subset]

    if donor_idx == tgt_idx:
        distances_k2, indices_k2 = donor_tree.query(target_coords, k=2)

        donor_idx = donor_candidates[indices_k2[best_idx_in_subset][1]]
        best_donor_node = donor_candidates[donor_idx]

        dist = distances_k2[best_idx_in_subset][1]

    # 6. Create the surgical bypass edge
    new_edge = {
        'node1id': best_donor_node,
        'node2id': tgt_idx,
        'avgRadiusAvg': radius,
        'length': dist,
        'capacity': (radius ** 4) / dist,
    }

    brainNet.graph.add_edge(new_edge['node1id'], new_edge['node2id'],
                            avgRadiusAvg=radius,
                            length=dist,
                            capacity=new_edge['capacity'])

    new_edge_row = {
        'source': best_donor_node,
        'target': tgt_idx,
        'avgRadiusAvg': float(radius),
        'length': float(dist),
        'capacity': float(new_edge['capacity']),
        'sediment_state': 0,
        'flow': 0.0,  # Will be updated in the next flow solver call
        'edge_cube_x': min(nodes.at[best_donor_node, 'cube_x'], nodes.at[tgt_idx, 'cube_x']),
        'edge_cube_y': min(nodes.at[best_donor_node, 'cube_y'], nodes.at[tgt_idx, 'cube_y']),
        'edge_cube_z': min(nodes.at[best_donor_node, 'cube_z'], nodes.at[tgt_idx, 'cube_z'])
    }

    new_row_df = pd.DataFrame([new_edge_row])
    updated_edges = pd.concat([edges, new_row_df], ignore_index=True)

    logger.info(f"Surgical anastomosis created: {best_donor_node} -> {tgt_idx}")

    return updated_edges

if __name__ == '__main__':
    brainNet = BrainNet('synthetic_graph_1')
    disease_simulation(brainNet, anastomosis_thr=0.6)
