import logging

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree

import analysis.flow as flow
from analysis import centralities
from brainNet import BrainNet
from simulation.disease_graphing import plot_cbf, plot_hypo_time

logger = logging.getLogger("ThrombosisAnalysis.disease")


def disease_simulation(brainNet: BrainNet, maxIter=1e9, random_selection=False, step_len=20, hypo_thr=0.4,
                       anastomosis_thr=-1.0):
    G = brainNet.graph

    nodes = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient='index')
    edges = nx.to_pandas_edgelist(G)

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

    stats_history = {
        "CBF": [],
        "hypo_frac": [],
        "cubes_state": [],
        "hypo_time": np.full(cube_sz, -1)
    }

    inlets, outlets = flow.find_inlet_outlets(G)
    new_edges, new_nodes = flow.calculate_flow_physics(edges, nodes, inlets, outlets, 100, 0)
    if new_edges is None:
        logger.warning(f"Simulation run aborted at iteration -1 due to solver failure.")
        return stats_history

    edges, nodes = new_edges, new_nodes

    thresholds = edges['flow'].abs() * 0.4

    if "sediment_state" not in edges.columns:
        edges["sediment_state"] = 0

    if "protected" not in edges.columns:
        edges["protected"] = 0.

    is_outlet_edge = edges['source'].isin(outlets) | edges['target'].isin(outlets)
    CBF = edges.loc[is_outlet_edge, 'flow'].abs().sum()
    stats_history["CBF"].append(CBF)

    dead = 0

    sediment_arr = edges['sediment_state'].values
    radius_arr = edges['avgRadiusAvg'].values
    capacity_arr = edges['capacity'].values
    length_arr = edges['length'].values
    flow_arr = edges['flow'].values

    weights_arr = 1 / (np.abs(flow_arr) + 0.1) ** 2
    weights_arr += 1e-5
    weights_arr[sediment_arr >= 2] = 0
    weights_arr[np.abs(flow_arr) < 1e-12] = 0.0

    for iters in range(int(maxIter // step_len)):
        current_batch_size = step_len
        candidate_indices = np.where((sediment_arr < 2) & (weights_arr > 0))[0]

        if len(candidate_indices) == 0: break
        if len(candidate_indices) < current_batch_size: current_batch_size = len(candidate_indices)

        if random_selection:
            targets = np.random.choice(candidate_indices, size=current_batch_size, replace=False)
        else:
            p = weights_arr[candidate_indices]
            p = p.astype(np.float64)
            p_sum = p.sum()

            if p_sum > 0:
                p /= p_sum
                # Floating point safety
                p[-1] = 1.0 - p[:-1].sum()
                if p[-1] < 0: p[-1] = 0; p /= p.sum()

                targets = np.random.choice(candidate_indices, size=current_batch_size, replace=False, p=p)
            else:
                targets = np.random.choice(candidate_indices, size=current_batch_size, replace=False)

        t_s0 = targets[sediment_arr[targets] == 0]
        t_s1 = targets[sediment_arr[targets] == 1]

        sediment_arr[t_s0] = 1
        radius_arr[t_s0] *= 0.5
        capacity_arr[t_s0] = (radius_arr[t_s0] ** 4) / length_arr[t_s0]

        sediment_arr[t_s1] = 2
        capacity_arr[t_s1] = 1e-12

        edges["sediment_state"] = sediment_arr
        edges['avgRadiusAvg'] = radius_arr
        edges['capacity'] = capacity_arr

        new_edges, new_nodes = flow.calculate_flow_physics(edges, nodes, inlets, outlets, 100, 0)

        if new_nodes is None:
            logger.warning(f"Simulation run aborted at iteration {iters} due to solver failure.")
            break

        edges, nodes = new_edges, new_nodes

        stats = flow.calculate_stats(edges, inlets, outlets, thresholds)

        stats_history["CBF"].append(stats['flow'])
        stats_history["hypo_frac"].append(stats['hypoperfused_vessel_fraction'])

        sum_grid.fill(0)
        count_grid.fill(0)

        np.add.at(sum_grid, (ex, ey, ez), stats['hypoperfused_vessel'].values.astype(float))
        np.add.at(count_grid, (ex, ey, ez), 1.0)

        cube_percentages = np.divide(sum_grid, count_grid, out=np.zeros_like(sum_grid), where=count_grid != 0)

        current_hypo_mask = (cube_percentages >= hypo_thr)

        recovery_mask = (stats_history["hypo_time"] == (iters - step_len)) & (~current_hypo_mask)
        stats_history["hypo_time"][recovery_mask] = -1

        hit_mask = current_hypo_mask & (stats_history["hypo_time"] == -1)
        stats_history["hypo_time"][hit_mask] = iters

        if anastomosis_thr > 0:
            if stats['flow'] < stats_history['CBF'][0] * anastomosis_thr:
                anastomosis_thr = -1.0
                new_edges = perform_anastomosis(brainNet, nodes, edges, stats_history['hypo_time'])

                if new_edges is not None:
                    edges, nodes = flow.calculate_flow_physics(new_edges, nodes, inlets, outlets, 100, 0)
                    dead = 0

                    thresholds[len(thresholds)] = (edges['flow'].abs())[len(edges) - 1] * 0.4

                    ex = edges['edge_cube_x'].values.astype(int)
                    ey = edges['edge_cube_y'].values.astype(int)
                    ez = edges['edge_cube_z'].values.astype(int)

        if stats['flow'] < stats_history["CBF"][0] * 0.001:
            dead += 1
            if dead >= 10: break  # No need to continue, brain is dead for at least 1000 steps

        sediment_arr = edges['sediment_state'].values
        radius_arr = edges['avgRadiusAvg'].values
        capacity_arr = edges['capacity'].values
        length_arr = edges['length'].values
        flow_arr = edges['flow'].values

        weights_arr = 1 / (np.abs(flow_arr) + 0.1) ** 2

        weights_arr += 1e-5

        weights_arr[sediment_arr >= 2] = 0
        weights_arr[np.abs(flow_arr) < 1e-12] = 0.0

        print(f"{iters} iters complete. Dead: {dead} | SUM: {weights_arr.sum()}")

    logger.info(f"Simulation complete.")
    plot_hypo_time(stats_history["hypo_time"])
    return stats_history


def disease_step(sediment, radius, capacity, length, weights, random_selection=False, constriction_factor=0.5):
    candidate_indices = np.where((sediment < 2) & (weights > 0))[0]

    if len(candidate_indices) == 0:
        return False

    p = weights[candidate_indices]
    p_sum = p.sum()
    if p_sum > 0:
        p /= p_sum
        p[-1] = 1.0 - p[:-1].sum()

        if p[-1] < 0:
            p[-1] = 0
            p /= p.sum()

        target_idx = np.random.choice(candidate_indices, p=p)
    else:
        target_idx = np.random.choice(candidate_indices)

    if sediment[target_idx] == 0:
        sediment[target_idx] = 1
        radius[target_idx] *= constriction_factor
        capacity[target_idx] = (radius[target_idx] ** 4) / length[target_idx]
    elif sediment[target_idx] == 1:
        sediment[target_idx] = 2
        capacity[target_idx] = 1e-12

    return True


def perform_anastomosis(brainNet: BrainNet, nodes: pd.DataFrame, edges: pd.DataFrame, hypo_times: np.ndarray,
                        r_scale: float = 1.1):
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

    if best_donor_node == tgt_idx:
        distances[best_idx_in_subset] = 1e6

        best_idx_in_subset = np.argmin(distances)

        donor_idx = indices[best_idx_in_subset]
        best_donor_node = donor_candidates[donor_idx]

        tgt_idx = tgt_indices[best_idx_in_subset]

        dist = distances[best_idx_in_subset]

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
        'sediment_state': -1,
        'flow': 0.0,  # Will be updated in the next flow solver call
        'edge_cube_x': min(nodes.at[best_donor_node, 'cube_x'], nodes.at[tgt_idx, 'cube_x']),
        'edge_cube_y': min(nodes.at[best_donor_node, 'cube_y'], nodes.at[tgt_idx, 'cube_y']),
        'edge_cube_z': min(nodes.at[best_donor_node, 'cube_z'], nodes.at[tgt_idx, 'cube_z'])
    }

    new_row_df = pd.DataFrame([new_edge_row])
    new_row_df.index += len(edges)
    updated_edges = pd.concat([edges, new_row_df], ignore_index=False)

    logger.info(f"Surgical anastomosis created: {best_donor_node} -> {tgt_idx}")

    return updated_edges


if __name__ == '__main__':
    brainNet = BrainNet('synthetic_graph_1')
    # stats = disease_simulation(brainNet, anastomosis_thr=-1)
    # disease_graphing.plot_hypo_time(stats['hypo_time'])

    CBF = []
    hypo = []
    for i in range(2):
        print(i)
        stats = disease_simulation(brainNet, anastomosis_thr=-1)
        CBF.append(stats['CBF'])
        hypo.append(stats['hypo_time'])

    CBF_a = []
    for i in range(2):
        print(i)
        stats = disease_simulation(brainNet, anastomosis_thr=0.8)
        CBF_a.append(stats['CBF'])

    plot_cbf(CBF, CBF_a, output='cbf.pdf')
