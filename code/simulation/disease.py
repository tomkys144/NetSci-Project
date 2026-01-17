import logging

import networkx as nx
import numpy as np
import pandas as pd

import analysis.flow as flow
from brainNet import BrainNet
from graphing import plot_hypo_time

logger = logging.getLogger("ThrombosisAnalysis.disease")


def disease_simulation(brainNet: BrainNet, maxIter=1e9, random_selection=False, step_len = 20, hypo_thr = 0.4):
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
            edges_post, nodes_post = flow.calculate_flow_physics(edges_post, nodes, inlets, outlets, 100, 0)
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


if __name__ == '__main__':
    brainNet = BrainNet('synthetic_graph_1')
    disease_simulation(brainNet)
