import logging

import networkx as nx
import numpy as np
import pandas as pd

import analysis.flow as flow
from brainNet import BrainNet

logger = logging.getLogger("ThrombosisAnalysis.disease")


def disease_simulation(brainNet: BrainNet, maxIter=1e9, random_selection=False, step_len = 20):
    G = brainNet.graph

    nodes = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient='index')
    edges = nx.to_pandas_edgelist(G)

    inlets, outlets = flow.find_inlet_outlets(G)
    edges, nodes = flow.calculate_flow_physics(edges, nodes, inlets, outlets, 100, 0)

    if "sediment_state" not in edges.columns:
        edges["sediment_state"] = 0

    stats_history = {
        "CBF": [],
        "CBF_drop": [],
        "hypo_frac": [],
        "flow_reversal_frac": []
    }

    iters = 0
    edges_pre = edges

    dead = 0

    for _ in range(int(maxIter)):
        edges_post = edges_pre.copy()
        active = disease_step(edges_post, random_selection)
        if not active:
            break

        if iters % step_len == 0:
            edges_post, nodes_post = flow.calculate_flow_physics(edges_post, nodes, inlets, outlets, 100, 0)
            stats = flow.calculate_stats(edges_pre, edges_post, inlets, outlets)
            if iters == 0:
                stats_history["CBF"].append(stats['baseline_flow'])

            stats_history["CBF"].append(stats['post_obstruction_flow'])
            stats_history["CBF_drop"].append(stats['global_cbf_drop'])
            stats_history["hypo_frac"].append(stats['hypoperfused_vessel_fraction'])
            stats_history["flow_reversal_frac"].append(stats['flow_reversal_fraction'])

            if stats['post_obstruction_flow'] < stats_history["CBF"][0] * 0.001:
                dead += 1
                if dead >= 10: break  # No need to continue, brain is dead for at least 1000 steps
        else:
            is_inlet_edge = edges['source'].isin(inlets) | edges['target'].isin(inlets)
            CBF = edges_post.loc[is_inlet_edge, 'flow'].abs().sum()
            stats_history["CBF"].append(CBF)
            stats_history["CBF_drop"].append((stats_history["CBF"][-2] - CBF) / stats_history["CBF"][-2])

        edges_pre = edges_post
        iters += 1

    logger.info(f"Simulation complete. Syncing {steps_taken} changes back to graph...")
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
        target_idx = df.loc[candidate_indices, 'flow'].abs().idxmin()
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
