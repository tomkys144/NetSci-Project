import pickle
from argparse import ArgumentParser
from sys import argv
from typing import List

import brainNet
import comunities
from brainNet import BrainNet
from aenum import Enum, auto

class Task(Enum):
    ALL = auto()
    LOAD = auto()
    COMMUNITIES = auto()
    DRAW = auto()


def main(tasks, dataset:str = "synthetic_graph_1"):
    if len(tasks) == 0:
        tasks = [Task.ALL]

    # Load data
    if (Task.ALL in tasks) or (Task.LOAD in tasks):
        brainNet=load(dataset, True)
    else:
        brainNet=load(dataset)

    if (Task.ALL in tasks) or (Task.DRAW in tasks):
        draw(dataset, brainNet)

    # Find communities
    if (Task.ALL in tasks) or (Task.COMMUNITIES in tasks):
        communities(dataset, brainNet)

    print("Done")

def load(dataset:str, load_gt = False):
    brainNet = BrainNet(dataset)

    if load_gt:
        brainNet.get_gt()

    return brainNet

def draw(dataset:str, brainNet:BrainNet):
    print("Printing raw graph...")
    brainNet.draw_gt(f"graph-{dataset}-xy.png", coords=(0, 1))
    brainNet.draw_gt(f"graph-{dataset}-xz.png", coords=(0, 2))
    brainNet.draw_gt(f"graph-{dataset}-yz.png", coords=(1, 2))

def communities(dataset:str, brainNet:BrainNet):
    print("Community detection...")
    sbmState = comunities.sbm(brainNet, nmcmc=100)

    print("Drawing CD results...")
    comunities.draw_sbm(sbmState, f"sbm-{dataset}.png")
    comunities.draw_sbm_shape(sbmState, brainNet, f"sbm-{dataset}-xy.png", coords=(0, 1), layer=1)
    comunities.draw_sbm_shape(sbmState, brainNet, f"sbm-{dataset}-xz.png", coords=(0, 2), layer=1)
    comunities.draw_sbm_shape(sbmState, brainNet, f"sbm-{dataset}-yz.png", coords=(1, 2), layer=1)

if __name__ == "__main__":
    parser = ArgumentParser(
        prog="NetSci Project",
        description="Net Sci Project"
    )
    parser.add_argument("--load", action="store_true", help="Flag to load graph-tool graph")
    parser.add_argument("--communities", action="store_true", help="Flag to run community detection")
    parser.add_argument("--draw", action="store_true", help="Flag to draw raw graph")

    args = parser.parse_args()

    tasks = []
    if args.load:
        tasks.append(Task.LOAD)
    if args.communities:
        tasks.append(Task.COMMUNITIES)
    if args.draw:
        tasks.append(Task.DRAW)
    main(tasks)
