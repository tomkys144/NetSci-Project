import comunities
from brainNet import BrainNet


def main():
    dataset = "synthetic_graph_1"
    # Load data
    brainNet = BrainNet(dataset)

    # Visualize source data
    # brainNet.visualize(outputFile="res.png", show=False)
    brainNet.get_gt()
    print("Printing raw graph...")
    brainNet.draw_gt(f"graph-{dataset}-xy.png", coords=(0, 1))
    brainNet.draw_gt(f"graph-{dataset}-xz.png", coords=(0, 2))
    brainNet.draw_gt(f"graph-{dataset}-yz.png", coords=(1, 2))

    # Find communities
    print("Community detection...")
    sbmState = comunities.sbm(brainNet, nmcmc=1000)
    print("Printing results...")
    comunities.draw_sbm(sbmState, f"sbm-{dataset}.png")
    comunities.draw_sbm_shape(sbmState, brainNet, f"sbm-{dataset}-xy.png", coords=(0, 1))
    comunities.draw_sbm_shape(sbmState, brainNet, f"sbm-{dataset}-xz.png", coords=(0, 2))
    comunities.draw_sbm_shape(sbmState, brainNet,f"sbm-{dataset}-yz.png", coords=(1,2))
    print("Done")


if __name__ == "__main__":
    main()
