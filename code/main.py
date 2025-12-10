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
    # brainNet.draw_gt(f"graph-{dataset}.png")

    # Find communities
    print("Community detection...")
    sbmState = comunities.sbm(brainNet, nmcmc=10)
    print("Printing results...")
    comunities.draw_sbm(sbmState, f"sbm-{dataset}.png")
    print("Done")


if __name__ == "__main__":
    main()
