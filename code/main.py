from brainNet import BrainNet
from centralities import compute_centralities


def main():
    # Load data
    brainNet = BrainNet('CD1-E_no2')

    # Visualize source data
    brainNet.visualize(outputFile='res.png', show=False)

    # Compute and visualize centralities
    compute_centralities(graph=brainNet.graph,methods=None,weighted=True)

if __name__ == '__main__':
    main()