from brainNet import BrainNet


def main():
    # Load data
    brainNet = BrainNet('synthetic_graph_1')

    # Visualize source data
    brainNet.visualize()

if __name__ == '__main__':
    main()