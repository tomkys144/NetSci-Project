from brainNet import BrainNet


def main():
    # Load data
    brainNet = BrainNet('CD1-E_no2')

    # Visualize source data
    brainNet.visualize(outputFile='res.png', show=False)

if __name__ == '__main__':
    main()