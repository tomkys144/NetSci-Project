import os.path as op
import math

from matplotlib import pyplot as plt


def _extraction_data_csv(filePath):
    file = open(filePath, "r")
    lines = file.readlines()
    file_dic = {}
    keys = lines[0].split(";")
    for k in range(len(keys)):
        keys[k] = keys[k].replace("\n", "")
    for i in lines[1:]:
        current_line = {}
        list_line = i.split(";")
        for j in range(1, len(list_line)):
            current_line[keys[j]] = float(list_line[j].replace("\n", ""))
        file_dic[int(list_line[0])] = current_line
    return (file_dic)


def _normalize_dic_by_key(dic, key):
    min_value = math.inf
    max_value = -math.inf
    for i in dic:
        if dic[i][key] < min_value :
            min_value = dic[i][key]
        if dic[i][key] > max_value :
            max_value = dic[i][key]
    for i in dic:
        dic[i][key] = (dic[i][key] - min_value) / (max_value-min_value)
    return(dic)


class BrainNet:
    nodes = {}
    edges = {}

    def __init__(self, dataset: str):
        root = op.dirname(__file__)
        if op.basename(root) != "code":
            raise Exception("Unexpected file location")
        csvEdgesPath = op.join(root,"datasets", dataset, "edges.csv")
        csvNodesPath = op.join(root,"datasets", dataset, "nodes.csv")

        self.nodes = _extraction_data_csv(csvNodesPath)
        self.edges = _extraction_data_csv(csvEdgesPath)

        # normalization between 0 and 100
        nodes_keys_to_normalize = ["pos_x", "pos_y", "pos_z"]
        for key in nodes_keys_to_normalize:
            self.nodes = _normalize_dic_by_key(self.nodes, key)

        edges_keys_to_normalize = ["avgRadiusAvg"]
        for key in edges_keys_to_normalize:
            self.edges = _normalize_dic_by_key(self.edges, key)

    def visualize(self, outputFile='', show=True):
        fig, ax = plt.subplots()
        ax = fig.add_subplot(projection='3d')

        print("Graphing...")
        progress = 0
        for e in self.edges:
            X = [self.nodes[self.edges[e]['node1id']]['pos_x'], self.nodes[self.edges[e]['node2id']]['pos_x']]
            Y = [self.nodes[self.edges[e]['node1id']]['pos_y'], self.nodes[self.edges[e]['node2id']]['pos_y']]
            Z = [self.nodes[self.edges[e]['node1id']]['pos_z'], self.nodes[self.edges[e]['node2id']]['pos_z']]
            ax.plot(X, Y, Z, color='red', alpha=0.4, linewidth=self.edges[e]['avgRadiusAvg'] * 5)
            if progress % 5000 == 0:
                print(str(100 * progress / len(self.edges)) + "%")
            progress += 1

        print("Done!")

        if outputFile:
            plt.savefig(outputFile)

        if show:
            plt.show()


if __name__ == "__main__":
    brainNet = BrainNet("CD1-E_no2")
    brainNet.visualize()