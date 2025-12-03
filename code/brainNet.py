import os.path as op
import os
import math
import numpy as np
from matplotlib import pyplot as plt


def _extraction_data_csv(filePath):
    file = open(filePath, "r")
    lines = file.readlines()
    fileDic = {}
    keys = lines[0].split(";")
    for k in range(len(keys)):
        keys[k] = keys[k].replace("\n", "")
    for i in lines[1:]:
        currentLine = {}
        listLine = i.split(";")
        for j in range(1, len(listLine)):
            currentLine[keys[j]] = float(listLine[j].replace("\n", ""))
        fileDic[int(listLine[0])] = currentLine
    return (fileDic)


def _normalize_dic_by_key(dic, key):
    minValue = math.inf
    maxValue = -math.inf
    for i in dic:
        if dic[i][key] < minValue :
            minValue = dic[i][key]
        if dic[i][key] > maxValue :
            maxValue = dic[i][key]
    for i in dic:
        dic[i][key] = (dic[i][key] - minValue) / (maxValue-minValue)
    return(dic)

def _edges2matrix(edges, numNodes, keyNode1, keyNode2, keyWeight='', undirected=True):
    adjMatrix = np.zeros((numNodes, numNodes))
    for edge in edges.values():
        node1 = int(edge[keyNode1])
        node2 = int(edge[keyNode2])
        weight = edge[keyWeight]
        adjMatrix[node1, node2] = weight
        if undirected:
            adjMatrix[node2, node1] = weight
    return adjMatrix


class BrainNet:
    nodes = None
    edges = None
    adjMatrix = None

    def __init__(self, dataset: str, useCache = True):
        root = op.dirname(__file__)
        if op.basename(root) != "code":
            raise Exception("Unexpected file location")
        csvEdgesPath = op.join(root,"datasets", dataset, "edges.csv")
        csvNodesPath = op.join(root,"datasets", dataset, "nodes.csv")

        self.nodes = _extraction_data_csv(csvNodesPath)
        self.edges = _extraction_data_csv(csvEdgesPath)

        # normalization between 0 and 100
        nodesKeysToNormalize = ["pos_x", "pos_y", "pos_z"]
        for key in nodesKeysToNormalize:
            self.nodes = _normalize_dic_by_key(self.nodes, key)

        edgesKeysToNormalize = ["avgRadiusAvg"]
        for key in edgesKeysToNormalize:
            self.edges = _normalize_dic_by_key(self.edges, key)

        if useCache:
            try:
                self.adjMatrix = np.load(op.join(root,"cache", f'{dataset}.npy'))
            except FileNotFoundError:
                self.adjMatrix = _edges2matrix(self.edges, len(self.nodes), 'node1id', 'node2id', 'avgRadiusAvg')
                if not op.exists(op.join(root,"cache")):
                    os.makedirs(op.join(root,"cache"))
                np.save(op.join(root,"cache", f'{dataset}.npy'), self.adjMatrix)
        else:
            self.adjMatrix = _edges2matrix(self.edges, len(self.nodes), 'node1id', 'node2id', 'avgRadiusAvg')


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
    brainNet = BrainNet("synthetic_graph_1")
    brainNet.visualize()