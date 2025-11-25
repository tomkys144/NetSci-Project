import os.path as op
import os
import math
import time
import matplotlib.pyplot as plt
import pathlib
import time

path = "synthetic_graph_1\\synthetic_graph_1\\1_b_3_0"
path = "CD1-E_no2\CD1-E_no2\CD1-E-no2_iso3um_stitched_segmentation_bulge_size_3.0"
ab_path = os.path.dirname(__file__)

def extraction_data_csv(path,name):
    file = open(op.join(path,name), "r")
    lines = file.readlines()
    file_dic = {}
    keys = lines[0].split(";")
    for k in range(len(keys)):
        keys[k] = keys[k].replace("\n","")
    for i in lines[1:]:
        current_line = {}
        list_line = i.split(";")
        for j in range(1,len(list_line)):
            current_line[keys[j]] = float(list_line[j].replace("\n",""))
        file_dic[int(list_line[0])] = current_line
    return(file_dic)

name_nodes = "CD1-E-no2_iso3um_stitched_segmentation_bulge_size_3.0_nodes_processed.csv"
name_edges = "CD1-E-no2_iso3um_stitched_segmentation_bulge_size_3.0_edges_processed.csv"

nodes = extraction_data_csv(path,name_nodes)

print("Node 0 : " + str(nodes[0]))
print("Number of nodes : " + str(len(nodes)))

edges = extraction_data_csv(path,name_edges)

print("Edge 0 : " + str(edges[0]))
print("Number of edges : " + str(len(edges)))

# normalization between 0 and 100

def normalize_dic_by_key(dic,key):
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

nodes_keys_to_normalize = ["pos_x","pos_y","pos_z"]
for k in nodes_keys_to_normalize:
    nodes = normalize_dic_by_key(nodes,k)

edges_keys_to_normalize = ["avgRadiusAvg"]
for k in edges_keys_to_normalize:
    edges = normalize_dic_by_key(edges,k)

# print 3D visualization

fig, ax = plt.subplots()
ax = fig.add_subplot(projection='3d')

print("Graphing...")
tic = time.time()

progress = 0
for e in edges:
    X = [ nodes[edges[e]['node1id']]['pos_x'], nodes[edges[e]['node2id']]['pos_x'] ]
    Y = [ nodes[edges[e]['node1id']]['pos_y'], nodes[edges[e]['node2id']]['pos_y'] ]
    Z = [ nodes[edges[e]['node1id']]['pos_z'], nodes[edges[e]['node2id']]['pos_z'] ]
    ax.plot(X, Y, Z, color='red', alpha=0.4, linewidth = edges[e]['avgRadiusAvg']*5)
    if progress%5000 == 0:
        print(str(100*progress/len(edges)) + "%")
    progress+=1

plt.show()

tac = time.time()
print("Time to graph : " + str(tac-tic))