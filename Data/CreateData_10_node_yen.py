import random
import csv
import pandas as pd
from YenAlgorithm import YenAlgorithm 
weight_map={
	1:{2:0,5:0,8:0},
	2:{1:0,3:0,8:0},
	3:{2:0,4:0,8:0,9:0},
	4:{3:0,9:0,10:0},
	5:{1:0,6:0,8:0},
	6:{5:0,7:0,8:0,9:0},
	7:{6:0,9:0,10:0},
	8:{1:0,2:0,3:0,5:0,6:0,9:0},
	9:{3:0,4:0,6:0,7:0,8:0,10:0},
	10:{4:0,7:0,9:0}
}
values = [10,20,30,40,50,60,70,80,90]
vertices = [1,2,3,4,5,6,7,8,9,10]
edges = []
for node_1 in weight_map.keys():
		for node_2 in weight_map[node_1].keys():
			column_1 = (node_1,node_2)
			column_2 = (node_2,node_1)
			if column_1 not in edges and column_2 not in edges:
				edges.append(column_1)
cols=[]
for x in edges:
	strx = "X"+repr(x)
	cols.append(strx)
cols_Y = []
for i in range(1,5):
	for x in edges:
		strx = "Y"+str(i)
		cols.append(strx)
		cols_Y.append(strx)
df = pd.DataFrame(columns=cols)
for ii in range(50000):
	if ii%1000 == 0:
		print(ii)
	for node_1 in weight_map.keys():
		for node_2 in weight_map[node_1].keys():
			if weight_map[node_2][node_1] != 0:
				weight_map[node_1][node_2] = weight_map[node_2][node_1]
			else:
				weight_map[node_1][node_2] = random.choice(values)
	data = {}
	for node_1 in weight_map.keys():
		for node_2 in weight_map[node_1].keys():
			column = (node_1,node_2)
			x = "X"+repr(column)
			for col in cols:
				if x == col:
					data[col] = weight_map[node_2][node_1]
	alg = YenAlgorithm(weight_map,vertices,1,9,4)
	paths_vertices = []
	paths_vertices = alg.compute_shortest_paths()
	paths_links = []
	for i in range(4):
		for j in range(len(paths_vertices[i])-1):
			y_1 = "Y"+str(i+1)+repr((paths_vertices[i][j],paths_vertices[i][j+1]))
			y_2 = "Y"+str(i+1)+repr((paths_vertices[i][j+1],paths_vertices[i][j]))
			if y_1 in cols:
				paths_links.append(y_1)
			elif y_2 in cols:
				paths_links.append(y_2)
	for i in cols_Y:
		data[i]=0
	for i in cols_Y:
		for j in paths_links:
			if i==j:
				data[i] = 1
	df = df.append(data, ignore_index = True)
	for node_1 in weight_map.keys():
		for node_2 in weight_map[node_1].keys():
			weight_map[node_1][node_2] = 0
df.to_csv('data_18_18_yen_3.csv',index=False)