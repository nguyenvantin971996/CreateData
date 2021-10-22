import random
import csv
import pandas as pd
from DijkstraAlgorithm import DijkstraAlgorithm 
weight_map={
	1:{2:0,3:0,4:0},
	2:{1:0,4:0,5:0},
	3:{1:0,4:0,5:0},
	4:{1:0,2:0,3:0,5:0},
	5:{2:0,3:0,4:0}
}
values = [10,20,30,40,50,60,70,80,90]
vertices = [1,2,3,4,5]
columns = []
for node_1 in weight_map.keys():
		for node_2 in weight_map[node_1].keys():
			column_1 = (node_1,node_2)
			column_2 = (node_2,node_1)
			if column_1 not in columns and column_2 not in columns:
				columns.append(column_1)
cols=[]
for x in columns:
	strx = "X"+repr(x)
	cols.append(strx)
for x in columns:
	strx = "Y"+repr(x)
	cols.append(strx)
	cols_Y.append(strx)
df = pd.DataFrame(columns=cols)
for i in range(10000):
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
	alg = DijkstraAlgorithm(weight_map,vertices)
	path_vertices = []
	path_vertices = alg.compute_shortest_path(1,5)
	path_links = []
	for i in range(len(path_vertices)-1):
		y_1 = "Y"+repr((path_vertices[i],path_vertices[i+1]))
		y_2 = "Y"+repr((path_vertices[i+1],path_vertices[i]))
		if y_1 in cols:
			path_links.append(y_1)
		elif y_2 in cols:
			path_links.append(y_2)
	for i in cols_Y:
		data[i]=0
	for i in cols_Y:
		for j in path_links:
			if i==j:
				data[i] = 1
	df = df.append(data, ignore_index = True)
	for node_1 in weight_map.keys():
		for node_2 in weight_map[node_1].keys():
			weight_map[node_1][node_2] = 0
df.to_csv('data_8.csv',index=False)