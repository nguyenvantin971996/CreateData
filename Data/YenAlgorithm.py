from DijkstraAlgorithm import DijkstraAlgorithm
import copy
class YenAlgorithm(object):

    def __init__(self, weight_map, vertices, src, dst, K):
        self._vertices = vertices
        self._weight_map = weight_map
        self._source_vertex = src
        self._destination_vertex = dst
        self.K = K

    def compute_shortest_paths(self):
        paths =[]
        alg = DijkstraAlgorithm(self._weight_map,self._vertices)
        path_0 = alg.compute_shortest_path(1,5)
        paths.append(path_0)
        for i in range(1,self.K):
            B = []
            for j in range(len(paths[i-1])-2):
                weight = copy.deepcopy(self._weight_map)
                rootPath = paths[i-1][:j]
                spurNode = paths[i-1][j]
                for m in range(i):
                    if (rootPath == paths[m][:j]):
                        weight[paths[m][j]][paths[m][j+1]] = float("inf")
                        weight[paths[m][j+1]][paths[m][j]] = float("inf")
                for m in range(j):
                    for node_2 in weight[rootPath[m]].keys():
                        weight[rootPath[m]][node_2] = float("inf")
                        weight[node_2][rootPath[m]] = float("inf")
                alg_d = DijkstraAlgorithm(weight,self._vertices)
                spurpath = alg_d.compute_shortest_path(spurNode,5)
                rootPath.extend(spurpath)
                if(rootPath not in B):
                    B.append(rootPath)
            
