from DijkstraAlgorithm import DijkstraAlgorithm
import copy
class Path(object):
    def __init__(self):
        self.path_vertices = []
        self.dictance = 0

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
        B = []
        for i in range(1,self.K):
            for j in range(len(paths[i-1])-1):
                path = Path()
                weight = copy.deepcopy(self._weight_map)
                rootPath = paths[i-1][:j+1]
                spurNode = paths[i-1][j]
                for m in range(i):
                    if (rootPath == paths[m][:j+1]):
                        weight[paths[m][j]][paths[m][j+1]] = 9999999999
                        weight[paths[m][j+1]][paths[m][j]] = 9999999999
                for m in range(j):
                    for node_2 in weight[rootPath[m]].keys():
                        weight[rootPath[m]][node_2] = 9999999999
                        weight[node_2][rootPath[m]] = 9999999999
                alg_d = DijkstraAlgorithm(weight,self._vertices)
                spurpath = alg_d.compute_shortest_path(spurNode,5)
                rootPath.pop()
                rootPath.extend(spurpath)
                path.path_vertices = copy.deepcopy(rootPath)
                for m in range(len(path.path_vertices)-1):
                    path.dictance += self._weight_map[path.path_vertices[m]][path.path_vertices[m+1]]
                dk = True
                for path_b in B:
                    if(path_b.path_vertices == path.path_vertices):
                        dk = False
                check = 0
                for m in range(len(spurpath)-1):
                    check += weight[spurpath[m]][spurpath[m+1]]
                if(check>=9999999999):
                    dk = False
                if(dk):
                    B.append(copy.deepcopy(path))
            B.sort(key=lambda x: x.dictance)
            paths.append(copy.deepcopy(B[0].path_vertices))
            B.pop(0)
        return paths