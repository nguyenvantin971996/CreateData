import random
import numpy as np
import pandas as pd
import copy
import time
from sklearn.preprocessing import StandardScaler

class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        
        # helper variables
        self.bias_nitem = sum(sizes[1:])
        self.weight_nitem = sum([self.weights[i].size for i in range(self.num_layers-2)])
        self.counter = 0

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = self.sigmoid(np.dot(w,a)+b)
        return a

    def sigmoid(self, z):
        return 1.0/(1.0+np.exp(-z))

    def score(self, X, y):
        total_score=0
        for i in range(X.shape[0]):
            predicted = self.feedforward(X[i].reshape(-1,1))
            actual = y[i].reshape(-1,1)
            total_score += np.sum(np.power(predicted-actual,2)/2)  # mean-squared error
        return total_score

    def accuracy(self, X, y):
        accuracy = 0
        for i in range(X.shape[0]):
            output = (self.feedforward(X[i].reshape(-1,1))).reshape(-1)
            condition = True
            for j in range(len(output)):
                output[j] = round(output[j])
            for j in range(len(output)):
                if(output[j]!=y[i][j]):
                    condition = False
                    break
            if condition:
                accuracy += 1
        return accuracy / X.shape[0] * 100

    def __str__(self):
        s = "\nBias:\n\n" + str(self.biases)
        s += "\nWeights:\n\n" + str(self.weights)
        s += "\n\n"
        return s

class NN_ACO_Algo:

    def __init__(self, n_pops, p, a, b, p0, Q, net_size, X, y):
        self.n_pops = n_pops
        self.net_size = net_size
        self.nets = [Network(self.net_size) for i in range(self.n_pops)]
        self.X = X[:]
        self.y = y[:]
        self.p = p
        self.a = a
        self.b = b
        self.p0 = p0
        self.Q = Q
        self.p = p
    
    def get_random_point(self, type):
        nn = self.nets[0]
        layer_index, point_index = random.randint(0, nn.num_layers-2), 0
        if type == 'weight':
            row = random.randint(0,nn.weights[layer_index].shape[0]-1)
            col = random.randint(0,nn.weights[layer_index].shape[1]-1)
            point_index = (row, col)
        elif type == 'bias':
            point_index = random.randint(0,nn.biases[layer_index].size-1)
        return (layer_index, point_index)

    def get_all_scores(self):
        return [net.score(self.X, self.y) for net in self.nets]

    def get_all_accuracy(self):
        return [net.accuracy(self.X, self.y) for net in self.nets]

    def create_paths(self):
        for i in range(self.n_pops):
            r = list(range(0,i)) + list(range(i+1,self.n_pops))
            coceg = random.choice(r)
            fi = random.random()
            nn = copy.deepcopy(self.nets[i])
            k_1 = random.randint(0,self.nets[0].bias_nitem)
            for _ in range(k_1):
                # get some random points
                layer, point = self.get_random_point('bias')
                # replace genetic (bias) with mother's value
                nn.biases[layer][point] = nn.biases[layer][point] + fi*self.nets[coceg].biases[layer][point]
            k_2 = random.randint(0,self.nets[0].weight_nitem)
            for _ in range(k_2):
                layer, point = self.get_random_point('weight')
                # replace genetic (weights) with mother's value
                nn.weights[layer][point] = nn.weights[layer][point] + fi*self.nets[coceg].weights[layer][point]
            if(nn.score_vector(self.X,self.y)>self.nets[i].score_vector(self.X,self.y)):
                self.nets[i] = copy.deepcopy(nn)
            else:
                self.nets[i].counter += 1
        
    def outlooked_phase(self):
        all_scores_vector = self.get_all_scores_vector()
        sum = np.sum(all_scores_vector)
        probability = all_scores_vector/sum
        for i in range(self.n_pops):
            index_solution = np.random.choice(range(self.n_pops),p=probability)
            r = list(range(0,index_solution)) + list(range(index_solution+1,self.n_pops))
            coceg = random.choice(r)
            fi = random.random()
            nn = copy.deepcopy(self.nets[index_solution])
            k_1 = random.randint(0,self.nets[0].bias_nitem)
            for _ in range(k_1):
                # get some random points
                layer, point = self.get_random_point('bias')
                # replace genetic (bias) with mother's value
                nn.biases[layer][point] = nn.biases[layer][point] + fi*self.nets[coceg].biases[layer][point]
            k_2 = random.randint(0,self.nets[0].weight_nitem)
            for _ in range(k_2):
                layer, point = self.get_random_point('weight')
                # replace genetic (weights) with mother's value
                nn.weights[layer][point] = nn.weights[layer][point] + fi*self.nets[coceg].weights[layer][point]
            if(nn.score_vector(self.X,self.y)>self.nets[index_solution].score_vector(self.X,self.y)):
                self.nets[index_solution] = copy.deepcopy(nn)
            else:
                self.nets[index_solution].counter += 1

    def scout_phase(self):
        # calculate score for each population of neural-net
        score_list = list(zip(self.nets, self.get_all_scores()))

        # sort the network using its score
        score_list.sort(key=lambda x: x[1])

        # exclude score as it is not needed anymore
        score_list = [obj[0] for obj in score_list]

        # keep only the best one
        retain_num = int(self.n_pops*0.5)
        self.nets = copy.deepcopy(score_list)
        for i in range(retain_num,self.n_pops):
            if self.nets[i].counter > int(self.limit/2):
                self.nets[i] = random.choice(self.nets)
    
    def sort_nets(self):
        # calculate score for each population of neural-net
        score_list = list(zip(self.nets, self.get_all_scores()))

        # sort the network using its score
        score_list.sort(key=lambda x: x[1])

        # exclude score as it is not needed anymore
        score_list = [obj[0] for obj in score_list]
        self.nets = copy.deepcopy(score_list)

    def evolve(self):
        self.employeed_phase()
        self.outlooked_phase()
        self.scout_phase()
        self.sort_nets()
def main():

    # load data from iris.csv into X and y
    df = pd.read_csv("data_(8-8).csv")
    X = df.iloc[:200, :8].values
    y = df.iloc[:200, 8:16].values
    sc = StandardScaler()
    X = sc.fit_transform(X)

    # parameters
    N_POPS = 50
    NET_SIZE = [8,8,8]

    # start our neural-net & optimize it using genetic algorithm
    nnabc = NN_ABC_Algo(N_POPS, NET_SIZE, X, y)

    start_time = time.time()
    
    # run for n iterations
    for i in range(500):
        if i % 10 == 0:
            print("Current iteration : {}".format(i+1))
            print("Time taken by far : %.1f seconds" % (time.time() - start_time))
            print("Current top member's network score: %.2f " % nnabc.get_all_scores()[0])
            print("Current top member's network accuracy: %.2f%%\n" % nnabc.get_all_accuracy()[0])

        # evolve the population
        nnabc.evolve()

if __name__ == "__main__":
    main()