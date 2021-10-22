
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

class NNGeneticAlgo:

    def __init__(self, n_pops, net_size, mutation_rate, crossover_rate, retain_rate, X, y):
        self.n_pops = n_pops
        self.net_size = net_size
        self.nets = [Network(self.net_size) for i in range(self.n_pops)]
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.retain_rate = retain_rate
        self.X = X[:]
        self.y = y[:]
    
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

    def crossover(self):
        for i in range(self.n_pops):
            if random.uniform(0,1) < self.crossover_rate:
                father = random.randint(0,self.n_pops-1)
                mother = random.randint(0,self.n_pops-1)
                # make a copy of father 'genetic' weights & biases information
                nn_1 = copy.deepcopy(self.nets[father])
                nn_2 = copy.deepcopy(self.nets[mother])
                # cross-over bias
                k_1 = random.randint(int(0.5*self.nets[0].bias_nitem),self.nets[0].bias_nitem)
                for _ in range(k_1):
                    # get some random points
                    layer, point = self.get_random_point('bias')
                    # replace genetic (bias) with mother's value
                    nn_1.biases[layer][point] = self.nets[mother].biases[layer][point]
                    nn_2.biases[layer][point] = self.nets[father].biases[layer][point]

                # cross-over weight
                k_2 = random.randint(int(0.5*self.nets[0].weight_nitem),self.nets[0].weight_nitem)
                for _ in range(k_2):
                    # get some random points
                    layer, point = self.get_random_point('weight')
                    # replace genetic (weight) with mother's value
                    nn_1.weights[layer][point] = self.nets[mother].weights[layer][point]
                    nn_2.weights[layer][point] = self.nets[father].weights[layer][point]
                self.nets[father] = copy.deepcopy(nn_1)
                self.nets[mother] = copy.deepcopy(nn_2)
        
    def mutation(self):
        for i in range(self.n_pops):
            if random.uniform(0,1) < self.mutation_rate:
                origin = random.randint(0,self.n_pops-1)
                nn = copy.deepcopy(self.nets[origin])

                # mutate bias
                k_1 = random.randint(int(0.5*self.nets[0].bias_nitem),self.nets[0].bias_nitem)
                for _ in range(k_1):
                    # get some random points
                    layer, point = self.get_random_point('bias')
                    # add some random value between -0.5 and 0.5
                    nn.biases[layer][point] += random.uniform(-0.5, 0.5)

                # mutate weight
                k_2 = random.randint(int(0.5*self.nets[0].weight_nitem),self.nets[0].weight_nitem)
                for _ in range(k_2):
                    # get some random points
                    layer, point = self.get_random_point('weight')
                    # add some random value between -0.5 and 0.5
                    nn.weights[layer][point[0], point[1]] += random.uniform(-0.5, 0.5)
                self.nets[origin] = copy.deepcopy(nn)

    def selection(self):
        # calculate score for each population of neural-net
        score_list = list(zip(self.nets, self.get_all_scores()))

        # sort the network using its score
        score_list.sort(key=lambda x: x[1])

        # exclude score as it is not needed anymore
        score_list = [obj[0] for obj in score_list]

        # keep only the best one
        retain_num = int(self.n_pops*self.retain_rate)
        score_list_top = score_list[:retain_num]
        for i in range(self.n_pops-retain_num):
            score_list_top.append(random.choice(score_list))
        self.nets = copy.deepcopy(score_list_top)
    
    def selection_2(self):
        nets_new=[]
        for i in range(self.n_pops):
            k_1 = random.randint(0,self.n_pops-1)
            k_2 = random.randint(0,self.n_pops-1)
            if(self.nets[k_1].score(self.X,self.y)<self.nets[k_2].score(self.X,self.y)):
                nets_new.append(self.nets[k_1])
            else:
                nets_new.append(self.nets[k_2])
        self.nets = copy.deepcopy(nets_new)
    
    def sort_nets(self):
        # calculate score for each population of neural-net
        score_list = list(zip(self.nets, self.get_all_scores()))

        # sort the network using its score
        score_list.sort(key=lambda x: x[1])

        # exclude score as it is not needed anymore
        score_list = [obj[0] for obj in score_list]
        self.nets = copy.deepcopy(score_list)

    def evolve(self):
        self.crossover()
        self.mutation()
        self.selection_2()
        self.sort_nets()
def main():

    # load data from iris.csv into X and y
    df = pd.read_csv("data_(8-8).csv")
    X = df.iloc[:1000, :8].values
    y = df.iloc[:1000, 8:16].values
    sc = StandardScaler()
    X = sc.fit_transform(X)

    # parameters
    N_POPS = 50
    NET_SIZE = [8,8,8]
    MUTATION_RATE = 0.2
    CROSSOVER_RATE = 0.6
    RETAIN_RATE = 0.5

    # start our neural-net & optimize it using genetic algorithm
    nnga = NNGeneticAlgo(N_POPS, NET_SIZE, MUTATION_RATE, CROSSOVER_RATE, RETAIN_RATE, X, y)

    start_time = time.time()
    
    # run for n iterations
    for i in range(100):
        if i % 10 == 0:
            print("Current iteration : {}".format(i+1))
            print("Time taken by far : %.1f seconds" % (time.time() - start_time))
            print("Current top member's network score: %.2f " % nnga.get_all_scores()[0])
            print("Current top member's network accuracy: %.2f%%\n" % nnga.get_all_accuracy()[0])

        # evolve the population
        nnga.evolve()

if __name__ == "__main__":
    main()