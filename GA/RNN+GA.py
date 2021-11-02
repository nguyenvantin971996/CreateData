import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
import copy
import time
from sklearn.preprocessing import StandardScaler
import random
class Simple_RNN(object):

    def __init__(self, sizes, steps):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.wx = np.random.randn(sizes[1], sizes[0])
        self.wh = np.random.randn(sizes[1], sizes[1])
        self.bh = np.random.randn(sizes[1],1)
        self.wy = np.random.randn(sizes[2], sizes[1])
        self.by = np.random.randn(sizes[2],1)
        self.h0 = np.zeros(sizes[1])
        self.steps = steps

    def feedforward(self, x):
        h_temp = np.tanh(np.dot(self.wx,x[0]) + self.h0 + self.bh.reshape(-1))
        for i in range(self.steps-1):
            h_temp = np.tanh(np.dot( self.wx,x[i+1]) + np.dot(self.wh,h_temp) + self.bh.reshape(-1))
        result = self.sigmoid(np.dot(self.wy,h_temp) + self.by.reshape(-1))
        return result 
    
    def sigmoid(self, z):
        return 1.0/(1.0+np.exp(-z))

    def score(self, X, y):
        total_score=0
        for i in range(X.shape[0]):
            predicted = self.feedforward(X[i])
            actual = y[i]
            total_score += np.sum(np.power(predicted-actual,2))  # mean-squared error
        return total_score/X.shape[0]

    def accuracy(self, X, y):
        accuracy = 0
        for i in range(X.shape[0]):
            output = self.feedforward(X[i])
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

class Simple_RNN_GA:

    def __init__(self, n_pops, net_size, mutation_rate, crossover_rate, X, y, X_test, y_test, steps):
        self.n_pops = n_pops
        self.steps = steps
        self.net_size = net_size
        self.nets = [Simple_RNN(self.net_size,self.steps) for i in range(self.n_pops)]
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.X = X[:]
        self.y = y[:]
        self.X_test = X_test[:]
        self.y_test = y_test[:]
        self.accuracy_train = []
        self.accuracy_test = []
        self.best = Simple_RNN(self.net_size,self.steps)
    
    def get_random_point(self, type):
        nn = self.nets[0]
        point_index = 0
        if type == 'wx':
            row = random.randint(0,nn.wx.shape[0]-1)
            col = random.randint(0,nn.wx.shape[1]-1)
            point_index = (row, col)
        elif type == 'wh':
            row = random.randint(0,nn.wh.shape[0]-1)
            col = random.randint(0,nn.wh.shape[1]-1)
            point_index = (row, col)
        elif type == 'bh':
            row = random.randint(0,nn.bh.shape[0]-1)
            col = random.randint(0,nn.bh.shape[1]-1)
            point_index = (row, col)
        elif type == 'wy':
            row = random.randint(0,nn.wy.shape[0]-1)
            col = random.randint(0,nn.wy.shape[1]-1)
            point_index = (row, col)
        elif type == 'by':
            row = random.randint(0,nn.by.shape[0]-1)
            col = random.randint(0,nn.by.shape[1]-1)
            point_index = (row, col)
        return point_index

    def get_all_scores(self,Xc,yc):
        return [net.score(Xc,yc) for net in self.nets]

    def get_all_accuracy(self,Xc,yc):
        return [net.accuracy(Xc,yc) for net in self.nets]

    def crossover(self):
        for i in range(self.n_pops):
            if random.uniform(0,1) < self.crossover_rate:
                father = random.randint(0,self.n_pops-1)
                mother = random.randint(0,self.n_pops-1)
                nn_1 = copy.deepcopy(self.nets[father])
                nn_2 = copy.deepcopy(self.nets[mother])
                
                k_1 = random.randint(int(0.5*self.nets[0].wx.size),self.nets[0].wx.size)
                for _ in range(k_1):
                    point = self.get_random_point('wx')
                    nn_1.wx[point] = self.nets[mother].wx[point]
                    nn_2.wx[point] = self.nets[father].wx[point]
                    
                k_2 = random.randint(int(0.5*self.nets[0].wh.size),self.nets[0].wh.size)
                for _ in range(k_2):
                    point = self.get_random_point('wh')
                    nn_1.wh[point] = self.nets[mother].wh[point]
                    nn_2.wh[point] = self.nets[father].wh[point]
                    
                k_3 = random.randint(int(0.5*self.nets[0].bh.size),self.nets[0].bh.size)
                for _ in range(k_3):
                    point = self.get_random_point('bh')
                    nn_1.bh[point] = self.nets[mother].bh[point]
                    nn_2.bh[point] = self.nets[father].bh[point]
                    
                k_4 = random.randint(int(0.5*self.nets[0].wy.size),self.nets[0].wy.size)
                for _ in range(k_4):
                    point = self.get_random_point('wy')
                    nn_1.wy[point] = self.nets[mother].wy[point]
                    nn_2.wy[point] = self.nets[father].wy[point]
                    
                k_5 = random.randint(int(0.5*self.nets[0].by.size),self.nets[0].by.size)
                for _ in range(k_5):
                    point = self.get_random_point('by')
                    nn_1.by[point] = self.nets[mother].by[point]
                    nn_2.by[point] = self.nets[father].by[point]
                    
                self.nets.append(copy.deepcopy(nn_1))
                self.nets.append(copy.deepcopy(nn_2))
        
    def mutation(self):
        for i in range(self.n_pops):
            if random.uniform(0,1) < self.mutation_rate:
                origin = random.randint(0,self.n_pops-1)
                nn = copy.deepcopy(self.nets[origin])

                k_1 = random.randint(int(0.5*self.nets[0].wx.size),self.nets[0].wx.size)
                for _ in range(k_1):
                    point = self.get_random_point('wx')
                    nn.wx[point] += random.uniform(-0.5, 0.5)

                k_2 = random.randint(int(0.5*self.nets[0].wh.size),self.nets[0].wh.size)
                for _ in range(k_2):
                    point = self.get_random_point('wh')
                    nn.wh[point] += random.uniform(-0.5, 0.5)
                    
                k_3 = random.randint(int(0.5*self.nets[0].bh.size),self.nets[0].bh.size)
                for _ in range(k_3):
                    point = self.get_random_point('bh')
                    nn.bh[point] += random.uniform(-0.5, 0.5)
                    
                k_4 = random.randint(int(0.5*self.nets[0].wy.size),self.nets[0].wy.size)
                for _ in range(k_4):
                    point = self.get_random_point('wy')
                    nn.wy[point] += random.uniform(-0.5, 0.5)
                    
                k_5 = random.randint(int(0.5*self.nets[0].by.size),self.nets[0].by.size)
                for _ in range(k_5):
                    point = self.get_random_point('by')
                    nn.by[point] += random.uniform(-0.5, 0.5)
                self.nets.append(copy.deepcopy(nn))
        
    def selection(self,Xc,yc):
        nets_new=[]
        for i in range(self.n_pops):
            k_1 = random.randint(0,len(self.nets)-1)
            k_2 = random.randint(0,len(self.nets)-1)
            if(self.nets[k_1].score(Xc,yc)<self.nets[k_2].score(Xc,yc)):
                nets_new.append(self.nets[k_1])
            else:
                nets_new.append(self.nets[k_2])
        self.nets = copy.deepcopy(nets_new)
    
    def sort_nets(self,Xc,yc):
        score_list = list(zip(self.nets, self.get_all_scores(Xc,yc)))
        score_list.sort(key=lambda x: x[1])
        score_list = [obj[0] for obj in score_list]
        self.nets = copy.deepcopy(score_list)
        if(self.best.accuracy(self.X,self.y)<self.nets[0].accuracy(self.X,self.y)):
            self.best = copy.deepcopy(self.nets[0])

    def evolve(self):
        start_time = time.time()
        for t in range(25):
            self.accuracy_train.append(self.best.accuracy(self.X,self.y))
            self.accuracy_test.append(self.best.accuracy(self.X_test,self.y_test))
            for i in range(20):
                j1=i*40
                j2=(1+i)*40
                Xc=self.X[j1:j2,:,:]
                yc=self.y[j1:j2,:]
                for k in range(25): 
                    self.crossover()
                    self.mutation()
                    self.selection(Xc,yc)
                    self.sort_nets(Xc,yc)
                print("Current iteration : {}, batch : {}".format(t+1,i+1))
                print("Time taken by far : %.1f seconds" % (time.time() - start_time))
                print("Current top member's network score: %.5f " % self.best.score(self.X,self.y))
                print("Current top member's network accuracy: %.2f%%\n" % self.best.accuracy(self.X,self.y))

df = pd.read_csv("../Data/data_(8-8).csv")
sc = StandardScaler()
X = df.iloc[:800, :8].values
y = df.iloc[:800, 8:16].values
X = sc.fit_transform(X)
X_test = df.iloc[800:1000, :8].values
y_test = df.iloc[800:1000, 8:16].values
X_test = sc.fit_transform(X_test)
X1= np.reshape(X, (X.shape[0], X.shape[1], 1))
y1= np.reshape(y, (y.shape[0], y.shape[1]))
X2= np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
y2= np.reshape(y_test, (y_test.shape[0], y_test.shape[1]))
N_POPS = 100
steps = X.shape[1]
NET_SIZE = [1,24,8]
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.8
simple_rnn_ga = Simple_RNN_GA(N_POPS, NET_SIZE, MUTATION_RATE, CROSSOVER_RATE, X1, y1, X2, y2, steps)
simple_rnn_ga.evolve()
