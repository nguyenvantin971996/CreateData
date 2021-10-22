import random
import numpy as np
import pandas as pd
import copy
import time
from sklearn.preprocessing import StandardScaler
class RBF(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.centers = np.random.uniform(low=-1, high=1,size=(sizes[1], sizes[0]))
        self.betas= np.random.uniform(low=1, high=2, size=(sizes[1], 1))
        self.biases = np.random.uniform(low=-1, high=1,size=(sizes[2], 1))
        self.weights = np.random.uniform(low=-1, high=1,size=(sizes[2], sizes[1]))
        
        # helper variables
        self.center_nitem = sizes[0]*sizes[1]
        self.beta_nitem = sizes[1]
        self.bias_nitem = sizes[2]
        self.weight_nitem = sizes[1]*sizes[2]
        self.counter = 0

    def feedforward(self, a):
        f = self.Gaussian(a)
        t = self.sigmoid(np.dot(self.weights,f)+self.biases)
        return t
    
    def Gaussian(self,x):
        g=(x-self.centers.T)*(x-self.centers.T)
        z = (np.sum(g,axis=0)*self.betas.T).T
        f = np.exp(-z)
        return f
    
    def sigmoid(self, z):
        return 1.0/(1.0+np.exp(-z))

    def score(self, X, y):
        total_score=0
        for i in range(X.shape[0]):
            predicted = self.feedforward(X[i].reshape(-1,1))
            actual = y[i].reshape(-1,1)
            total_score += np.sum(np.power(predicted-actual,2))  # mean-squared error
        return total_score/X.shape[0]
    
    def score_vector(self, X, y):
        total_score=0
        for i in range(X.shape[0]):
            predicted = self.feedforward(X[i].reshape(-1,1))
            actual = y[i].reshape(-1,1)
            total_score += np.sum(np.power(predicted-actual,2))
        total_score=total_score/X.shape[0] # mean-squared error
        return (1/(1+total_score))

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
class RBF_ABC_Algo:

    def __init__(self, n_pops, net_size, X, y):
        self.n_pops = n_pops
        self.net_size = net_size
        self.nets = [RBF(self.net_size) for i in range(self.n_pops)]
        self.X = X[:]
        self.y = y[:]
        self.limit = self.n_pops
        self.best = RBF(self.net_size)
    
    def get_random_point(self, type):
        nn = self.nets[0]
        point_index = 0
        if type == 'weight':
            row = random.randint(0,nn.weights.shape[0]-1)
            col = random.randint(0,nn.weights.shape[1]-1)
            point_index = (row, col)
        elif type == 'bias':
            point_index = random.randint(0,nn.biases.size-1)
        elif type == 'center':
            row = random.randint(0,nn.centers.shape[0]-1)
            col = random.randint(0,nn.centers.shape[1]-1)
            point_index = (row, col)
        elif type == 'beta':
            point_index = random.randint(0,nn.betas.size-1)
        return point_index

    def get_all_scores(self,Xc,yc):
        return [net.score(Xc, yc) for net in self.nets]

    def get_all_scores_vector(self,Xc,yc):
        return [net.score_vector(Xc, yc) for net in self.nets]

    def get_all_accuracy(self,Xc,yc):
        return [net.accuracy(Xc, yc) for net in self.nets]
    
    def initialization_phase(self):
        for i in range(self.n_pops):
            weight = copy.deepcopy(self.nets[i].weights)
            for x in range(len(weight)):
                for y in range(len(weight[x])):
                        weight[x][y]=self.nets[i].weights[x].min() + np.random.rand()*(self.nets[i].weights[x].max()-self.nets[i].weights[x].min())
            self.nets[i].weights = copy.deepcopy(weight)
            
            biases = copy.deepcopy(self.nets[i].biases)
            for x in range(len(biases)):
                for y in range(len(biases[x])):
                        biases[x][y]=self.nets[i].biases.min() + np.random.rand()*(self.nets[i].biases.max()-self.nets[i].biases.min())
            self.nets[i].biases = copy.deepcopy(biases)
            
            center = copy.deepcopy(self.nets[i].centers)
            for x in range(len(center)):
                for y in range(len(center[x])):
                        center[x][y]=self.nets[i].centers[x].min() + np.random.rand()*(self.nets[i].centers[x].max()-self.nets[i].centers[x].min())
            self.nets[i].centers = copy.deepcopy(center)
            
            beta = copy.deepcopy(self.nets[i].betas)
            for x in range(len(beta)):
                for y in range(len(beta[x])):
                        beta[x][y]=self.nets[i].betas.min() + np.random.rand()*(self.nets[i].betas.max()-self.nets[i].betas.min())
            self.nets[i].betas = copy.deepcopy(beta)

    def employeed_phase(self,Xc,yc):
        nets = copy.deepcopy(self.nets)
        for i in range(self.n_pops):
            r = list(range(0,i)) + list(range(i+1,self.n_pops))
            coceg = random.choice(r)
            fi1 = random.uniform(-1,1)
            fi2 = random.uniform(-1,1)
            fi3 = random.uniform(-1,1)
            nn = copy.deepcopy(self.nets[i])
            for t in range(1):
                # get some random points
                point1 = self.get_random_point('bias')
                # replace genetic (bias) with mother's value
                nn.biases[point1] = nn.biases[point1] + fi1*(nn.biases[point1]-self.nets[coceg].biases[point1])
                point2 = self.get_random_point('weight')
                # replace genetic (weights) with mother's value
                nn.weights[point2] = nn.weights[point2] + fi2*(nn.weights[point2]-self.nets[coceg].weights[point2])
                point3 = self.get_random_point('center')
                # replace genetic (weights) with mother's value
                nn.centers[point3] = nn.centers[point3] + fi3*(nn.centers[point3]-self.nets[coceg].centers[point3])
                point4 = self.get_random_point('beta')
                # replace genetic (weights) with mother's value
                while(True):
                    fi4 = random.uniform(-1,1)
                    temp = nn.betas[point4] + fi4*(nn.betas[point4]-self.nets[coceg].betas[point4])
                    if(temp>0):
                        nn.betas[point4] = temp
                        break                    
            if(nn.score_vector(Xc,yc)>nets[i].score_vector(Xc,yc)):
                nn.counter = 0
                nets[i] = copy.deepcopy(nn)
            else:
                nets[i].counter += 1
        self.nets = copy.deepcopy(nets)
        
    def onlooked_phase(self,Xc,yc):
        all_scores_vector = self.get_all_scores_vector(Xc,yc)
        sum = np.sum(all_scores_vector)
        probability = all_scores_vector/sum
        nets = copy.deepcopy(self.nets)
        for i in range(self.n_pops):
            index_solution = np.random.choice(list(range(self.n_pops)),p=probability)
            r = list(range(0,index_solution)) + list(range(index_solution+1,self.n_pops))
            coceg = random.choice(r)
            fi1 = random.uniform(-1,1)
            fi2 = random.uniform(-1,1)
            fi3 = random.uniform(-1,1)
            nn = copy.deepcopy(self.nets[index_solution])
            for t in range(1):
                # get some random points
                point1 = self.get_random_point('bias')
                # replace genetic (bias) with mother's value
                nn.biases[point1] = nn.biases[point1] + fi1*(nn.biases[point1]-self.nets[coceg].biases[point1])
                point2 = self.get_random_point('weight')
                # replace genetic (weights) with mother's value
                nn.weights[point2] = nn.weights[point2] + fi2*(nn.weights[point2]-self.nets[coceg].weights[point2])
                point3 = self.get_random_point('center')
                # replace genetic (weights) with mother's value
                nn.centers[point3] = nn.centers[point3] + fi3*(nn.centers[point3]-self.nets[coceg].centers[point3])
                point4 = self.get_random_point('beta')
                # replace genetic (weights) with mother's value
                while(True):
                    fi4 = random.uniform(-1,1)
                    temp = nn.betas[point4] + fi4*(nn.betas[point4]-self.nets[coceg].betas[point4])
                    if(temp>0):
                        nn.betas[point4] = temp
                        break  
            if(nn.score_vector(Xc,yc)>nets[index_solution].score_vector(Xc,yc)):
                nn.counter = 0
                nets[index_solution] = copy.deepcopy(nn)
            else:
                nets[index_solution].counter += 1
        self.nets = copy.deepcopy(nets)

    def scout_phase(self):
        for i in range(self.n_pops):
            if self.nets[i].counter > self.limit:
                weight = copy.deepcopy(self.nets[i].weights)
                for x in range(len(weight)):
                    for y in range(len(weight[x])):
                            weight[x][y]=self.nets[i].weights[x].min() + np.random.rand()*(self.nets[i].weights[x].max()-self.nets[i].weights[x].min())
                self.nets[i].weights = copy.deepcopy(weight)

                biases = copy.deepcopy(self.nets[i].biases)
                for x in range(len(biases)):
                    for y in range(len(biases[x])):
                            biases[x][y]=self.nets[i].biases.min() + np.random.rand()*(self.nets[i].biases.max()-self.nets[i].biases.min())
                self.nets[i].biases = copy.deepcopy(biases)

                center = copy.deepcopy(self.nets[i].centers)
                for x in range(len(center)):
                    for y in range(len(center[x])):
                            center[x][y]=self.nets[i].centers[x].min() + np.random.rand()*(self.nets[i].centers[x].max()-self.nets[i].centers[x].min())
                self.nets[i].centers = copy.deepcopy(center)

                beta = copy.deepcopy(self.nets[i].betas)
                for x in range(len(beta)):
                    for y in range(len(beta[x])):
                            beta[x][y]=self.nets[i].betas.min() + np.random.rand()*(self.nets[i].betas.max()-self.nets[i].betas.min())
                self.nets[i].betas = copy.deepcopy(beta)
                self.nets[i].counter = 0
                
    def sort_nets(self,Xc,yc):
        # calculate score for each population of neural-net
        score_list = list(zip(self.nets, self.get_all_scores(Xc,yc)))

        # sort the network using its score
        score_list.sort(key=lambda x: x[1])

        # exclude score as it is not needed anymore
        score_list = [obj[0] for obj in score_list]
        self.nets = copy.deepcopy(score_list)
        if(self.best.accuracy(self.X,self.y)<self.nets[0].accuracy(self.X,self.y)):
            self.best = copy.deepcopy(self.nets[0])

    def evolve(self):
        start_time = time.time()
        self.initialization_phase()
        for t in range(15):
            for i in range(20):
                j1=i*40
                j2=(1+i)*40
                Xc=self.X[j1:j2,:]
                yc=self.y[j1:j2,:]
                for k in range(25):  
                    self.employeed_phase(Xc,yc)
                    self.onlooked_phase(Xc,yc)
                    self.scout_phase()
                    self.sort_nets(Xc,yc)
                print("Current iteration : {}, batch : {}".format(t+1,i+1))
                print("Time taken by far : %.1f seconds" % (time.time() - start_time))
                print("Current top member's network score: %.5f " % self.best.score(self.X,self.y))
                print("Current top member's network accuracy: %.2f%%\n" % self.best.accuracy(self.X,self.y))
df = pd.read_csv("../Data/data_(8-8).csv")
X = df.iloc[:800, :8].values
y = df.iloc[:800, 8:16].values
sc = StandardScaler()
X = sc.fit_transform(X)
N_POPS = 100
NET_SIZE = [8,8,8]
rbfabc = RBF_ABC_Algo(N_POPS, NET_SIZE, X, y)
rbfabc.evolve()