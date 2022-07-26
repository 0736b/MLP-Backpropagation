from random import random
from utils.activationfunc import tanh, d_tanh, e
from dataclasses import dataclass

# Fully-connected Multi-layer Perceptron (Backpropagation)

@dataclass
class Neuron:
    actv: float
    out_weights: list
    bias: float
    z: float
    dactv: float
    dw: list
    dbias: float
    dz: float

@dataclass
class Layer:
    num_neuron: int
    neuron: list
    
def create_layer(num_of_neuron):
    lst = [None] * num_of_neuron
    layer = Layer(-1,lst)
    return layer
    
def create_neuron(num_out_weights):
    o_weights = [None] * num_out_weights
    d_o_weights = [None] * num_out_weights
    neuron = Neuron(0.0,o_weights,0.0,0.0,0.0,d_o_weights,0.0,0.0)
    return neuron
    
class MLP:
    def __init__(self,num_layers,num_neurons,learning_rate,momentum_rate,epoch,lengthdata,data_inputs,desired_outputs):
        self.nowEpoch = 0
        self.num_layers = num_layers
        self.num_neurons = num_neurons 
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.epoch = epoch
        self.lengthdata = lengthdata
        # self.data_inputs = [[0,0], [0,1], [1,0], [1,1]]
        self.data_inputs = data_inputs
        # self.desired_outputs = [[0], [1], [1], [0]]
        self.desired_outputs = desired_outputs
        self.cost = [None] * num_neurons[num_layers - 1]
        self.full_cost = 0.0
        self.n = 1.0
    
    def create_network(self):
        self.layer = [None] * self.num_layers
        for i in range(0,self.num_layers,1):
            self.layer[i] = create_layer(self.num_neurons[i])
            self.layer[i].num_neuron = self.num_neurons[i]
            for j in range(0,self.num_neurons[i],1):
                if (i < (self.num_layers - 1)):
                    self.layer[i].neuron[j] = create_neuron(self.num_neurons[i+1])
                elif (i == self.num_layers - 1):
                    self.layer[i].neuron[j] = create_neuron(self.num_neurons[self.num_layers - 1])
            # print(self.layer[i],'\n')
        self.init_weight()
    
    def init_weight(self):
        for i in range(0,self.num_layers - 1,1):
            for j in range(0,self.num_neurons[i],1):
                for k in range(0,self.num_neurons[i+1],1):
                    # init output weight for each neuron
                    self.layer[i].neuron[j].out_weights[k] = random()
                    self.layer[i].neuron[j].dw[k] = 0.0
                if i > 0 :
                    self.layer[i].neuron[j].bias = 1.0
        for j in range(0,self.num_neurons[self.num_layers - 1],1):
            self.layer[self.num_layers - 1].neuron[j].bias = 1.0
        for z in range(0, self.num_neurons[self.num_layers-1],1):
            self.layer[self.num_layers-1].neuron[z].out_weights[0] = 0.0
            self.layer[self.num_layers-1].neuron[z].dw[0] = 0.0
        
    def update_weights(self):
        for i in range(0,self.num_layers - 1,1):
            for j in range(0,self.num_neurons[i],1):
                for k in range(0,self.num_neurons[i+1],1):
                    # update weights
                    self.layer[i].neuron[j].out_weights[k] = (self.layer[i].neuron[j].out_weights[k]) - (self.learning_rate * self.layer[i].neuron[j].dw[k])
                    
    def forward(self, test):
        for i in range(1,self.num_layers,1):
            for j in range(0,self.num_neurons[i],1):
                self.layer[i].neuron[j].z = self.layer[i].neuron[j].bias
                for k in range(0,self.num_neurons[i-1],1):
                    # if len(self.layer[i-1].neuron[k].out_weights) > 0:
                    # print(self.layer[i-1].neuron[k].out_weights[j])
                    # print('self.layer[',i,'].neuron[j].z =',self.layer[i].neuron[j].z,'\n')
                    # print('self.layer[',i-1,'].neuron[k].out_weights[j] =',self.layer[i-1].neuron[k].out_weights[j],'\n')
                    # print('self.layer[',i-1,',].neuron[k].actv =',self.layer[i-1].neuron[k].actv,'\n')
                    self.layer[i].neuron[j].z = self.layer[i].neuron[j].z + ((self.layer[i-1].neuron[k].out_weights[j]) * (self.layer[i-1].neuron[k].actv))
                # for Hidden layers
                if i < (self.num_layers - 1):
                    
                    # This is ReLU
                    if (self.layer[i].neuron[j].z < 0.0):
                        self.layer[i].neuron[j].actv = 0.0
                    else:
                        self.layer[i].neuron[j].actv = self.layer[i].neuron[j].z
                    
                    # !!! This is Tanh
                    # self.layer[i].neuron[j].actv = tanh(self.layer[i].neuron[j].z)
                
                # for Output layers
                else:
                    # This is Sigmoid
                    self.layer[i].neuron[j].actv = 1.0 / (1.0 + (e ** (-1.0 * self.layer[i].neuron[j].z)))
                   
                    # !!! This is Tanh
                    # self.layer[i].neuron[j].actv = tanh(self.layer[i].neuron[j].z)
                    
                    if test :
                        print('Output:', (self.layer[i].neuron[j].actv))
                        print('Output (Rounded):', round(self.layer[i].neuron[j].actv),'\n')
    
    def compute_cost(self,i):
        tmpcost = 0.0
        tcost = 0.0
        for j in range(0,self.num_neurons[self.num_layers - 1],1):
            tmpcost = self.desired_outputs[i][j] - self.layer[self.num_layers - 1].neuron[j].actv
            self.cost[j] = (tmpcost * tmpcost) / 2.0
            tcost = tcost + self.cost[j]
        self.full_cost = (self.full_cost + tcost) / self.n
        self.n += 1.0
        
    def backprop(self,p):
        # Output layer
        for j in range(0,self.num_neurons[self.num_layers - 1],1):
            
            # This is Derivative Sigmoid
            self.layer[self.num_layers - 1].neuron[j].dz = (self.layer[self.num_layers - 1].neuron[j].actv - self.desired_outputs[p][j]) * (self.layer[self.num_layers - 1].neuron[j].actv) * (1.0 - self.layer[self.num_layers - 1].neuron[j].actv)
            
            # This is Derivative_Tanh
            # self.layer[self.num_layers - 1].neuron[j].dz = (self.layer[self.num_layers - 1].neuron[j].actv - self.desired_outputs[p][j]) * d_tanh(self.layer[self.num_layers - 1].neuron[j].z)
            
            for k in range(0,self.num_neurons[self.num_layers - 2],1):
                self.layer[self.num_layers - 2].neuron[k].dw[j] = self.momentum_rate * (self.layer[self.num_layers - 1].neuron[j].dz * self.layer[self.num_layers - 2].neuron[k].actv)
                self.layer[self.num_layers - 2].neuron[k].dactv = self.layer[self.num_layers - 2].neuron[k].out_weights[j] * self.layer[self.num_layers - 1].neuron[j].dz
            # self.layer[self.num_layers - 1].neuron[j].dbias = self.layer[self.num_layers - 1].neuron[j].dz
        
        # Hidden Layer
        for i in range((self.num_layers - 2),0,-1):
            for j in range(0,self.num_neurons[i],1):
                
                # This is Derivative ReLU
                if (self.layer[i].neuron[j].z >= 0):
                    self.layer[i].neuron[j].dz = self.layer[i].neuron[j].dactv
                else:
                    self.layer[i].neuron[j].dz = 0.0
                
                # This is Derivative_Tanh
                # self.layer[i].neuron[j].dz = d_tanh(self.layer[i].neuron[j].z)
                
                for k in range(0,self.num_neurons[i-1],1):
                    self.layer[i - 1].neuron[k].dw[j] = self.momentum_rate * self.layer[i].neuron[j].dz * self.layer[i-1].neuron[k].actv
                    if i > 1:
                        self.layer[i - 1].neuron[k].dactv = self.layer[i - 1].neuron[k].out_weights[j] * self.layer[i].neuron[j].dz
                # self.layer[i].neuron[j].dbias = self.layer[i].neuron[j].dz
    
    def load_input(self,i):
        for j in range(0,self.num_neurons[0],1):
            self.layer[0].neuron[j].actv = self.data_inputs[i][j]
        # print(self.layer[0])
        
    def train(self):
        for iterator in range(0,(self.epoch + 1),1):
            for i in range(0,self.lengthdata,1):
                self.load_input(i)
                self.forward(False)
                self.compute_cost(i)
                self.backprop(i)
                self.update_weights()
            self.nowEpoch = iterator
    
    def printlayer(self):
        print('---------------------------------------------- Neural Network Details Epoch =',self.nowEpoch,'----------------------------------------------\n')
        for i in range(0, len(self.layer),1):
            print(self.layer[i],'\n')
        print('---------------------------------------------- Neural Network Details Epoch =',self.nowEpoch,'----------------------------------------------\n')
            
    def test(self):
        i = 0
        print('---------------------------------------------- Testing trained neural network ----------------------------------------------\n')
        for k in range(self.lengthdata):
            for i in range(self.num_neurons[0]):
                print('Input',i,':', self.data_inputs[k][i])
                self.layer[0].neuron[i].actv = self.data_inputs[k][i]
            self.forward(True)
        print('---------------------------------------------- Testing trained neural network ----------------------------------------------\n')