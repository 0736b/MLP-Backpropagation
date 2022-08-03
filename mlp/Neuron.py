from mlp.ActivationFunc import ActivationFunc
from random import random

class Neuron:
    def __init__(self, weightCount, activationFn):
        self.output = 0.0
        self.weights = [0.0] * weightCount
        self.bias = 0.0
        self.biasDifference = 0.0
        self.delta = 0.0
        self.weightDifference = [0.0] * weightCount
        self.fn = ActivationFunc(activationFn)
        
    def initialize(self):
        self.bias = round(random(), 2)
        for i in range(0, len(self.weights),1):
            self.weights[i] = round(random(), 2)
            
    def compute(self, inp):
        sum = self.bias
        for i in range(0, len(self.weights),1):
            sum = sum + inp[i] * self.weights[i]
        self.output = self.fn.activate(sum)
        return self.output
    
    def getOutput(self):
        return self.output
    
    def getWeights(self):
        return self.weights
    
    def setWeight(self, i, weight):
        self.weights[i] = weight
        
    def setDelta(self, error):
        self.delta = error * self.fn.activateDerivative(self.output)
        
    def getDelta(self):
        return self.delta
    
    def getBias(self):
        return self.bias
    
    def updateWeights(self, momentum, learning_rate, prevOutput):
        self.biasDifference = momentum * self.biasDifference + learning_rate * self.delta
        self.bias = self.bias + self.biasDifference
        for i in range(0, len(self.weights),1):
            self.weightDifference[i] = momentum * self.weightDifference[i] + learning_rate * self.delta * prevOutput[i]
            self.weights[i] = self.weights[i] + self.weightDifference[i]