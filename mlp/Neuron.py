from mlp.ActivationFunc import ActivationFunc
from random import uniform

class Neuron:
    def __init__(self, weightCount, actFn):
        self.output = 0.0
        self.weights = [0.0] * weightCount
        self.bias = 0.0
        self.delta = 0.0
        self.weightDiff = [0.0] * weightCount
        self.fn = ActivationFunc(actFn)
        
    def init(self):
        self.bias = 1.0
        for i in range(0, len(self.weights),1):
            t_weight = round(uniform(-1,1), 2)
            if t_weight == 0.00:
                t_weight = 0.01
            self.weights[i] = t_weight
            
    def calc(self, inp):
        sum = self.bias
        for i in range(0, len(self.weights),1):
            sum = sum + inp[i] * self.weights[i]
        self.output = self.fn.act(sum)
        return self.output
    
    def getOutput(self):
        return self.output
    
    def getWeights(self):
        return self.weights
    
    def setWeight(self, i, weight):
        self.weights[i] = weight
        
    def setDelta(self, error):
        self.delta = error * self.fn.actDeriv(self.output)
        
    def getDelta(self):
        return self.delta
    
    def getBias(self):
        return self.bias
    
    def updateWeights(self, momentum, learning_rate, prevOutput):
        for i in range(0, len(self.weights),1):
            self.weightDiff[i] = momentum * self.weightDiff[i] + (learning_rate * self.delta * prevOutput[i])
            self.weights[i] = self.weights[i] + self.weightDiff[i]