from mlp.Neuron import Neuron

class Layer:
    def __init__(self, neuronCount, weightCount, actFn):
        self.neuronCount = neuronCount
        self.weightCount = weightCount
        self.neurons = [None] * neuronCount
        for i in range(0, neuronCount, 1):
            self.neurons[i] = Neuron(weightCount, actFn)
        
    def init(self, l):
        if l == 0:
            for i in range(0, len(self.neurons), 1):
                self.neurons[i].setWeight(i, 1.0)
        for i in range(0, len(self.neurons),1):
            self.neurons[i].init()
            
    def calc(self, inp):
        self.output = [None] * len(self.neurons)
        for i in range(0, len(self.neurons),1):
            self.output[i] = self.neurons[i].calc(inp)
        return self.output
    
    def getNeuronCount(self):
        return self.neuronCount
    
    def getOutput(self):
        return self.output
    
    def setDelta(self, error):
        for i in range(0, len(self.neurons),1):
            self.neurons[i].setDelta(error[i])
            
    def calcError(self):
        self.error = [None] * self.weightCount
        for i in range(0, self.weightCount,1):
            for j in range(0, len(self.neurons),1):
                self.error[i] = self.neurons[j].getDelta() * self.neurons[j].getWeights()[i]
                
    def getError(self):
        return self.error
    
    def updateWeights(self, momentum, learning_rate, output):
        for i in range(0, len(self.neurons),1):
            self.neurons[i].updateWeights(momentum, learning_rate, output)