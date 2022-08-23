from mlp.Layer import Layer
from math import sqrt
import copy

class MultilayerPerceptron:
    def __init__(self, neuronCount, activationFnList, momentum, learning_rate):
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.layers = [None] * len(neuronCount)
        self.diff_sum = 0.0
        prevNeuronCount = neuronCount[0]
        for i in range(0, len(neuronCount),1):
            self.layers[i] = Layer(neuronCount[i], prevNeuronCount, activationFnList[i])
            self.layers[i].init(i)
            prevNeuronCount = neuronCount[i]

    def forward_pass(self,inp):
        output = inp
        for i in range(0, len(self.layers),1):
            output = self.layers[i].calc(inp)
            inp = output
        return output
        
    def backward_pass(self, actualOutput, desiredOutput):
        outputLayer = self.layers[len(self.layers) - 1]
        error = [None] * outputLayer.getNeuronCount()
        for i in range(0, len(error),1):
            error[i] = desiredOutput[i] - actualOutput[i]
        t_error = copy.deepcopy(error)
        for i in range(len(self.layers) - 1, 0, -1):
            self.layers[i].setDelta(error)
            self.layers[i].calcError()
            error = self.layers[i].getError()
            self.layers[i].updateWeights(self.momentum, self.learning_rate, self.layers[i - 1].getOutput())
        return t_error
    
    def train(self, inp, desiredOutput, datapos, lengthdata, mode, cm):
        if mode == 'regression':
            error = 0.0
            actualOutput = self.forward_pass(inp)
            n = datapos + 1
            t_err = self.backward_pass(actualOutput, desiredOutput)
            for e in t_err:
                self.diff_sum += e ** 2
            error = sqrt(self.diff_sum / n)
            if n == lengthdata:
                error = sqrt(self.diff_sum / n)
                self.diff_sum = 0.0
                return error
        elif mode == 'classification':
            acc = 0.0
            actualOutput = self.forward_pass(inp)
            t_actual = actualOutput.copy()
            self.backward_pass(actualOutput, desiredOutput)
            if t_actual[0] > t_actual[1]:
                t_actual[0] = 1
                t_actual[1] = 0
            elif t_actual[0] < t_actual[1]:
                t_actual[0] = 0
                t_actual[1] = 1
            cm.add_data(desiredOutput, t_actual)
            if (datapos + 1) == lengthdata:
                cm.calc_column()
                acc = cm.get_accuracy()
                return acc
            
    def test(self, inp):
        return self.forward_pass(inp)    