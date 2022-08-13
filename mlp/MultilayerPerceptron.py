from mlp.Layer import Layer
from math import sqrt

class MultilayerPerceptron:
    def __init__(self, neuronCount, activationFnList, momentum, learning_rate):
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.layers = [None] * len(neuronCount)
        self.diff_sum = 0.0
        prevNeuronCount = neuronCount[0]
        for i in range(0, len(neuronCount),1):
            self.layers[i] = Layer(neuronCount[i], prevNeuronCount, activationFnList[i])
            self.layers[i].initialize(i)
            prevNeuronCount = neuronCount[i]

    def feedForward(self,inp):
        output = inp
        for i in range(0, len(self.layers),1):
            output = self.layers[i].compute(inp)
            inp = output
        return output
        
    def backpropagate(self, actualOutput, expectedOutput):
        outputLayer = self.layers[len(self.layers) - 1]
        error = [None] * outputLayer.getNeuronCount()
        for i in range(0, len(error),1):
            error[i] = expectedOutput[i] - actualOutput[i]
        for i in range(len(self.layers) - 1, 0, -1):
            self.layers[i].setDelta(error)
            self.layers[i].calculateError()
            error = self.layers[i].getError()
            self.layers[i].updateWeights(self.momentum, self.learning_rate, self.layers[i - 1].getOutput())
    
    def train(self, inp, expectedOutput, datapos, lengthdata, epoch, mode, cm):
        if mode == 'predict':
            error = 0.0
            actualOutput = self.feedForward(inp)
            n = datapos + 1
            self.backpropagate(actualOutput, expectedOutput)
            for i in range(0, len(expectedOutput), 1):
                self.diff_sum += pow(expectedOutput[i] - actualOutput[i], 2)
            if datapos + 1 == lengthdata:
                # print('epoch :',epoch,'error :', sqrt((1.0 / n) * self.diff_sum))
                error = sqrt((1.0 / n) * self.diff_sum)
                self.diff_sum = 0.0
                return error
        
        elif mode == 'classify':
            actualOutput = self.feedForward(inp)
            t_actual = actualOutput.copy()
            self.backpropagate(actualOutput, expectedOutput)
            if t_actual[0] > t_actual[1]:
                t_actual[0] = 1
                t_actual[1] = 0
            elif t_actual[0] < t_actual[1]:
                t_actual[0] = 0
                t_actual[1] = 1
            cm.add_data(expectedOutput, t_actual)
            if datapos + 1 == lengthdata:
                cm.calc_column()
                acc = cm.get_accuracy()
                return acc
            
        
    def test(self, inp):
        return self.feedForward(inp)    