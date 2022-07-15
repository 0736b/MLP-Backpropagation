import random as rn
import utils.activationfunc as ActFn

# Multilayer Perceptron (Backpropagation)

class MLP:
    
    inputs = 8
    hiddens = 3
    outputs = 1
    
    def __init__(self,learning_rate,momentum_rate):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
    
    def init_weight(self):
        self.weights = {
            "input-to-hidden": [],
            "hidden-to-output": []
        }
        self.bias = 1
        for i in range(self.inputs):
            for j in range(self.hiddens):
                self.weights["input-to-hidden"].append(rn.random())     
        for i in range(self.hiddens):
            for j in range(self.outputs):
                self.weights["hidden-to-output"].append(rn.random())
        
        
    