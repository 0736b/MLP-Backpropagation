from MLP import *
from utils.datareader import get_datafromfile

def main():
    
    # Setup Neural Network
    
    data_inputs, desired_outputs, lengthdata, inps, outs = get_datafromfile('xor.txt')
    
    layers = 4
    neurons_per_layer = [inps,3,3,outs]
    learning_rate = 0.15
    momentum_rate = 0.5
    max_epoch = 20000
    
    # Running Neural Network
    nn = MLP(layers, neurons_per_layer, learning_rate, momentum_rate, max_epoch, lengthdata, data_inputs, desired_outputs)
    nn.create_network()
    nn.printlayer()
    nn.train()
    nn.test()
    nn.printlayer()
    
    

if __name__ == '__main__':
    main()