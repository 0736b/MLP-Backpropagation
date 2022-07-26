from MLP import *

def main():
    
    nn = MLP(4,[2,3,3,1],4,0.15,0.15,20000)
    nn.create_network()
    # nn.printlayer()
    # nn.get_inputs()
    # nn.get_desired_output()
    nn.train()
    nn.test()
    # nn.printlayer()
    
if __name__ == '__main__':
    main()