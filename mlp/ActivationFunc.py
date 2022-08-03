from utils.activationfunc import *

class ActivationFunc:
    def __init__(self, activationFn):
        self.activationFn = activationFn
    
    def activate(self,value):
        if self.activationFn == 1:
            return Linear(value)
        elif self.activationFn == 2:
            return Sigmoid(value)
        elif self.activationFn == 3:
            return Threshold(value)
        elif self.activationFn == 4:
            return Tanh(value)
        else:
            return 0.0
        
    def activateDerivative(self,value):
        if self.activationFn == 1:
            return dLinear(value)
        elif self.activationFn == 2:
            return dSigmoid(value)
        elif self.activationFn == 3:
            return dThreshold(value)
        elif self.activationFn == 4:
            return dTanh(value)
        else:
            return 0.0
        
def get_fn_name(act_num):
    if act_num == 1:
        return 'Linear'
    elif act_num == 2:
        return 'Sigmoid'
    elif act_num == 3:
        return 'Threshold'
    elif act_num == 4:
        return 'Hyperbolic Tangent'
    else:
        return 'None'        