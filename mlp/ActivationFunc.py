from utils.activationfunc import *

class ActivationFunc:
    def __init__(self, actFn):
        self.actFn = actFn
    
    def act(self,value):
        if self.actFn == 1:
            return Linear(value)
        elif self.actFn == 2:
            return Sigmoid(value)
        elif self.actFn == 3:
            return Tanh(value)
        elif self.actFn == 4:
            return Relu(value)
        else:
            return value
        
    def actDeriv(self,value):
        if self.actFn == 1:
            return dLinear(value)
        elif self.actFn == 2:
            return dSigmoid(value)
        elif self.actFn == 3:
            return dTanh(value)
        elif self.actFn == 4:
            return dRelu(value)
        else:
            return 1.0
        
def get_fn_name(act_num):
    if act_num == 1:
        return 'Linear'
    elif act_num == 2:
        return 'Sigmoid'
    elif act_num == 3:
        return 'Hyperbolic Tangent'
    elif act_num == 4:
        return 'ReLU'
    else:
        return 'None'        