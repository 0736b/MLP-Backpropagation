import math

def Tanh(value):
    up = math.exp(value) - math.exp(-value)
    down = math.exp(value) + math.exp(-value)
    return float(up / down)

def dTanh(value):
    return float(1.0 - pow(Tanh(value), 2))

def Linear(value):
    return float(value)

def dLinear(value):
    return 1.0

def Sigmoid(value):
    return float((1 / (1 + math.exp(-value))))

def dSigmoid(value):
    return float((value * (1 - value)))

def Relu(value):
    return max(0, value)

def dRelu(value):
    if value <= 0:
        return 0
    else:
        return 1