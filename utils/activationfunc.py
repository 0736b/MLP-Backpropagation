e = 2.718281828459045

# Hyperbolic Tangent function
def tanh(x):
    x = float(x)
    minus_x = x * -1.0
    return (((e ** x) - (e ** minus_x)) / ((e ** x) + (e ** minus_x)))

# Derivative of Hyperbolic Tangent
def d_tanh(x):
    x = float(x)
    return ((1.0 - (tanh(x) * tanh(x))))
