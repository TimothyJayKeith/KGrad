import numpy as np
import math
from HDual import dual

def pow(x, p):
    """
        Calculates x^p, where x is a float or dual and p is an integer. Optimized for use with dual numbers.
    """
    if type(x) == dual:
        return dual(math.pow(x.re, p), x.im*p*math.pow(x.re, p-1))
    return math.pow(x, p)

def exp(x):
    """
        Calculates e^x, where x is a float or dual.
    """
    if type(x) == dual:
        return dual(np.exp(x.re), x.im*np.exp(x.re))
    return np.exp(x)

def sin(x):
    """
        Calculates sin(x), where x is a float or dual.
    """
    if type(x) == dual:
        return dual(np.sin(x.re), x.im*np.cos(x.re))
    return np.sin(x)
    
def cos(x):
    """
        Calculates cos(x), where x is a float or dual.
    """
    if type(x) == dual:
        return dual(np.cos(x.re), -x.im*np.sin(x.re))
    return np.cos(x)

if __name__ == "__main__":
    pass