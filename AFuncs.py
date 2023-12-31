import numpy as np
import cmath
import math
from HDual import dual, hdual, hdual_basis

def pow(x, p):
    """
        Calculates x^p, where x is a float or dual and p is an integer.
    """
    if type(x) == hdual_basis:
        if x.index*p >= x.dim:
            return hdual_basis(0, x.dim, 0)
        return hdual_basis(x.index*p, x.dim, (math.factorial(p*x.index)/pow(math.factorial(x.index), p))*pow(x.component, p))
    if type(x) == hdual:
        if p == 0:
            return hdual([1] + [0]*(x.dim - 1))
        return np.prod([x]*p)
    if type(x) == dual:
        return dual(pow(x.re, p), x.im*p*pow(x.re, p-1))
    return x**p

def exp(x, complex=False):
    """
        Calculates e^x, where x is a float or dual.
    """
    if isinstance(x, hdual):
        return np.prod([exp(x[0], complex=complex)] + 
                       [np.sum([pow(hdual_basis(i, x.dim, x[i]), j)/math.factorial(j) 
                                for j in range(math.floor(x.dim/i) + 1)]) 
                                for i in range(1, x.dim)])
    if type(x) == dual:
        return dual(exp(x.re), x.im*exp(x.re))
    if complex:
        return cmath.exp(x)
    return math.exp(x)

def sin(x):
    """
        Calculates sin(x), where x is a float or dual.
    """
    if type(x) == hdual:
        if x.dim == 1:
            return sin(x[0])
        cexp = exp(1j*x, complex=True)
        return hdual([cexp[i].imag for i in range(x.dim)])
    if type(x) == dual:
        return dual(np.sin(x.re), x.im*np.cos(x.re))
    return math.sin(x)
    
def cos(x):
    """
        Calculates cos(x), where x is a float or dual.
    """
    if type(x) == hdual:
        if x.dim == 1:
            return cos(x[0])
        cexp = exp(1j*x, complex=True)
        return hdual([cexp[i].real for i in range(x.dim)])
    if type(x) == dual:
        return dual(np.cos(x.re), -x.im*np.sin(x.re))
    return math.cos(x)

if __name__ == "__main__":
    print(-sin(hdual([1, 1])))