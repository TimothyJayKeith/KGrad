from HDual import dual

def oderiv1(f, x):
    """
        Calculates the derivative of a function f at point x using automatic differentiation
        with dual numbers. The functions should be formed from the functions in this file.
    """
    return f(dual(x, 1)).im

if __name__ == "__main__":
    from AFuncs import *
    import numpy as np
    f = lambda x: pow(cos(x), 3)
    print(f(np.pi))
    print(oderiv1(f, np.pi))