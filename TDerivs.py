from logging import warning
from typing import Iterable
from HDual import dual
import numpy as np

def oderiv1(f, x):
    """
        Calculates the derivative of a function f at point x using automatic differentiation
        with dual numbers. The functions should be formed from the functions in this file.
    """
    return f(dual(x, 1)).im

def pderiv1(f, inputs, partial_index=0):
    """
        Calculates the first partial derivative of a multivariate function f at a point in 
        n-dimensional space.

        Parameters
        ----------

        f: function
            The function whose derivative is being calculated.
        inputs: list
            The list of inputs at which the derivative is calculated. It must equal the number
            of inputs for f.
        partial_index: int
            The index of the variable whose partial derivative is being taken. Defaults to 0.

        Returns
        -------

        The value of the partial derivative of f at the given point in the given direction.
    """
    inputs[partial_index] = dual(inputs[partial_index], 1)
    return f(*inputs).im

def grad(f, inputs):
    """
        Calculates the derivative of a 1D function or the gradient of an ND function.

        Parameters
        ----------

        f: function
            The function whose derivative is being calculated.
        inputs: number or list
            If f is 1D, input a number. If f is ND, then input a list of inputs whose length 
            is the same as the number of inputs of f.
        
        Returns
        -------

        The value of a the derivative/gradient of a function at a point.
    """
    if type(inputs) == list:
        return [pderiv1(f, inputs, i) for i in range(len(inputs))]
    return oderiv1(f, inputs)

def dderiv(f, inputs, direction=None):
    """
        Derivative of a multivariate function f at a point in a given direction.

        Parameters
        ----------

        f: function
            The function whose derivative is to be calculated.
        inputs: list
            The list of points at which the derivative is evaluated. Its length must be equal 
            to the number of inputs of f.
        direction: iterable or None
            The direction of the derivative. If None, it evaluates in the direction of f (i.e.
            the direction of steepest ascent).
        
        Returns
        -------
        The value of the derivative of f in the given direction.
    """
    fgrad = grad(f, inputs)
    if direction is None:
        direction = fgrad
    direction_vector = np.array(direction)
    direction_vector = direction_vector/np.linalg.norm(direction_vector)
    return np.dot(fgrad, direction_vector)

if __name__ == "__main__":
    pass