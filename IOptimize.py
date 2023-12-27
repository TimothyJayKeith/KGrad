import TDerivs as TD
import numpy as np

def newton(funcs, guess, iters=50):
    """
        Performs Newton's rootfinding method on a collection of functions. The
        number of functions should equal the number of inputs of each function.

        Parameters
        ----------

        funcs: function or list
            A function or list of functions.
        guess: number or list
            The initial guess for Newton's method.
        iters: int
            The number of iterations before termination. Defaults to 50.

        Returns
        -------

        A numpy ndarray with approximation for root.
    """
    x = guess
    try:
        for i in range(iters):
            dmatrix = [TD.grad(f, x) for f in funcs]
            x = x - np.matmul(np.linalg.inv(dmatrix), [f(*x) for f in funcs])
    except TypeError:
        for i in range(iters):
            df = TD.oderiv1(f, x)
            x = x - f(x)/df
    return x

if __name__ == "__main__":
    pass