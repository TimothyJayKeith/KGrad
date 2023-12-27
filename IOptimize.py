import TDerivs as TD
import numpy as np

def newton(funcs, guess, iters=50):
    """
        Performs Newton's rootfinding method on a collection of functions. The
        number of functions should equal the number of inputs of each function.

        Parameters
        ----------

        funcs: list
            A list of functions.
        guess: list
            A list containing the initial guess for Newton's method.
        iters: int
            The number of iterations before termination. Defaults to 50.

        Returns
        -------

        A numpy ndarray with approximation for root.
    """
    x = guess
    for i in range(iters):
        nmatrix = [TD.grad(f, x) for f in funcs]
        x = x - np.matmul(np.linalg.inv(nmatrix), [f(*x) for f in funcs])
    return x

if __name__ == "__main__":
    from AFuncs import*
    f1 = lambda x, y: x*y
    f2 = lambda x, y: pow(x, 2) + pow(y, 2)
    x = newton([f1, f2], [2, 1])
    print(x)
    print(f1(*x))
    print(f2(*x))