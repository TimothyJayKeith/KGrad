from AFuncs import pow
import TDerivs as TD
import numpy as np
import math
from HDual import hdual
import warnings

class tseries(object):
    def __init__(self, coeffs, center=0, num_terms=5):
        """
            A Taylor Series object.

            Attributes
            ----------
            coeffs: function or iterable
                Can specify either a function whose Taylor coefficients to solve for or an iterable
                whose entries are the coefficients. If it is a function, it should be built from the
                functions in AFuncs.
            center: float
                The center of the Taylor series. Defaults to 0 (Maclaurin series).
            num_terms: int
                Number of terms wanted in Taylor series if coeffs is a function. Unneeded if coeffs
                is an iterable. Defaults to 5.
        """
        if callable(coeffs):
            coeffs = TD.oderivn(coeffs, center, n=num_terms-1, return_lower_derivs=True)
        self.coeffs = coeffs
        self.center = center
        self.num_terms = len(coeffs)
    
    def __call__(self, x):
        return np.sum([self.coeffs[i]*pow(x - self.center, i)/math.factorial(i) for i in range(self.num_terms)])
    
    def __str__(self):
        if self.center != 0:
            return f"{self.coeffs[0]}" + "".join(f" + {self.coeffs[i]}(x - {self.center})^{i}/{i}!" for i in range(1, self.num_terms))
        return f"{self.coeffs[0]}" + "".join(f" + {self.coeffs[i]}x^{i}/{i}!" for i in range(1, self.num_terms))
    
    def __repr__(self):
        return self.__str__

def ssolve(coeff_funcs, initial_values=[], center=0, num_terms=5):
    """
        Produces a series solution for a linear ordinary differential equation (ODE) of the form 
        f^(n)(x) + g_n(x)f^(n - 1)(x) + ... + g_2(x)f'(x) + g_1(x)f(x) = g_0(x)

        Parameters
        ----------
        coeff_funcs: function or list of functions.
            The functions g_0, g_1, g_2, ..., g_n in the ODE above. if only a single
            function is specified, it will simply return the Taylor series for that function.
        intial_values: list
            A list of the initial values for f(x_0), f'(x_0), ..., f^(n - 1)(x_0). There should always
            be exactly one less initial value than coeff_func.
        center: float
            The initial point of the ODE x_0. Defaults to 0.
        num_terms: int
            The number of terms in the Taylor series. Defaults to 5. Keep in mind the first few terms will 
            always be the initial values of the ODE.

        Returns
        -------
        The taylor series of the function f in the ODE centered on x_0 in the form of a KGrad tseries object.
    """
    try:
        iter(coeff_funcs)
    except TypeError:
        return tseries(coeff_funcs, center=center, num_terms=num_terms)
    if len(coeff_funcs) == 1:
        return tseries(coeff_funcs[0])
    
    if len(initial_values) != len(coeff_funcs) - 1:
        raise Exception("The number of initial values should be one less than the number of coefficient functions.")
    coeffs = initial_values.copy()
    if len(coeffs) >= num_terms:
        warnings.warn("Number of initial conditions equals or exceeds number of terms specified. Returning initial values only.")
        return coeffs
    
    deriv_list = [center, 1] + [0]*(num_terms-len(coeffs)-1)
    for i in range(num_terms - len(coeffs)):
        tdual = hdual(deriv_list[0:i+1])
        coeffs.append(np.sum([coeff_funcs[0](tdual)] +
        [-coeff_funcs[j](tdual)*hdual(coeffs[j-1:j+i]) 
         for j in range(1, len(coeff_funcs))])[i])
    return tseries(coeffs, center=center)
        
if __name__ == "__main__":
    pass
