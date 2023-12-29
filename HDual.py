import numpy as np
from math import comb

class dual(object):
    """
        Class for dual numbers in python. These are numbers of the form
        a + be where a and b are both real numbers and e^2=0, similar to
        complex numbers. These numbers are helpful in automatic derivatives.

        Attributes
        ----------
        re: real part of dual number ("a" in "a + be").
        im: imaginary part of dual number ("b" in "a + be"). Defaults to 0.
    """
    def __init__(self, re, im=0):
        self.re = re
        self.im = im

    def is_real(self):
        if np.is_close(self.im, 0):
            return True
        return False
        
    def __str__(self):
        return f"{self.re} + {self.im}e"
    
    def __repr__(self):
        return str(self)
    
    def __add__(self, other):
        if type(other) == dual:
            return dual(self.re + other.re, self.im + other.im)
        return dual(self.re + other, self.im)
        
    def __radd__(self, other):
        return self.__add__(other)
    
    def __neg__(self):
        return dual(-self.re, -self.im)
    
    def __sub__(self, other):
        if type(other) == dual:
            return dual(self.re - other.re, self.im - other.im)
        return dual(self.re - other, self.im)
    
    def __rsub__(self, other):
        return -self.__sub__(other)
        
    def __mul__(self, other):
        if type(other) == dual:
            return dual(self.re*other.re, self.im*other.re + self.re*other.im)
        return dual(self.re*other, self.im*other)
        
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def conj(self):
        return dual(self.re, -self.im)
    
    def __truediv__(self, other):
        if type(other) == dual:
            return dual(self.re/other.re, (self.im*other.re - self.re*other.im)/other.re**2)
        return dual(self.re/other, self.im/other)
    
    def __rtruediv__(self, other):
        if type(other) == dual:
            return other.__truediv__(self)
        return dual(other/self.re, -other*self.im/self.re**2)
    
    def __getitem(self, item):
        return [self.re, self.im][item]
    
class hdual(object):
    """
        Class for hyperdual numbers in Python. These are numbers of the form z_01 + z_1e1 + z_2e2 + ... + z_(n - 1)e(n - 1) + z_nen.
        According to Szirzay-Kalos in 2020, these can be multiplied with the rule 
        z*w = sum_{j = 0}^n sum_{k = 0}^{n - j} choose(j + k, k) z_jw_k e_{j + k}, a generalization of the multiplication rule for
        regular dual numbers. They are helpful for finding higher order derivatives.

        Attributes
        ----------
        value: iterable
            A list of the components of the number, starting with the real part then each of the imaginary parts.
    """
    def __init__(self, value):
        self.value = np.array(value)
        self.dim = len(self.value)

    def is_real(self):
        if np.allclose(self.value[1:], np.zeros(self.dim-1)):
            return True
        return False
    
    def __str__(self):
        return f"{self.value[0]}" + "".join(f" + {self.value[i]}e{i}" for i in range(1, self.dim))
    
    def __repr__(self):
        return self.__str__
    
    def __add__(self, other):
        if isinstance(other, hdual) and other.dim == self.dim: #TODO: decide what to do when adding with dual numbers of different dimensions
            return hdual(self.value + other.value)
        return hdual([self.value[0] + other, *self.value[1:]])
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __neg__(self):
        return dual(-self.value)
    
    def __sub__(self, other):
        if isinstance(other, hdual) and other.dim == self.dim:
            return hdual(self.value - other.value)
        return hdual([self.value[0] - other, *self.value[1:]])
    
    def __rsub__(self, other):
        return -self.__sub__(other)
    
    def __mul__(self, other):
        if isinstance(other, hdual) and other.dim == self.dim:
            prod = np.zeros(self.dim, dtype=self.value.dtype)
            for j in range(self.dim):
                for k in range(self.dim - j):
                    prod[j + k] += comb(j + k, k)*self.value[j]*other.value[k]
            return hdual(prod)
        return hdual(other*self.value)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def conj(self):
        return hdual([self.value[0], *(-self.value[1:])])
    
    def __truediv__(self, other):
        if isinstance(other, hdual) and other.dim == self.dim:
            if other.is_real():
                return hdual(self.value/other.value[0])
            else:
                return (self*other.conj())/(other*other.conj())
        return hdual(self.value/other)
    
    def __rtruediv__(self, other):
        if isinstance(other, hdual) and other.dim == self.dim:
            return other/self
        return (other*self.conj())/(self*self.conj())
    
    def __getitem__(self, item):
        return self.value[item]

class hdual_basis(hdual):
    def __init__(self, index, dim, component=1):
        self.value = np.zeros(dim)
        self.value[index] = component
        self.index = index
        self.dim = dim
        self.component = component
    
    def __str__(self):
        return f"{self.component}e{self.index}"
    
    def __repr__(self):
        return self.__str__

if __name__ == "__main__":
    pass
