import numpy as np
from warnings import warn
from math import comb

class dual(hdual):
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
        self.dim = 2
        self.value = np.array([re, im])

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
            if np.allclose([self.re, other.re], [0,0]):
                return self.im/other.im
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
        try:
            l = sorted((self.value, other.value), key=len)
            c = l[1].copy()
            c[:len(l[0])] += l[0] #This code is meant to handle adding of arrays of unlike dimension. This is courtesy of Tobia Marucci. Try and find a faster way to do this.
            return hdual(c)
        except AttributeError:
            return hdual([self.value[0] + other, *self.value[1:]])
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __neg__(self):
        return hdual(-self.value)
    
    def __sub__(self, other):
        return self + -other
    
    def __rsub__(self, other):
        return -self.__sub__(other)
    
    def __mul__(self, other):
        try:
            prod = np.zeros(max(self.dim, other.dim), dtype=self.value.dtype)
            for j in range(self.dim):
                try:
                    for k in range(max(self.dim - j, other.dim)):
                        prod[j + k] += comb(j + k, k)*self.value[j]*other.value[k]
                except IndexError:
                    continue
            return hdual(prod)
        except AttributeError:
            return hdual(other*self.value)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def conj(self):
        return hdual([self.value[0], *(-self.value[1:])])
    
    def __truediv__(self, other):
        try:
            if np.allclose([self[0], other[0]], [0, 0]):
                warn("0/0 encountered. Handling automatically.")
                return hdual(self[1:])/hdual(other[1:])
            if other.is_real():
                return self/other[0]
            return (self*other.conj())/(other*other.conj())
        except IndexError:
            return hdual(self.value/other)
    
    def __rtruediv__(self, other):
        if isinstance(other, hdual) and other.dim == self.dim:
            return other/self
        return (other*self.conj())/(self*self.conj())
    
    def __getitem__(self, item):
        return self.value[item]

class hdual_basis(hdual):
    def __init__(self, index, dim, component=1):
        self.value = np.zeros(dim, dtype=type(component))
        self.value[index] = component
        self.index = index
        self.dim = dim
        self.component = component
    
    def __str__(self):
        return f"{self.component}e{self.index}"
    
    def __repr__(self):
        return self.__str__

if __name__ == "__main__":
    z = hdual([0, 1, 2, 3, 4, 5])
    w = hdual([0, 2, 3])
    print(z/w)
