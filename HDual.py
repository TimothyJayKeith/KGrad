from decimal import DivisionByZero
import numpy as np

class dual(object):
    """
        Class for dual numbers in python. These are numbers of the form
        a + be where a and b are both real numbers and e^2=0, similar to
        complex numbers. These numbers are helpful in automatic derivatives.
        
        ...

        Attributes
        ----------
        re: real part of dual number ("a" in "a + be").
        im: imaginary part of dual number ("b" in "a + be"). Defaults to 0.
    """
    def __init__(self, re, im=0):
        self.value = np.array([re, im])
        
    def __str__(self):
        return f"{self.value[0]} + {self.value[1]}e"
    
    def __repr__(self):
        return str(self)
    
    def __add__(self, other):
        if type(other) == dual:
            re, im = self.value + other.value
            return dual(re, im)
        return dual(self.value[0] + other, self.value[1])
        
    def __radd__(self, other):
        return self.__add__(other)
    
    def __neg__(self):
        re, im = -self.value
        return dual(re, im)
    
    def __sub__(self, other):
        if type(other) == dual:
            re, im = self.value - other.value
            return dual(re, im)
        return dual(self.value[0] - other, self.value[1])
    
    def __rsub__(self, other):
        return -self.__sub__(other)
        
    def __mul__(self, other):
        if type(other) == dual:
            return dual(self.value[0]*other.value[0], self.value[1]*other.value[0] + self.value[0]*other.value[1])
        re, im = other*self.value
        return dual(re, im)
        
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        if type(other) == dual:
            if np.isclose(other.value[0], 0):
                raise DivisionByZero("Real part of dual quotient must be non-zero")
            return dual(self.value[0]/other.value[0], (self.value[1]*other.value[0] - self.value[0]*other.value[1])/other.value[0]**2)
        re, im = self.value/other
        return dual(re, im)
    
    def __rtruediv__(self, other):
        if type(other) == dual:
            return other.__truediv__(self)
        return dual(other/self.value[0], other*self.value[1]/self.value[0]**2)
    
if __name__ == "__main__":
    pass
