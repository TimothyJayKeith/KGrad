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
        self.re = re
        self.im = im
        
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
    
    def __truediv__(self, other):
        if type(other) == dual:
            return dual(self.re/other.re, (self.im*other.re - self.re*other.im)/other.re**2)
        return dual(self.re/other, self.im/other)
    
    def __rtruediv__(self, other):
        if type(other) == dual:
            return other.__truediv__(self)
        return dual(other/self.re, -other*self.im/self.re**2)
    
if __name__ == "__main__":
    pass
