#!/usr/bin/python3.11

### TODO. Realize arbitrary finite field arithmetic

from sympy import gcdex
from decorators import cached

### This is stupid. REDO. On the other hand, we have only small primes?
@cached
def _is_prime(n):
    if not isinstance(n, int) or n < 2:
        return False
    if n % 2 == 0:
        return n == 2

    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2

    return True



class _PrimeFieldElementBase(int):
    """Base class for elements of a prime field."""

    __slots__ = ()

    def __new__(cls, value=0):
        return super().__new__(cls, int(value) % cls.order)

    def __repr__(self):
        return f"{int(self)} (mod {self.order})"

    def __str__(self):
        return f"{int(self)}"

    def _coerce2int(self, other):
        # Do not mix elements of different fields.
        if isinstance(type(other), PrimeField) and type(other) is not type(self):
            raise TypeError(
                f"cannot mix elements of {type(self)} and {type(other)}"
            )
        return int(other)% self.order

    def __add__(self, other):
        other = self._coerce2int(other)
        return type(self)(int(self) + other)

    __radd__ = __add__

    def __sub__(self, other):
        other = self._coerce2int(other)
        return type(self)(int(self) - other)

    def __rsub__(self, other):
        other = self._coerce2int(other)
        if other is NotImplemented:
            return NotImplemented
        return type(self)(other - int(self))

    def __mul__(self, other):
        other = self._coerce2int(other)
        return type(self)(int(self) * other)

    __rmul__ = __mul__

    def __neg__(self):
        return type(self)(-int(self))

    def inverse(self):
        if self == 0:
            raise ZeroDivisionError("0 has no multiplicative inverse")

        x, y, g = gcdex(self, self.order)
        return type(self)(x)

    def __truediv__(self, other):
        other = self._coerce2int(other)
        return self * type(self)(other).inverse()

    def __rtruediv__(self, other):
        other = self._coerce2int(other)
        return type(self)(other) * self.inverse()

    def __pow__(self, exponent):
        if not isinstance(exponent, int):
            raise ValueError("Exponent must me integer, not {type(exponent)}")
        if exponent > 0:
            return type(self)(pow(int(self), exponent, self.order))

        if self == 0:
            raise ZeroDivisionError(
                "0 cannot be raised to a nonpositive power"
            )
        return type(self)(pow(int(self.inverse()), -exponent, self.order))


    
class PrimeField(type):
    """Metaclass whose instances are prime-field element classes."""

    ### We will cache previous calls and return cached values for
    ### repeated calls with the same parameter.
    _cache = dict()
    
    def __new__(mcls, order):
        if not _is_prime(order):
            raise ValueError(f"{order} is not prime")

        ### include mcls in cache key, in case we subclass later
        key = (mcls,order)

        try:
            return mcls._cache[key]
        except KeyError:
            pass
        
        namespace = {
            "order": order,
            # "range": range(order),
            "__slots__": (),
            "__module__": __name__,
        }

        mcls._cache[key] = super().__new__(mcls,
                                           f"F_{order}",
                                           (_PrimeFieldElementBase,),
                                           namespace,
                                           )
        return mcls._cache[key]
    
    def __repr__(cls):
        return f"F_{cls.order}"
    
    __str__ = __repr__

    def range(self,nonzero=False):
        if nonzero:
            return range(1,self.order)
        return range(self.order)
