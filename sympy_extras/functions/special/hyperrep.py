"""The representing functions of hyperexpand's formula tables: a
``HyperRep_*(params, z)`` is a hypergeometric function whose closed form
depends on the branch of ``z`` (``z`` polar: ``z*exp_polar(2*pi*I*n)``),
kept unevaluated while hyperexpand computes, and replaced by its closed
form at the end (``rewrite_nonrep``: the small-argument form, or a
``Piecewise`` of the big-argument form for ``|z| > 1`` and the small one).

Ported from SymPy's ``sympy.functions.special.hyper``. The four closed
forms of a class are classmethods taking the parameters (a tuple, empty
for the functions without one), the argument and, for the big forms, the
branch number; SymPy's ``rewrite('nonrep')`` and
``rewrite('nonrepsmall')`` are :func:`rewrite_nonrep`.

Examples
========

>>> from sympy_extras.core import Symbol, Rational
>>> from sympy_extras.functions.special.hyperrep import HyperRep_power1, HyperRep_atanh, rewrite_nonrep
>>> a, z = Symbol('a'), Symbol('z')
>>> rewrite_nonrep(HyperRep_power1(a, z)), rewrite_nonrep(HyperRep_atanh(z))
((1 - z)**a, atanh(sqrt(z))/sqrt(z))
>>> rewrite_nonrep(HyperRep_power1(a, z), small=False)
Piecewise(((z - 1)**a*exp(-I*pi*a), Abs(z) > 1), ((1 - z)**a, True))
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/functions/special/hyper.py (HyperRep and its
# subclasses) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: typed; the closed forms take the parameters as a tuple; the
# rewrites 'nonrep' and 'nonrepsmall' are the function rewrite_nonrep.
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Optional

from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.function import DefinedFunction
from sympy_extras.core.numbers import I, Integer, Rational, pi
from sympy_extras.core.power import sqrt

__all__ = ['HyperRep', 'HyperRep_power1', 'HyperRep_power2', 'HyperRep_log1', 'HyperRep_atanh', 'HyperRep_asin1',
           'HyperRep_asin2', 'HyperRep_sqrts1', 'HyperRep_sqrts2', 'HyperRep_log2', 'HyperRep_cosasin',
           'HyperRep_sinasin', 'rewrite_nonrep']

Params = tuple[Expr, ...]
ONE = Integer(1)
HALF = Rational(1, 2)


# PORTED FROM SYMPY: sympy/functions/special/hyper.py:HyperRep
class HyperRep(DefinedFunction):
    """A hypergeometric function in terms of elementary functions, with
    the closed forms for small and big arguments and their branches;
    the parameters (all but the last argument) unpolarified."""

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        from sympy_extras.functions.elementary.complexes import unpolarify
        newargs = tuple(unpolarify(a) for a in args[:-1]) + args[-1:]
        if newargs != args:
            return cls(*newargs)
        return None

    @classmethod
    def _small(cls, p: Params, x: Expr) -> Expr:
        raise NotImplementedError

    @classmethod
    def _small_minus(cls, p: Params, x: Expr) -> Expr:
        raise NotImplementedError

    @classmethod
    def _big(cls, p: Params, x: Expr, n: Expr) -> Expr:
        raise NotImplementedError

    @classmethod
    def _big_minus(cls, p: Params, x: Expr, n: Expr) -> Expr:
        raise NotImplementedError

    def _params(self) -> Params:
        return tuple(a for a in self.args[:-1] if isinstance(a, Expr))

    def _argument(self) -> Expr:
        z = self.args[-1]
        assert isinstance(z, Expr)
        return z

    # PORTED FROM SYMPY: sympy/functions/special/hyper.py:HyperRep._eval_rewrite_as_nonrep
    def nonrep(self) -> Expr:
        """The closed form on every branch: the big form for ``|x| > 1``."""
        from sympy_extras.functions.elementary.complexes import abs_
        from sympy_extras.functions.elementary.piecewise import piecewise
        x, n = self._argument().extract_branch_factor(allow_half=True)
        p = self._params()
        if not isinstance(n, Integer):
            n = n - HALF
            small, big = self._small_minus(p, x), self._big_minus(p, x, n)
        else:
            small, big = self._small(p, x), self._big(p, x, n)
        if big == small:
            return small
        return piecewise((big, abs_(x) > 1), (small, True))

    # PORTED FROM SYMPY: sympy/functions/special/hyper.py:HyperRep._eval_rewrite_as_nonrepsmall
    def nonrepsmall(self) -> Expr:
        """The closed form for small arguments."""
        x, n = self._argument().extract_branch_factor(allow_half=True)
        if not isinstance(n, Integer):
            return self._small_minus(self._params(), x)
        return self._small(self._params(), x)

    def _eval_facts(self) -> dict[str, bool]:
        return {'commutative': True}


def _exp(e: Expr) -> Expr:
    from sympy_extras.functions.elementary.exponential import exp
    return exp(e)


def _log(e: Expr) -> Expr:
    from sympy_extras.functions.elementary.exponential import log
    return log(e)


# PORTED FROM SYMPY: sympy/functions/special/hyper.py:HyperRep_power1
class HyperRep_power1(HyperRep):
    """``(1 - z)**a``."""
    function_name = 'HyperRep_power1'
    nargs = frozenset({2})
    call_nargs = frozenset({2})

    @classmethod
    def _small(cls, p: Params, x: Expr) -> Expr:
        return (1 - x)**p[0]

    @classmethod
    def _small_minus(cls, p: Params, x: Expr) -> Expr:
        return (1 + x)**p[0]

    @classmethod
    def _big(cls, p: Params, x: Expr, n: Expr) -> Expr:
        a = p[0]
        if a.is_integer is True:
            return cls._small(p, x)
        return (x - 1)**a * _exp((2*n - 1)*pi*I*a)

    @classmethod
    def _big_minus(cls, p: Params, x: Expr, n: Expr) -> Expr:
        a = p[0]
        if a.is_integer is True:
            return cls._small_minus(p, x)
        return (1 + x)**a * _exp(2*n*pi*I*a)


# PORTED FROM SYMPY: sympy/functions/special/hyper.py:HyperRep_power2
class HyperRep_power2(HyperRep):
    """``2**(2*a - 1)*(1 + sqrt(1 - z))**(1 - 2*a)``."""
    function_name = 'HyperRep_power2'
    nargs = frozenset({2})
    call_nargs = frozenset({2})

    @classmethod
    def _small(cls, p: Params, x: Expr) -> Expr:
        a = p[0]
        return 2**(2*a - 1) * (1 + sqrt(1 - x))**(1 - 2*a)

    @classmethod
    def _small_minus(cls, p: Params, x: Expr) -> Expr:
        a = p[0]
        return 2**(2*a - 1) * (1 + sqrt(1 + x))**(1 - 2*a)

    @classmethod
    def _big(cls, p: Params, x: Expr, n: Expr) -> Expr:
        a = p[0]
        sgn = -1
        if n.is_odd is True:
            sgn = 1
            n = n - 1
        return 2**(2*a - 1) * (1 + sgn*I*sqrt(x - 1))**(1 - 2*a) * _exp(-2*n*pi*I*a)

    @classmethod
    def _big_minus(cls, p: Params, x: Expr, n: Expr) -> Expr:
        a = p[0]
        sgn = 1
        if n.is_odd is True:
            sgn = -1
        return sgn * 2**(2*a - 1) * (sqrt(1 + x) + sgn)**(1 - 2*a) * _exp(-2*pi*I*a*n)


# PORTED FROM SYMPY: sympy/functions/special/hyper.py:HyperRep_log1
class HyperRep_log1(HyperRep):
    """``log(1 - z)``."""
    function_name = 'HyperRep_log1'
    nargs = frozenset({1})

    @classmethod
    def _small(cls, p: Params, x: Expr) -> Expr:
        return _log(1 - x)

    @classmethod
    def _small_minus(cls, p: Params, x: Expr) -> Expr:
        return _log(1 + x)

    @classmethod
    def _big(cls, p: Params, x: Expr, n: Expr) -> Expr:
        return _log(x - 1) + (2*n - 1)*pi*I

    @classmethod
    def _big_minus(cls, p: Params, x: Expr, n: Expr) -> Expr:
        return _log(1 + x) + 2*n*pi*I


# PORTED FROM SYMPY: sympy/functions/special/hyper.py:HyperRep_atanh
class HyperRep_atanh(HyperRep):
    """``atanh(sqrt(z))/sqrt(z)``."""
    function_name = 'HyperRep_atanh'
    nargs = frozenset({1})

    @classmethod
    def _small(cls, p: Params, x: Expr) -> Expr:
        from sympy_extras.functions.elementary.hyperbolic import atanh
        return atanh(sqrt(x)) / sqrt(x)

    @classmethod
    def _small_minus(cls, p: Params, x: Expr) -> Expr:
        from sympy_extras.functions.elementary.trigonometric import atan
        return atan(sqrt(x)) / sqrt(x)

    @classmethod
    def _big(cls, p: Params, x: Expr, n: Expr) -> Expr:
        from sympy_extras.functions.elementary.hyperbolic import acoth
        if n.is_even is True:
            return (acoth(sqrt(x)) + I*pi/2) / sqrt(x)
        return (acoth(sqrt(x)) - I*pi/2) / sqrt(x)

    @classmethod
    def _big_minus(cls, p: Params, x: Expr, n: Expr) -> Expr:
        from sympy_extras.functions.elementary.trigonometric import atan
        if n.is_even is True:
            return atan(sqrt(x)) / sqrt(x)
        return (atan(sqrt(x)) - pi) / sqrt(x)


# PORTED FROM SYMPY: sympy/functions/special/hyper.py:HyperRep_asin1
class HyperRep_asin1(HyperRep):
    """``asin(sqrt(z))/sqrt(z)``."""
    function_name = 'HyperRep_asin1'
    nargs = frozenset({1})

    @classmethod
    def _small(cls, p: Params, x: Expr) -> Expr:
        from sympy_extras.functions.elementary.trigonometric import asin
        return asin(sqrt(x)) / sqrt(x)

    @classmethod
    def _small_minus(cls, p: Params, x: Expr) -> Expr:
        from sympy_extras.functions.elementary.hyperbolic import asinh
        return asinh(sqrt(x)) / sqrt(x)

    @classmethod
    def _big(cls, p: Params, x: Expr, n: Expr) -> Expr:
        from sympy_extras.functions.elementary.hyperbolic import acosh
        return Integer(-1)**n * ((HALF - n)*pi/sqrt(x) + I*acosh(sqrt(x))/sqrt(x))

    @classmethod
    def _big_minus(cls, p: Params, x: Expr, n: Expr) -> Expr:
        from sympy_extras.functions.elementary.hyperbolic import asinh
        return Integer(-1)**n * (asinh(sqrt(x))/sqrt(x) + n*pi*I/sqrt(x))


# PORTED FROM SYMPY: sympy/functions/special/hyper.py:HyperRep_asin2
class HyperRep_asin2(HyperRep):
    """``asin(sqrt(z))/sqrt(z)/sqrt(1 - z)``."""
    function_name = 'HyperRep_asin2'
    nargs = frozenset({1})

    @classmethod
    def _small(cls, p: Params, x: Expr) -> Expr:
        return HyperRep_asin1._small((), x) / HyperRep_power1._small((HALF,), x)

    @classmethod
    def _small_minus(cls, p: Params, x: Expr) -> Expr:
        return HyperRep_asin1._small_minus((), x) / HyperRep_power1._small_minus((HALF,), x)

    @classmethod
    def _big(cls, p: Params, x: Expr, n: Expr) -> Expr:
        return HyperRep_asin1._big((), x, n) / HyperRep_power1._big((HALF,), x, n)

    @classmethod
    def _big_minus(cls, p: Params, x: Expr, n: Expr) -> Expr:
        return HyperRep_asin1._big_minus((), x, n) / HyperRep_power1._big_minus((HALF,), x, n)


# PORTED FROM SYMPY: sympy/functions/special/hyper.py:HyperRep_sqrts1
class HyperRep_sqrts1(HyperRep):
    """``((1 - sqrt(z))**(2*a) + (1 + sqrt(z))**(2*a))/2``."""
    function_name = 'HyperRep_sqrts1'
    nargs = frozenset({2})
    call_nargs = frozenset({2})

    @classmethod
    def _small(cls, p: Params, x: Expr) -> Expr:
        a = p[0]
        return ((1 - sqrt(x))**(2*a) + (1 + sqrt(x))**(2*a)) / 2

    @classmethod
    def _small_minus(cls, p: Params, x: Expr) -> Expr:
        from sympy_extras.functions.elementary.trigonometric import atan, cos
        a = p[0]
        return (1 + x)**a * cos(2*a*atan(sqrt(x)))

    @classmethod
    def _big(cls, p: Params, x: Expr, n: Expr) -> Expr:
        a = p[0]
        if n.is_even is True:
            return ((sqrt(x) + 1)**(2*a)*_exp(2*pi*I*n*a) + (sqrt(x) - 1)**(2*a)*_exp(2*pi*I*(n - 1)*a)) / 2
        n = n - 1
        return ((sqrt(x) - 1)**(2*a)*_exp(2*pi*I*a*(n + 1)) + (sqrt(x) + 1)**(2*a)*_exp(2*pi*I*a*n)) / 2

    @classmethod
    def _big_minus(cls, p: Params, x: Expr, n: Expr) -> Expr:
        from sympy_extras.functions.elementary.trigonometric import atan, cos
        a = p[0]
        if n.is_even is True:
            return (1 + x)**a * _exp(2*pi*I*n*a) * cos(2*a*atan(sqrt(x)))
        return (1 + x)**a * _exp(2*pi*I*n*a) * cos(2*a*atan(sqrt(x)) - 2*pi*a)


# PORTED FROM SYMPY: sympy/functions/special/hyper.py:HyperRep_sqrts2
class HyperRep_sqrts2(HyperRep):
    """``sqrt(z)*((1 - sqrt(z))**(2*a) - (1 + sqrt(z))**(2*a))/2``."""
    function_name = 'HyperRep_sqrts2'
    nargs = frozenset({2})
    call_nargs = frozenset({2})

    @classmethod
    def _small(cls, p: Params, x: Expr) -> Expr:
        a = p[0]
        return sqrt(x) * ((1 - sqrt(x))**(2*a) - (1 + sqrt(x))**(2*a)) / 2

    @classmethod
    def _small_minus(cls, p: Params, x: Expr) -> Expr:
        from sympy_extras.functions.elementary.trigonometric import atan, sin
        a = p[0]
        return sqrt(x) * (1 + x)**a * sin(2*a*atan(sqrt(x)))

    @classmethod
    def _big(cls, p: Params, x: Expr, n: Expr) -> Expr:
        a = p[0]
        if n.is_even is True:
            return sqrt(x)/2 * ((sqrt(x) - 1)**(2*a)*_exp(2*pi*I*a*(n - 1)) - (sqrt(x) + 1)**(2*a)*_exp(2*pi*I*a*n))
        n = n - 1
        return sqrt(x)/2 * ((sqrt(x) - 1)**(2*a)*_exp(2*pi*I*a*(n + 1)) - (sqrt(x) + 1)**(2*a)*_exp(2*pi*I*a*n))

    @classmethod
    def _big_minus(cls, p: Params, x: Expr, n: Expr) -> Expr:
        from sympy_extras.functions.elementary.trigonometric import atan, sin
        a = p[0]
        if n.is_even is True:
            return (1 + x)**a * _exp(2*pi*I*n*a) * sqrt(x) * sin(2*a*atan(sqrt(x)))
        return (1 + x)**a * _exp(2*pi*I*n*a) * sqrt(x) * sin(2*a*atan(sqrt(x)) - 2*pi*a)


# PORTED FROM SYMPY: sympy/functions/special/hyper.py:HyperRep_log2
class HyperRep_log2(HyperRep):
    """``log(1/2 + sqrt(1 - z)/2)``."""
    function_name = 'HyperRep_log2'
    nargs = frozenset({1})

    @classmethod
    def _small(cls, p: Params, x: Expr) -> Expr:
        return _log(HALF + sqrt(1 - x)/2)

    @classmethod
    def _small_minus(cls, p: Params, x: Expr) -> Expr:
        return _log(HALF + sqrt(1 + x)/2)

    @classmethod
    def _big(cls, p: Params, x: Expr, n: Expr) -> Expr:
        from sympy_extras.functions.elementary.trigonometric import asin
        if n.is_even is True:
            return (n - HALF)*pi*I + _log(sqrt(x)/2) + I*asin(1/sqrt(x))
        return (n - HALF)*pi*I + _log(sqrt(x)/2) - I*asin(1/sqrt(x))

    @classmethod
    def _big_minus(cls, p: Params, x: Expr, n: Expr) -> Expr:
        if n.is_even is True:
            return pi*I*n + _log(HALF + sqrt(1 + x)/2)
        return pi*I*n + _log(sqrt(1 + x)/2 - HALF)


# PORTED FROM SYMPY: sympy/functions/special/hyper.py:HyperRep_cosasin
class HyperRep_cosasin(HyperRep):
    """``cos(2*a*asin(sqrt(z)))``."""
    function_name = 'HyperRep_cosasin'
    nargs = frozenset({2})
    call_nargs = frozenset({2})

    @classmethod
    def _small(cls, p: Params, x: Expr) -> Expr:
        from sympy_extras.functions.elementary.trigonometric import asin, cos
        return cos(2*p[0]*asin(sqrt(x)))

    @classmethod
    def _small_minus(cls, p: Params, x: Expr) -> Expr:
        from sympy_extras.functions.elementary.hyperbolic import asinh, cosh
        return cosh(2*p[0]*asinh(sqrt(x)))

    @classmethod
    def _big(cls, p: Params, x: Expr, n: Expr) -> Expr:
        from sympy_extras.functions.elementary.hyperbolic import acosh, cosh
        a = p[0]
        return cosh(2*a*acosh(sqrt(x)) + a*pi*I*(2*n - 1))

    @classmethod
    def _big_minus(cls, p: Params, x: Expr, n: Expr) -> Expr:
        from sympy_extras.functions.elementary.hyperbolic import asinh, cosh
        a = p[0]
        return cosh(2*a*asinh(sqrt(x)) + 2*a*pi*I*n)


# PORTED FROM SYMPY: sympy/functions/special/hyper.py:HyperRep_sinasin
class HyperRep_sinasin(HyperRep):
    """``sqrt(z)/sqrt(1 - z)*sin(2*a*asin(sqrt(z)))``."""
    function_name = 'HyperRep_sinasin'
    nargs = frozenset({2})
    call_nargs = frozenset({2})

    @classmethod
    def _small(cls, p: Params, x: Expr) -> Expr:
        from sympy_extras.functions.elementary.trigonometric import asin, sin
        return sqrt(x)/sqrt(1 - x) * sin(2*p[0]*asin(sqrt(x)))

    @classmethod
    def _small_minus(cls, p: Params, x: Expr) -> Expr:
        from sympy_extras.functions.elementary.hyperbolic import asinh, sinh
        return -sqrt(x)/sqrt(1 + x) * sinh(2*p[0]*asinh(sqrt(x)))

    @classmethod
    def _big(cls, p: Params, x: Expr, n: Expr) -> Expr:
        from sympy_extras.functions.elementary.hyperbolic import acosh, sinh
        a = p[0]
        return -1/sqrt(1 - 1/x) * sinh(2*a*acosh(sqrt(x)) + a*pi*I*(2*n - 1))

    @classmethod
    def _big_minus(cls, p: Params, x: Expr, n: Expr) -> Expr:
        from sympy_extras.functions.elementary.hyperbolic import asinh, sinh
        a = p[0]
        return -1/sqrt(1 + 1/x) * sinh(2*a*asinh(sqrt(x)) + 2*a*pi*I*n)


def rewrite_nonrep(e: ExprLike, small: bool = True) -> Expr:
    """``e`` with its ``HyperRep`` nodes replaced by their closed forms
    (SymPy's ``rewrite('nonrepsmall')``, or ``rewrite('nonrep')`` when
    ``small`` is false), bottom up."""
    def walk(b: Basic) -> Basic:
        if not b.args:
            return b
        args = tuple(walk(a) for a in b.args)
        rebuilt = b._evaluated(args) if args != b.args else b
        if isinstance(rebuilt, HyperRep):
            return rebuilt.nonrepsmall() if small else rebuilt.nonrep()
        return rebuilt
    result = walk(to_expr(e))
    assert isinstance(result, Expr)
    return result
