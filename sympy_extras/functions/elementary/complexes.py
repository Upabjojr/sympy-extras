"""Complex parts on the typed core: ``Re``, ``Im``, ``Abs``, ``Sign``,
``Arg``, ``Conjugate`` (unevaluated nodes) with ``re``, ``im``, ``Abs``
(evaluating ``abs_``), ``sign``, ``arg``, ``conjugate``, and ``atan2``
(``Atan2``); the complex expansion :func:`as_real_imag` and the
conjugation :func:`conjugate_of` of every class of the core.

Ported from SymPy's ``sympy.functions.elementary.complexes`` (``re``,
``im``, ``sign``, ``Abs``, ``arg``, ``conjugate``), ``atan2`` from
``trigonometric.py``, and the ``as_real_imag``/``_eval_conjugate``
methods of SymPy's ``Expr``, ``Add``, ``Mul``, ``Pow`` and ``Symbol``.
Where SymPy reaches further than the typed core, the node stays
unevaluated:

* ``Abs`` uses ``signsimp``: only ``Abs(-z) = Abs(z)`` is applied;
* ``atan2`` answers with ``Heaviside`` or ``Piecewise`` for some
  arguments (``atan2(y, -oo)``, ``atan2(0, x)``): ported where they
  evaluate (their conditions decided), the node stays otherwise;
* the complex expansion of a power with a symbolic exponent relies on
  ``expand``, which in the typed core does not split ``(-p)**(I*r)``
  into ``(-1)**(I*r)*p**(I*r)`` as SymPy's does;

Examples
========

>>> from sympy_extras.core import Symbol, I, pi
>>> from sympy_extras.functions.elementary.complexes import re, im, Abs, abs_, sign, arg, conjugate, atan2
>>> x, y = Symbol('x'), Symbol('y')
>>> r = Symbol('r', real=True)
>>> re(2 + 3*I), im(x + 2*I), re(r), abs_(-3), abs_(3 + 4*I), sign(-2*r**2 - 1), arg(I)
(2, im(x) + 2, r, 3, 5, -1, pi/2)
>>> conjugate(x + I*r), atan2(1, 1), Abs(-3), (x + I*y).as_real_imag()
(-I*r + conjugate(x), pi/4, Abs(-3), (re(x) - im(y), re(y) + im(x)))
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/functions/elementary/complexes.py, sympy/functions/elementary/trigonometric.py
# (atan2), sympy/core/{expr,add,mul,power,symbol}.py (as_real_imag, _eval_conjugate) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: unevaluating classes with evaluating functions, strict typing,
# the per-class methods as typed dispatchers, the subset of the rules the
# typed core supports.
# ---------------------------------------------------------------------

from __future__ import annotations

from math import comb
from typing import Optional, Sequence

import mpmath

from sympy_extras.core.add import Add, add, sub
from sympy_extras.core.assumptions import facts_of
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.function import AppliedUndef, Derivative, DefinedFunction, diff, expand_mul
from sympy_extras.core.mul import Mul, div, mul
from sympy_extras.core.numbers import (ComplexInfinity, I, ImaginaryUnit, Infinity, Integer, NaN, NegativeInfinity,
                                       Number, Rational, nan, oo, pi)
from sympy_extras.core.power import Pow, pow, sqrt
from sympy_extras.core.symbol import Dummy, Symbol
from sympy_extras.functions.elementary.exponential import Exp, exp, log
from sympy_extras.functions.elementary.trigonometric import atan, cos, sin

__all__ = ['Re', 'Im', 'Abs', 'Sign', 'Arg', 'Conjugate', 'Atan2', 're', 'im', 'abs_', 'sign', 'arg', 'conjugate',
           'atan2', 'as_real_imag', 'conjugate_of', 'PolarLift', 'PeriodicArgument', 'PrincipalBranch', 'polar_lift',
           'periodic_argument', 'unbranched_argument', 'principal_branch', 'polarify', 'unpolarify', 'Adjoint',
           'Transpose', 'adjoint', 'transpose']

ZERO = Integer(0)
ONE = Integer(1)
MINUS_ONE = Integer(-1)

RealImag = tuple[Expr, Expr]


def _terms(e: Expr) -> tuple[Expr, ...]:
    return e.args if isinstance(e, Add) else (e,)


# ---------------------------------------------------------------------
# conjugation

# PORTED FROM SYMPY: sympy/core/expr.py:Expr._eval_conjugate (and Add, Mul, Pow, numbers)
def conjugate_of(e: Expr) -> Optional[Expr]:
    """The conjugate of ``e`` when a rule of its class gives it (SymPy's
    ``_eval_conjugate``), else ``None``."""
    if isinstance(e, ImaginaryUnit):
        return mul(MINUS_ONE, I)
    if isinstance(e, ComplexInfinity):
        return None  # not a Number in SymPy: conjugate(zoo) stays
    if isinstance(e, Number):
        return e
    if isinstance(e, DefinedFunction):
        return e._eval_conjugate()
    from .piecewise import Piecewise
    if isinstance(e, Piecewise):
        return e.map_expressions(conjugate)
    if isinstance(e, (Add, Mul)):
        parts = [conjugate(t) for t in e.args]
        return add(*parts) if isinstance(e, Add) else mul(*parts)
    if isinstance(e, Pow):
        integer, positive = e.exp.is_integer, e.base.is_positive
        if integer:
            return pow(conjugate(e.base), e.exp)
        if positive:
            return pow(e.base, conjugate(e.exp))
        if e.is_extended_real:
            return e
        return None
    if e.is_extended_real:
        return e
    if e.is_imaginary:
        return mul(MINUS_ONE, e)
    return None


# ---------------------------------------------------------------------
# complex expansion

def _binomial_parts(n: int, a: Expr, b: Expr) -> RealImag:
    """The real and imaginary parts of ``(a + b*I)**n`` for real ``a``, ``b``
    and ``n >= 0``, from the binomial expansion: the even powers of ``b``
    are real (with the sign of ``I**k``), the odd ones imaginary."""
    real_terms: list[Expr] = []
    imaginary_terms: list[Expr] = []
    for k in range(n + 1):
        c = comb(n, k)
        term = mul(Integer(c), pow(a, Integer(n - k)), pow(b, Integer(k)))
        if k % 2 == 0:
            real_terms.append(mul(Integer(1 if k % 4 == 0 else -1), term))
        else:
            imaginary_terms.append(mul(Integer(1 if k % 4 == 1 else -1), term))
    return add(*real_terms), add(*imaginary_terms)


# PORTED FROM SYMPY: sympy/core/mul.py:Mul.as_real_imag
def _mul_real_imag(e: Mul, deep: bool, ignore: Optional[Expr]) -> Optional[RealImag]:
    other: list[Expr] = []
    coeff_real: list[Expr] = []
    coeff_imaginary: list[Expr] = []
    add_terms: Expr = ONE
    for a in e.args:
        parts = as_real_imag(a, deep)
        if parts is None:
            return None
        r, i = parts
        if i.is_zero:
            coeff_real.append(r)
        elif r.is_zero:
            coeff_imaginary.append(mul(i, I))
        elif a.is_commutative:
            a_conjugate = conjugate(a) if other else None
            for index, x in enumerate(other):
                if x == a_conjugate:
                    coeff_real.append(pow(abs_(x), Integer(2)))
                    del other[index]
                    break
            else:
                if isinstance(a, Add):
                    add_terms = mul(add_terms, a)
                else:
                    other.append(a)
        else:
            other.append(a)
    m = mul(*other)
    if ignore is not None and ignore == m:
        return None
    imaginary_coefficient: Expr = ZERO
    if len(coeff_imaginary) % 2:
        imaginary_coefficient = im(coeff_imaginary.pop(0))
    real_coefficient = mul(*(coeff_real + coeff_imaginary))
    r, i = mul(real_coefficient, re(m)), mul(real_coefficient, im(m))
    if add_terms == 1:
        if m == 1:
            if imaginary_coefficient.is_zero:
                return (real_coefficient, ZERO)
            return (ZERO, mul(real_coefficient, imaginary_coefficient))
        if imaginary_coefficient == 0:
            return (r, i)
        return (mul(MINUS_ONE, imaginary_coefficient, i), mul(imaginary_coefficient, r))
    sum_parts = as_real_imag(expand_mul(add_terms, deep=False), deep)
    if sum_parts is None:
        return None
    sum_real, sum_imaginary = sum_parts
    if imaginary_coefficient == 0:
        return (add(mul(r, sum_real), mul(MINUS_ONE, i, sum_imaginary)), add(mul(i, sum_real), mul(r, sum_imaginary)))
    r, i = mul(MINUS_ONE, imaginary_coefficient, i), mul(imaginary_coefficient, r)
    return (add(mul(r, sum_real), mul(MINUS_ONE, i, sum_imaginary)), add(mul(r, sum_imaginary), mul(i, sum_real)))


# PORTED FROM SYMPY: sympy/core/power.py:Pow.as_real_imag
def _pow_real_imag(e: Pow, deep: bool, ignore: Optional[Expr]) -> Optional[RealImag]:
    from sympy_extras.core.function import expand_multinomial
    exponent = e.exp
    if isinstance(exponent, Rational) and exponent.q == 1:
        parts = as_real_imag(e.base, deep)
        if parts is None:
            return None
        re_e, im_e = parts
        if im_e == 0:
            return (e, ZERO)
        n = exponent.p
        if n >= 0:
            if isinstance(re_e, Number) and isinstance(im_e, Number):
                expanded = expand_multinomial(pow(e.base, exponent))
                if expanded != e:
                    return as_real_imag(expanded, deep)
            return _binomial_parts(n, re_e, im_e)
        magnitude = add(pow(re_e, Integer(2)), pow(im_e, Integer(2)))
        re_e, im_e = mul(re_e, pow(magnitude, MINUS_ONE)), mul(MINUS_ONE, im_e, pow(magnitude, MINUS_ONE))
        if isinstance(re_e, Number) and isinstance(im_e, Number):
            expanded = expand_multinomial(pow(add(re_e, mul(im_e, I)), Integer(-n)))
            if expanded != e:
                return as_real_imag(expanded, deep)
        return _binomial_parts(-n, re_e, im_e)
    if isinstance(exponent, Rational):
        parts = as_real_imag(e.base, deep)
        if parts is None:
            return None
        re_e, im_e = parts
        if im_e.is_zero and exponent == Rational(1, 2):
            if re_e.is_extended_nonnegative:
                return (e, ZERO)
            if re_e.is_extended_nonpositive:
                return (ZERO, pow(mul(MINUS_ONE, e.base), exponent))
        modulus = pow(add(pow(re_e, Integer(2)), pow(im_e, Integer(2))), Rational(1, 2))
        angle = atan2(im_e, re_e)
        rp, tp = pow(modulus, exponent), mul(angle, exponent)
        return (mul(rp, cos(tp)), mul(rp, sin(tp)))
    from sympy_extras.core.numbers import Exp1
    if isinstance(e.base, Exp1):
        parts = as_real_imag(exponent, deep)
        if parts is None:
            return None
        re_e, im_e = parts
        if deep:
            re_e, im_e = re_e.expand(), im_e.expand()
        return (mul(exp(re_e), cos(im_e)), mul(exp(re_e), sin(im_e)))
    if deep:
        expanded = e.expand(complex=False)
        if ignore is not None and ignore == expanded:
            return None
        return (re(expanded), im(expanded))
    return (re(e), im(e))


# PORTED FROM SYMPY: sympy/core/expr.py:Expr.as_real_imag (and Add, Mul, Pow, Symbol)
def as_real_imag(e: Expr, deep: bool = True, ignore: Optional[Expr] = None) -> Optional[RealImag]:
    """The real and imaginary parts of ``e`` by complex expansion; ``None``
    when they would only be ``re(ignore)``, ``im(ignore)`` (SymPy's
    ``ignore`` hint, which keeps ``re`` and ``im`` from looping)."""
    if isinstance(e, Number) and e.exact_value() is not None:
        return (e, ZERO)
    if e.is_extended_real is True:
        return (e, ZERO)
    if isinstance(e, Add):
        real_parts: list[Expr] = []
        imaginary_parts: list[Expr] = []
        for term in e.args:
            parts = as_real_imag(term, deep)
            if parts is None:
                return None
            real_parts.append(parts[0])
            imaginary_parts.append(parts[1])
        return (add(*real_parts), add(*imaginary_parts))
    if isinstance(e, Mul):
        return _mul_real_imag(e, deep, ignore)
    if isinstance(e, Pow):
        return _pow_real_imag(e, deep, ignore)
    if isinstance(e, DefinedFunction):
        return e._eval_as_real_imag(deep, ignore)
    if ignore is not None and ignore == e:
        return None
    return (re(e), im(e))


# ---------------------------------------------------------------------
# re and im

def _split_real_imag(arg: Expr, real: bool) -> Optional[Expr]:
    """SymPy's ``re.eval`` (``real``) and ``im.eval`` for a sum: the terms
    whose parts are known taken out."""
    included: list[Expr] = []
    reverted: list[Expr] = []
    excluded: list[Expr] = []
    terms = _terms(arg)
    for term in terms:
        coefficient = term.as_coefficient(I)
        if coefficient is not None:
            if not coefficient.is_extended_real:
                reverted.append(coefficient)
            elif not real:
                excluded.append(coefficient)
        elif real and not term.has(I) and term.is_extended_real:
            excluded.append(term)
        elif real or term.has(I) or not term.is_extended_real:
            parts = as_real_imag(term, ignore=arg)
            if parts is not None:
                excluded.append(parts[0] if real else parts[1])
            else:
                included.append(term)
    if len(terms) == len(included):
        return None
    a, b, c = add(*included), add(*reverted), add(*excluded)
    # SymPy: cls(a) - im(b) + c, cls(a) + re(b) + c (cls evaluates)
    if real:
        return add(re(a), mul(MINUS_ONE, im(b)), c)
    return add(im(a), re(b), c)


# PORTED FROM SYMPY: sympy/functions/elementary/complexes.py:re
class Re(DefinedFunction):
    """The real part, unevaluated (``re`` evaluates)."""

    unbranched = True

    function_name = 're'

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        [arg] = args
        if isinstance(arg, (NaN, ComplexInfinity)):
            return nan
        if arg.is_extended_real:
            return arg
        if arg.is_imaginary or mul(I, arg).is_extended_real:
            return ZERO
        if isinstance(arg, Conjugate):
            return re(arg.args[0])
        return _split_real_imag(arg, True)

    def _eval_as_real_imag(self, deep: bool, ignore: Optional[Expr]) -> Optional[RealImag]:
        return (self, ZERO)

    def _eval_conjugate(self) -> Optional[Expr]:
        return self

    def _eval_derivative(self, s: Symbol) -> Optional[Expr]:
        a = self.args[0]
        if s.is_extended_real or a.is_extended_real:
            return re(diff(a, s))
        if s.is_imaginary or a.is_imaginary:
            return mul(MINUS_ONE, I, im(diff(a, s)))
        return Derivative(self, s)

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        return mpmath.mpc(mpmath.re(values[0]))

    def _eval_facts(self) -> dict[str, bool]:
        facts = {'commutative': True, 'extended_real': True, 'imaginary': False}
        if facts_of(self.args[0]).get('finite'):
            facts.update(real=True, finite=True)
        return facts


# PORTED FROM SYMPY: sympy/functions/elementary/complexes.py:im
class Im(DefinedFunction):
    """The imaginary part, unevaluated (``im`` evaluates)."""

    unbranched = True

    function_name = 'im'

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        [arg] = args
        if isinstance(arg, (NaN, ComplexInfinity)):
            return nan
        if arg.is_extended_real:
            return ZERO
        if arg.is_imaginary or mul(I, arg).is_extended_real:
            return mul(MINUS_ONE, I, arg)
        if isinstance(arg, Conjugate):
            return mul(MINUS_ONE, im(arg.args[0]))
        return _split_real_imag(arg, False)

    def _eval_as_real_imag(self, deep: bool, ignore: Optional[Expr]) -> Optional[RealImag]:
        return (self, ZERO)

    def _eval_conjugate(self) -> Optional[Expr]:
        return self

    def _eval_derivative(self, s: Symbol) -> Optional[Expr]:
        a = self.args[0]
        if s.is_extended_real or a.is_extended_real:
            return im(diff(a, s))
        if s.is_imaginary or a.is_imaginary:
            return mul(MINUS_ONE, I, re(diff(a, s)))
        return Derivative(self, s)

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        return mpmath.mpc(mpmath.im(values[0]))

    def _eval_facts(self) -> dict[str, bool]:
        facts = {'commutative': True, 'extended_real': True, 'imaginary': False}
        if facts_of(self.args[0]).get('finite'):
            facts.update(real=True, finite=True)
        return facts


# ---------------------------------------------------------------------
# sign

# PORTED FROM SYMPY: sympy/functions/elementary/complexes.py:sign
class Sign(DefinedFunction):
    """The complex sign ``z/|z|`` (``0`` at ``0``), unevaluated (``sign``
    evaluates)."""

    function_name = 'sign'

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        [arg] = args
        if isinstance(arg, Mul):
            c, factors = arg.as_coeff_mul()
            unknown: list[Expr] = []
            s: Expr = sign(c)
            for a in factors:
                if a.is_extended_negative:
                    s = mul(MINUS_ONE, s)
                elif a.is_extended_positive:
                    pass
                elif a.is_imaginary:
                    ai = im(a)
                    if ai.is_comparable:
                        s = mul(s, I)
                        if ai.is_extended_negative:
                            s = mul(MINUS_ONE, s)
                    else:
                        unknown.append(a)
                else:
                    unknown.append(a)
            if c == 1 and len(unknown) == len(factors):
                return None
            rest: Expr = ONE if not unknown else (unknown[0] if len(unknown) == 1 else Mul(*unknown))
            return mul(s, sign(rest)) if unknown else s
        if isinstance(arg, NaN):
            return nan
        if arg.is_zero:
            return ZERO
        if arg.is_extended_positive:
            return ONE
        if arg.is_extended_negative:
            return MINUS_ONE
        if isinstance(arg, Sign):
            return arg
        if arg.is_imaginary:
            if isinstance(arg, Pow) and arg.exp == Rational(1, 2):
                return I
            rotated = mul(MINUS_ONE, I, arg)
            if rotated.is_extended_positive:
                return I
            if rotated.is_extended_negative:
                return mul(MINUS_ONE, I)
        return None

    def _eval_conjugate(self) -> Optional[Expr]:
        return sign(conjugate(self.args[0]))

    def _eval_derivative(self, s: Symbol) -> Optional[Expr]:
        from sympy_extras.functions.special.delta_functions import dirac_delta
        a = self.args[0]
        if a.is_extended_real:
            return mul(Integer(2), diff(a, s), dirac_delta(a))
        if a.is_imaginary:
            return mul(Integer(2), diff(a, s), dirac_delta(mul(MINUS_ONE, I, a)))
        return Derivative(self, s)

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        z = values[0]
        return mpmath.mpc(0) if z == 0 else mpmath.mpc(z/abs(z))

    def _eval_facts(self) -> dict[str, bool]:
        facts = {'commutative': True, 'finite': True, 'complex': True}
        a = facts_of(self.args[0])
        if a.get('extended_real'):
            facts.update(real=True, extended_real=True, integer=True, rational=True, imaginary=False)
        if a.get('zero') is not None:
            facts['zero'] = a['zero']
        return facts


# ---------------------------------------------------------------------
# Abs

# PORTED FROM SYMPY: sympy/functions/elementary/complexes.py:Abs
class Abs(DefinedFunction):
    """The absolute value, unevaluated (``abs_`` evaluates; Python's
    ``abs()`` of a core expression calls it)."""

    unbranched = True

    function_name = 'Abs'

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        [arg] = args
        from .exponential import ExpPolar
        if isinstance(arg, ExpPolar):
            return exp(re(arg.args[0]))  # SymPy: exp_polar._eval_Abs
        if isinstance(arg, PolarLift):
            return abs_(arg.args[0])  # SymPy: polar_lift._eval_Abs
        if isinstance(arg, Sign) and arg.args[0].is_zero is False:
            return ONE
        if isinstance(arg, Conjugate):
            return abs_(arg.args[0])
        # SymPy: signsimp; here only Abs(-z) = Abs(z)
        if arg.could_extract_minus_sign():
            return abs_(mul(MINUS_ONE, arg))
        n, d = arg.as_numer_denom()
        if d.free_symbols and not n.free_symbols:
            return mul(abs_(n), pow(abs_(d), MINUS_ONE))
        if isinstance(arg, Mul):
            known: list[Expr] = []
            unknown: list[Expr] = []
            for t in arg.args:
                if isinstance(t, Pow) and t.exp.is_integer and t.exp.is_negative:
                    new_base = abs_(t.base)
                    if isinstance(new_base, Abs):
                        unknown.append(t)
                    else:
                        known.append(pow(new_base, t.exp))
                else:
                    new_factor = abs_(t)
                    if isinstance(new_factor, Abs):
                        unknown.append(t)
                    else:
                        known.append(new_factor)
            rest: Expr = cls(mul(*unknown)) if unknown else ONE
            return mul(mul(*known), rest)
        if isinstance(arg, NaN):
            return nan
        if isinstance(arg, ComplexInfinity):
            return oo
        if isinstance(arg, Pow):
            base, exponent = arg.as_base_exp()
            if base.is_extended_real:
                if exponent.is_integer:
                    if exponent.is_even:
                        return arg
                    if base == -1:
                        return ONE
                    return pow(abs_(base), exponent)
                if base.is_extended_nonnegative:
                    return pow(base, re(exponent))
                if base.is_extended_negative:
                    return mul(pow(mul(MINUS_ONE, base), re(exponent)), exp(mul(MINUS_ONE, pi, im(exponent))))
                return None
            if not any(isinstance(s, Symbol) for s in base.free_symbols):
                parts = as_real_imag(log(base))
                if parts is not None:
                    z = add(parts[0], mul(I, parts[1]))
                    return exp(re(mul(exponent, z)))
        if isinstance(arg, Exp):
            return exp(re(arg.args[0]))
        if isinstance(arg, AppliedUndef):
            if arg.is_positive:
                return arg
            if arg.is_negative:
                return mul(MINUS_ONE, arg)
            return None
        if isinstance(arg, Add) and (arg.has(oo) or arg.has(NegativeInfinity())):
            parts = as_real_imag(arg)
            if parts is not None and any(p.is_infinite for p in parts):
                return oo
        if arg.is_zero:
            return ZERO
        if arg.is_extended_nonnegative:
            return arg
        if arg.is_extended_nonpositive:
            return mul(MINUS_ONE, arg)
        if arg.is_imaginary:
            rotated = mul(MINUS_ONE, I, arg)
            if rotated.is_extended_nonnegative:
                return rotated
        if arg.is_extended_real:
            return None
        conj = conjugate(arg)
        new_conjugates = _atoms_of(conj, Conjugate) - _atoms_of(arg, Conjugate)
        if new_conjugates and all(arg.has(c.args[0]) for c in new_conjugates):
            return None
        if arg != conj and arg != mul(MINUS_ONE, conj):
            replaced: dict[Basic, Basic] = {a: Dummy('a', real=True) for a in _atoms_of(arg, Abs)}
            abs_free = arg.xreplace(replaced)
            unknown_symbols = [s for s in abs_free.free_symbols if s.is_extended_real is None]
            if not unknown_symbols or not all(conj.has(Conjugate(s)) for s in unknown_symbols):
                return sqrt(expand_mul(mul(arg, conj)))
        return None

    def _eval_as_real_imag(self, deep: bool, ignore: Optional[Expr]) -> Optional[RealImag]:
        return (self, ZERO)

    def _eval_conjugate(self) -> Optional[Expr]:
        return self

    def _eval_derivative(self, s: Symbol) -> Optional[Expr]:
        a = self.args[0]
        if a.is_extended_real or a.is_imaginary:
            return mul(diff(a, s), sign(conjugate(a)))
        # SymPy rewrites the result with sign: 1/Abs(a) = sign(a)/a
        return mul(add(mul(re(a), diff(re(a), s)), mul(im(a), diff(im(a), s))), sign(a), pow(a, MINUS_ONE))

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        return mpmath.mpc(abs(values[0]))

    def _eval_facts(self) -> dict[str, bool]:
        a = facts_of(self.args[0])
        facts = {'commutative': True, 'extended_real': True, 'imaginary': False, 'extended_nonnegative': True}
        if a.get('finite'):
            facts.update(real=True, finite=True, nonnegative=True, complex=True)
        if a.get('extended_real'):
            if a.get('integer') is not None:
                facts['integer'] = a['integer']
            if a.get('rational') is not None:
                facts['rational'] = a['rational']
        if a.get('zero') is not None:
            facts['zero'] = a['zero']
        return facts


def _atoms_of(e: Expr, cls: type[DefinedFunction]) -> set[DefinedFunction]:
    """The nodes of the class ``cls`` in ``e``."""
    found: set[DefinedFunction] = set()
    stack: list[Basic] = [e]
    while stack:
        node = stack.pop()
        if isinstance(node, cls):
            found.add(node)
        stack.extend(node.args)
    return found


# ---------------------------------------------------------------------
# arg

# PORTED FROM SYMPY: sympy/functions/elementary/complexes.py:arg
class Arg(DefinedFunction):
    """The argument (principal value in ``(-pi, pi]``), unevaluated
    (``arg`` evaluates)."""

    function_name = 'arg'

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        from sympy_extras.core.exprtools import factor_terms
        from .trigonometric import _gt, _mod
        [z] = args
        a = z
        for i in range(3):
            if isinstance(a, Arg):
                a = a.args[0]
            else:
                if i == 2 and a.is_extended_real:
                    return nan
                break
        else:
            return nan
        if isinstance(z, Exp):
            angle = im(z.args[0])
            if angle.is_comparable:
                angle = _mod(angle, mul(Integer(2), pi))
                if _gt(angle, pi):
                    angle = add(angle, mul(Integer(-2), pi))
                return angle
        if z.args:
            c, rest = factor_terms(z).as_coeff_Mul()
            if isinstance(rest, Mul):
                rest = mul(*[t if sign(t) not in (ONE, MINUS_ONE) else sign(t) for t in rest.args])
            rest = mul(sign(c), rest)
        else:
            rest = z
        if any(u.is_extended_positive is None for u in _atoms_of_undef(rest)):
            return None
        parts = as_real_imag(rest)
        if parts is None:
            return None
        x, y = parts
        value = atan2(y, x)
        if value.is_number:
            return value
        if rest != z:
            return cls(rest)
        return None

    def _eval_derivative(self, s: Symbol) -> Optional[Expr]:
        parts = as_real_imag(self.args[0])
        if parts is None:
            return Derivative(self, s)
        x, y = parts
        return mul(add(mul(x, diff(y, s)), mul(MINUS_ONE, y, diff(x, s))),
                   pow(add(pow(x, Integer(2)), pow(y, Integer(2))), MINUS_ONE))

    def _eval_as_real_imag(self, deep: bool, ignore: Optional[Expr]) -> Optional[RealImag]:
        return (self, ZERO)

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        if values[0] == 0:
            return None
        return mpmath.mpc(mpmath.arg(values[0]))

    def _eval_facts(self) -> dict[str, bool]:
        return {'commutative': True, 'real': True, 'extended_real': True, 'finite': True, 'complex': True,
                'imaginary': False}


def _atoms_of_undef(e: Expr) -> list[AppliedUndef]:
    found: list[AppliedUndef] = []
    stack: list[Basic] = [e]
    while stack:
        node = stack.pop()
        if isinstance(node, AppliedUndef):
            found.append(node)
        stack.extend(node.args)
    return found


# ---------------------------------------------------------------------
# conjugate

# PORTED FROM SYMPY: sympy/functions/elementary/complexes.py:conjugate
class Conjugate(DefinedFunction):
    """The complex conjugate, unevaluated (``conjugate`` evaluates)."""

    function_name = 'conjugate'

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        [arg] = args
        return conjugate_of(arg)

    def _eval_conjugate(self) -> Optional[Expr]:
        return self.args[0]

    def _eval_derivative(self, s: Symbol) -> Optional[Expr]:
        if s.is_real:
            return conjugate(diff(self.args[0], s))
        if s.is_imaginary:
            return mul(MINUS_ONE, conjugate(diff(self.args[0], s)))
        return Derivative(self, s)

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        return mpmath.mpc(mpmath.conj(values[0]))

    def _eval_facts(self) -> dict[str, bool]:
        return _commutative_like(self.args[0])


# ---------------------------------------------------------------------
# atan2

# PORTED FROM SYMPY: sympy/functions/elementary/trigonometric.py:atan2
class Atan2(DefinedFunction):
    """``atan2(y, x)``, the argument of ``x + I*y``, unevaluated (``atan2``
    evaluates)."""

    function_name = 'atan2'
    nargs = frozenset({2})
    call_nargs = frozenset({2})

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        y, x = args
        if isinstance(x, NegativeInfinity):
            if y.is_zero:
                return pi
            # SymPy: 2*pi*Heaviside(re(y)) - pi, evaluated when the sign is known
            real_y = re(y)
            if real_y.is_extended_positive:
                return pi
            if real_y.is_extended_negative:
                return mul(MINUS_ONE, pi)
            if real_y.is_zero:
                return ZERO  # Heaviside(0) = 1/2
            return None
        if isinstance(x, Infinity):
            return ZERO
        if x.is_imaginary and y.is_imaginary and x.is_number and y.is_number:
            x, y = im(x), im(y)
        if x.is_extended_real and y.is_extended_real:
            if x.is_positive:
                return atan(mul(y, pow(x, MINUS_ONE)))
            if x.is_negative:
                if y.is_negative:
                    return add(atan(mul(y, pow(x, MINUS_ONE))), mul(MINUS_ONE, pi))
                if y.is_nonnegative:
                    return add(atan(mul(y, pow(x, MINUS_ONE))), pi)
            elif x.is_zero:
                if y.is_positive:
                    return mul(pi, Rational(1, 2))
                if y.is_negative:
                    return mul(pi, Rational(-1, 2))
                if y.is_zero:
                    return nan
        if y.is_zero:
            if x.is_extended_nonzero:
                # SymPy: pi*(1 - Heaviside(x)), evaluated when the sign is known
                if x.is_extended_positive:
                    return ZERO
                if x.is_extended_negative:
                    return pi
                return None
            if x.is_number:
                # SymPy: Piecewise((pi, re(x) < 0), (0, Ne(x, 0)), (nan, True)),
                # evaluated when its conditions are decided
                negative = re(x).is_extended_negative
                if negative is True:
                    return pi
                if negative is False:
                    if x.is_zero is False:
                        return ZERO
                    if x.is_zero is True:
                        return nan
                return None
        if x.is_number and y.is_number:
            modulus = sqrt(add(pow(x, Integer(2)), pow(y, Integer(2))))
            return mul(MINUS_ONE, I, log(mul(add(x, mul(I, y)), pow(modulus, MINUS_ONE))))
        return None

    def _eval_conjugate(self) -> Optional[Expr]:
        return atan2(conjugate(self.args[0]), conjugate(self.args[1]))

    def fdiff(self, argindex: int) -> Expr:
        y, x = self.args
        denominator = add(pow(x, Integer(2)), pow(y, Integer(2)))
        if argindex == 1:
            return mul(x, pow(denominator, MINUS_ONE))
        return mul(MINUS_ONE, y, pow(denominator, MINUS_ONE))

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        y, x = values
        if mpmath.im(y) != 0 or mpmath.im(x) != 0 or (x == 0 and y == 0):
            return None
        return mpmath.mpc(mpmath.atan2(mpmath.re(y), mpmath.re(x)))

    def _eval_facts(self) -> dict[str, bool]:
        facts = {'commutative': True}
        if all(facts_of(a).get('extended_real') for a in self.args):
            facts['extended_real'] = True
        return facts


def re(x: ExprLike) -> Expr:
    """The real part of ``x``, evaluated."""
    return Re.apply(to_expr(x))


def im(x: ExprLike) -> Expr:
    """The imaginary part of ``x``, evaluated."""
    return Im.apply(to_expr(x))


def abs_(x: ExprLike) -> Expr:
    """The absolute value of ``x``, evaluated (the class is ``Abs``, as in
    SymPy; ``abs()`` of a core expression calls this)."""
    return Abs.apply(to_expr(x))


def sign(x: ExprLike) -> Expr:
    """The complex sign of ``x``, evaluated."""
    return Sign.apply(to_expr(x))


def arg(x: ExprLike) -> Expr:
    """The argument of ``x``, evaluated."""
    return Arg.apply(to_expr(x))


def conjugate(x: ExprLike) -> Expr:
    """The complex conjugate of ``x``, evaluated."""
    return Conjugate.apply(to_expr(x))


def atan2(y: ExprLike, x: ExprLike) -> Expr:
    """``atan2(y, x)``, evaluated."""
    return Atan2.apply(to_expr(y), to_expr(x))




# ---------------------------------------------------------------------
# the polar numbers

# PORTED FROM SYMPY: sympy/functions/elementary/complexes.py:polar_lift
class PolarLift(DefinedFunction):
    """``polar_lift(z)``: ``z`` lifted to the Riemann surface of the
    logarithm, unevaluated (``polar_lift`` evaluates)."""

    function_name = 'polar_lift'

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        from .exponential import exp_polar
        [z] = args
        if z.is_number:
            angle = arg(z)
            if angle in (ZERO, mul(pi, Rational(1, 2)), mul(pi, Rational(-1, 2)), pi):
                return mul(exp_polar(mul(I, angle)), abs_(z))
        factors = z.args if isinstance(z, Mul) else (z,)
        included: list[Expr] = []
        excluded: list[Expr] = []
        positive: list[Expr] = []
        for f in factors:
            if f.is_polar:
                included.append(f)
            elif f.is_positive:
                positive.append(f)
            else:
                excluded.append(f)
        if len(excluded) < len(factors):
            if excluded:
                return mul(mul(*(included + positive)), polar_lift(mul(*excluded)))
            if included:
                return mul(*(included + positive))
            return mul(mul(*positive), exp_polar(ZERO))
        return None

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        return values[0]

    def _eval_facts(self) -> dict[str, bool]:
        return {'commutative': True, 'polar': True}


def _unbranched(z: Expr) -> Optional[Expr]:
    """SymPy's ``periodic_argument._getunbranched``: the argument of ``z``
    on the Riemann surface of the logarithm, ``None`` when unknown."""
    from .exponential import ExpPolar, log
    factors = z.args if isinstance(z, Mul) else (z,)
    total: Expr = ZERO
    for a in factors:
        if not a.is_polar:
            total = add(total, arg(a))
        elif isinstance(a, ExpPolar):
            parts = as_real_imag(a.args[0])
            if parts is None:
                return None
            total = add(total, parts[1])
        elif isinstance(a, Pow):
            parts = as_real_imag(a.exp)
            if parts is None:
                return None
            total = add(total, mul(parts[0], unbranched_argument(a.base)), mul(parts[1], log(abs_(a.base))))
        elif isinstance(a, PolarLift):
            total = add(total, arg(a.args[0]))
        else:
            return None
    return total


# PORTED FROM SYMPY: sympy/functions/elementary/complexes.py:periodic_argument
class PeriodicArgument(DefinedFunction):
    """The argument of ``z`` reduced into an interval of length ``period``
    (``oo`` for the unbranched argument), unevaluated
    (``periodic_argument`` evaluates)."""

    function_name = 'periodic_argument'
    nargs = frozenset({2})
    call_nargs = frozenset({2})

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        from sympy_extras.core.logic import true
        from sympy_extras.core.relational import ge
        from .integers import Ceiling, ceiling
        from .trigonometric import Atan
        z, period = args
        if not period.is_extended_positive:
            return None
        if period == oo and isinstance(z, PrincipalBranch):
            return periodic_argument(*z.args)
        if isinstance(z, PolarLift) and ge(period, mul(Integer(2), pi)) == true:
            return periodic_argument(z.args[0], period)
        if isinstance(z, Mul):
            kept = [x for x in z.args if not x.is_positive]
            if len(kept) != len(z.args):
                return periodic_argument(mul(*kept), period)
        unbranched = _unbranched(z)
        if unbranched is None:
            return None
        if unbranched.has(PeriodicArgument) or unbranched.has(Atan2) or unbranched.has(Atan):
            return None
        if period == oo:
            return unbranched
        n = mul(ceiling(sub(div(unbranched, period), Rational(1, 2))), period)
        if not n.has(Ceiling):
            return sub(unbranched, n)
        return None

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        return None

    def _eval_facts(self) -> dict[str, bool]:
        return {'commutative': True}


def unbranched_argument(z: ExprLike) -> Expr:
    """``periodic_argument(z, oo)``."""
    return periodic_argument(z, oo)


# PORTED FROM SYMPY: sympy/functions/elementary/complexes.py:principal_branch
class PrincipalBranch(DefinedFunction):
    """The polar number ``z`` reduced to the branch of arguments of length
    ``period``, unevaluated (``principal_branch`` evaluates)."""

    function_name = 'principal_branch'
    nargs = frozenset({2})
    call_nargs = frozenset({2})

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        from sympy_extras.core.logic import true
        from sympy_extras.core.relational import lt
        from .exponential import ExpPolar, exp_polar
        x, period = args
        if isinstance(x, PolarLift):
            return principal_branch(x.args[0], period)
        if period == oo:
            return x
        whole = periodic_argument(x, oo)
        branch = periodic_argument(x, period)
        if whole != branch and not whole.has(PeriodicArgument) and not branch.has(PeriodicArgument):
            lifted = polar_lift(x)
            replaced = _replace_polar_lifts(lifted)
            whole = periodic_argument(replaced, oo)
            if not replaced.has(PolarLift):
                result = mul(exp_polar(mul(I, sub(branch, whole))), replaced) if whole != branch else replaced
                if not result.is_polar and not result.has(ExpPolar):
                    result = mul(result, exp_polar(ZERO))
                return result
        if not x.free_symbols:
            c: Expr = x
            rest: tuple[Expr, ...] = ()
        else:
            coefficient_part = [f for f in (x.args if isinstance(x, Mul) else (x,)) if not f.free_symbols]
            c = mul(*coefficient_part)
            rest = tuple(f for f in (x.args if isinstance(x, Mul) else (x,)) if f.free_symbols)
        others: list[Expr] = []
        for y in rest:
            if y.is_positive:
                c = mul(c, y)
            else:
                others.append(y)
        angle = periodic_argument(c, period)
        if angle.has(PeriodicArgument):
            return None
        if angle.is_number and (unbranched_argument(c) != angle or (angle == 0 and others and c != ONE)):
            if angle == 0:
                return mul(abs_(c), principal_branch(mul(*others), period))
            return mul(principal_branch(mul(exp_polar(mul(I, angle)), *others), period), abs_(c))
        if angle.is_number and (lt(abs_(angle), div(period, Integer(2))) == true or angle == div(period, Integer(2))) \
                and not others:
            return mul(exp_polar(mul(angle, I)), abs_(c))
        return None

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        return None

    def _eval_facts(self) -> dict[str, bool]:
        return {'commutative': True, 'polar': True}


def _replace_polar_lifts(e: Expr) -> Expr:
    """``polar_lift(expr)`` kept around non-symbols only (SymPy's
    ``pl.replace(polar_lift, mr)``)."""
    if isinstance(e, PolarLift):
        inner = _replace_polar_lifts(e.args[0])
        return inner if isinstance(inner, Symbol) else polar_lift(inner)
    if not e.args:
        return e
    rebuilt = e._evaluated(tuple(_replace_polar_lifts(a) if isinstance(a, Expr) else a for a in e.args))
    return rebuilt if isinstance(rebuilt, Expr) else e


def polar_lift(z: ExprLike) -> Expr:
    """``polar_lift(z)``, evaluated."""
    return PolarLift.apply(to_expr(z))


def periodic_argument(z: ExprLike, period: ExprLike) -> Expr:
    """``periodic_argument(z, period)``, evaluated."""
    return PeriodicArgument.apply(to_expr(z), to_expr(period))


def principal_branch(z: ExprLike, period: ExprLike) -> Expr:
    """``principal_branch(z, period)``, evaluated."""
    return PrincipalBranch.apply(to_expr(z), to_expr(period))


# PORTED FROM SYMPY: sympy/functions/elementary/complexes.py:_polarify
def _polarify(e: Expr, lift: bool, pause: bool = False) -> Expr:
    from sympy_extras.core.basic import Atom
    from sympy_extras.core.numbers import Exp1
    if e.is_polar:
        return e
    if e.is_number and not pause:
        return polar_lift(e)
    if isinstance(e, Symbol) and not pause and lift:
        return polar_lift(e)
    if isinstance(e, Atom):
        return e
    if isinstance(e, Add):
        rebuilt = add(*[_polarify(a, lift, pause=True) for a in e.args])
        return polar_lift(rebuilt) if lift else rebuilt
    if isinstance(e, Pow) and isinstance(e.base, Exp1):
        return pow(e.base, _polarify(e.exp, lift))
    if isinstance(e, DefinedFunction):
        return type(e).apply(*[_polarify(a, lift) for a in e.args])
    result = e._evaluated(tuple(_polarify(a, lift, pause=pause) if isinstance(a, Expr) else a for a in e.args))
    return result if isinstance(result, Expr) else e


# PORTED FROM SYMPY: sympy/functions/elementary/complexes.py:polarify
def polarify(e: ExprLike, subs: bool = True, lift: bool = False) -> tuple[Expr, dict[Symbol, Symbol]]:
    """``e`` with its numbers (and, with ``lift``, its symbols) turned into
    polar numbers; with ``subs`` its symbols are replaced by polar dummies,
    returned with the map back (as SymPy returns them). Without ``subs``
    the map is empty."""
    if lift:
        subs = False
    result = _polarify(to_expr(e), lift)
    if not subs:
        return (result, {})
    replacements = {x: Dummy(x.name, polar=True) for x in result.free_symbols if isinstance(x, Symbol)}
    substituted = result.subs(list(replacements.items()))
    if not isinstance(substituted, Expr):
        raise TypeError("an expression is expected")
    return (substituted, {r: x for x, r in replacements.items()})


# PORTED FROM SYMPY: sympy/functions/elementary/complexes.py:_unpolarify
def _unpolarify(e: Basic, exponents_only: bool, pause: bool = False) -> Basic:
    from sympy_extras.core.basic import Atom
    from sympy_extras.core.logic import Boolean
    from sympy_extras.core.relational import Eq, Ne, Relational
    from .exponential import ExpPolar
    if isinstance(e, Atom) or not e.args:
        return e
    if not pause:
        if isinstance(e, ExpPolar):
            inner = _unpolarify(e.args[0], exponents_only)
            return exp(inner) if isinstance(inner, Expr) else e
        if isinstance(e, PrincipalBranch) and e.args[1] == mul(Integer(2), pi):
            return _unpolarify(e.args[0], exponents_only)
        relation_ok = isinstance(e, Relational) and (not isinstance(e, (Eq, Ne)) or ZERO in e.args)
        if isinstance(e, (Add, Mul)) or (isinstance(e, Boolean) and not isinstance(e, Relational)) or relation_ok:
            return e._evaluated(tuple(_unpolarify(a, exponents_only) for a in e.args))
        if isinstance(e, PolarLift):
            return _unpolarify(e.args[0], exponents_only)
    if isinstance(e, Pow):
        exponent = _unpolarify(e.exp, exponents_only)
        if not isinstance(exponent, Expr):
            return e
        base = _unpolarify(e.base, exponents_only, not (exponent.is_integer and not pause))
        return pow(base, exponent) if isinstance(base, Expr) else e
    if isinstance(e, DefinedFunction) and e.unbranched:
        return e._evaluated(tuple(_unpolarify(a, exponents_only, exponents_only) for a in e.args))
    return e._evaluated(tuple(_unpolarify(a, exponents_only, True) for a in e.args))


# PORTED FROM SYMPY: sympy/functions/elementary/complexes.py:unpolarify
def unpolarify(e: ExprLike, exponents_only: bool = False) -> Expr:
    """``e`` with its polar numbers turned into ordinary ones where that
    does not change its value: ``exp_polar`` into ``exp``, ``polar_lift``
    removed, ``principal_branch(z, 2*pi)`` into ``z``, repeated until
    nothing changes."""
    from .exponential import exp_polar
    current: Basic = to_expr(e)
    pause = exponents_only
    while True:
        result = _unpolarify(current, exponents_only, pause)
        if result == current:
            break
        current = result
    replaced = current.subs([(exp_polar(ZERO), ONE), (polar_lift(ZERO), ZERO)])
    if not isinstance(replaced, Expr):
        raise TypeError("an expression is expected")
    return replaced



# ---------------------------------------------------------------------
# adjoint and transpose

# PORTED FROM SYMPY: sympy/core/expr.py:Expr._eval_transpose (and transpose, adjoint)
def _commutative_like(e: Expr) -> dict[str, bool]:
    """The commutativity of a node taken from its argument."""
    commutative = facts_of(e).get('commutative')
    return {} if commutative is None else {'commutative': commutative}


def transpose_of(e: Expr) -> Optional[Expr]:
    """The transpose by the rules of ``e``'s class (SymPy's
    ``_eval_transpose``): ``e`` when commutative, the reversed product of
    the transposes for a product, the sum of the transposes, the power of
    the transpose for an integer power, its conjugate when hermitian, minus
    it when antihermitian; ``None`` when unknown."""
    if isinstance(e, Transpose):
        return e.args[0]
    if isinstance(e, Adjoint):
        return conjugate(e.args[0])
    if isinstance(e, Conjugate):
        return adjoint(e.args[0])
    if e.is_commutative:
        return e
    if isinstance(e, Mul):
        return mul(*[transpose(t) for t in reversed(e.args)])
    if isinstance(e, Add):
        return add(*[transpose(t) for t in e.args])
    if isinstance(e, Pow) and e.exp.is_integer:
        return pow(transpose(e.base), e.exp)
    if e.is_commutative:
        return e
    if e.is_hermitian:
        return conjugate(e)
    if e.is_antihermitian:
        return mul(MINUS_ONE, conjugate(e))
    return None


# PORTED FROM SYMPY: sympy/core/expr.py:Expr._eval_adjoint
def adjoint_of(e: Expr) -> Optional[Expr]:
    """The adjoint by the rules of ``e``'s class (SymPy's
    ``_eval_adjoint``)."""
    if isinstance(e, Adjoint):
        return e.args[0]
    if isinstance(e, Transpose):
        return conjugate(e.args[0])
    if isinstance(e, Conjugate):
        return transpose(e.args[0])
    if isinstance(e, Mul) and not e.is_commutative:
        return mul(*[adjoint(t) for t in reversed(e.args)])
    if isinstance(e, Add) and not e.is_commutative:
        return add(*[adjoint(t) for t in e.args])
    if isinstance(e, Pow) and e.exp.is_integer and not e.is_commutative:
        return pow(adjoint(e.base), e.exp)
    if e.is_hermitian:
        return e
    if e.is_antihermitian:
        return mul(MINUS_ONE, e)
    conjugated = conjugate_of(e)
    if conjugated is not None:
        return transpose(conjugated)
    transposed = transpose_of(e)
    if transposed is not None:
        return conjugate(transposed)
    return None


# PORTED FROM SYMPY: sympy/functions/elementary/complexes.py:transpose
class Transpose(DefinedFunction):
    """The transpose, unevaluated (``transpose`` evaluates)."""

    function_name = 'transpose'

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        return transpose_of(args[0])

    def _eval_conjugate(self) -> Optional[Expr]:
        return adjoint(self.args[0])

    def _eval_facts(self) -> dict[str, bool]:
        return _commutative_like(self.args[0])


# PORTED FROM SYMPY: sympy/functions/elementary/complexes.py:adjoint
class Adjoint(DefinedFunction):
    """The conjugate transpose, unevaluated (``adjoint`` evaluates)."""

    function_name = 'adjoint'

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        own = adjoint_of(args[0])
        if own is not None:
            return own
        transposed = transpose_of(args[0])
        if transposed is not None:
            return conjugate(transposed)
        return None

    def _eval_conjugate(self) -> Optional[Expr]:
        return transpose(self.args[0])

    def _eval_facts(self) -> dict[str, bool]:
        return _commutative_like(self.args[0])


def transpose(x: ExprLike) -> Expr:
    """The transpose of ``x``, evaluated."""
    return Transpose.apply(to_expr(x))


def adjoint(x: ExprLike) -> Expr:
    """The adjoint (conjugate transpose) of ``x``, evaluated."""
    return Adjoint.apply(to_expr(x))
