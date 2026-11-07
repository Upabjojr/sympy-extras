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
* ``sign``'s derivative is a ``DiracDelta``: not ported;

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

from sympy_extras.core.add import Add, add
from sympy_extras.core.assumptions import facts_of
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.function import AppliedUndef, Derivative, DefinedFunction, diff, expand_mul
from sympy_extras.core.mul import Mul, mul
from sympy_extras.core.numbers import (ComplexInfinity, I, ImaginaryUnit, Infinity, Integer, NaN, NegativeInfinity,
                                       Number, Rational, nan, oo, pi)
from sympy_extras.core.power import Pow, pow, sqrt
from sympy_extras.core.symbol import Dummy, Symbol
from sympy_extras.functions.elementary.exponential import Exp, exp, log
from sympy_extras.functions.elementary.trigonometric import atan, cos, sin

__all__ = ['Re', 'Im', 'Abs', 'Sign', 'Arg', 'Conjugate', 'Atan2', 're', 'im', 'abs_', 'sign', 'arg', 'conjugate',
           'atan2', 'as_real_imag', 'conjugate_of']

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
    if real:
        return add(Re(a) if a != 0 else ZERO, mul(MINUS_ONE, im(b)), c)
    return add(Im(a) if a != 0 else ZERO, re(b), c)


# PORTED FROM SYMPY: sympy/functions/elementary/complexes.py:re
class Re(DefinedFunction):
    """The real part, unevaluated (``re`` evaluates)."""

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
            return mul(s, cls(rest)) if unknown else s
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
        return Derivative(self, s)  # SymPy: 2*Derivative(arg)*DiracDelta(arg), not ported

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

    function_name = 'Abs'

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        [arg] = args
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
        return {'commutative': True}


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

