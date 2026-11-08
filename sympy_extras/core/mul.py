"""Products: the class ``Mul`` and the evaluating function ``mul``.

``Mul(*factors)`` is a node with the given factors in the given order,
not simplified. The function ``mul(*factors)`` returns SymPy's canonical
product: nested products are flattened, the numbers are multiplied into
one coefficient, the powers of the same base whose exponents have the
same non-numerical part are collected (``mul(x, x)`` is ``x**2``,
``mul(x**y, x**(2*y))`` is ``x**(3*y)``, while ``x**y*x**z`` is kept),
numerical bases with the same exponent are multiplied (``sqrt(2)*sqrt(3)``
is ``sqrt(6)``), a zero coefficient gives zero, a coefficient alone in
front of a sum is distributed (``mul(2, x + y)`` is ``2*x + 2*y``), the
commutative factors are sorted with the coefficient first, and the
noncommutative ones keep their order (adjacent powers of the same base
are merged). The operators of ``Expr`` call ``mul``; ``Mul(...).doit()``
is ``mul(...)``.

References
==========

.. [1] SymPy Development Team, *SymPy documentation: Mul*,
       https://docs.sympy.org/latest/modules/core.html#sympy.core.mul.Mul

The content-primitive split is ported from SymPy (``sympy/core/mul.py``).
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/core/mul.py (Mul.as_content_primitive, _keep_coeff) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: typed; the rest of the module is original code.
# ---------------------------------------------------------------------
from __future__ import annotations

from fractions import Fraction
from typing import Callable, TYPE_CHECKING, Iterable, Optional

from .add import Add, add
from .basic import Basic
from .expr import Expr, ExprLike, to_expr
from .numbers import (ComplexInfinity, Exp1, Infinity, Integer, NaN, NegativeInfinity, Number, Rational, nan,
    number_from_fraction)
from .operations import AssocOp
from .parameters import global_parameters
from .sorting import ClassKey, compare_key

if TYPE_CHECKING:
    from .printing import StrPrinter
    from .symbol import Symbol

__all__ = ['Mul', 'mul', 'div', 'neg', 'prod']

#: the passes of ``mul`` on a product whose rebuilt factors need
#: flattening again (each pass simplifies strictly; the bound is a guard)
_MAX_PASSES = 8


class Mul(AssocOp):
    """An unevaluated product.

    Examples
    ========

    >>> from sympy_extras.core import symbols, Mul
    >>> x, y = symbols('x y')
    >>> p = Mul(x, 2, x)
    >>> p
    x*2*x
    >>> p.args
    (x, 2, x)
    >>> p.doit()
    2*x**2
    """

    __slots__ = ()

    is_Mul = True
    identity = Integer(1)

    @classmethod
    def class_key(cls) -> ClassKey:
        return (3, 0, 'Mul')

    def _eval_facts(self) -> dict[str, bool]:
        from .assumptions import mul_facts
        return mul_facts(self._args)

    # PORTED FROM SYMPY: sympy/core/mul.py:Mul._eval_is_meromorphic
    def _eval_is_meromorphic(self, x: Symbol, a: Expr) -> Optional[bool]:
        from .fuzzy import fuzzy_group
        return fuzzy_group((t.is_meromorphic(x, a) for t in self._args), quick_exit=True)

    # PORTED FROM SYMPY: sympy/core/mul.py:Mul._eval_nseries
    def _eval_nseries(self, x: Symbol, n: int, logx: Optional[Expr], cdir: Expr) -> Expr:
        from itertools import product
        from .add import _ceiling_int
        from .coreerrors import PoleError
        from .order import Order, order
        from .power import pow
        from .series import _less
        from sympy_extras.simplify.powsimp import powsimp

        def coeff_exp(term: Expr) -> tuple[Expr, Expr]:
            lt = term.as_coeff_exponent(x)
            if lt[0].has(x):
                try:
                    lt = term.leadterm(x)
                except ValueError:
                    return term, Integer(0)
            return lt

        ords: list[tuple[Expr, Expr]] = []
        nn: Expr = Integer(n)
        facs: list[Expr] = []
        try:
            for t in self._args:
                coeff, exp = t.leadterm(x)
                if coeff.has(x):
                    raise ValueError
                ords.append((t, exp))
            n0 = add(*[m for _, m in ords if m.is_number])
            for t, m in ords:
                n1 = _ceiling_int(add(nn, mul(Integer(-1), n0), m if m.is_number else Integer(0)))
                series = t.nseries(x, n=n1, logx=logx, cdir=cdir)
                ns = series.getn()
                if ns is not None and _less(ns, Integer(n1)):
                    # fewer terms than expected: n reduced
                    nn = add(nn, mul(Integer(-1), add(Integer(n1), mul(Integer(-1), ns))))
                facs.append(series)
        except (ValueError, NotImplementedError, TypeError, PoleError):
            n0 = add(*[m for _, m in ords if m.is_number])
            if n0.is_nonnegative is True:
                n0 = Integer(0)
            k = _ceiling_int(add(nn, mul(Integer(-1), n0)))
            facs = [t.nseries(x, n=k, logx=logx, cdir=cdir) for t in self._args]
            fallback = powsimp(mul(*facs).expand(), combine='exp', deep=True)
            if fallback.has(Order):
                fallback = add(fallback, order(pow(x, nn), x))
            return fallback

        res: Expr = Integer(0)
        ords2 = [f._args if isinstance(f, Add) else (f,) for f in facs]
        for fac in product(*ords2):
            ords3 = [coeff_exp(term) for term in fac]
            power = add(*[p for _, p in ords3])
            if add(power, mul(Integer(-1), nn)).is_negative is True:
                res = add(res, mul(mul(*[c for c, _ in ords3]), pow(x, power)))

        if self.is_polynomial(x) is True:
            from sympy_extras.polys.polyerrors import PolynomialError
            try:
                if not _less(_max_degree(self, x), nn) or _degree(self, x) != _degree(res, x):
                    res = add(res, order(pow(x, nn), x))
            except PolynomialError:
                pass
            else:
                return res

        if res != self:
            if add(self, mul(Integer(-1), res)).subs(x, Integer(0)) == Integer(0) and _less(Integer(0), nn):
                lt = self._eval_as_leading_term(x, logx, cdir)
                if lt == Integer(0):
                    return res
            res = add(res, order(pow(x, nn), x))
        return res

    # PORTED FROM SYMPY: sympy/core/mul.py:Mul._eval_as_leading_term
    def _eval_as_leading_term(self, x: Symbol, logx: Optional[Expr], cdir: Expr) -> Expr:
        return mul(*[t.as_leading_term(x, logx=logx, cdir=cdir) for t in self._args])

    # PORTED FROM SYMPY: sympy/core/mul.py:Mul.as_content_primitive
    def as_content_primitive(self, radical: bool = False, clear: bool = True) -> tuple[Number, Expr]:
        content = Fraction(1)
        primitives: list[Expr] = []
        for a in self._args:
            c, p = a.as_content_primitive(radical=radical, clear=clear)
            value = c.exact_value()
            content *= value if value is not None else Fraction(1)
            if not (isinstance(p, Integer) and p.p == 1):
                primitives.append(p)
        return (number_from_fraction(content), mul(*primitives))

    def _evaluated(self, args: tuple[Basic, ...]) -> Basic:
        return mul(*[_as_expr(a) for a in args])

    def _sympystr(self, printer: StrPrinter) -> str:
        return printer.print_mul(self._args)

    def _subs_partial(self, old: Basic, new: Basic) -> Optional[Basic]:
        if not (isinstance(old, Mul) and isinstance(new, Expr)):
            return None
        remaining = list(self._args)
        for factor in old.args:
            if factor not in remaining:
                return None
            remaining.remove(factor)
        return mul(new, *[_as_expr(f._subs(old, new)) for f in remaining])

    def as_coeff_Mul(self, rational: bool = False) -> tuple[Number, Expr]:
        number: Number = Integer(1)
        rest: list[Expr] = []
        for a in self._args:
            if isinstance(a, Number):
                number = number.number_mul(a)
            else:
                rest.append(a)
        if isinstance(number, ComplexInfinity):
            return (Integer(1), self)  # as SymPy: zoo*z + 3*z stays
        return (number, _raw(rest))

    def as_two_terms(self) -> tuple[Expr, Expr]:
        return (self._args[0], _raw(list(self._args[1:])))

    def as_base_exp(self) -> tuple[Expr, Expr]:
        # x**2*y**2 is (x*y)**2, as in SymPy, when every factor has the
        # same integer exponent
        pairs = [f.as_base_exp() for f in self._args]
        exponents = {e for _, e in pairs}
        if len(exponents) == 1:
            exponent = pairs[0][1]
            if isinstance(exponent, Rational) and exponent.q == 1 and exponent.p != 1:
                return (mul(*[b for b, _ in pairs]), exponent)
        return (self, Integer(1))

    def as_numer_denom(self) -> tuple[Expr, Expr]:
        parts = [a.as_numer_denom() for a in self._args]
        return (mul(*[n for n, _ in parts]), mul(*[d for _, d in parts]))

    def could_extract_minus_sign(self) -> bool:
        # SymPy's _coeff_isneg: an extended negative coefficient (-oo*I counts)
        return self.as_coeff_Mul()[0].is_extended_negative is True

    def primitive(self) -> tuple[Number, Expr]:
        # SymPy: the absolute rational coefficient and the rest with the sign
        c, rest = self.as_coeff_Mul()
        if not isinstance(c, Rational) or c.p == 0:
            return (Integer(1), self)
        if c.p < 0:
            return (Rational(-c.p, c.q), mul(Integer(-1), rest))
        return (c, rest)


def _raw(factors: list[Expr]) -> Expr:
    if not factors:
        return Integer(1)
    if len(factors) == 1:
        return factors[0]
    return Mul(*factors)


def _as_expr(b: Basic) -> Expr:
    if not isinstance(b, Expr):
        raise TypeError("an expression is expected, got %s" % (b,))
    return b


#: the commutative factors during flattening, keyed by base and by the
#: non-numerical part of the exponent: the numerical coefficients of the
#: exponents, and the factor itself when it is the only one of its key
_Collected = dict[tuple[Expr, Expr], tuple[list[Number], Expr]]


def mul(*args: ExprLike) -> Expr:
    """The evaluated product of the arguments.

    The factors are taken as already canonical, as SymPy's ``Mul`` does:
    ``mul(Pow(x, 0))`` is ``x**0``, while ``Mul(Pow(x, 0)).doit()``
    evaluates bottom up.

    Examples
    ========

    >>> from sympy_extras.core import symbols, mul, sqrt, Rational
    >>> x, y = symbols('x y')
    >>> mul(x, x)
    x**2
    >>> mul(2, x, 3, y, x)
    6*x**2*y
    >>> mul(2, x + y)
    2*x + 2*y
    >>> mul(sqrt(2), sqrt(6))
    2*sqrt(3)
    >>> mul(x, 0), mul()
    (0, 1)
    """
    factors = [to_expr(a) for a in args]
    if not global_parameters.evaluate:
        return _raw(_flattened(factors))
    # SymPy's AssocOp.__new__: the identities dropped, a single argument
    # returned as it is (Mul(8*(x + 1)) keeps the unevaluated product)
    nonidentity = [f for f in factors if not (isinstance(f, Integer) and f.p == 1)]
    if len(nonidentity) == 1:
        return nonidentity[0]
    if any(f.is_Order for f in _flattened(factors)):
        return _with_orders(factors)
    if any(_is_bounds(f) for f in _flattened(factors)):
        return _with_bounds(factors)
    two = _two_argument_rule(factors)
    if two is not None:
        return two
    for _ in range(_MAX_PASSES):
        result, again = _mul_pass(factors)
        if again is None:
            return result
        factors = again
    return _raw(factors)


# PORTED FROM SYMPY: sympy/core/mul.py:Mul.flatten (the 2-arg hack)
def _two_argument_rule(factors: list[Expr]) -> Optional[Expr]:
    """SymPy's rule for exactly two factors (the factors ``1`` dropped
    first), a nonzero rational ``a`` and ``b = r*s`` with ``s`` a sum:
    ``a*r*s`` stays an undistributed product when ``r != 1`` (``-(1 -
    x)/5`` for ``1/5`` and ``-(1 - x)``), and ``a`` is distributed over
    ``s`` when ``r = 1``; ``None`` otherwise."""
    rest = [f for f in factors if not (isinstance(f, Integer) and f.p == 1)]
    if len(rest) != 2:
        return None
    a, b = rest
    if isinstance(b, Rational):
        a, b = b, a
    if not isinstance(a, Rational) or a.p == 0:
        return None
    r, s = b.as_coeff_Mul()
    if not isinstance(s, Add):
        return None
    if not _is_one(r):
        ar = a.number_mul(r)
        return s if _is_one(ar) else Mul(ar, s)
    if global_parameters.distribute and s.is_commutative is not False:
        return add(*[_keep_coeff(a, t) for t in s.args])
    return None


def _is_one(n: Number) -> bool:
    return isinstance(n, Rational) and n.p == 1 and n.q == 1


def _mul_pass(args: list[Expr]) -> tuple[Expr, Optional[list[Expr]]]:
    """One pass of flattening and collecting; the second component is the
    list of factors to flatten again when a rebuilt factor is a product
    or a number."""
    pow = _pow
    coefficient: Number = Integer(1)
    collected: _Collected = {}
    noncommutative: list[Expr] = []
    for a in _flattened(args):
        if isinstance(a, Number):
            coefficient = coefficient.number_mul(a)
            continue
        if a.is_commutative is False:
            noncommutative.append(a)
            continue
        base, exponent = a.as_base_exp()
        c, rest = exponent.as_coeff_Mul()
        key = (base, rest)
        if key in collected:
            collected[key][0].append(c)
        else:
            collected[key] = ([c], a)
    if isinstance(coefficient, NaN):
        return (nan, None)
    # the commutative factors rebuilt: positive numerical bases with the
    # same exponent are multiplied together
    rebuilt: list[Expr] = []
    numerical: dict[Expr, list[Rational]] = {}
    # a combined power whose base changed (sqrt(I)*sqrt(I) is I, collected
    # as (-1)**(1/2)) is flattened again with the other factors
    rebased = False
    for (base, rest), (cs, factor) in collected.items():
        total: Number = cs[0]
        for c in cs[1:]:
            total = total.number_add(c)
        exponent = rest if _is_one(total) else mul(total, rest)
        if isinstance(base, Rational) and _groupable(base, exponent, factor):
            if base.p > 0 and isinstance(exponent, Rational) and exponent.p < 0:
                base, exponent = Rational(base.q, base.p), Rational(-exponent.p, exponent.q)
            numerical.setdefault(exponent, []).append(base)
            continue
        if len(cs) == 1:
            # SymPy's _gather rebuilds b**(c*t) with c*t evaluated: an
            # unevaluated exponent 2*(n + 1) becomes 2*n + 2
            if factor.is_Pow and factor.as_base_exp()[1] != exponent:
                rebuilt.append(pow(base, exponent))
            elif isinstance(base, Exp1) and factor.args:
                # SymPy rebuilds E**e evaluated: an unevaluated
                # exp(2*log(p)) times x is p**2*x
                from sympy_extras.functions.elementary.exponential import exp
                rebuilt.append(exp(exponent))
            else:
                rebuilt.append(factor)
            continue
        combined = pow(base, exponent)
        if combined.as_base_exp()[0] != base:
            rebased = True
        rebuilt.append(combined)
    paired = _pair_radicals(numerical)
    if paired is not None:
        pairs, numerical = paired
        rebased = True  # the products of the pairs are flattened again
        rebuilt.extend(pairs)
    for exponent, bases in numerical.items():
        if len(bases) == 1:
            rebuilt.append(_single_power(collected, bases[0], exponent))
            continue
        # the bases with the same exponent multiplied: (-5)**(2/3)*2**(2/3)
        # is (-10)**(2/3), (-2)**(1/3)*(-3)**(1/3) is (-1)**(2/3)*6**(1/3)
        negatives = sum(1 for b in bases if b.p < 0)
        product: Number = Integer(1)
        for b in bases:
            product = product.number_mul(Rational(abs(b.p), b.q))
        if negatives == 0:
            rebuilt.append(pow(product, exponent))
        elif negatives == 1:
            rebuilt.append(pow(product.number_mul(Integer(-1)), exponent))
        elif not isinstance(exponent, Number):
            # an integer symbolic exponent: (-1)**(k*m) is 1 for an even k
            sign = Integer(1) if negatives % 2 == 0 else Integer(-1)
            rebuilt.append(pow(product.number_mul(sign), exponent))
        else:
            from .power import minus_one_pow
            if not isinstance(exponent, Rational):
                raise TypeError("a rational exponent is expected")
            rebuilt.append(minus_one_pow(negatives * exponent.as_fraction()))
            rebuilt.append(pow(product, exponent))
    # adjacent noncommutative powers of the same base are merged
    merged: list[Expr] = []
    for f in noncommutative:
        if merged:
            b1, e1 = merged[-1].as_base_exp()
            b2, e2 = f.as_base_exp()
            if b1 == b2 and e1.as_coeff_Mul()[1] == e2.as_coeff_Mul()[1]:
                merged[-1] = pow(b1, add(e1, e2))
                continue
        merged.append(f)
    again = [f for f in rebuilt + merged if f.is_Mul or isinstance(f, Number)]
    if again or rebased:
        return (Integer(0), [coefficient] + rebuilt + merged)
    if isinstance(coefficient, (Infinity, NegativeInfinity)):
        # oo absorbs the factors of known sign
        kept: list[Expr] = []
        for f in rebuilt:
            if f.is_positive is True:
                continue
            if f.is_negative is True:
                coefficient = coefficient.number_mul(Integer(-1))
                continue
            kept.append(f)
        rebuilt = kept
    elif isinstance(coefficient, ComplexInfinity):
        # zoo absorbs the nonzero factors known real or known non-real
        rebuilt = [f for f in rebuilt if not (f.is_zero is False and f.is_extended_real is not None)]
    rebuilt.sort(key=compare_key)
    if not rebuilt and not merged:
        return (coefficient, None)
    zero = coefficient.exact_value()
    if zero is not None and zero == 0:
        # SymPy's Mul.flatten: zero times a factor known to be infinite is nan
        if any(f.is_finite is False for f in rebuilt + merged):
            return (nan, None)
        # SymPy distributes a rational coefficient over a single sum first:
        # 0*(a + zoo) is 0*a + 0*zoo = nan
        others = rebuilt + merged
        if len(others) == 1 and isinstance(others[0], Add) and global_parameters.distribute \
                and any(isinstance(t, (Infinity, NegativeInfinity, ComplexInfinity)) or t.is_finite is False
                        for t in others[0].args):
            return (nan, None)
        return (Integer(0), None)
    one = _is_one(coefficient)
    if one and len(rebuilt) + len(merged) == 1:
        return ((rebuilt + merged)[0], None)
    if not one and len(rebuilt) == 1 and not merged and coefficient.is_finite_number \
            and global_parameters.distribute:
        only = rebuilt[0]
        if isinstance(only, Add):
            return (add(*[mul(coefficient, t) for t in only.args]), None)
    factors = ([] if one else [coefficient]) + rebuilt + merged
    return (Mul(*factors), None)


# PORTED FROM SYMPY: sympy/core/mul.py:Mul.flatten (the order factors) and AssocOp.__new__
def _with_orders(factors: list[Expr]) -> Expr:
    """A product with ``Order`` factors: the product of their expressions
    and of the other factors, in the order term of their merged variables."""
    from .order import Order, order
    order_symbols: Optional[tuple[tuple[Symbol, Expr], ...]] = None
    plain: list[Expr] = []
    for f in _flattened(factors):
        if isinstance(f, Order):
            e, order_symbols = f.as_expr_variables(order_symbols)
            plain.append(e)
        else:
            plain.append(f)
    product = mul(*plain)
    if order_symbols is None:
        return product
    from .containers import Tuple
    return order(product, *[Tuple(v, p) for v, p in order_symbols])


def _degree(e: Expr, x: Symbol) -> int:
    """The degree of ``e`` in ``x`` (SymPy's ``degree``; through SymPy over
    the domains of coefficients not ported yet)."""
    from sympy_extras.polys.polyerrors import CoercionFailed
    from sympy_extras.polys.polytools import degree
    try:
        return degree(e, x)
    except CoercionFailed:
        import sympy
        from sympy_extras.polys.polyerrors import PolynomialError
        from .interop import to_sympy
        try:
            value = sympy.degree(to_sympy(e), to_sympy(x))
        except sympy.PolynomialError as error:
            raise PolynomialError(str(error)) from error
        if not isinstance(value, (int, sympy.Integer)):
            raise PolynomialError("the degree of %s is %s" % (e, value))
        return int(value)


# PORTED FROM SYMPY: sympy/core/mul.py:Mul._eval_nseries (max_degree)
def _max_degree(e: Expr, x: Symbol) -> Expr:
    from .power import Pow
    if e == x:
        return Integer(1)
    if not e.args:
        return Integer(0)
    if isinstance(e, Add):
        degrees = [_max_degree(_expr_arg(a), x) for a in e.args]
        best = degrees[0]
        from .series import _less
        for d in degrees[1:]:
            if _less(best, d):
                best = d
        return best
    if isinstance(e, Mul):
        return add(*[_max_degree(_expr_arg(a), x) for a in e.args])
    if isinstance(e, Pow):
        return mul(_max_degree(e.base, x), e.exp)
    return Integer(0)


def _expr_arg(b: Basic) -> Expr:
    if not isinstance(b, Expr):
        raise TypeError("an expression is expected, got %s" % (b,))
    return b


def _flattened(args: list[Expr]) -> list[Expr]:
    """The factors with the nested products opened, in order."""
    result: list[Expr] = []
    for a in args:
        if isinstance(a, Mul):
            result.extend(_flattened(list(a.args)))
        else:
            result.append(a)
    return result


# PORTED FROM SYMPY: sympy/core/mul.py:Mul.flatten (the gcd pairing of num_rat)
def _pair_radicals(numerical: dict[Expr, list[Rational]]
                   ) -> Optional[tuple[list[Expr], dict[Expr, list[Rational]]]]:
    """SymPy's combination of rational powers of positive numbers with
    different exponents through the gcd of their bases: ``sqrt(2)*6**(1/4)``
    is ``2**(3/4)*3**(1/4)``. Returns the new factors and the groups left
    alone, or ``None`` when nothing combines."""
    from math import gcd
    from .power import pow
    candidates = [(e, bases) for e, bases in numerical.items()
                  if isinstance(e, Rational) and e.q > 1 and all(b.p > 0 and b.q == 1 for b in bases)]
    if len(candidates) < 2:
        return None
    num_rat: list[tuple[int, Fraction]] = []
    for e, bases in candidates:
        product = 1
        for b in bases:
            product *= b.p
        num_rat.append((product, e.as_fraction()))
    changed = False
    coefficient_parts: list[Expr] = []
    pnew: dict[Fraction, list[int]] = {}
    i = 0
    while i < len(num_rat):
        bi, ei = num_rat[i]
        if bi == 1:
            i += 1
            continue
        grow: list[tuple[int, Fraction]] = []
        for j in range(i + 1, len(num_rat)):
            bj, ej = num_rat[j]
            g = gcd(bi, bj)
            if g != 1:
                changed = True
                es = ei + ej
                if es.denominator == 1:
                    coefficient_parts.append(pow(Integer(g), Integer(es.numerator)))
                else:
                    if es.numerator > es.denominator:
                        e_i, ep = divmod(es.numerator, es.denominator)
                        coefficient_parts.append(pow(Integer(g), Integer(e_i)))
                        es = Fraction(ep, es.denominator)
                    grow.append((g, es))
                num_rat[j] = (bj // g, ej)
                bi = bi // g
                if bi == 1:
                    break
        if bi != 1:
            pnew.setdefault(ei, []).append(bi)
        num_rat.extend(grow)
        i += 1
    if not changed:
        return None
    rest = {e: bases for e, bases in numerical.items() if (e, bases) not in candidates}
    factors = coefficient_parts
    for ef, bs in pnew.items():
        base_product = 1
        for bv in bs:
            base_product *= bv
        factors.append(pow(Integer(base_product), number_from_fraction(ef)))
    return factors, rest


def _groupable(base: Rational, exponent: Expr, factor: Expr) -> bool:
    """Whether the numerical ``base**exponent`` may be multiplied with the
    other numerical bases of the same exponent: a positive base with a
    non-integer rational or a symbolic exponent (``sqrt(2)*sqrt(3)``,
    ``2**x*3**x``), a negative base under a rational exponent of
    denominator at least 3 (``I`` and ``sqrt(2)`` stay apart)."""
    from .power import Pow
    if base.p > 0:
        if isinstance(exponent, Rational):
            return exponent.q > 1
        return not isinstance(exponent, Number)
    if base.p < 0 and isinstance(factor, Pow):
        if not isinstance(exponent, Number) and exponent.is_integer is True:
            return True  # SymPy: (-1)**m*2**m is (-2)**m for an integer m
        return isinstance(exponent, Rational) and exponent.q > 2 and exponent.p > 0
    return False


def _single_power(collected: _Collected, base: Rational, exponent: Expr) -> Expr:
    """``base**exponent`` for a numerical base alone in its exponent group:
    the factor as given when its exponent did not change."""
    from .power import Pow, pow
    single = collected.get((base, exponent.as_coeff_Mul()[1]))
    if single is not None and len(single[0]) == 1:
        factor = single[1]
        if isinstance(factor, Pow) and factor.base == base and factor.exp == exponent:
            return factor
    return pow(base, exponent)


def div(a: ExprLike, b: ExprLike) -> Expr:
    """The evaluated quotient ``a / b``.

    Examples
    ========

    >>> from sympy_extras.core import symbols, div, Integer
    >>> x, y = symbols('x y')
    >>> div(x, y), div(Integer(1), 2), div(x, x)
    (x/y, 1/2, 1)
    """
    from .power import pow
    numerator = to_expr(a)
    if isinstance(numerator, Rational) and numerator.p == 1 and numerator.q == 1:
        return pow(b, Integer(-1))
    return mul(numerator, pow(b, Integer(-1)))


def prod(a: Iterable[ExprLike], start: ExprLike = 1) -> Expr:
    """The product of the elements of ``a`` times ``start``.

    Examples
    ========

    >>> from sympy_extras.core import symbols, prod
    >>> x, y = symbols('x y')
    >>> prod([x, y, 2]), prod([], 3)
    (2*x*y, 3)
    """
    return mul(start, *a)


def neg(a: ExprLike) -> Expr:
    """The evaluated opposite ``-a``.

    Examples
    ========

    >>> from sympy_extras.core import symbols, neg
    >>> x, y = symbols('x y')
    >>> neg(x), neg(x + y), neg(-2*x)
    (-x, -x - y, 2*x)
    """
    from .add import Add, add
    from .expr import to_expr
    e = to_expr(a)
    if isinstance(e, Number):
        return mul(Integer(-1), e)
    if e.is_Order:
        # PORTED FROM SYMPY: sympy/series/order.py:Order.__neg__
        return e
    if isinstance(e, Add):
        # PORTED FROM SYMPY: sympy/core/add.py:Add.__neg__ (distribute)
        return add(*[neg(t) for t in e.args if isinstance(t, Expr)])
    if isinstance(e, Mul):
        # PORTED FROM SYMPY: sympy/core/mul.py:Mul.__neg__ (no flattening)
        factors = list(e.args)
        first = factors[0]
        if isinstance(first, Number):
            negated = mul(Integer(-1), first)
            if _is_rational_value(negated, 1):
                rest = [f for f in factors[1:] if isinstance(f, Expr)]
                return rest[0] if len(rest) == 1 else Mul(*rest)
            return Mul(negated, *factors[1:])
        return Mul(Integer(-1), *factors)
    # PORTED FROM SYMPY: sympy/core/expr.py:Expr.__neg__ (Mul._from_args)
    return Mul(Integer(-1), e)

_POW: Optional[Callable[[ExprLike, ExprLike], Expr]] = None


def _pow(b: ExprLike, e: ExprLike) -> Expr:
    """``pow`` of the sibling module, bound at the first call (it imports this one)."""
    global _POW
    if _POW is None:
        from .power import pow
        _POW = pow
    return _POW(b, e)


def _is_rational_value(e: Expr, value: int) -> bool:
    """Whether ``e`` is the rational number ``value`` (an ``Integer`` or a
    ``Rational`` with denominator 1: SymPy's ``S.One``, ``S.NegativeOne``)."""
    return isinstance(e, Rational) and e.q == 1 and e.p == value


# PORTED FROM SYMPY: sympy/core/mul.py:_keep_coeff
def _keep_coeff(coeff: Expr, factors: Expr, clear: bool = True, sign: bool = False) -> Expr:
    """``coeff*factors`` without distributing a number over a sum
    (``2*(x + y)``), the number put first in a product."""
    if not isinstance(coeff, Number):
        if isinstance(factors, Number):
            factors, coeff = coeff, factors
        else:
            return mul(coeff, factors)
    if _is_rational_value(factors, 1):
        return coeff
    if _is_rational_value(coeff, 1):
        return factors
    if _is_rational_value(coeff, -1) and not sign:
        return mul(Integer(-1), factors)
    if isinstance(factors, Add):
        if not clear and isinstance(coeff, Rational) and coeff.q != 1:
            pairs = [t.as_coeff_Mul() for t in factors.args]
            scaled = [(_keep_coeff(c, coeff), m) for c, m in pairs]
            if any(isinstance(c, Integer) for c, _ in scaled):
                return Add(*[m if _is_rational_value(c, 1) else Mul(c, m) for c, m in scaled])
        return Mul(coeff, factors)
    if isinstance(factors, Mul):
        margs = list(factors.args)
        first = margs[0]
        if isinstance(first, Number):
            product = mul(first, coeff)
            if _is_rational_value(product, 1):
                margs.pop(0)
            else:
                margs[0] = product
        else:
            margs.insert(0, coeff)
        return Mul(*margs) if len(margs) != 1 else margs[0]
    m = mul(coeff, factors)
    if isinstance(m, Number) and not isinstance(factors, Number):
        return Mul(coeff, factors)
    return m


def _is_bounds(t: Basic) -> bool:
    from .accumbounds import AccumBounds
    return isinstance(t, AccumBounds)


# PORTED FROM SYMPY: sympy/core/mul.py:Mul.flatten (the AccumBounds coefficient)
def _with_bounds(factors: list[Expr]) -> Expr:
    """A product with accumulation bounds: the numbers and the bounds make
    the coefficient (``2*AccumBounds(-1, 1)`` is ``AccumBounds(-2, 2)``)."""
    from .accumbounds import AccumBounds, ab_mul
    coeff: Expr = Integer(1)
    others: list[Expr] = []
    for o in _flattened(factors):
        if isinstance(o, Number):
            if isinstance(coeff, Number):
                coeff = coeff.number_mul(o)
            elif isinstance(coeff, AccumBounds):
                coeff = ab_mul(coeff, o)
        elif isinstance(o, AccumBounds):
            coeff = ab_mul(o, coeff)
        else:
            others.append(o)
    rest = mul(*others)
    if not isinstance(coeff, AccumBounds):
        return mul(coeff, rest)
    if isinstance(rest, Integer) and rest.p == 1:
        return coeff
    return Mul(coeff, *(list(rest.args) if isinstance(rest, Mul) else [rest]))
