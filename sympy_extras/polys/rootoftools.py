"""``RootSum``: the sum of a function over the roots of a polynomial, on
the typed core.

Ported from SymPy's ``sympy.polys.rootoftools`` (``RootSum``): the node
:class:`RootSum` holds the polynomial (an expression in its bound
variable), the function (a :class:`~sympy_extras.core.function.Lambda`)
and the variable, and never evaluates; the function :func:`root_sum`
evaluates as SymPy's constructor does. The polynomial is preprocessed
(``preprocess_roots``: denominators and content cleared, a common power
of a parameter or an integer basis scaled out of the roots), the terms of
the function free of the variable are taken out, and the polynomial is
factored: the sum over the roots of a linear factor (of a quadratic one
with ``quadratic=True``) is written with the roots, a factor of higher
degree gives a ``RootSum``, and a function rational in the variable gives
a rational function of the coefficients (the trace of the function in
``K[t]/(q)``, by Newton's sums, where SymPy uses ``symmetrize`` and
``viete``).

``RootSum.doit`` writes the sum with the roots (``roots``) when they are
all found, the derivative is a ``RootSum`` of the derivative, and a
``RootSum`` without free symbols has a numerical value (the roots by
``mpmath.polyroots``). Coefficients which are not polynomials over ``ZZ``
or ``QQ`` in parameters (``sqrt(2)``) are taken as elements of ``EX``: the
polynomial is not factored, as in SymPy.

Examples
========

>>> from sympy_extras.core import symbols, Lambda
>>> from sympy_extras.functions import log
>>> from sympy_extras.polys.rootoftools import RootSum, root_sum
>>> t, x = symbols('t x')
>>> r = root_sum(t**3 - t - 1, Lambda(t, t*log(x + t)), t)
>>> r
RootSum(t**3 - t - 1, Lambda(t, t*log(t + x)))
>>> r.diff(x)
-(2*x - 3)/(x**3 - x + 1)
>>> root_sum(t**3 - 1, Lambda(t, t*log(x + t)), t, quadratic=True)
log(x + 1) + (-1/2 - sqrt(3)*I/2)*log(x - 1/2 - sqrt(3)*I/2) + (-1/2 + sqrt(3)*I/2)*log(x - 1/2 + sqrt(3)*I/2)
>>> root_sum(t**3 - t - 1, Lambda(t, 1/(x + t)), t)
(3*x**2 - 1)/(x**3 - x + 1)
>>> RootSum(t**2 - 2, Lambda(t, t**2), t).doit()
4
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/polys/rootoftools.py (RootSum, _pure_factors) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: an unevaluating node RootSum and the evaluating function
# root_sum; the rational case by the trace in K[t]/(q) instead of
# symmetrize and viete; the derivative of a function constant in the
# variable is the degree times it (SymPy returns the function itself).
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Final, Optional

import mpmath

from sympy_extras.core.add import add
from sympy_extras.core.basic import Basic, NodePrinter, ReplacementRule
from sympy_extras.core.expr import Expr, as_expr
from sympy_extras.core.function import Lambda
from sympy_extras.core.mul import mul
from sympy_extras.core.numbers import Float, I, Integer, number_from_fraction
from sympy_extras.core.power import Pow, pow
from sympy_extras.core.symbol import Symbol

from .densearith import dup_mul, dup_rem
from .densebasic import dup_degree, dup_strip
from .domains.expressiondomain import EX, ex_normal
from .euclidtools import dup_invert
from .polyerrors import BasePolynomialError
from .polytools import Poly, cancel, factor

__all__ = ['RootSum', 'root_sum']

_ZERO: Final[Expr] = Integer(0)
_ONE: Final[Expr] = Integer(1)

#: what the polynomial routines raise on coefficients they cannot take
_FAILURES = (BasePolynomialError, NotImplementedError, ValueError, TypeError, ZeroDivisionError)


def polynomial_coefficients(f: Expr, t: Symbol) -> list[Expr]:
    """The coefficients of the polynomial ``f`` in ``t``, highest degree
    first (SymPy's ``Poly(f, t).all_coeffs()`` over any domain: the
    expanded terms grouped by their power of ``t``); ``ValueError`` when
    ``f`` is no polynomial in ``t``."""
    from .polytools import _decompose_with_gens
    from .polyerrors import PolynomialError
    try:
        decomposed = _decompose_with_gens(f, (t,))
    except PolynomialError as error:
        raise ValueError(str(error))
    terms: dict[int, Expr] = {}
    for monomial, parts in decomposed.items():
        value = add(*parts)
        if value != _ZERO:
            terms[monomial[0]] = value
    if not terms:
        return []
    n = max(terms)
    return [terms.get(k, _ZERO) for k in range(n, -1, -1)]


def expr_of_coefficients(coefficients: list[Expr], t: Expr) -> Expr:
    """``sum c_k t**k`` (SymPy's ``Poly.as_expr``) of coefficients given
    highest degree first."""
    n = len(coefficients) - 1
    terms: list[Expr] = []
    for i, c in enumerate(coefficients):
        if c == _ZERO:
            continue
        k = n - i
        terms.append(mul(c, pow(t, Integer(k))) if k else c)
    return add(*terms) if terms else _ZERO


def _xreplace(e: Expr, rule: ReplacementRule) -> Expr:
    value = e._xreplace_evaluated(rule)
    if not isinstance(value, Expr):
        raise TypeError("%s is not an expression" % (value,))
    return value


def _call(fun: Lambda, value: Expr) -> Expr:
    """SymPy's ``Lambda.__call__``: the body with the variable replaced
    (``xreplace``)."""
    return _xreplace(fun.expr, {fun.variables[0]: value})


def _is_radical_number(e: Basic) -> bool:
    """A number which SymPy's ``construct_domain`` takes as algebraic (the
    domain ``EX``): a radical of a number, ``I``."""
    from sympy_extras.core.numbers import ImaginaryUnit, Rational
    if isinstance(e, ImaginaryUnit):
        return True
    return isinstance(e, Pow) and isinstance(e.exp, Rational) and e.exp.q != 1 and not e.base.free_symbols


def has_radical_number(e: Expr) -> bool:
    """Whether ``e`` has a radical of a number or ``I``."""
    from sympy_extras.core.basic import preorder_traversal
    return any(_is_radical_number(node) for node in preorder_traversal(e))


def generic_constants(e: Expr, radicals: bool = False) -> tuple[Expr, dict[Basic, Basic]]:
    """``e`` with its constants which are not rational numbers (``pi``,
    ``tan(1)``, ``exp(2)``; with ``radicals`` also ``sqrt(2)`` and ``I``)
    replaced by dummies, as SymPy's ``construct_domain`` takes the
    transcendental ones as generators of a polynomial domain, and the map
    back."""
    from sympy_extras.core.add import Add
    from sympy_extras.core.mul import Mul
    from sympy_extras.core.numbers import Rational
    from sympy_extras.core.symbol import Dummy
    forward: dict[Basic, Basic] = {}
    back: dict[Basic, Basic] = {}

    def walk(node: Expr) -> None:
        if isinstance(node, Rational) or node.free_symbols and not node.args:
            return
        if not node.free_symbols and not isinstance(node, (Add, Mul)) and \
                (radicals or not has_radical_number(node)) and \
                not (isinstance(node, Pow) and isinstance(node.exp, Rational) and node.exp.q == 1):
            if node not in forward:
                d = Dummy('c%d' % len(forward))
                forward[node] = d
                back[d] = node
            return
        for a in node.args:
            if isinstance(a, Expr):
                walk(a)

    walk(e)
    if not forward:
        return e, back
    return _xreplace(e, forward), back


class _Polynomial:
    """The polynomial of a ``RootSum`` after ``preprocess_roots``: a
    native ``Poly`` over ``ZZ``, ``QQ`` or ``ZZ[params]`` (the
    transcendental constants of the coefficients replaced by dummies,
    ``back`` restores them), or the coefficients over ``EX`` (``poly`` is
    ``None``). ``coefficients`` are the restored ones."""

    def __init__(self, poly: Optional[Poly], coefficients: list[Expr], t: Symbol,
                 back: Optional[dict[Basic, Basic]] = None) -> None:
        self.poly: Final[Optional[Poly]] = poly
        self.back: Final[dict[Basic, Basic]] = back if back is not None else {}
        self.coefficients: Final[list[Expr]] = [self.restored(c) for c in coefficients]
        self.t: Final[Symbol] = t

    def restored(self, e: Expr) -> Expr:
        return _xreplace(e, self.back) if self.back else e

    @property
    def degree(self) -> int:
        return len(self.coefficients) - 1

    def as_expr(self) -> Expr:
        if self.poly is not None:
            return self.restored(self.poly.as_expr())
        return expr_of_coefficients(self.coefficients, self.t)


def _transformed(f: Expr, t: Symbol) -> tuple[Expr, _Polynomial]:
    """SymPy's ``RootSum._transform``: ``(coeff, poly)``, the roots of
    ``f`` being ``coeff`` times those of ``poly`` (``preprocess_roots``);
    over ``EX`` (radicals of numbers) the denominators cleared only."""
    from .polyroots import preprocess_roots
    if len(polynomial_coefficients(f, t)) < 1:
        raise ValueError('the zero polynomial has no roots to sum over')
    numerator, _ = cancel(f, t).as_numer_denom()
    if has_radical_number(numerator):
        return _ONE, _Polynomial(None, polynomial_coefficients(numerator, t), t)
    generic, back = generic_constants(numerator)
    try:
        native = Poly(generic, t)
    except _FAILURES:
        return _ONE, _Polynomial(None, polynomial_coefficients(numerator, t), t)
    if native.gens != (t,):
        return _ONE, _Polynomial(None, polynomial_coefficients(numerator, t), t)
    coeff, poly = preprocess_roots(native)
    p = _Polynomial(poly, [c for c in poly.all_coeffs()], t, back)
    return p.restored(coeff), p


def _pure_factors(p: _Polynomial) -> list[tuple[_Polynomial, int]]:
    """SymPy's ``_pure_factors``: the irreducible factors over the domain
    (the polynomial itself over ``EX``, which SymPy does not factor)."""
    if p.poly is None:
        return [(p, 1)]
    _, factors = p.poly.factor_list()
    return [(_Polynomial(g, list(g.all_coeffs()), p.t, p.back), k) for g, k in factors]


def _simplified(e: Expr) -> Expr:
    """SymPy's ``simplify`` of a root over ``EX`` (within the time limit)."""
    from sympy_extras._timeout import attempt
    from sympy_extras.settings import settings
    from sympy_extras.simplify.simplify import simplify
    limit = settings.timeout
    found = attempt(lambda: simplify(e), None if limit is None else limit/10, native=True)
    return e if found is None else found


# PORTED FROM SYMPY: sympy/polys/polyroots.py:roots_linear (the domain EX)
def _linear_root(p: _Polynomial) -> Expr:
    from .polyroots import roots_linear
    if p.poly is not None:
        return p.restored(roots_linear(p.poly)[0])
    b, c = p.coefficients
    return _simplified(mul(Integer(-1), c, pow(b, Integer(-1))))


# PORTED FROM SYMPY: sympy/polys/polyroots.py:roots_quadratic (the domain EX)
def _quadratic_roots(p: _Polynomial) -> list[Expr]:
    from .polyroots import _sqrt, roots_quadratic
    from sympy_extras.core.exprtools import factor_terms
    if p.poly is not None:
        return [p.restored(r) for r in roots_quadratic(p.poly)]
    a, b, c = p.coefficients
    if c == _ZERO:
        return [_ZERO, _simplified(mul(Integer(-1), b, pow(a, Integer(-1))))]
    if b == _ZERO:
        R = _sqrt(_simplified(mul(Integer(-1), c, pow(a, Integer(-1)))))
        return [mul(Integer(-1), R), R]
    d = _simplified(add(pow(b, Integer(2)), mul(Integer(-4), a, c)))
    A = mul(Integer(2), a)
    B = _simplified(mul(Integer(-1), b, pow(A, Integer(-1))))
    D = factor_terms(mul(_sqrt(d), pow(A, Integer(-1))))
    r0, r1 = add(B, mul(Integer(-1), D)), add(B, D)
    if a.is_negative is True:
        r0, r1 = r1, r0
    else:
        from .polyroots import _expand_2arg
        r0, r1 = _expand_2arg(r0), _expand_2arg(r1)
    return [r0, r1]


def _rational_case(p: _Polynomial, fun: Lambda) -> Optional[Expr]:
    """The sum of the rational function ``fun`` over the roots of the
    irreducible ``p``: the trace of ``N/D`` in ``K[t]/(p)``, ``N/D`` reduced
    to a polynomial ``S`` of degree below ``n = deg p`` (``D`` inverted
    modulo ``p``) and ``sum s_k p_k`` with Newton's power sums ``p_k``;
    factored, as SymPy factors its symmetric reduction. ``None`` when
    ``D`` vanishes at the roots."""
    var = fun.variables[0]
    n = p.degree
    numerator, denominator = cancel(fun.expr, var).as_numer_denom()
    try:
        N = [EX.from_expr(c) for c in polynomial_coefficients(numerator, var)]
        D = [EX.from_expr(c) for c in polynomial_coefficients(denominator, var)]
    except ValueError:
        return None
    lc = p.coefficients[0]
    q = dup_strip([EX.exquo(c, lc) for c in p.coefficients], EX)
    try:
        inverse = dup_invert(dup_rem(D, q, EX), q, EX)
    except (ValueError, ZeroDivisionError):
        return None
    S = dup_rem(dup_mul(N, inverse, EX), q, EX)
    # Newton's sums of the monic q = t**n + c[1] t**(n-1) + ... + c[n]
    c = q
    sums: list[Expr] = [Integer(n)]
    for k in range(1, n):
        total = mul(Integer(k), c[k])
        for i in range(1, k):
            total = EX.add(total, EX.mul(c[i], sums[k - i]))
        sums.append(EX.neg(total) if total != _ZERO else _ZERO)
    trace: Expr = _ZERO
    m = dup_degree(S)
    for i, s in enumerate(S):
        trace = EX.add(trace, EX.mul(s, sums[m - i]))
    return factor(ex_normal(trace))


def _root_sum(poly: _Polynomial, fun: Lambda, auto: bool) -> Expr:
    """SymPy's ``RootSum.new``: a function free of the variable times the
    degree, the rational case, or the node."""
    var = fun.variables[0]
    if not fun.expr.has(var):
        return mul(Integer(poly.degree), fun.expr)
    if auto and fun.expr.is_rational_function(var) is True:
        value = _rational_case(poly, fun)
        if value is not None:
            return value
    return RootSum(poly.as_expr(), fun, poly.t)


# PORTED FROM SYMPY: sympy/polys/rootoftools.py:RootSum.__new__
def root_sum(f: Expr, fun: Optional[Lambda] = None, t: Optional[Symbol] = None, auto: bool = True,
             quadratic: bool = False) -> Expr:
    """The sum of ``fun`` over the roots of the polynomial ``f`` in ``t``
    (counted with multiplicity), evaluated as SymPy's ``RootSum``
    evaluates: the sums over the linear factors (and the quadratic ones,
    with ``quadratic``) written with the roots, the rational functions of
    the roots written in the coefficients (with ``auto``), the rest as
    :class:`RootSum` nodes of the irreducible factors.

    >>> from sympy_extras.core import symbols, Lambda
    >>> from sympy_extras.functions import log
    >>> from sympy_extras.polys.rootoftools import root_sum
    >>> t, x = symbols('t x')
    >>> root_sum(4*t**3 - 2*t - 2, Lambda(t, t*log(x + t)), t)
    log(x + 1) + RootSum(2*t**2 + 2*t + 1, Lambda(t, t*log(t + x)))
    >>> root_sum(t**3 - t - 1, Lambda(t, 2*t*log(x + t) + 3), t)
    2*RootSum(t**3 - t - 1, Lambda(t, t*log(t + x))) + 9
    """
    if t is None:
        free = sorted(f.free_symbols, key=lambda s: s.sort_key())
        if len(free) != 1:
            raise ValueError("the variable of %s is needed" % (f,))
        t = free[0]
    coeff, poly = _transformed(f, t)
    if fun is None:
        fun = Lambda(t, t)
    var, expr = fun.variables[0], fun.expr
    if coeff != _ONE:
        expr = expr.subs(var, mul(coeff, var))
    deg = poly.degree
    if not expr.has(var):
        return mul(Integer(deg), expr)
    from sympy_extras.core.add import Add
    from sympy_extras.core.mul import Mul
    add_const: Expr = _ZERO
    mul_const: Expr = _ONE
    if isinstance(expr, Add):
        add_const, expr = expr.as_independent(var, as_Add=True)
    if isinstance(expr, Mul):
        mul_const, expr = expr.as_independent(var, as_Add=False)
    func = Lambda(var, expr)
    rational = expr.is_rational_function(var) is True
    terms: list[Expr] = []
    for factor_poly, k in _pure_factors(poly):
        term: Expr
        if factor_poly.degree == 1:
            term = _call(func, _linear_root(factor_poly))
        elif quadratic and factor_poly.degree == 2:
            term = add(_ZERO, *[_call(func, r) for r in _quadratic_roots(factor_poly)])
        else:
            value = _rational_case(factor_poly, func) if rational and auto else None
            term = value if value is not None else RootSum(factor_poly.as_expr(), func, t)
        terms.append(mul(Integer(k), term))
    return add(mul(mul_const, add(*terms)), mul(Integer(deg), add_const))


# PORTED FROM SYMPY: sympy/polys/rootoftools.py:RootSum
class RootSum(Expr):
    """The sum of the function ``fun`` (a :class:`Lambda` of one variable)
    over the roots of the polynomial ``f`` in the bound variable ``t``,
    counted with multiplicity. The constructor checks and stores its
    arguments (:func:`root_sum` evaluates).

    >>> from sympy_extras.core import symbols, Lambda
    >>> from sympy_extras.functions import log
    >>> from sympy_extras.polys.rootoftools import RootSum
    >>> t, x = symbols('t x')
    >>> r = RootSum(t**3 + t + 1, Lambda(t, t*log(x - t)), t)
    >>> r, r.args, r.free_symbols
    (RootSum(t**3 + t + 1, Lambda(t, t*log(-t + x))), (t**3 + t + 1, Lambda(t, t*log(-t + x)), t), {x})
    >>> r.subs(x, 2)
    RootSum(t**3 + t + 1, Lambda(t, t*log(2 - t)))
    >>> r.subs(x, 2).evalf(10)
    1.227313065
    """

    __slots__ = ('_polynomial', '_function', '_variable')

    def __init__(self, f: Expr, fun: Lambda, t: Symbol) -> None:
        if not isinstance(f, Expr):
            raise TypeError("an expression is expected, got %s" % (f,))
        if not isinstance(fun, Lambda) or fun.nargs != 1:
            raise ValueError("expected a univariate function, got %s" % (fun,))
        if not isinstance(t, Symbol):
            raise TypeError("RootSum(f, fun, t) takes the variable t of f")
        self._polynomial: Final[Expr] = f
        self._function: Final[Lambda] = fun
        self._variable: Final[Symbol] = t
        Basic.__init__(self)

    @property
    def args(self) -> tuple[Basic, ...]:
        return (self._polynomial, self._function, self._variable)

    @property
    def expr(self) -> Expr:
        """The polynomial, in the bound variable."""
        return self._polynomial

    @property
    def fun(self) -> Lambda:
        """The function summed over the roots."""
        return self._function

    @property
    def variable(self) -> Symbol:
        """The variable of the polynomial (bound)."""
        return self._variable

    @property
    def coefficients(self) -> list[Expr]:
        """The coefficients of the polynomial, highest degree first."""
        return polynomial_coefficients(self._polynomial, self._variable)

    @property
    def degree(self) -> int:
        return len(self.coefficients) - 1

    @property
    def free_symbols(self) -> set[Symbol]:
        return (self._polynomial.free_symbols - {self._variable}) | self._function.free_symbols

    @property
    def bound_symbols(self) -> tuple[Symbol, ...]:
        return (self._variable,)

    def _is_number_node(self) -> Optional[bool]:
        return not self.free_symbols

    @staticmethod
    def _checked(args: tuple[Basic, ...]) -> tuple[Expr, Lambda, Symbol]:
        f, fun, t = args
        if not (isinstance(f, Expr) and isinstance(fun, Lambda) and isinstance(t, Symbol)):
            raise TypeError("invalid arguments of a RootSum: %s" % (args,))
        return f, fun, t

    def _rebuild(self, args: tuple[Basic, ...]) -> Basic:
        f, fun, t = self._checked(args)
        return RootSum(f, fun, t)

    def _evaluated(self, args: tuple[Basic, ...]) -> Basic:
        f, fun, t = self._checked(args)
        return root_sum(f, fun, t)

    def _replaced(self, f: Basic, body: Basic, evaluated: bool) -> Basic:
        if f is self._polynomial and body is self._function.expr:
            return self
        if not (isinstance(f, Expr) and isinstance(body, Expr)):
            raise TypeError("the replacement made %s a non-expression" % (self,))
        fun = Lambda(self._function.variables[0], body)
        return root_sum(f, fun, self._variable) if evaluated else RootSum(f, fun, self._variable)

    def _subs(self, old: Basic, new: Basic) -> Basic:
        # the variables are bound: a substitution of them leaves them alone
        if self == old:
            return new
        f: Basic = self._polynomial if old == self._variable else self._polynomial._subs(old, new)
        body: Basic = self._function.expr
        if old != self._function.variables[0]:
            body = body._subs(old, new)
        return self._replaced(f, body, True)

    def _xreplace(self, rule: ReplacementRule) -> Basic:
        if self in rule:
            return rule[self]
        return self._replaced(self._polynomial._xreplace(rule), self._function.expr._xreplace(rule), False)

    def _xreplace_evaluated(self, rule: ReplacementRule) -> Basic:
        if self in rule:
            return rule[self]
        return self._replaced(self._polynomial._xreplace_evaluated(rule),
                              self._function.expr._xreplace_evaluated(rule), True)

    def doit(self) -> Expr:
        """The sum written with the roots (``roots``) when they are all
        found, the node otherwise (SymPy's ``RootSum.doit``)."""
        from .polyroots import roots
        try:
            found = roots(self._polynomial, self._variable, multiple=True)
        except _FAILURES:
            return self
        if not isinstance(found, list) or len(found) < self.degree:
            return self
        return add(*[_call(self._function, r) for r in found])

    def _eval_derivative_node(self, s: Symbol) -> Optional[Expr]:
        # PORTED FROM SYMPY: sympy/polys/rootoftools.py:RootSum._eval_derivative
        var = self._function.variables[0]
        fun = Lambda(var, as_expr(self._function.expr.diff(s)))
        poly = _Polynomial(None, self.coefficients, self._variable)
        return _root_sum(poly, fun, True)

    def _numeric_node(self, prec: int) -> Optional[mpmath.mpc]:
        """The sum of the function at the numerical roots."""
        if self.free_symbols:
            return None
        from sympy_extras.core.evalf import mpf_to_fraction, numeric_value
        working = prec + 30
        values: list[mpmath.mpc] = []
        for c in self.coefficients:
            value = numeric_value(c, working)
            if value is None:
                return None
            values.append(value)
        if len(values) < 2:
            return mpmath.mpc(0)
        with mpmath.workprec(working):
            try:
                found = mpmath.polyroots(values, maxsteps=500, extraprec=2*working)
            except (mpmath.libmp.libhyper.NoConvergence, ValueError, ZeroDivisionError):
                return None
            total = mpmath.mpc(0)
            for r in found:
                z = mpmath.mpc(r)
                re = Float(number_from_fraction(mpf_to_fraction(mpmath.mpf(z.real))), precision=working)
                im = Float(number_from_fraction(mpf_to_fraction(mpmath.mpf(z.imag))), precision=working)
                value_at = numeric_value(_call(self._function, add(re, mul(im, I))), working)
                if value_at is None:
                    return None
                total += value_at
            return total

    def _sympystr(self, printer: NodePrinter) -> str:
        return 'RootSum(%s, %s)' % (printer.doprint(self._polynomial), printer.doprint(self._function))
