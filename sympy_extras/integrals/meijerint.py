"""Integration by rewriting the integrand as Meijer G-functions.

Ported from SymPy's ``sympy.integrals.meijerint``. A function is written
as a sum of terms `C x^s G(a x^b)` by a table of representations of the
elementary and special functions as G-functions [P]_ (section 8.4), or,
for a function of the table of :mod:`.mellin`, by the inversion of its
Mellin transform as a Mellin–Barnes integral (:func:`rewrite_single`);
the integrals are then theorems on G-functions:

* the indefinite integral of `t^{\\rho - 1} G(t)` is `t^\\rho` times a
  G-function with one more pair of parameters [R]_ (section 5)
  (:func:`meijerint_indefinite`);
* the integral over `(0, \\infty)` of one G-function is a quotient of
  gamma functions, its Mellin transform at `s = 1` [L]_ (section 5.6.1),
  under the conditions of [P]_ (section 2.24.2);
* the integral over `(0, \\infty)` of the product of two G-functions is a
  G-function of the ratio of their arguments [L]_ (section 5.6.2), under
  the 35 cases of conditions of [P]_ (section 2.24.1) and four of
  [L]_ (:func:`meijerint_definite`).

Other ranges are brought to `(0, \\infty)` by translations, reflections
and Heaviside factors. The conditions come with the values: a value
holds where its condition holds.

The module works on the objects of the core (:mod:`sympy_extras.core`,
the functions of :mod:`sympy_extras.functions`) with the native
``hyperexpand``, ``gammasimp``, ``powdenest``, ``cancel`` and Mellin
transforms, and imports no SymPy. SymPy's recursive Mellin transform of
a product (the Mellin transform computed by the two-function theorem,
then inverted) is not ported: the representations which are not in the
table come from the Mellin transforms of the kernels of :mod:`.mellin`,
inverted by :func:`.slater.mellin_barnes`, and are checked numerically
when they have no parameters. Not ported either: ``meijerint_inversion``
(the inverse Laplace transform).

Examples
========

>>> from sympy_extras.core import symbols, oo, Integer
>>> from sympy_extras.functions import exp, sin, besselj
>>> from sympy_extras.integrals.meijerint import meijerint_definite, meijerint_indefinite
>>> x = symbols('x')
>>> meijerint_indefinite(sin(x), x)
(-cos(x), True)
>>> meijerint_definite(exp(-x**2), x, -oo, oo)
(sqrt(pi), True)
>>> meijerint_definite(exp(-x)*sin(x), x, Integer(0), oo)
(1/2, True)
>>> meijerint_indefinite(besselj(Integer(1), x), x)
(-besselj(0, x), True)

References
==========

.. [L] Y. L. Luke, *The special functions and their approximations*,
   volume 1, Academic Press, 1969.
.. [R] K. B. Roach, *Meijer G function representations*, in ISSAC '97,
   pp. 205-211, ACM, 1997.
.. [P] A. P. Prudnikov, Yu. A. Brychkov, O. I. Marichev, *Integrals and
   series*, vol. 3: More special functions, Gordon and Breach, 1990.
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/integrals/meijerint.py, sympy/integrals/integrals.py
# (Integral._eval_integral, the meijerg=True path) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: strict typing on the core; the table entries are typed records
# with callables for the computed terms and the hints; results and
# conditions are pairs instead of Piecewise with an unevaluated Integral;
# the recursive Mellin transform replaced by the kernels of .mellin;
# meijerint_inversion not ported.
# ---------------------------------------------------------------------
from __future__ import annotations

import itertools
from math import lcm
from typing import Callable, Final, Optional, Sequence

from sympy_extras._numeric import reliable_value
from sympy_extras.core.add import Add, add
from sympy_extras.core.basic import Atom, Basic, MatchDict, preorder_traversal
from sympy_extras.core.cache import cacheit
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.exprtools import factor_terms
from sympy_extras.core.function import expand, expand_mul, expand_power_base, expand_trig
from sympy_extras.core.logic import (And, Boolean, BooleanFalse, BooleanFunction, BooleanLike, Or, and_, false, or_,
                                     to_boolean, true)
from sympy_extras.core.matching import match
from sympy_extras.core.mul import Mul, mul
from sympy_extras.core.numbers import I, Infinity, Integer, NegativeInfinity, Rational, oo, pi
from sympy_extras.core.power import Pow, sqrt
from sympy_extras.core.relational import Eq, Ge, Gt, Le, Lt, Ne, Relational, eq, ge, gt, le, lt, ne
from sympy_extras.core.singleton import S
from sympy_extras.core.sorting import SortKey, default_sort_key
from sympy_extras.core.symbol import Dummy, Symbol, Wild
from sympy_extras.functions.combinatorial.factorials import factorial
from sympy_extras.functions.elementary.complexes import (PeriodicArgument, PrincipalBranch, _unpolarify, abs_,
                                                          arg as arg_, polar_lift, polarify, principal_branch, re, sign,
                                                          unbranched_argument, unpolarify)
from sympy_extras.functions.elementary.exponential import Exp, exp, exp_polar, log
from sympy_extras.functions.elementary.hyperbolic import Cosh, Coth, Csch, Sech, Sinh, Tanh, cosh, sinh
from sympy_extras.functions.elementary.piecewise import Piecewise, piecewise, piecewise_fold
from sympy_extras.functions.elementary.trigonometric import Cos, Cot, Csc, Sec, Sin, Tan, cos, sin, sinc
from sympy_extras.functions.special.bessel import besseli, besselj, besselk, bessely
from sympy_extras.functions.special.delta_functions import DiracDelta, Heaviside, heaviside
from sympy_extras.functions.special.elliptic_integrals import elliptic_e, elliptic_k
from sympy_extras.functions.special.error_functions import (Chi, Ci, Ei, Shi, Si, erf, erfc, erfi, expint,
                                                            fresnelc, fresnels)
from sympy_extras.functions.special.gamma_functions import gamma
from sympy_extras.functions.special.hyper import Hyper, MeijerG, meijerg
from sympy_extras.functions.special.singularity_functions import SingularityFunctionNode
from sympy_extras.polys.polytools import cancel, factor
from sympy_extras.simplify.fu import sincos_to_sum
from sympy_extras.simplify.gammasimp import gammasimp
from sympy_extras.simplify.hyperexpand import hyperexpand
from sympy_extras.simplify.powsimp import powdenest, powsimp
from sympy_extras.simplify.radsimp import collect

__all__ = ['meijerint_indefinite', 'meijerint_definite', 'meijer_antiderivative', 'rewrite_single',
           'lookup_table', 'TableEntry', 'GTerm', 'Rewriting', 'ConditionalExpr']

#: a term ``C * x**s * G`` of a rewriting: ``(C, s, G)``
GTerm = tuple[Expr, Expr, MeijerG]
#: the terms of a rewriting and the condition under which they hold
Rewriting = tuple[list[GTerm], Boolean]
#: a value and the condition under which it holds
ConditionalExpr = tuple[Expr, Boolean]
#: the terms ``(factor, G)`` of a table entry, computed from the bindings of its wilds
TermsBuilder = Callable[[MatchDict], list[tuple[Expr, MeijerG]]]
#: whether a table entry applies to the bindings of its wilds
Hint = Callable[[MatchDict], bool]

# the variable of the formulas of the table (keep it at the top, as SymPy does)
z: Final = Dummy('z')

ZERO = Integer(0)
ONE = Integer(1)
HALF = Rational(1, 2)


class _CoeffExpValueError(ValueError):
    """Raised by :func:`_get_coeff_exp` when the expression is not ``c*x**b``."""


def _expr(b: Basic) -> Expr:
    if not isinstance(b, Expr):
        raise TypeError("an expression is expected, got %s" % (b,))
    return b


def _g(an: Sequence[ExprLike], aother: Sequence[ExprLike], bm: Sequence[ExprLike], bother: Sequence[ExprLike],
       argument: ExprLike) -> MeijerG:
    """``meijerg`` as a :class:`MeijerG` (``ValueError`` on parameters which
    SymPy's constructor refuses), the parameters unpolarified as SymPy's
    constructor does (``_prep_tuple``)."""
    def prep(params: Sequence[ExprLike]) -> list[Expr]:
        return [unpolarify(to_expr(p)) for p in params]
    g = meijerg(prep(an), prep(aother), prep(bm), prep(bother), argument)
    if not isinstance(g, MeijerG):
        raise TypeError("meijerg did not return a G-function")
    return g


def _as_g(e: Basic) -> MeijerG:
    if not isinstance(e, MeijerG):
        raise TypeError("a G-function is expected, got %s" % (e,))
    return e


def _subs_g(g: MeijerG, rule: MatchDict) -> MeijerG:
    """``g`` with the bindings substituted (``ValueError`` when the new
    parameters are refused)."""
    lists = [[_expr(p.subs(rule)) for p in params] for params in (g.an, g.aother, g.bm, g.bother)]
    return _g(lists[0], lists[1], lists[2], lists[3], _expr(g.argument.subs(rule)))


def _replace_symbol(g: MeijerG, old: Symbol, new: Expr) -> MeijerG:
    return _subs_g(g, {old: new})


def _has_infinity(e: Basic) -> bool:
    return e.has(S.Infinity, S.ComplexInfinity, S.NegativeInfinity)


def _coeff_mul(e: Expr, x: Symbol) -> tuple[Expr, tuple[Expr, ...]]:
    """SymPy's ``e.as_coeff_mul(x)``: the factors free of ``x`` multiplied,
    and those with ``x``."""
    if isinstance(e, Mul):
        free: list[Expr] = []
        dependent: list[Expr] = []
        for factor_ in e.args:
            f_ = _expr(factor_)
            (dependent if f_.has(x) else free).append(f_)
        return mul(*free), tuple(dependent)
    if not e.has(x):
        return e, ()
    return ONE, (e,)


def _coeff_add(e: Expr, x: Symbol) -> tuple[Expr, tuple[Expr, ...]]:
    """SymPy's ``e.as_coeff_add(x)``: the terms free of ``x`` added, and
    those with ``x``."""
    free: Expr = ZERO
    dependent: list[Expr] = []
    for term in Add.make_args(e):
        if term.has(x):
            dependent.append(term)
        else:
            free = free + term
    return free, tuple(dependent)


def _node_count(e: Basic) -> int:
    return 1 + sum(_node_count(a) for a in e.args)


# PORTED FROM SYMPY: sympy/integrals/meijerint.py:_has
def _has(res: Expr, *f: type[Basic]) -> bool:
    """Whether ``res`` has one of ``f``; for a ``Piecewise`` whether every
    piece has one."""
    res = piecewise_fold(res)
    if isinstance(res, Piecewise):
        return all(pair.has(*f) for pair in res.args)
    return res.has(*f)


# ---------------------------------------------------------------------------
# the table of representations

class TableEntry:
    """A formula of the table: ``formula`` (a pattern in the dummy ``z``
    with wilds) is ``sum(factor * G)`` over ``terms(bindings)`` when
    ``cond`` holds and ``hint(bindings)``."""

    __slots__ = ('formula', 'terms', 'cond', 'hint')

    def __init__(self, formula: Expr, terms: TermsBuilder, cond: Boolean, hint: Hint) -> None:
        self.formula: Final[Expr] = formula
        self.terms: Final[TermsBuilder] = terms
        self.cond: Final[Boolean] = cond
        self.hint: Final[Hint] = hint

    def __repr__(self) -> str:
        return "TableEntry(%s)" % (self.formula,)


TableKey = tuple[type[Basic], ...]
Table = dict[TableKey, list[TableEntry]]


def _type_key(t: type[Basic]) -> tuple[tuple[int, int, str], str]:
    return (t.class_key(), t.__name__)


# PORTED FROM SYMPY: sympy/integrals/meijerint.py:_mytype
def _mytype(f: Basic, x: Symbol) -> TableKey:
    """A hashable description of the types of the functions of ``f`` with
    ``x``."""
    if x not in f.free_symbols:
        return ()
    if f.is_Function:
        return (type(f),)
    return tuple(sorted((t for a in f.args for t in _mytype(a, x)), key=_type_key))


def _always(subs: MatchDict) -> bool:
    return True


def _fixed(terms: list[tuple[Expr, MeijerG]]) -> TermsBuilder:
    def build(subs: MatchDict) -> list[tuple[Expr, MeijerG]]:
        return terms
    return build


def _positive_integer(k: Basic) -> bool:
    return isinstance(k, Integer) and k.p > 0


def _wild(name: str) -> Wild:
    return Wild(name, exclude=[z])


# PORTED FROM SYMPY: sympy/integrals/meijerint.py:_create_lookup_table
def _create_lookup_table(table: Table) -> None:
    """Add the formulae of the function -> G-function table."""
    p, q, a, b = _wild('p'), _wild('q'), _wild('a'), _wild('b')
    n = Wild('n', properties=[_positive_integer])
    t = p * z**q

    def add(formula: Expr, an: Sequence[ExprLike], ap: Sequence[ExprLike], bm: Sequence[ExprLike],
            bq: Sequence[ExprLike], arg: ExprLike = t, fac: ExprLike = ONE, cond: BooleanLike = true,
            hint: Hint = _always) -> None:
        table.setdefault(_mytype(formula, z), []).append(
            TableEntry(formula, _fixed([(to_expr(fac), _g(an, ap, bm, bq, arg))]), to_boolean(cond), hint))

    def addi(formula: Expr, terms: TermsBuilder, cond: BooleanLike = true) -> None:
        table.setdefault(_mytype(formula, z), []).append(TableEntry(formula, terms, to_boolean(cond), _always))

    def constant(c: Expr) -> list[tuple[Expr, MeijerG]]:
        return [(c, _g([1], [], [], [0], z)), (c, _g([], [1], [0], [], z))]
    table[()] = [TableEntry(a, _fixed(constant(a)), true, _always)]

    def not_nonpositive_integer(subs: MatchDict) -> bool:
        # SymPy's hint Not(IsNonPositiveInteger(a)): False only for an
        # integer a <= 0
        value = unpolarify(_expr(a.subs(subs)))
        return not (isinstance(value, Integer) and value.p <= 0)

    # [P], Section 8.4.2
    add(heaviside(t - b) * (t - b)**(a - 1), [a], [], [], [0], t / b, gamma(a) * b**(a - 1), gt(b, 0))
    add(heaviside(b - t) * (b - t)**(a - 1), [], [a], [0], [], t / b, gamma(a) * b**(a - 1), gt(b, 0))
    add(heaviside(z - (b / p)**(1 / q)) * (t - b)**(a - 1), [a], [], [], [0], t / b, gamma(a) * b**(a - 1),
        gt(b, 0))
    add(heaviside((b / p)**(1 / q) - z) * (b - t)**(a - 1), [], [a], [0], [], t / b, gamma(a) * b**(a - 1),
        gt(b, 0))
    add((b + t)**(-a), [1 - a], [], [0], [], t / b, b**(-a) / gamma(a), hint=not_nonpositive_integer)
    add(abs_(b - t)**(-a), [1 - a], [(1 - a) / 2], [0], [(1 - a) / 2], t / b,
        2 * sin(pi * a / 2) * gamma(1 - a) * abs_(b)**(-a), lt(re(a), 1))
    add((t**a - b**a) / (t - b), [0, a], [], [0, a], [], t / b, b**(a - 1) * sin(a * pi) / pi)

    # 12
    def A1(r: Expr, sgn: int, nu: Expr) -> Expr:
        return pi**Rational(-1, 2) * (-sgn * nu / 2)**(1 - 2 * r)

    def tmpadd12(r: Expr, sgn: int) -> None:
        add((sqrt(a**2 + t) + sgn * a)**b / (a**2 + t)**r,
            [(1 + b) / 2, 1 - 2 * r + b / 2], [],
            [(b - sgn * b) / 2], [(b + sgn * b) / 2], t / a**2,
            a**(b - 2 * r) * A1(r, sgn, b))
    tmpadd12(ZERO, 1)
    tmpadd12(ZERO, -1)
    tmpadd12(HALF, 1)
    tmpadd12(HALF, -1)

    # 13
    def tmpadd13(r: Expr, sgn: int) -> None:
        add((sqrt(a + p * z**q) + sgn * sqrt(p) * z**(q / 2))**b / (a + p * z**q)**r,
            [1 - r + sgn * b / 2], [1 - r - sgn * b / 2], [0, HALF], [],
            p * z**q / a, a**(b / 2 - r) * A1(r, sgn, b))
    tmpadd13(ZERO, 1)
    tmpadd13(ZERO, -1)
    tmpadd13(HALF, 1)
    tmpadd13(HALF, -1)

    # Section 8.4.3: SymPy's exp(polar_lift(-1)*t) with the factor
    # polar_lift(-1) moved into the argument of the G-function (the
    # matcher of the core does not divide the binding of p by it)
    add(exp(t), [], [], [0], [], arg=exp_polar(-I * pi) * t)

    # 8.4.4 (hyperbolic functions)
    add(sinh(t), [], [1], [HALF], [1, 0], t**2 / 4, pi**Rational(3, 2))
    add(cosh(t), [], [HALF], [0], [HALF, HALF], t**2 / 4, pi**Rational(3, 2))

    # Section 8.4.5
    add(sin(t), [], [], [HALF], [0], t**2 / 4, sqrt(pi))
    add(cos(t), [], [], [0], [HALF], t**2 / 4, sqrt(pi))

    # Section 8.4.6 (sinc function)
    add(sinc(t), [], [], [0], [Rational(-1, 2)], t**2 / 4, sqrt(pi) / 2)

    # Section 8.5.5
    def order(subs: MatchDict) -> int:
        N = subs[n]
        if not isinstance(N, Integer):
            raise TypeError("the order of the logarithm must be an integer")
        return N.p

    def make_log1(subs: MatchDict) -> list[tuple[Expr, MeijerG]]:
        N = order(subs)
        return [(Integer(-1)**N * factorial(N), _g([], [1] * (N + 1), [0] * (N + 1), [], t))]

    def make_log2(subs: MatchDict) -> list[tuple[Expr, MeijerG]]:
        N = order(subs)
        return [(factorial(N), _g([1] * (N + 1), [], [], [0] * (N + 1), t))]

    def make_log3(subs: MatchDict) -> list[tuple[Expr, MeijerG]]:
        return make_log1(subs) + make_log2(subs)
    addi(log(t)**n * heaviside(1 - t), make_log1)
    addi(log(t)**n * heaviside(t - 1), make_log2)
    addi(log(t)**n, make_log3)
    addi(log(t + a), _fixed(constant(log(a)) + [(ONE, _g([1, 1], [], [1], [0], t / a))]))
    addi(log(abs_(t - a)), _fixed(constant(log(abs_(a))) + [(pi, _g([1, 1], [HALF], [1], [0, HALF], t / a))]))

    # Section 8.4.11
    addi(Ei(t), _fixed(constant(-I * pi) + [(Integer(-1), _g([], [1], [0, 0], [], t * polar_lift(-1)))]))

    # Section 8.4.12
    add(Si(t), [1], [], [HALF], [0, 0], t**2 / 4, sqrt(pi) / 2)
    add(Ci(t), [], [1], [0, 0], [HALF], t**2 / 4, -sqrt(pi) / 2)

    # Section 8.4.13
    add(Shi(t), [HALF], [], [0], [Rational(-1, 2), Rational(-1, 2)], polar_lift(-1) * t**2 / 4,
        t * sqrt(pi) / 4)
    add(Chi(t), [], [HALF, 1], [0, 0], [HALF, HALF], t**2 / 4, -pi**Rational(3, 2) / 2)

    # generalized exponential integral
    add(expint(a, t), [], [a], [a - 1, 0], [], t)

    # Section 8.4.14
    add(erf(t), [1], [], [HALF], [0], t**2, 1 / sqrt(pi))
    add(erfc(t), [], [1], [0, HALF], [], t**2, 1 / sqrt(pi))
    add(erfi(t), [HALF], [], [0], [Rational(-1, 2)], -t**2, t / sqrt(pi))

    # Fresnel Integrals
    add(fresnels(t), [1], [], [Rational(3, 4)], [0, Rational(1, 4)], pi**2 * t**4 / 16, HALF)
    add(fresnelc(t), [1], [], [Rational(1, 4)], [0, Rational(3, 4)], pi**2 * t**4 / 16, HALF)

    # Section 8.4.19
    add(besselj(a, t), [], [], [a / 2], [-a / 2], t**2 / 4)
    # Section 8.4.20
    add(bessely(a, t), [], [-(a + 1) / 2], [a / 2, -a / 2], [-(a + 1) / 2], t**2 / 4)
    # Section 8.4.22
    add(besseli(a, t), [], [(1 + a) / 2], [a / 2], [-a / 2, (1 + a) / 2], t**2 / 4, pi)
    # Section 8.4.23
    add(besselk(a, t), [], [], [a / 2, -a / 2], [], t**2 / 4, HALF)

    # Complete elliptic integrals K(z) and E(z)
    add(elliptic_k(t), [HALF, HALF], [], [0], [0], -t, HALF)
    add(elliptic_e(t), [HALF, 3 * HALF], [], [0], [0], -t, Rational(-1, 2) / 2)


_lookup_table: Table = {}


def lookup_table() -> Table:
    """The table of representations, by the types of the functions of the
    formulas (built on first use).

    >>> from sympy_extras.functions import Sin
    >>> from sympy_extras.integrals.meijerint import lookup_table
    >>> lookup_table()[(Sin,)]
    [TableEntry(sin(_z**q_*p_))]
    """
    if not _lookup_table:
        _create_lookup_table(_lookup_table)
    return _lookup_table


# ---------------------------------------------------------------------------
# helpers

# PORTED FROM SYMPY: sympy/integrals/meijerint.py:_get_coeff_exp
def _get_coeff_exp(expr: Expr, x: Symbol) -> tuple[Expr, Expr]:
    """``(c, b)`` for ``expr = c*x**b``; :class:`_CoeffExpValueError` when
    it is not of that form.

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.integrals.meijerint import _get_coeff_exp
    >>> x, a, b = symbols('x a b')
    >>> _get_coeff_exp(a*x**b, x), _get_coeff_exp(x, x), _get_coeff_exp(2*x, x), _get_coeff_exp(x**3, x)
    ((a, b), (1, 1), (2, 1), (1, 3))
    """
    c, m = _coeff_mul(expand_power_base(powsimp(expr)), x)
    if not m:
        return c, ZERO
    if len(m) > 1:
        raise _CoeffExpValueError('expr not of form a*x**b: %s' % (expr,))
    factor_ = m[0]
    if isinstance(factor_, Pow):
        if factor_.base != x:
            raise _CoeffExpValueError('expr not of form a*x**b')
        return c, factor_.exp
    if factor_ == x:
        return c, ONE
    raise _CoeffExpValueError('expr not of form a*x**b: %s' % (expr,))


# PORTED FROM SYMPY: sympy/integrals/meijerint.py:_exponents
def _exponents(expr: Expr, x: Symbol) -> set[Expr]:
    """The exponents of ``x`` (other than zero) in ``expr``.

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.functions import sin
    >>> from sympy_extras.integrals.meijerint import _exponents
    >>> x, y = symbols('x y')
    >>> sorted(_exponents(x**3*sin(x + x**y) + 1/x, x), key=str)
    [-1, 1, 3, y]
    """
    res: set[Expr] = set()

    def walk(e: Basic) -> None:
        if e == x:
            res.add(ONE)
            return
        if isinstance(e, Pow) and e.base == x:
            res.add(e.exp)
            return
        for argument in e.args:
            walk(argument)
    walk(expr)
    return res


# PORTED FROM SYMPY: sympy/integrals/meijerint.py:_functions
def _functions(expr: Expr, x: Symbol) -> set[type[Basic]]:
    """The types of the functions of ``expr`` with ``x``."""
    return {type(e) for e in preorder_traversal(expr) if e.is_Function and x in e.free_symbols}


# PORTED FROM SYMPY: sympy/integrals/meijerint.py:_find_splitting_points
def _find_splitting_points(expr: Expr, x: Symbol) -> set[Expr]:
    """The numbers ``a`` such that the linear substitution ``x -> x + a``
    would (hopefully) simplify ``expr``.

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.functions import sin
    >>> from sympy_extras.integrals.meijerint import _find_splitting_points as fsp
    >>> x = symbols('x')
    >>> fsp(x, x), fsp((x - 1)**3, x), sorted(fsp(sin(x + 3)*x, x), key=str)
    ({0}, {1}, [-3, 0])
    """
    p, q = Wild('p', exclude=[x]), Wild('q', exclude=[x])
    res: set[Expr] = set()

    def innermost(e: Basic) -> None:
        if not isinstance(e, Expr):
            return
        m = e.match(p * x + q)
        if m and m.get(p, ZERO) != 0:
            res.add(-_expr(m.get(q, ZERO)) / _expr(m[p]))
            return
        if isinstance(e, Atom):
            return
        for argument in e.args:
            innermost(argument)
    innermost(expr)
    return res


# PORTED FROM SYMPY: sympy/integrals/meijerint.py:_split_mul
def _split_mul(f: Expr, x: Symbol) -> tuple[Expr, Expr, Expr]:
    """``f`` as ``fac * po * g`` with ``fac`` constant, ``po = x**s`` and
    ``g`` the rest.

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.functions import sin
    >>> from sympy_extras.integrals.meijerint import _split_mul
    >>> s, x = symbols('s x')
    >>> _split_mul((3*x)**s*sin(x**2)*x, x)
    (3**s, x*x**s, sin(x**2))
    """
    fac: Expr = ONE
    po: Expr = ONE
    g: Expr = ONE
    f = expand_power_base(f)
    for a in Mul.make_args(f):
        if a == x:
            po = po * x
        elif x not in a.free_symbols:
            fac = fac * a
        else:
            if isinstance(a, Pow) and x not in a.exp.free_symbols:
                c, t = _coeff_mul(a.base, x)
                if t != (x,):
                    c, t = _coeff_mul(expand_mul(a.base), x)
                if t == (x,):
                    po = po * x**a.exp
                    fac = fac * unpolarify(polarify(c**a.exp, subs=False)[0])
                    continue
            g = g * a
    return fac, po, g


# PORTED FROM SYMPY: sympy/integrals/meijerint.py:_mul_args
def _mul_args(f: Expr) -> list[Expr]:
    """The factors of ``f``, an integer power repeated."""
    gs: list[Expr] = []
    for g in Mul.make_args(f):
        if isinstance(g, Pow) and isinstance(g.exp, Integer):
            n = g.exp.p
            base = g.base
            if n < 0:
                n = -n
                base = 1 / base
            gs += [base] * n
        else:
            gs.append(g)
    return gs


# PORTED FROM SYMPY: sympy/integrals/meijerint.py:_mul_as_two_parts
def _mul_as_two_parts(f: Expr) -> Optional[list[tuple[Expr, Expr]]]:
    """All the ways to write ``f`` as a product of two factors; ``None``
    when it has fewer than two."""
    gs = _mul_args(f)
    if len(gs) < 2:
        return None
    if len(gs) == 2:
        return [(gs[0], gs[1])]
    # the partitions of the multiset into two blocks, the first block
    # holding the first factor (SymPy's multiset_partitions(gs, 2))
    found: list[tuple[Expr, Expr]] = []
    seen: set[tuple[Expr, Expr]] = set()
    rest = list(range(1, len(gs)))
    for size in range(0, len(rest)):
        for chosen in itertools.combinations(rest, size):
            first = [gs[0]] + [gs[i] for i in chosen]
            second = [gs[i] for i in rest if i not in chosen]
            pair = (mul(*first), mul(*second))
            if pair not in seen:
                seen.add(pair)
                found.append(pair)
    return found


# PORTED FROM SYMPY: sympy/integrals/meijerint.py:_inflate_g
def _inflate_g(g: MeijerG, n: int) -> tuple[Expr, MeijerG]:
    """``(C, h)`` with ``h`` a G-function of argument ``z**n`` and
    ``g = C*h`` ([L], page 150, equation (5))."""
    def inflate(params: Sequence[Expr]) -> list[Expr]:
        return [(a + i) / n for a, i in itertools.product(params, range(n))]
    v = Integer(len(g.ap) - len(g.bq))
    C: Expr = Integer(n)**(1 + g.nu + v / 2)
    C = C / (2 * pi)**((n - 1) * g.delta)
    return C, _g(inflate(g.an), inflate(g.aother), inflate(g.bm), inflate(g.bother),
                 g.argument**n * Integer(n)**(n * v))


# PORTED FROM SYMPY: sympy/integrals/meijerint.py:_flip_g
def _flip_g(g: MeijerG) -> MeijerG:
    """The G-function of the inverse argument ([L], section 5.2)."""
    def tr(params: Sequence[Expr]) -> list[Expr]:
        return [1 - a for a in params]
    return _g(tr(g.bm), tr(g.bother), tr(g.an), tr(g.aother), 1 / g.argument)


_dummies: dict[tuple[str, str], Dummy] = {}


# PORTED FROM SYMPY: sympy/integrals/meijerint.py:_dummy
def _dummy(name: str, token: str, expr: Expr, positive: bool = False) -> Dummy:
    """The dummy of ``name`` and ``token`` (the same one each time, for the
    caches), a new one when it is in ``expr``."""
    d = _dummy_(name, token, positive)
    if d in expr.free_symbols:
        return Dummy(name, positive=True) if positive else Dummy(name)
    return d


def _dummy_(name: str, token: str, positive: bool = False) -> Dummy:
    key = (name, token + (':positive' if positive else ''))
    if key not in _dummies:
        _dummies[key] = Dummy(name, positive=True) if positive else Dummy(name)
    return _dummies[key]


# PORTED FROM SYMPY: sympy/integrals/meijerint.py:_is_analytic
def _is_analytic(f: Expr, x: Symbol) -> bool:
    """Whether ``f``, written by G-functions on the positive reals, agrees
    with them almost everywhere."""
    from sympy_extras.functions.elementary.complexes import Abs
    return not any(x in e.free_symbols for e in f.atoms(Heaviside, Abs))


# ---------------------------------------------------------------------------
# the simplification of the conditions

_RELATIONS: Final[dict[type[Relational], Callable[[ExprLike, ExprLike], Boolean]]] = {
    Eq: eq, Ne: ne, Lt: lt, Le: le, Gt: gt, Ge: ge}


def _relation(cls: type[Relational], lhs: Expr, rhs: Expr) -> Boolean:
    return _RELATIONS[cls](lhs, rhs)


# PORTED FROM SYMPY: sympy/core/relational.py:_canonical_coeff
def _canonical_coeff(node: Basic) -> Basic:
    """``-2*x + 1 < 0`` as ``x > 1/2``."""
    if not isinstance(node, Relational):
        return node
    rel = node.canonical
    b, rest = rel.lhs.as_coeff_Add()
    m, lhs = rest.as_coeff_Mul(rational=True)
    if m == 0 or not isinstance(m, Rational):
        return rel
    rhs = (rel.rhs - b) / m
    if m.is_negative:
        return _relation(type(rel.reversed), lhs, rhs)
    return _relation(type(rel), lhs, rhs)


_P: Final = Wild('p')
_Q: Final = Wild('q')
_R: Final = Wild('r')
_rules: list[tuple[Boolean, Boolean]] = []


def _condsimp_rules() -> list[tuple[Boolean, Boolean]]:
    """The rewriting rules of :func:`_condsimp` (built on first use)."""
    if _rules:
        return _rules
    p, q = _P, _Q
    _rules.extend([
        (Or(Lt(p, q), Eq(p, q)), le(p, q)),  # 0
        (And(Le(abs_(arg_(p)), pi), Le(abs_(arg_(p) - 2 * pi), pi)), eq(arg_(p) - pi, 0)),  # 1
        (And(Le(abs_(2 * arg_(p) + pi), pi), Le(abs_(2 * arg_(p) - pi), pi)), eq(arg_(p), 0)),  # 2
        (And(Lt(abs_(2 * arg_(p) + pi), pi), Le(abs_(2 * arg_(p) - pi), pi)), false),  # 3
        (And(Le(abs_(arg_(p) - pi / 2), pi / 2), Le(abs_(arg_(p) + pi / 2), pi / 2)), eq(arg_(p), 0)),  # 4
        (And(Le(abs_(arg_(p) - pi / 2), pi / 2), Lt(abs_(arg_(p) + pi / 2), pi / 2)), false),  # 5
        (And(Lt(abs_(arg_(p**2 / 2 + 1)), pi), Ne(abs_(arg_(p**2 / 2 + 1)), pi)), true),  # 6
        (Or(Lt(abs_(arg_(p**2 / 2 + 1)), pi), Ne(1 / (p**2 / 2 + 1), 0)), true),  # 7
        (And(Le(abs_(unbranched_argument(p)), pi),
             Le(abs_(unbranched_argument(exp_polar(-2 * pi * I) * p)), pi)),
         eq(unbranched_argument(exp_polar(-I * pi) * p), 0)),  # 8
        (And(Le(abs_(unbranched_argument(p)), pi / 2),
             Le(abs_(unbranched_argument(exp_polar(-pi * I) * p)), pi / 2)),
         eq(unbranched_argument(exp_polar(-I * pi / 2) * p), 0)),  # 9
        (And(Ne(p**2, 1), Gt(p**2, 1)), gt(p**2, 1)),  # 11
        (And(Ne(1 / p, 1), Gt(cos(abs_(arg_(p))) * abs_(p), 1)), gt(abs_(p), 1)),  # 12
        (And(Ne(p, 2), Gt(cos(abs_(arg_(p))) * abs_(p), 2)), gt(abs_(p), 2)),  # 13
        (And(Lt(abs_(arg_(p)), pi / 2), Gt(cos(abs_(arg_(p))) * sqrt(abs_(p**2)), 1)), gt(p**2, 1)),  # 14
    ])
    # rule 10 of SymPy, Or(p <= q, And(p < q, r)) -> p <= q, has a boolean
    # wild, which the matcher of the core does not take: left out
    return _rules


def _rel_touchup(node: Basic) -> Basic:
    """``Eq(arg(p)**q, 0)`` (or of the unbranched argument) as ``p > 0``."""
    if not isinstance(node, Eq) or node.rhs != 0:
        return node
    lhs = node.lhs
    m = lhs.match(arg_(_P)**_Q)
    if not m:
        m = lhs.match(unbranched_argument(polar_lift(_P)**_Q))
    if not m:
        if isinstance(lhs, PeriodicArgument) and _expr(lhs.args[0]).is_polar is not True and lhs.args[1] == oo:
            return gt(_expr(lhs.args[0]), 0)
        return node
    return gt(_expr(m[_P]), 0)


# PORTED FROM SYMPY: sympy/integrals/meijerint.py:_condsimp
def _condsimp(cond: Boolean, first: bool = True) -> Boolean:
    """Naive simplifications of ``cond`` (ad hoc rules, added as the need
    arose, as in SymPy).

    >>> from sympy_extras.core import symbols, Or, Eq
    >>> from sympy_extras.integrals.meijerint import _condsimp
    >>> x, y = symbols('x y')
    >>> _condsimp(Or(x < y, Eq(x, y)))
    x <= y
    """
    if first:
        cond = to_boolean(cond.replace(lambda node: isinstance(node, Relational), _canonical_coeff))
        first = False
    if not isinstance(cond, BooleanFunction):
        return cond
    cond = to_boolean(cond._evaluated(tuple(_condsimp(to_boolean(a), first) for a in cond.args)))
    change = True
    while change:
        change = False
        for fro, to in _condsimp_rules():
            if type(fro) is not type(cond):
                continue
            for n, arg1 in enumerate(cond.args):
                if _R in fro.args[0].free_symbols:
                    m = arg1.match(fro.args[1])
                    num = 1
                else:
                    num = 0
                    m = arg1.match(fro.args[0])
                if not m:
                    continue
                otherargs = [to_boolean(x.subs(m)) for x in fro.args[:num] + fro.args[num + 1:]]
                otherlist = [n]
                for arg2 in otherargs:
                    for k, arg3 in enumerate(cond.args):
                        if k in otherlist:
                            continue
                        if arg2 == arg3:
                            otherlist += [k]
                            break
                if len(otherlist) != len(otherargs) + 1:
                    continue
                newargs = [to_boolean(arg_k) for (k, arg_k) in enumerate(cond.args) if k not in otherlist]
                newargs.append(to_boolean(to.subs(m)))
                cond = to_boolean(cond._evaluated(tuple(newargs)))
                change = True
                break
            if not isinstance(cond, BooleanFunction):
                break
        if not isinstance(cond, BooleanFunction):
            break
    return to_boolean(cond.replace(lambda node: isinstance(node, Relational), _rel_touchup))


# PORTED FROM SYMPY: sympy/integrals/meijerint.py:_eval_cond
def _eval_cond(cond: Boolean) -> Boolean:
    """The conditions evaluated again."""
    return _condsimp(to_boolean(cond.doit()))


def _unpolarify_boolean(cond: Boolean) -> Boolean:
    """:func:`unpolarify` on a condition."""
    current: Basic = cond
    while True:
        result = _unpolarify(current, False)
        if result == current:
            break
        current = result
    return to_boolean(current.subs([(exp_polar(ZERO), ONE), (polar_lift(ZERO), ZERO)]))


# PORTED FROM SYMPY: sympy/integrals/meijerint.py:_my_unpolarify
def _my_unpolarify(f: Expr) -> Expr:
    """``f`` without polar numbers, the conditions of its ``Piecewise``
    simplified."""
    e = unpolarify(f).doit()
    replaced = e.replace(lambda node: isinstance(node, Relational), _canonical_coeff)
    return _expr(replaced)


def _my_unpolarify_cond(cond: Boolean) -> Boolean:
    return _eval_cond(_unpolarify_boolean(cond))


def _is_false(cond: Boolean) -> bool:
    return isinstance(cond, BooleanFalse)


# ---------------------------------------------------------------------------
# the theorems

# PORTED FROM SYMPY: sympy/integrals/meijerint.py:_my_principal_branch
def _my_principal_branch(expr: Expr, period: Expr, full_pb: bool = False) -> Expr:
    """``expr`` brought nearer to its principal branch by removing
    superfluous factors (to the principal branch with ``full_pb``)."""
    res = principal_branch(expr, period)
    if not full_pb:
        res = _expr(res.replace(PrincipalBranch, _first_argument))
    return res


def _first_argument(*args: Basic) -> Basic:
    return args[0]


# PORTED FROM SYMPY: sympy/integrals/meijerint.py:_rewrite_saxena_1
def _rewrite_saxena_1(fac: Expr, po: Expr, g: MeijerG, x: Symbol) -> tuple[Expr, MeijerG]:
    """The integral of ``fac*po*g`` over ``(0, oo)`` as that of ``C*G``,
    ``G`` with argument ``a*x`` (``po = x**s``)."""
    _, s = _get_coeff_exp(po, x)
    a, b = _get_coeff_exp(g.argument, x)
    a = _my_principal_branch(a, g.get_period())
    # the substitution t = x**b
    C = fac / (abs_(b) * a**((s + 1) / b - 1))

    def tr(params: Sequence[Expr]) -> list[Expr]:
        return [c + (1 + s) / b - 1 for c in params]
    return C, _g(tr(g.an), tr(g.aother), tr(g.bm), tr(g.bother), a * x)


# PORTED FROM SYMPY: sympy/integrals/meijerint.py:_check_antecedents_1
def _check_antecedents_1(g: MeijerG, x: Symbol, helper: bool = False) -> Boolean:
    """The condition under which the Mellin transform of ``g`` exists at
    ``s = 1``, the integral of ``g`` over ``(0, oo)`` ([L], section 5.6.1);
    with ``helper`` whether the integral over ``(1, oo)`` exists."""
    delta = g.delta
    eta, _ = _get_coeff_exp(g.argument, x)
    m, n, p, q = len(g.bm), len(g.an), len(g.ap), len(g.bq)
    if p > q:
        def tr(params: Sequence[Expr]) -> list[Expr]:
            return [1 - c for c in params]
        return _check_antecedents_1(_g(tr(g.bm), tr(g.bother), tr(g.an), tr(g.aother), x / eta), x)

    tmp: list[Boolean] = [lt(-re(b), 1) for b in g.bm] + [lt(1, 1 - re(a)) for a in g.an]
    cond_3 = and_(*tmp)
    tmp += [lt(-re(b), 1) for b in g.bother]
    tmp += [lt(1, 1 - re(a)) for a in g.aother]
    cond_4 = gt(-re(g.nu) + Rational(q + 1 - p, 2), q - p)

    conds: list[Boolean] = []
    # case 1
    tmp1: list[Boolean] = [to_boolean(1 <= n), to_boolean(p < q), to_boolean(1 <= m)]
    tmp2: list[Boolean] = [to_boolean(1 <= p), to_boolean(1 <= m), to_boolean(q == p + 1),
                           to_boolean(not (n == 0 and m == p + 1))]
    tmp3: list[Boolean] = [to_boolean(1 <= p), to_boolean(q == p)]
    # ceiling(delta/2)
    for k in range(-((-delta.p) // (2 * delta.q)) + 1):
        tmp3 += [ne(abs_(unbranched_argument(eta)), (delta - 2 * k) * pi)]
    common = [gt(delta, 0), lt(abs_(unbranched_argument(eta)), delta * pi)]
    extra: list[Boolean] = [] if helper else [ne(eta, 0), cond_3]
    for t in [tmp1, tmp2, tmp3]:
        conds.append(and_(*(t + common + extra)))
    # case 2
    extra = [] if helper else [cond_3]
    conds.append(and_(n == 0, p + 1 <= m, m <= q, lt(abs_(unbranched_argument(eta)), delta * pi), *extra))
    # case 3
    extra = [] if helper else [cond_3, cond_4]
    conds.append(and_(p < q, 1 <= m, gt(delta, 0), eq(abs_(unbranched_argument(eta)), delta * pi), *extra))
    conds.append(and_(p <= q - 2, eq(delta, 0), eq(abs_(unbranched_argument(eta)), 0), *extra))
    # the extra case from the Wolfram functions site (Prudnikov, section 2.24.2)
    case_extra: list[Boolean] = [to_boolean(p == q), eq(delta, 0), eq(unbranched_argument(eta), 0), ne(eta, 0)]
    if not helper:
        case_extra += [cond_3]
    total: Expr = ZERO
    for a, b in zip(g.ap, g.bq):
        total = total + (b - a)
    case_extra += [lt(re(total), 0)]
    conds.append(and_(*case_extra))
    case_extra_2: list[Boolean] = [and_(gt(delta, 0), lt(abs_(unbranched_argument(eta)), delta * pi))]
    if not helper:
        case_extra_2 += [cond_3]
    conds.append(and_(*case_extra_2))
    return or_(*conds)


# PORTED FROM SYMPY: sympy/integrals/meijerint.py:_int0oo_1
def _int0oo_1(g: MeijerG, x: Symbol) -> Expr:
    """The integral of ``g`` over ``(0, oo)``, the conditions assumed
    ([L], section 5.6.1, with s = 1).

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.integrals.meijerint import _g, _int0oo_1
    >>> a, b, c, d, x, y = symbols('a b c d x y')
    >>> _int0oo_1(_g([a], [b], [c], [d], x*y), x)
    gamma(-a)*gamma(c + 1)/(y*gamma(-d)*gamma(b + 1))
    """
    eta, _ = _get_coeff_exp(g.argument, x)
    res: Expr = 1 / eta
    for b in g.bm:
        res = res * gamma(b + 1)
    for a in g.an:
        res = res * gamma(1 - a - 1)
    for b in g.bother:
        res = res / gamma(1 - b - 1)
    for a in g.aother:
        res = res / gamma(a + 1)
    return gammasimp(unpolarify(res))


# PORTED FROM SYMPY: sympy/integrals/meijerint.py:_rewrite_saxena
def _rewrite_saxena(fac: Expr, po: Expr, g1: MeijerG, g2: MeijerG, x: Symbol,
                    full_pb: bool = False) -> Optional[tuple[Expr, MeijerG, MeijerG]]:
    """The integral of ``fac*po*g1*g2`` over ``(0, oo)`` as that of
    ``C*f1*f2`` with G-functions of arguments ``c*x``; ``None`` when the
    exponents are not rational."""
    def pb(g: MeijerG) -> MeijerG:
        a, b = _get_coeff_exp(g.argument, x)
        return _g(g.an, g.aother, g.bm, g.bother, _my_principal_branch(a, g.get_period(), full_pb) * x**b)

    _, s = _get_coeff_exp(po, x)
    _, b1 = _get_coeff_exp(g1.argument, x)
    _, b2 = _get_coeff_exp(g2.argument, x)
    if lt(b1, 0) is true:
        b1 = -b1
        g1 = _flip_g(g1)
    if lt(b2, 0) is true:
        b2 = -b2
        g2 = _flip_g(g2)
    if not isinstance(b1, Rational) or not isinstance(b2, Rational) or b1 == 0 or b2 == 0:
        return None
    m1, n1 = b1.p, b1.q
    m2, n2 = b2.p, b2.q
    tau = lcm(m1 * n2, m2 * n1)
    r1 = tau // (m1 * n2)
    r2 = tau // (m2 * n1)
    C1, g1 = _inflate_g(g1, r1)
    C2, g2 = _inflate_g(g2, r2)
    g1 = pb(g1)
    g2 = pb(g2)
    fac = fac * C1 * C2
    a1, b = _get_coeff_exp(g1.argument, x)
    a2, _ = _get_coeff_exp(g2.argument, x)
    # arbitrarily the x**s part goes to g1
    exponent = (s + 1) / b - 1
    fac = fac / (abs_(b) * a1**exponent)

    def tr(params: Sequence[Expr]) -> list[Expr]:
        return [c + exponent for c in params]
    f1 = _g(tr(g1.an), tr(g1.aother), tr(g1.bm), tr(g1.bother), a1 * x)
    f2 = _g(g2.an, g2.aother, g2.bm, g2.bother, a2 * x)
    return powdenest(fac, polar=True), f1, f2


def _positive(e: Expr) -> bool:
    return e.is_positive is True


def _negative(e: Expr) -> bool:
    return e.is_negative is True


# PORTED FROM SYMPY: sympy/integrals/meijerint.py:_check_antecedents
def _check_antecedents(g1: MeijerG, g2: MeijerG, x: Symbol) -> Boolean:
    """The condition under which the theorem for the integral of the
    product of two G-functions applies: the cases of [P], section 2.24.1
    (reproduced on the Wolfram functions site, MeijerG/21/02/03; with
    k = l = r = alpha = 1), and four of [L]."""
    sigma, _ = _get_coeff_exp(g1.argument, x)
    omega, _ = _get_coeff_exp(g2.argument, x)
    s, t, u, v = (Integer(len(g1.bm)), Integer(len(g1.an)), Integer(len(g1.ap)), Integer(len(g1.bq)))
    m, n, p, q = (Integer(len(g2.bm)), Integer(len(g2.an)), Integer(len(g2.ap)), Integer(len(g2.bq)))
    bstar = s + t - (u + v) / 2
    cstar = m + n - (p + q) / 2
    rho = g1.nu + (u - v) / 2 + 1
    mu = g2.nu + (p - q) / 2 + 1
    phi = q - p - (v - u)
    eta = 1 - (v - u) - mu - rho
    psi = (pi * (q - m - n) + abs_(unbranched_argument(omega))) / (q - p)
    theta = (pi * (v - s - t) + abs_(unbranched_argument(sigma))) / (v - u)

    def _c1() -> bool:
        for g in [g1, g2]:
            for i, j in itertools.product(g.an, g.bm):
                diff = i - j
                if diff.is_integer and diff.is_positive:
                    return False
        return True
    c1 = _c1()
    c2 = and_(*[gt(re(1 + i + j), 0) for i in g1.bm for j in g2.bm])
    c3 = and_(*[lt(re(1 + i + j), 1 + 1) for i in g1.an for j in g2.an])
    c4 = and_(*[gt((p - q) * re(1 + i - 1) - re(mu), Rational(-3, 2)) for i in g1.an])
    c5 = and_(*[gt((p - q) * re(1 + i) - re(mu), Rational(-3, 2)) for i in g1.bm])
    c6 = and_(*[gt((u - v) * re(1 + i - 1) - re(rho), Rational(-3, 2)) for i in g2.an])
    c7 = and_(*[gt((u - v) * re(1 + i) - re(rho), Rational(-3, 2)) for i in g2.bm])
    c8 = gt(abs_(phi) + 2 * re((rho - 1) * (q - p) + (v - u) * (q - p) + (mu - 1) * (v - u)), 0)
    c9 = gt(abs_(phi) - 2 * re((rho - 1) * (q - p) + (v - u) * (q - p) + (mu - 1) * (v - u)), 0)
    c10 = lt(abs_(unbranched_argument(sigma)), bstar * pi)
    c11 = eq(abs_(unbranched_argument(sigma)), bstar * pi)
    c12 = lt(abs_(unbranched_argument(omega)), cstar * pi)
    c13 = eq(abs_(unbranched_argument(omega)), cstar * pi)

    # the condition c14 as other people have read [P] (it differs from the
    # implementation of REDUCE, see SymPy's comment)
    z0 = exp(-(bstar + cstar) * pi * I)
    zos = unpolarify(z0 * omega / sigma)
    zso = unpolarify(z0 * sigma / omega)
    c14: Boolean
    if zos == 1 / zso:
        c14 = and_(eq(phi, 0), le(bstar + cstar, 1),
                   or_(ne(zos, 1), lt(re(mu + rho + v - u), 1), lt(re(mu + rho + q - p), 1)))
    else:
        def _cond(w: Expr) -> Boolean:
            # abs(arg(1 - w)) < pi, avoiding arg(0)
            if w == 1:
                return false
            return lt(abs_(arg_(1 - w)), pi)
        c14 = and_(eq(phi, 0), le(bstar - 1 + cstar, 0),
                   or_(and_(ne(zos, 1), _cond(zos)), and_(lt(re(mu + rho + v - u), 1), eq(zos, 1))))
        c14_alt = and_(eq(phi, 0), le(cstar - 1 + bstar, 0),
                       or_(and_(ne(zso, 1), _cond(zso)), and_(lt(re(mu + rho + q - p), 1), eq(zso, 1))))
        # all the symmetric cases are enumerated below, so the dual
        # condition is the one with (g1, g2) exchanged
        c14 = or_(c14, c14_alt)

    c15: Boolean
    try:
        lambda_c = (q - p) * abs_(omega)**(1 / (q - p)) * cos(psi) + (v - u) * abs_(sigma)**(1 / (v - u)) * cos(theta)
        if not _is_false(_eval_cond(gt(lambda_c, 0))):
            c15 = gt(lambda_c, 0)
        else:
            def lambda_s0(k1: Expr, k2: Expr) -> Expr:
                return k1 * (q - p) * abs_(omega)**(1 / (q - p)) * sin(psi) \
                    + k2 * (v - u) * abs_(sigma)**(1 / (v - u)) * sin(theta)
            us, uo = unbranched_argument(sigma), unbranched_argument(omega)
            lambda_s = piecewise(
                (lambda_s0(ONE, ONE) * lambda_s0(-ONE, -ONE), and_(eq(us, 0), eq(uo, 0))),
                (lambda_s0(sign(uo), ONE) * lambda_s0(sign(uo), -ONE), and_(eq(us, 0), ne(uo, 0))),
                (lambda_s0(ONE, sign(us)) * lambda_s0(-ONE, sign(us)), and_(ne(us, 0), eq(uo, 0))),
                (lambda_s0(sign(uo), sign(us)), true))
            c15 = or_(gt(lambda_c, 0),
                      and_(eq(lambda_c, 0), ne(lambda_s, 0), gt(re(eta), -1)),
                      and_(eq(lambda_c, 0), eq(lambda_s, 0), gt(re(eta), 0)))
    except TypeError:
        c15 = false

    conds: list[Boolean] = []
    conds.append(and_(m * n * s * t != 0, _positive(bstar), _positive(cstar), c1, c2, c3, c10, c12))  # 1
    conds.append(and_(eq(u, v), eq(bstar, 0), _positive(cstar), _positive(sigma), lt(re(rho), 1),
                      c1, c2, c3, c12))  # 2
    conds.append(and_(eq(p, q), eq(cstar, 0), _positive(bstar), _positive(omega), lt(re(mu), 1),
                      c1, c2, c3, c10))  # 3
    conds.append(and_(eq(p, q), eq(u, v), eq(bstar, 0), eq(cstar, 0), _positive(sigma), _positive(omega),
                      lt(re(mu), 1), lt(re(rho), 1), ne(sigma, omega), c1, c2, c3))  # 4
    conds.append(and_(eq(p, q), eq(u, v), eq(bstar, 0), eq(cstar, 0), _positive(sigma), _positive(omega),
                      lt(re(mu + rho), 1), ne(omega, sigma), c1, c2, c3))  # 5
    conds.append(and_(gt(p, q), _positive(s), _positive(bstar), ge(cstar, 0), c1, c2, c3, c5, c10, c13))  # 6
    conds.append(and_(lt(p, q), _positive(t), _positive(bstar), ge(cstar, 0), c1, c2, c3, c4, c10, c13))  # 7
    conds.append(and_(gt(u, v), _positive(m), _positive(cstar), ge(bstar, 0), c1, c2, c3, c7, c11, c12))  # 8
    conds.append(and_(lt(u, v), _positive(n), _positive(cstar), ge(bstar, 0), c1, c2, c3, c6, c11, c12))  # 9
    conds.append(and_(gt(p, q), eq(u, v), eq(bstar, 0), ge(cstar, 0), _positive(sigma), lt(re(rho), 1),
                      c1, c2, c3, c5, c13))  # 10
    conds.append(and_(lt(p, q), eq(u, v), eq(bstar, 0), ge(cstar, 0), _positive(sigma), lt(re(rho), 1),
                      c1, c2, c3, c4, c13))  # 11
    conds.append(and_(eq(p, q), gt(u, v), ge(bstar, 0), eq(cstar, 0), _positive(omega), lt(re(mu), 1),
                      c1, c2, c3, c7, c11))  # 12
    conds.append(and_(eq(p, q), lt(u, v), ge(bstar, 0), eq(cstar, 0), _positive(omega), lt(re(mu), 1),
                      c1, c2, c3, c6, c11))  # 13
    conds.append(and_(lt(p, q), gt(u, v), ge(bstar, 0), ge(cstar, 0), c1, c2, c3, c4, c7, c11, c13))  # 14
    conds.append(and_(gt(p, q), lt(u, v), ge(bstar, 0), ge(cstar, 0), c1, c2, c3, c5, c6, c11, c13))  # 15
    conds.append(and_(gt(p, q), gt(u, v), ge(bstar, 0), ge(cstar, 0),
                      c1, c2, c3, c5, c7, c8, c11, c13, c14))  # 16
    conds.append(and_(lt(p, q), lt(u, v), ge(bstar, 0), ge(cstar, 0),
                      c1, c2, c3, c4, c6, c9, c11, c13, c14))  # 17
    conds.append(and_(eq(t, 0), _positive(s), _positive(bstar), _positive(phi), c1, c2, c10))  # 18
    conds.append(and_(eq(s, 0), _positive(t), _positive(bstar), _negative(phi), c1, c3, c10))  # 19
    conds.append(and_(eq(n, 0), _positive(m), _positive(cstar), _negative(phi), c1, c2, c12))  # 20
    conds.append(and_(eq(m, 0), _positive(n), _positive(cstar), _positive(phi), c1, c3, c12))  # 21
    conds.append(and_(eq(s * t, 0), _positive(bstar), _positive(cstar), c1, c2, c3, c10, c12))  # 22
    conds.append(and_(eq(m * n, 0), _positive(bstar), _positive(cstar), c1, c2, c3, c10, c12))  # 23

    # the cases of [Luke1969], not covered by Prudnikov's: the integral
    # exists from 0 to a > 0, G1 decays exponentially at infinity and the
    # Mellin transform of G2 exists
    mt1_exists = _check_antecedents_1(g1, x, helper=True)
    mt2_exists = _check_antecedents_1(g2, x, helper=True)
    conds.append(and_(mt2_exists, eq(t, 0), lt(u, s), _positive(bstar), c10, c1, c2, c3))  # E1
    conds.append(and_(mt2_exists, eq(s, 0), lt(v, t), _positive(bstar), c10, c1, c2, c3))  # E2
    conds.append(and_(mt1_exists, eq(n, 0), lt(p, m), _positive(cstar), c12, c1, c2, c3))  # E3
    conds.append(and_(mt1_exists, eq(m, 0), lt(q, n), _positive(cstar), c12, c1, c2, c3))  # E4

    # a short cut when one of these holds: the rest are corner cases
    r = or_(*conds)
    if not _is_false(_eval_cond(r)):
        return r

    conds.append(and_(gt(m + n, p), eq(t, 0), eq(phi, 0), _positive(s), _positive(bstar), _negative(cstar),
                      lt(abs_(unbranched_argument(omega)), (m + n - p + 1) * pi),
                      c1, c2, c10, c14, c15))  # 24
    conds.append(and_(gt(m + n, q), eq(s, 0), eq(phi, 0), _positive(t), _positive(bstar), _negative(cstar),
                      lt(abs_(unbranched_argument(omega)), (m + n - q + 1) * pi),
                      c1, c3, c10, c14, c15))  # 25
    conds.append(and_(eq(p, q - 1), eq(t, 0), eq(phi, 0), _positive(s), _positive(bstar),
                      ge(cstar, 0), lt(cstar * pi, abs_(unbranched_argument(omega))),
                      c1, c2, c10, c14, c15))  # 26
    conds.append(and_(eq(p, q + 1), eq(s, 0), eq(phi, 0), _positive(t), _positive(bstar),
                      ge(cstar, 0), lt(cstar * pi, abs_(unbranched_argument(omega))),
                      c1, c3, c10, c14, c15))  # 27
    conds.append(and_(lt(p, q - 1), eq(t, 0), eq(phi, 0), _positive(s), _positive(bstar),
                      ge(cstar, 0), lt(cstar * pi, abs_(unbranched_argument(omega))),
                      lt(abs_(unbranched_argument(omega)), (m + n - p + 1) * pi),
                      c1, c2, c10, c14, c15))  # 28
    conds.append(and_(gt(p, q + 1), eq(s, 0), eq(phi, 0), _positive(t), _positive(bstar), ge(cstar, 0),
                      lt(cstar * pi, abs_(unbranched_argument(omega))),
                      lt(abs_(unbranched_argument(omega)), (m + n - q + 1) * pi),
                      c1, c3, c10, c14, c15))  # 29
    conds.append(and_(eq(n, 0), eq(phi, 0), gt(s + t, 0), _positive(m), _positive(cstar), _negative(bstar),
                      lt(abs_(unbranched_argument(sigma)), (s + t - u + 1) * pi),
                      c1, c2, c12, c14, c15))  # 30
    conds.append(and_(eq(m, 0), eq(phi, 0), gt(s + t, v), _positive(n), _positive(cstar), _negative(bstar),
                      lt(abs_(unbranched_argument(sigma)), (s + t - v + 1) * pi),
                      c1, c3, c12, c14, c15))  # 31
    conds.append(and_(eq(n, 0), eq(phi, 0), eq(u, v - 1), _positive(m), _positive(cstar),
                      ge(bstar, 0), lt(bstar * pi, abs_(unbranched_argument(sigma))),
                      lt(abs_(unbranched_argument(sigma)), (bstar + 1) * pi),
                      c1, c2, c12, c14, c15))  # 32
    conds.append(and_(eq(m, 0), eq(phi, 0), eq(u, v + 1), _positive(n), _positive(cstar),
                      ge(bstar, 0), lt(bstar * pi, abs_(unbranched_argument(sigma))),
                      lt(abs_(unbranched_argument(sigma)), (bstar + 1) * pi),
                      c1, c3, c12, c14, c15))  # 33
    conds.append(and_(eq(n, 0), eq(phi, 0), lt(u, v - 1), _positive(m), _positive(cstar), ge(bstar, 0),
                      lt(bstar * pi, abs_(unbranched_argument(sigma))),
                      lt(abs_(unbranched_argument(sigma)), (s + t - u + 1) * pi),
                      c1, c2, c12, c14, c15))  # 34
    conds.append(and_(eq(m, 0), eq(phi, 0), gt(u, v + 1), _positive(n), _positive(cstar), ge(bstar, 0),
                      lt(bstar * pi, abs_(unbranched_argument(sigma))),
                      lt(abs_(unbranched_argument(sigma)), (s + t - v + 1) * pi),
                      c1, c3, c12, c14, c15))  # 35
    return or_(*conds)


# PORTED FROM SYMPY: sympy/integrals/meijerint.py:_int0oo
def _int0oo(g1: MeijerG, g2: MeijerG, x: Symbol) -> Expr:
    """The integral of ``g1*g2`` over ``(0, oo)`` as a G-function, the
    conditions assumed ([L], section 5.6.2, equation (1)).

    >>> from sympy_extras.core import symbols, S
    >>> from sympy_extras.integrals.meijerint import _g, _int0oo
    >>> s, t, m = symbols('s t m')
    >>> g1 = _g([], [], [-S.Half, 0], [], s**2*t/4)
    >>> g2 = _g([], [], [m/2], [-m/2], t/4)
    >>> _int0oo(g1, g2, t)
    4*meijerg(((0, 1/2), ()), ((m/2,), (-m/2,)), s**(-2))/s**2
    """
    eta, _ = _get_coeff_exp(g1.argument, x)
    omega, _ = _get_coeff_exp(g2.argument, x)

    def neg(params: Sequence[Expr]) -> list[Expr]:
        return [-c for c in params]
    a1 = neg(g1.bm) + list(g2.an)
    a2 = list(g2.aother) + neg(g1.bother)
    b1 = neg(g1.an) + list(g2.bm)
    b2 = list(g2.bother) + neg(g1.aother)
    return _g(a1, a2, b1, b2, omega / eta) / eta


# ---------------------------------------------------------------------------
# the rewriting as G-functions

def _match(f: Expr, pattern: Expr) -> Optional[MatchDict]:
    """The bindings of ``pattern`` for ``f``: the matcher of the core, and
    for a product the factors matched one by one, the bindings substituted
    into the factors left (the matcher misses ``z`` against ``p*z**q`` with
    ``p`` bound to 1)."""
    found = f.match(pattern)
    if found is not None:
        return found
    if not isinstance(pattern, Mul) or not isinstance(f, Mul):
        return None
    patterns = list(Mul.make_args(pattern))
    factors = list(Mul.make_args(f))
    if len(patterns) != len(factors):
        return None

    def assign(i: int, used: frozenset[int], bindings: MatchDict) -> Optional[MatchDict]:
        if i == len(patterns):
            return bindings
        pat = _expr(patterns[i]._xreplace_evaluated(bindings)) if bindings else patterns[i]
        for j, factor_ in enumerate(factors):
            if j in used:
                continue
            m = match(factor_, pat)
            if m is None:
                continue
            merged = dict(bindings)
            merged.update(m)
            result = assign(i + 1, used | {j}, merged)
            if result is not None:
                return result
        return None
    return assign(0, frozenset(), {})


def _polar_binding(value: Basic) -> Basic:
    """A binding with its numbers lifted to polar ones and its exponents
    unpolarified (SymPy's ``unpolarify(polarify(to, lift=True),
    exponents_only=True)``)."""
    if not isinstance(value, Expr):
        return value
    return unpolarify(polarify(value, lift=True)[0], exponents_only=True)


def _rewrite_table(f: Expr, x: Symbol) -> Optional[Rewriting]:
    """``f`` as a sum of G-functions by the table."""
    fz = f.subs(x, z)
    entries = lookup_table().get(_mytype(fz, z))
    if entries is None:
        return None
    for entry in entries:
        matched = _match(fz, entry.formula)
        if not matched:
            continue
        subs: MatchDict = {k: _polar_binding(v) for k, v in matched.items()}
        try:
            if not entry.hint(subs):
                continue
        except TypeError:
            continue
        cond = _unpolarify_boolean(to_boolean(entry.cond.subs(subs)))
        if _is_false(_eval_cond(cond)):
            continue
        res: list[GTerm] = []
        for fac, g in entry.terms(subs):
            try:
                coeff, power = _get_coeff_exp(unpolarify(_expr(fac.subs(subs)).subs(z, x), exponents_only=True), x)
                h = _replace_symbol(_subs_g(g, subs), z, x)
            except ValueError:
                continue
            if _has_infinity(coeff) or _has_infinity(power) or _has_infinity(h):
                continue
            h = _g(h.an, h.aother, h.bm, h.bother, unpolarify(h.argument, exponents_only=True))
            res.append((coeff, power, h))
        if res:
            return res, cond
    return None


def _rewrite_mellin(f: Expr, x: Symbol) -> Optional[Rewriting]:
    """``f`` as one G-function by its Mellin transform in the table of
    :mod:`.mellin` (one kernel), inverted as a Mellin–Barnes integral:
    ``f(x) = 1/(2 pi i) Integral(F(s) x**(-s), s)`` along a line of the
    strip. The representation is checked at a sample point when ``f`` has
    no parameters."""
    from .mellin import GammaQuotient, decompose_integrand, poles_separated
    from .slater import mellin_barnes
    product = decompose_integrand(f, x)
    if product is None or product.one_minus_power or product.log_power or len(product.matches) != 1:
        return None
    try:
        quotient = product.quotient()
    except ValueError:
        return None
    if quotient is None or not quotient.is_gamma_quotient():
        return None
    if lt(quotient.lower, quotient.upper) is not true or poles_separated(quotient) is not True:
        return None
    inverse = GammaQuotient(quotient.constant, quotient.bases + ((x, -ONE),), quotient.numerator,
                            quotient.denominator, quotient.lower, quotient.upper, quotient.condition,
                            quotient.extra)
    barnes = mellin_barnes(inverse)
    if barnes is None:
        return None
    try:
        g = _g(barnes.an, barnes.ap, barnes.bm, barnes.bq, barnes.z)
        coeff, power = _get_coeff_exp(_expr(barnes.prefactor), x)
        _get_coeff_exp(g.argument, x)
    except ValueError:
        return None
    if _has_infinity(coeff) or _has_infinity(g):
        return None
    if f.free_symbols == {x}:
        point = Rational(7, 10)
        value = reliable_value(_expr(f.subs(x, point)), 15)
        represented = reliable_value(_expr((coeff * x**power * g).subs(x, point)), 15)
        if value is None or represented is None or abs(complex(value) - complex(represented)) \
                > 1e-8 * max(1.0, abs(complex(value))):
            return None
    return [(coeff, power, g)], quotient.condition


# PORTED FROM SYMPY: sympy/integrals/meijerint.py:_rewrite_single
@cacheit
def rewrite_single(f: Expr, x: Symbol, recursive: bool = True) -> Optional[Rewriting]:
    """``f`` as a sum of single G-functions ``C*x**s*G(a*x**b)``, ``b``
    rational and ``C`` free of ``x``: the terms ``(C, s, G)`` and the
    condition under which they hold; ``None`` on failure. ``recursive``
    allows the Mellin transforms of :mod:`.mellin` when the table has no
    representation.

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.functions import sin, exp
    >>> from sympy_extras.integrals.meijerint import rewrite_single
    >>> x = symbols('x')
    >>> rewrite_single(sin(x), x)
    ([(sqrt(pi), 0, meijerg(((), ()), ((1/2,), (0,)), x**2/4))], True)
    >>> rewrite_single(exp(-x), x)
    ([(1, 0, meijerg(((), ()), ((0,), ()), x))], True)
    """
    if isinstance(f, MeijerG):
        coeff, m = _coeff_mul(factor(f.argument, x), x)
        if len(m) != 1:
            return None
        power = m[0]
        if isinstance(power, Pow):
            if power.base != x or not isinstance(power.exp, Rational):
                return None
        elif power != x:
            return None
        return [(ONE, ZERO, _g(f.an, f.aother, f.bm, f.bother, coeff * power))], true
    found = _rewrite_table(f, x)
    if found is not None:
        return found
    if not recursive:
        return None
    return _rewrite_mellin(f, x)


# PORTED FROM SYMPY: sympy/integrals/meijerint.py:_rewrite1
def _rewrite1(f: Expr, x: Symbol, recursive: bool = True) -> Optional[tuple[Expr, Expr, list[GTerm], Boolean]]:
    """``f`` as ``fac*po*g``, ``g`` a sum of single G-functions."""
    fac, po, g = _split_mul(f, x)
    found = rewrite_single(g, x, recursive)
    if found:
        return fac, po, found[0], found[1]
    return None


def _pair_key(pair: tuple[Expr, Expr], x: Symbol) -> tuple[int, int, int, int, SortKey, SortKey]:
    a, b = pair
    return (max(len(_exponents(a, x)), len(_exponents(b, x))),
            max(len(_functions(a, x)), len(_functions(b, x))),
            max(len(_find_splitting_points(a, x)), len(_find_splitting_points(b, x))),
            _node_count(a) + _node_count(b), default_sort_key(a), default_sort_key(b))


# PORTED FROM SYMPY: sympy/integrals/meijerint.py:_rewrite2
def _rewrite2(f: Expr, x: Symbol) -> Optional[tuple[Expr, Expr, list[GTerm], list[GTerm], Boolean]]:
    """``f`` as ``fac*po*g1*g2``, ``g1`` and ``g2`` sums of single
    G-functions."""
    fac, po, g = _split_mul(f, x)
    if any(rewrite_single(e, x, False) is None for e in _mul_args(g)):
        return None
    parts = _mul_as_two_parts(g)
    if not parts:
        return None
    parts = sorted(parts, key=lambda pair: _pair_key(pair, x))
    for recursive, (fac1, fac2) in itertools.product((False, True), parts):
        g1 = rewrite_single(fac1, x, recursive)
        g2 = rewrite_single(fac2, x, recursive)
        if g1 and g2:
            cond = and_(g1[1], g2[1])
            if not _is_false(cond):
                return fac, po, g1[0], g2[0], cond
    return None


# ---------------------------------------------------------------------------
# indefinite integration

def _rewrite_hyperbolics_as_exp(f: Expr) -> Expr:
    """The hyperbolic functions of ``f`` as exponentials."""
    rule: dict[Basic, Basic] = {h: h.rewrite(Exp) for h in f.atoms(Sinh, Cosh, Tanh, Coth, Sech, Csch)}
    return _expr(f.xreplace(rule))


def _ordered_first(results: list[ConditionalExpr]) -> ConditionalExpr:
    return min(results, key=lambda r: (_node_count(r[0]) + _node_count(r[1]), default_sort_key(r[0]),
                                       default_sort_key(r[1])))


# PORTED FROM SYMPY: sympy/integrals/meijerint.py:meijerint_indefinite
def meijerint_indefinite(f: ExprLike, x: Symbol) -> Optional[ConditionalExpr]:
    """An antiderivative of ``f`` by rewriting it as G-functions, with the
    condition under which it holds; ``None`` when ``f`` cannot be
    rewritten.

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.functions import sin, exp, besselj
    >>> from sympy_extras.integrals.meijerint import meijerint_indefinite
    >>> x = symbols('x')
    >>> meijerint_indefinite(sin(x), x)
    (-cos(x), True)
    >>> meijerint_indefinite(x*besselj(0, x), x)
    (x*besselj(1, x), True)
    """
    e = to_expr(f)
    results: list[ConditionalExpr] = []
    for a in sorted(_find_splitting_points(e, x) | {ZERO}, key=default_sort_key):
        found = _meijerint_indefinite_1(e.subs(x, x + a), x)
        if found is None:
            continue
        res = (found[0].subs(x, x - a), found[1])
        # SymPy's Piecewise((res, cond), (Integral(f, x), True)) has the
        # functions in every piece when cond is True, or when the
        # integrand has them too
        if _has(res[0], Hyper, MeijerG) and (res[1] is true or e.has(Hyper, MeijerG)):
            results.append(res)
        else:
            return res
    if e.has(Sinh, Cosh, Tanh, Coth, Sech, Csch):
        rv = meijerint_indefinite(_rewrite_hyperbolics_as_exp(e), x)
        if rv is not None:
            value = factor_terms(rv[0])
            exponentials = sorted(value.atoms(Exp), key=default_sort_key)
            return collect(value, [_expr(a) for a in exponentials]), rv[1]
    if results:
        return _ordered_first(results)
    return None


# PORTED FROM SYMPY: sympy/integrals/meijerint.py:_meijerint_indefinite_1
def _meijerint_indefinite_1(f: Expr, x: Symbol) -> Optional[ConditionalExpr]:
    """:func:`meijerint_indefinite` without the substitutions."""
    gs = _rewrite1(f, x)
    if gs is None:
        return None
    fac, po, gl, cond = gs
    res: Expr = ZERO
    for C, s, g in gl:
        a, b = _get_coeff_exp(g.argument, x)
        _, c = _get_coeff_exp(po, x)
        c = c + s
        # the substitution t = a*x**b, the integrand fac*t**rho*g, and
        # t**rho*G(params, t) = G(params + rho, t) ([L], page 150, (4));
        # the integral of G(params, t) is G(1, params + 1, 0, t) or the one
        # with 1 and 0 exchanged, whichever is well defined ([R], section 5)
        fac_ = fac * C * x**(1 + c) / b
        rho = (c + 1) / b
        t = _dummy('t', 'meijerint-indefinite', ONE)
        shifted = [p + rho for p in g.bm]
        if any(p.is_integer and le(p, 0) is true for p in shifted):
            r = -_g(list(g.an), list(g.aother) + [1 - rho], list(g.bm) + [-rho], list(g.bother), t)
        else:
            r = _g(list(g.an) + [1 - rho], list(g.aother), list(g.bm), list(g.bother) + [-rho], t)
        # the antiderivative is most often expected to be defined near 0
        at_zero = f.subs(x, ZERO)
        place: Optional[Expr] = ZERO if b.is_extended_nonnegative and not at_zero.has(S.NaN, S.ComplexInfinity) \
            else None
        expanded = hyperexpand(r.subs(t, a * x**b), place=place)
        res = res + powdenest(fac_ * expanded, polar=True)

    def clean(e: Expr) -> Expr:
        # the superfluous powers of x multiplied out and the constants
        # chopped off
        e = expand_mul(cancel(e), deep=False)
        return add(*_coeff_add(e, x)[1])

    res = piecewise_fold(res, evaluate=None)
    if isinstance(res, Piecewise):
        res = Piecewise(*[(_my_unpolarify(clean(pair.expr)), pair.cond) for pair in res.args])
    else:
        res = _my_unpolarify(clean(res))
    return res, _my_unpolarify_cond(cond)


def meijer_antiderivative(f: ExprLike, x: Symbol) -> Optional[Expr]:
    """SymPy's ``integrate(f, x, meijerg=True, risch=False)``: each term of
    the sum, its constant factor out, by :func:`meijerint_indefinite` when
    the condition of the antiderivative is ``True``, else by the native
    ``manualintegrate``; a lone term whose products of ``sin`` and ``cos``
    are sums is integrated as that sum. ``None`` when a term has no
    antiderivative.

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.functions import sin, cos, besselj
    >>> from sympy_extras.integrals.meijerint import meijer_antiderivative
    >>> x = symbols('x')
    >>> meijer_antiderivative(3*sin(x) + x*besselj(0, x), x)
    x*besselj(1, x) - 3*cos(x)
    """
    from .manualintegrate import manualintegrate
    from .integral import Integral, is_unevaluated
    e = to_expr(f)
    if isinstance(e, Piecewise):
        return None
    if not e.has(x):
        return e * x
    terms = Add.make_args(e)
    parts: list[Expr] = []
    for term in terms:
        coeff, g = term.as_independent(x, as_Add=False)
        h: Optional[Expr] = None
        found = meijerint_indefinite(g, x)
        if found is not None:
            if found[1] is not true:
                # SymPy's Piecewise with the unevaluated integral
                return None
            h = found[0]
        else:
            manual = manualintegrate(g, x)
            if not is_unevaluated(manual) and not manual.has(Integral):
                h = manual
        if h is None and len(terms) == 1:
            expanded = expand(sincos_to_sum(e), mul=True, deep=False)
            if isinstance(expanded, Add):
                return meijer_antiderivative(expanded, x)
        if h is None:
            return None
        parts.append(coeff * h)
    return add(*parts)


# ---------------------------------------------------------------------------
# definite integration

def _minus(value: Optional[ConditionalExpr]) -> Optional[ConditionalExpr]:
    return None if value is None else (-value[0], value[1])


# PORTED FROM SYMPY: sympy/integrals/meijerint.py:meijerint_definite
def meijerint_definite(f: ExprLike, x: Symbol, a: ExprLike, b: ExprLike) -> Optional[ConditionalExpr]:
    """The integral of ``f`` over ``[a, b]`` by rewriting it as a product
    of two G-functions or as one: the value and its convergence
    condition, ``None`` when no theorem applies.

    The integral is brought to ``(0, oo)`` (by translations, reflections
    and Heaviside factors), the integrand written in several forms
    (:func:`_guess_expansion`), and each form rewritten as G-functions.

    Examples
    ========

    >>> from sympy_extras.core import symbols, oo, Integer
    >>> from sympy_extras.functions import exp, sinc
    >>> from sympy_extras.integrals.meijerint import meijerint_definite
    >>> x = symbols('x')
    >>> meijerint_definite(exp(-x**2), x, -oo, oo)
    (sqrt(pi), True)
    >>> meijerint_definite(sinc(x)**2, x, -oo, oo)
    (pi, True)
    >>> meijerint_definite((x + 1)**3*exp(-x), x, Integer(0), oo)
    (16, True)
    """
    e, lower, upper = to_expr(f), to_expr(a), to_expr(b)
    if e.has(DiracDelta) or e.has(SingularityFunctionNode):
        return None
    e_, x_, a_, b_ = e, x, lower, upper
    # a dummy in case a bound has x
    d = Dummy('x')
    e = e.subs(x, d)
    x = d
    if lower == upper:
        return ZERO, true
    results: list[ConditionalExpr] = []
    if isinstance(lower, NegativeInfinity) and not isinstance(upper, Infinity):
        return meijerint_definite(e.subs(x, -x), x, -upper, -lower)
    elif isinstance(lower, NegativeInfinity):
        # over (-oo, oo): split at a sensible point
        innermost = _find_splitting_points(e, x)
        for c in sorted(innermost, key=default_sort_key, reverse=True) + [ZERO]:
            if not c.is_extended_real:
                continue
            res1 = _meijerint_definite_2(e.subs(x, x + c), x)
            if res1 is None:
                continue
            res2 = _meijerint_definite_2(e.subs(x, c - x), x)
            if res2 is None:
                continue
            cond = _condsimp(and_(res1[1], res2[1]))
            if _is_false(cond):
                continue
            return res1[0] + res2[0], cond
    elif isinstance(lower, Infinity):
        return _minus(meijerint_definite(e, x, upper, oo))
    elif lower == 0 and isinstance(upper, Infinity):
        res = _meijerint_definite_2(e, x)
        if res is not None:
            if _has(res[0], MeijerG):
                results.append(res)
            else:
                return res
    else:
        if isinstance(upper, Infinity):
            for split in _find_splitting_points(e, x):
                if ge(lower - split, 0) is true:
                    res = _meijerint_definite_2(e.subs(x, x + split) * heaviside(x + split - lower), x)
                    if res is not None:
                        if _has(res[0], MeijerG):
                            results.append(res)
                        else:
                            return res
        e = e.subs(x, x + lower)
        upper = upper - lower
        lower = ZERO
        if not isinstance(upper, Infinity):
            phi = exp(I * arg_(upper))
            upper = abs_(upper)
            e = e.subs(x, phi * x)
            e = e * heaviside(upper - x) * phi
            upper = oo
        res = _meijerint_definite_2(e, x)
        if res is not None:
            if _has(res[0], MeijerG):
                results.append(res)
            else:
                return res
    if e_.has(Sinh, Cosh, Tanh, Coth, Sech, Csch):
        rv = meijerint_definite(_rewrite_hyperbolics_as_exp(e_), x_, a_, b_)
        if rv is not None:
            value = factor_terms(rv[0])
            exponentials = sorted(value.atoms(Exp), key=default_sort_key)
            return collect(value, [_expr(a) for a in exponentials]), rv[1]
    if results:
        return _ordered_first(results)
    return None


# PORTED FROM SYMPY: sympy/integrals/meijerint.py:_guess_expansion
def _guess_expansion(f: Expr, x: Symbol) -> list[Expr]:
    """Sensible rewritings of the integrand: as it is, ``expand_mul``,
    ``expand``, the trigonometric expansion and the power reduction of
    ``sin`` and ``cos``."""
    res = [f]
    saw = {f}

    def push(e: Expr) -> None:
        if e not in saw:
            res.append(e)
            saw.add(e)
    push(expand_mul(f))
    push(expand(f))
    if f.has(Sin, Cos, Tan, Cot, Sec, Csc, Sinh, Cosh, Tanh, Coth, Sech, Csch):
        push(expand_mul(expand_trig(f)))
    if f.has(Sin, Cos):
        push(sincos_to_sum(f))
    return res


# PORTED FROM SYMPY: sympy/integrals/meijerint.py:_meijerint_definite_2
def _meijerint_definite_2(f: Expr, x: Symbol) -> Optional[ConditionalExpr]:
    """The integral of ``f`` over ``(0, oo)``, the forms of
    :func:`_guess_expansion` in turn."""
    dummy = _dummy('x', 'meijerint-definite2', f, positive=True)
    f = f.subs(x, dummy)
    x = dummy
    if f == 0:
        return ZERO, true
    for g in _guess_expansion(f, x):
        res = _meijerint_definite_3(g, x)
        if res is not None:
            return res
    return None


# PORTED FROM SYMPY: sympy/integrals/meijerint.py:_meijerint_definite_3
def _meijerint_definite_3(f: Expr, x: Symbol) -> Optional[ConditionalExpr]:
    """The integral of ``f`` over ``(0, oo)``, by linearity when the whole
    fails."""
    res = _meijerint_definite_4(f, x)
    if res is not None and not _is_false(res[1]):
        return res
    if isinstance(f, Add):
        found = [_meijerint_definite_4(_expr(g), x) for g in f.args]
        values: list[Expr] = []
        conds: list[Boolean] = []
        for r in found:
            if r is None:
                return None
            values.append(r[0])
            conds.append(r[1])
        c = and_(*conds)
        if not _is_false(c):
            return add(*values), c
    return None


# PORTED FROM SYMPY: sympy/integrals/meijerint.py:_meijerint_definite_4
def _meijerint_definite_4(f: Expr, x: Symbol, only_double: bool = False) -> Optional[ConditionalExpr]:
    """The integral of ``f`` over ``(0, oo)`` by the theorems: ``f`` as one
    G-function (not with ``only_double``) or as a product of two."""
    if not only_double:
        gs = _rewrite1(f, x, recursive=False)
        if gs is not None:
            fac, po, g, cond = gs
            res: Expr = ZERO
            for C, s, h in g:
                if C == 0:
                    continue
                C, h = _rewrite_saxena_1(fac * C, po * x**s, h, x)
                res = res + C * _int0oo_1(h, x)
                cond = and_(cond, _check_antecedents_1(h, x))
                if _is_false(cond):
                    break
            cond = _my_unpolarify_cond(cond)
            if not _is_false(cond):
                return _my_unpolarify(hyperexpand(res)), cond
    gs2 = _rewrite2(f, x)
    if gs2 is not None:
        for full_pb in [False, True]:
            fac, po, g1, g2, cond = gs2
            res = ZERO
            done = False
            for C1, s1, f1 in g1:
                for C2, s2, f2 in g2:
                    r = _rewrite_saxena(fac * C1 * C2, po * x**(s1 + s2), f1, f2, x, full_pb)
                    if r is None:
                        # non-rational exponents
                        return None
                    C, f1_, f2_ = r
                    cond = and_(cond, _check_antecedents(f1_, f2_, x))
                    if _is_false(cond):
                        done = True
                        break
                    res = res + C * _int0oo(f1_, f2_, x)
                if done:
                    break
            cond = _my_unpolarify_cond(cond)
            if not _is_false(cond):
                if only_double:
                    return res, cond
                return _my_unpolarify(hyperexpand(res)), cond
    return None
