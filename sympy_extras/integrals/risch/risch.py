"""
The Risch Algorithm for transcendental function integration, on the typed
core.

The core algorithms for the Risch algorithm are here.  The subproblem
algorithms are in the rde.py and prde.py files for the Risch
Differential Equation solver and the parametric problems solvers,
respectively.  All important information concerning the differential extension
for an integrand is stored in a DifferentialExtension object, which in the code
is usually called DE.  Throughout the code and Inside the DifferentialExtension
object, the conventions/attribute names are that the base domain is QQ and each
differential extension is x, t0, t1, ..., tn-1 = DE.t. DE.x is the variable of
integration (Dx == 1), DE.D is a list of the derivatives of
x, t1, t2, ..., tn-1 = t, DE.T is the list [x, t1, t2, ..., tn-1], DE.t is the
outer-most variable of the differential extension at the given level (the level
can be adjusted using DE.increment_level() and DE.decrement_level()),
k is the field C(x, t0, ..., tn-2), where C is the constant field.  The
numerator of a fraction is denoted by a and the denominator by
d.  If the fraction is named f, fa == numer(f) and fd == denom(f).
Fractions are returned as tuples (fa, fd).  DE.d and DE.t are used to
represent the topmost derivation and extension variable, respectively.
The docstring of a function signifies whether an argument is in k[t], in
which case it will just return a TowerPoly in t, or in k(t), in which case it
will return the fraction (fa, fd). Other variable names probably come
from the names used in Bronstein's book.

Ported from SymPy's ``sympy.integrals.risch`` (in the version of Aaron
Meurer's pull requests, see the notice below). The polynomials of the
tower are :class:`~sympy_extras.integrals.risch.tower.TowerPoly` (in one
generator over the field of the lower levels) and their coefficients
:class:`~sympy_extras.integrals.risch.tower.RationalFunction` (SymPy's
``ZZ(x, t0)``, ``QQ_I(x)`` and ``EX``); the expressions are those of the
typed core, and the face :mod:`sympy_extras.integrals.risch.risch_sympy`
converts SymPy's.

Examples
========

>>> from sympy_extras.core import symbols
>>> from sympy_extras.functions import exp, log
>>> from sympy_extras.integrals.risch.risch import risch_integrate
>>> x = symbols('x')
>>> risch_integrate(exp(x)*exp(exp(x)), x)
exp(exp(x))
>>> risch_integrate(exp(-x**2), x)
Integral(exp(-x**2), x)

References
==========

.. [Bronstein] M. Bronstein, *Symbolic Integration I: Transcendental
   Functions*, 2nd edition, Springer, 2005.
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/integrals/risch.py (SymPy 1.14 and Aaron Meurer's pull requests)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: the typed core and the polynomials of integrals.risch.tower
# (TowerPoly over RationalFunction) instead of SymPy's Poly and domains;
# NonElementaryIntegral as a node of the core; typed DifferentialExtension.
# ---------------------------------------------------------------------
# Ported from SymPy (sympy.integrals.risch) and from Aaron Meurer's pull
# requests to SymPy: sympy/sympy#30180 and #30221 ("Risch: Implement most
# remaining exp-log cases from Bronstein", branch risch-rde-cancellation
# @d7dafa43) and #30292 ("Hypertangent cases in the Risch algorithm", branch
# risch-hypertangent @66bc72d6, the version taken here), by Claude Fable 5.1
# for sympy-extras; rewritten on the typed core by Claude Opus 5.5. The
# algorithm is Bronstein's (Symbolic Integration I: Transcendental Functions,
# 2nd edition, Springer 2005) as implemented by Aaron Meurer. SymPy's licence
# (BSD 3-clause, copyright the SymPy Development Team) applies to this file:
# see LICENSE-SymPy in this directory.
from __future__ import annotations

from fractions import Fraction
from itertools import count
from math import lcm as integer_lcm
from types import TracebackType
from typing import Callable, Final, Iterable, Iterator, Literal, Optional, Sequence, TypeVar, Union

from sympy_extras.core.add import Add, add
from sympy_extras.core.basic import Basic, NodePrinter
from sympy_extras.core.expr import Expr, as_expr
from sympy_extras.core.function import Lambda, count_ops, expand, expand_trig
from sympy_extras.core.mul import Mul, mul
from sympy_extras.core.numbers import Float, I, Integer, Rational
from sympy_extras.core.power import Pow, pow
from sympy_extras.core.relational import ne
from sympy_extras.core.sorting import default_sort_key, ordered
from sympy_extras.core.symbol import Dummy, Symbol
from sympy_extras.functions.elementary.exponential import Exp, Log, exp, log
from sympy_extras.functions.elementary.hyperbolic import Cosh, Coth, Sinh, Tanh
from sympy_extras.functions.elementary.piecewise import piecewise
from sympy_extras.functions.elementary.trigonometric import (Acos, Acot, Asin, Atan, Cos, Cot, Csc, Sec, Sin, Tan,
                                                             atan, cos, sin, tan)
from sympy_extras.polys.polyerrors import PolynomialError
from sympy_extras.polys.rings import PolyElement, PolyRing
from .tower import TOWER, RationalFunction, TowerPoly, UnsupportedConstant

__all__ = ['DifferentialExtension', 'DecrementLevel', 'NonElementaryIntegral', 'NonElementaryIntegralException',
           'LaurentPolynomial', 'risch_integrate', 'integer_powers', 'gcdex_diophantine', 'frac_in', 'as_poly_1t',
           'derivation', 'derivation_element', 'get_case', 'splitfactor', 'splitfactor_sqf',
           'canonical_representation', 'hermite_reduce', 'polynomial_reduce', 'laurent_series',
           'recognize_derivative', 'recognize_log_derivative', 'residue_reduce', 'residue_reduce_to_basic',
           'residue_reduce_derivation', 'integrate_primitive_polynomial', 'integrate_primitive',
           'integrate_hyperexponential_polynomial', 'integrate_hyperexponential',
           'integrate_hypertangent_polynomial', 'integrate_hypertangent_reduced', 'integrate_hypertangent',
           'integrate_nonlinear_no_specials', 'unique_dummy']

#: the type of an extension of the tower, exts[i] describing T[i + 1]
Extension = Literal['exp', 'log', 'tan', 'atan']
#: the case of a derivation, as classified by get_case()
Case = Literal['base', 'primitive', 'exp', 'tan', 'other_linear', 'other_nonlinear']
#: an element of k(t): an expression, an element of the field, a polynomial or a fraction (fa, fd)
FractionLike = Union[Expr, RationalFunction, TowerPoly, tuple[TowerPoly, TowerPoly]]
#: the functions of the generators of the tower (for the back substitution)
TFunc = Callable[[Expr], Expr]
#: the back substitutions (old, new)
Substitutions = list[tuple[Expr, Expr]]

_T = TypeVar('_T')

_ZERO: Final[Expr] = Integer(0)
_ONE: Final[Expr] = Integer(1)

_DUMMIES: Iterator[int] = count()


def unique_dummy(name: str) -> Dummy:
    """A dummy with a name of its own (``name_n``): the typed polynomials
    of :mod:`sympy_extras.polys` identify their symbols by name."""
    return Dummy('%s_%d' % (name, next(_DUMMIES)))


def _arg(e: Basic) -> Expr:
    """The (first) argument of the function application ``e``."""
    return as_expr(e.args[0])


def _expr(b: Basic) -> Expr:
    return as_expr(b)


def _xrep(e: Expr, rule: dict[Basic, Basic]) -> Expr:
    """``xreplace`` with the rebuilt nodes evaluated (SymPy's)."""
    return as_expr(e._xreplace_evaluated(rule))


def _subs(e: Expr, pairs: Sequence[tuple[Expr, Expr]]) -> Expr:
    """``e.subs(pairs)`` (in sequence)."""
    if not pairs:
        return e
    return e.subs([(a, b) for a, b in pairs])


def _element(e: Expr) -> RationalFunction:
    return RationalFunction.from_expr(e)


def _is_rational_function(e: Expr, T: Sequence[Symbol]) -> bool:
    return e.is_rational_function(*T) is True


def _has(e: Expr, T: Sequence[Expr]) -> bool:
    return any(e.has(v) for v in T)


def _ratio(a: Expr, b: Expr) -> Optional[Fraction]:
    """``a/b`` when it is a rational number (SymPy's
    ``cancel(a/b).is_Rational``)."""
    try:
        q = _element(a) / _element(b)
    except (UnsupportedConstant, ZeroDivisionError):
        return None
    return q.as_fraction()


def _fraction_of(e: Expr) -> Fraction:
    if isinstance(e, Rational):
        return Fraction(e.p, e.q)
    raise TypeError('a rational number is expected, got %s' % (e,))


def _number(q: Fraction) -> Expr:
    if q.denominator == 1:
        return Integer(q.numerator)
    return Rational(q.numerator, q.denominator)


def integer_powers(exprs: Iterable[Expr]) -> list[tuple[Expr, list[tuple[Expr, Expr]]]]:
    """
    Rewrites a list of expressions as integer multiples of each other.

    Explanation
    ===========

    For example, if you have [x, x/2, x**2 + 1, 2*x/3], then you can rewrite
    this as [(x/6) * 6, (x/6) * 3, (x**2 + 1) * 1, (x/6) * 4]. This is useful
    in the Risch integration algorithm, where we must write exp(x) + exp(x/2)
    as (exp(x/2))**2 + exp(x/2), but not as exp(x) + sqrt(exp(x)) (this is
    because only the transcendental case is implemented and we therefore cannot
    integrate algebraic extensions). The integer multiples returned by this
    function for each term are the smallest possible (their content equals 1).

    Returns a list of tuples where the first element is the base term and the
    second element is a list of `(item, factor)` terms, where `factor` is the
    integer multiplicative factor that must multiply the base term to obtain
    the original item.

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.integrals.risch.risch import integer_powers
    >>> x = symbols('x')
    >>> integer_powers([x, x/2, x**2 + 1, 2*x/3])
    [(x/6, [(x, 6), (x/2, 3), (2*x/3, 4)]), (x**2 + 1, [(x**2 + 1, 1)])]
    """
    terms: dict[Expr, list[tuple[Expr, Expr]]] = {}
    for term in exprs:
        for trm, trm_list in terms.items():
            a = _ratio(term, trm)
            if a is not None:
                trm_list.append((term, _number(a)))
                break
        else:
            terms[term] = [(term, _ONE)]

    newterms: dict[Expr, list[tuple[Expr, Expr]]] = {}
    for term, term_list in terms.items():
        common_denom = 1
        for _, i in term_list:
            if isinstance(i, Rational):
                common_denom = integer_lcm(common_denom, i.q)
        newterm = mul(term, Rational(1, common_denom))
        newmults = [(i, _number(_fraction_of(j) * common_denom)) for i, j in term_list]
        newterms[newterm] = newmults

    return sorted(newterms.items(), key=lambda item: item[0].sort_key())


# ---------------------------------------------------------------------
# the half-angle tangents back to sines and cosines

def _poly_in(e: Expr, gen: Expr) -> TowerPoly:
    return TowerPoly.from_expr(e, gen)


def _bivariate(e: Expr, s: Symbol, c: Symbol, R: PolyRing[RationalFunction]) -> PolyElement[RationalFunction]:
    """The polynomial ``e`` in ``s`` and ``c`` with coefficients in the
    field of the other generators."""
    terms: dict[tuple[int, ...], RationalFunction] = {}
    p = _poly_in(e, s)
    for i, cs in p.terms():
        q = TowerPoly.from_element_strict(cs, c)
        for j, cc in q.terms():
            terms[(i, j)] = cc
    return R.from_dict(terms)


def _bivariate_expr(f: PolyElement[RationalFunction], s: Symbol, c: Symbol) -> Expr:
    return add(*[mul(v.as_expr(), pow(s, Integer(m[0])), pow(c, Integer(m[1]))) for m, v in f.terms.items()])


def _half_angle_to_sincos(expr: Expr, tang: Expr, theta: Expr,
        x: Optional[Symbol] = None) -> Expr:
    """
    Rewrite the rational functions of ``tang`` == tan(theta/2) in ``expr``
    as rational functions of sin(theta) and cos(theta).

    Explanation
    ===========

    tan(theta/2) == sin(theta)/(1 + cos(theta)), so a rational function of
    the half-angle tangent is a rational function of s = sin(theta) and
    c = cos(theta), and the two agree as meromorphic functions: the
    rewriting is exact, also inside the arguments of other functions.
    Modulo s**2 + c**2 == 1, the numerator and the denominator are each
    reduced to be linear in s (or in c), and the denominator is optionally
    made free of s (or of c) by multiplying with its conjugate.  Of those
    four forms, the one with the fewest operations is used, unless the
    tangent form has fewer still.

    Each maximal rational subexpression is rewritten as a whole
    (rational terms of a sum, rational factors of a product), except
    for the argument of an arc-tangent that is a polynomial in ``tang``:
    atan(c*tan(theta/2) + d) is the conventional form, and the one
    whose spurious discontinuities are removed by Integral.doit().

    If ``x`` is given, ``expr`` is an antiderivative with respect to x,
    and its presentation may change by a locally constant function:
    additive constants are dropped from its rational part, and its
    logarithmic terms with polynomial arguments in ``tang`` and
    commensurable constant coefficients are combined into a single
    logarithm, constant factors being dropped from its argument, whenever
    that is shorter (e.g. log(tan(x/2)**2 + 1) - log(tan(x/2)**2 + 3)
    becomes -log(cos(x) + 2)).
    """
    T, s, c = unique_dummy('T'), unique_dummy('s'), unique_dummy('c')
    sincos: dict[Basic, Basic] = {s: sin(theta), c: cos(theta)}
    R: PolyRing[RationalFunction] = PolyRing(TOWER, ['s', 'c'])

    def ops(e: Expr) -> int:
        n = count_ops(e)
        return n.p if isinstance(n, Integer) else 0

    def dropconst(e: Expr) -> Expr:
        if x is None:
            return e
        return add(*[i for i in Add.make_args(e) if i.has(x)])

    def forms(e: Expr, top: bool = False) -> list[Expr]:
        # The candidate forms of the rational function e of T: the
        # sine/cosine forms, then the tangent form
        fe = _element(e)
        n0 = _poly_in(fe.numerator().as_expr(), T)
        d0 = _poly_in(fe.denominator().as_expr(), T)
        m = max(n0.degree(), d0.degree())
        # t == s/(1 + c), with the powers of 1 + c cleared
        n, d = [add(*[mul(coeff.as_expr(), pow(s, Integer(k)), pow(add(_ONE, c), Integer(m - k)))
            for k, coeff in p.terms()]) for p in (n0, d0)]
        cands: list[Expr] = []
        for lin, other in ((s, c), (c, s)):
            G = _poly_in(add(pow(lin, Integer(2)), pow(other, Integer(2)), Integer(-1)), lin)
            nl = _poly_in(n, lin).rem(G)
            dl = _poly_in(d, lin).rem(G)
            cands.append(mul(nl.as_expr(), pow(dl.as_expr(), Integer(-1))))
            if dl.degree(lin) == 1:
                d1, d0_ = dl.nth(1), dl.nth(0)
                conj = add(d0_.as_expr(), mul(Integer(-1), d1.as_expr(), lin))
                nl = _poly_in(mul(nl.as_expr(), conj), lin).rem(G)
                dl = _poly_in(mul(dl.as_expr(), conj), lin).rem(G)
                cands.append(mul(nl.as_expr(), pow(dl.as_expr(), Integer(-1))))
        out: list[Expr] = []
        for i in cands:
            fi = _element(i)
            ni, di = fi.numerator(), fi.denominator()
            if not di.has(s, c):
                i = expand(fi.as_expr())
            elif top:
                # Split off the polynomial part, so that the constant
                # term can be dropped
                Pn = _bivariate(ni.as_expr(), s, c, R)
                Pd = _bivariate(di.as_expr(), s, c, R)
                (q,), r = Pn.div([Pd])
                i = add(expand(_bivariate_expr(q, s, c)), mul(_bivariate_expr(r, s, c), pow(di.as_expr(), Integer(-1))))
            else:
                i = mul(ni.as_expr(), pow(di.as_expr(), Integer(-1)))
            i = _xrep(i, sincos)
            if top:
                i = dropconst(i)
            out.append(i)
        out.append(_xrep(e, {T: tang}))
        return out

    def pick(cands: Sequence[_T], key: Callable[[_T], int]) -> _T:
        # The first shortest sine/cosine form, unless the tangent form is
        # strictly shorter
        best = min(cands[:-1], key=key)
        if key(cands[-1]) < key(best):
            return cands[-1]
        return best

    def rational(e: Expr, top: bool = False) -> Expr:
        return pick(forms(e, top), ops)

    def logterm(coeff: Expr, p: Expr) -> Expr:
        # coeff*log(p) up to a constant: a constant factor and a constant
        # numerator are pulled out of p
        def strip(q: Expr) -> tuple[Expr, Expr]:
            n, d = [i.primitive()[1] for i in q.as_numer_denom()]
            if x is not None:
                from sympy_extras.core.exprtools import factor_terms
                n = factor_terms(n).as_independent(x, as_Add=False)[1]
                d = factor_terms(d).as_independent(x, as_Add=False)[1]
            if x is not None and not n.has(x):
                return (Integer(-1), d)
            return (_ONE, mul(n, pow(d, Integer(-1))))
        sgn, stripped = pick([strip(q) for q in forms(p)], lambda i: ops(i[1]))
        return mul(sgn, coeff, log(stripped))

    def rw(e: Expr) -> Expr:
        if not e.has(T):
            return e
        if _is_rational_function(e, [T]):
            return rational(e)
        args = [as_expr(i) for i in e.args]
        if isinstance(e, (Add, Mul)):
            rat: list[Expr] = []
            rest: list[Expr] = []
            for i in args:
                if i.has(T) and _is_rational_function(i, [T]):
                    rat.append(i)
                else:
                    rest.append(rw(i))
            if rat:
                rest.append(rational(add(*rat) if isinstance(e, Add) else mul(*rat)))
            return add(*rest) if isinstance(e, Add) else mul(*rest)
        if isinstance(e, Atan) and args[0].is_polynomial(T) is True:
            return _xrep(e, {T: tang})
        return _expr(e._evaluated(tuple(rw(i) for i in args)))

    def top_level(e: Expr) -> Expr:
        rat: list[Expr] = []
        logs: list[tuple[Expr, Expr]] = []
        rest: list[Expr] = []
        for i in Add.make_args(e):
            coeff, l = i.as_independent(T)
            if not i.has(T):
                rest.append(i)
            elif _is_rational_function(i, [T]):
                rat.append(i)
            elif (isinstance(l, Log) and _arg(l).is_polynomial(T) is True and
                    x is not None and not coeff.has(x)):
                logs.append((coeff, _arg(l)))
            else:
                rest.append(rw(i))
        if rat:
            rest.append(rational(add(*rat), top=True))
        # Group the logarithms by commensurable coefficients
        groups: list[tuple[Expr, list[tuple[Expr, Expr]]]] = []
        for coeff, p in logs:
            for group in groups:
                ratio = _ratio(coeff, group[0])
                if ratio is not None:
                    group[1].append((_number(ratio), p))
                    break
            else:
                if coeff.could_extract_minus_sign():
                    groups.append((mul(Integer(-1), coeff), [(Integer(-1), p)]))
                else:
                    groups.append((coeff, [(_ONE, p)]))
        for r, items in groups:
            separate = add(*[logterm(mul(r, n), p) for n, p in items])
            l_ = 1
            for n, _ in items:
                if isinstance(n, Rational):
                    l_ = integer_lcm(l_, n.q)
            combined = logterm(mul(r, Rational(1, l_)), mul(*[pow(p, mul(n, Integer(l_))) for n, p in items]))
            if ops(combined) <= ops(separate):
                rest.append(combined)
            else:
                rest.append(separate)
        return add(*rest)

    expr = _xrep(expr, {tang: T})
    try:
        if x is None:
            return rw(expr)
        return top_level(expr)
    except (UnsupportedConstant, PolynomialError, ZeroDivisionError):
        return _xrep(expr, {T: tang})


# ---------------------------------------------------------------------
# the rewriting of the trigonometric functions through tangents

def _tan_form(e: Expr) -> Expr:
    """SymPy's ``rewrite(tan)`` of ``sin``, ``cos``, ``sec``, ``csc``
    (through the tangent of the half angle) and ``cot``."""
    u = _arg(e)
    if isinstance(e, Cot):
        return pow(tan(u), Integer(-1))
    h = tan(mul(Rational(1, 2), u))
    h2 = pow(h, Integer(2))
    if isinstance(e, Sin):
        return mul(Integer(2), h, pow(add(h2, _ONE), Integer(-1)))
    if isinstance(e, Cos):
        return mul(add(_ONE, mul(Integer(-1), h2)), pow(add(h2, _ONE), Integer(-1)))
    if isinstance(e, Sec):
        return mul(add(h2, _ONE), pow(add(_ONE, mul(Integer(-1), h2)), Integer(-1)))
    return mul(add(h2, _ONE), pow(mul(Integer(2), h), Integer(-1)))


def _numbered_dummies(prefix: str) -> Iterator[Symbol]:
    for i in count():
        yield Dummy('%s%d' % (prefix, i))


def _numbered_symbols(prefix: str) -> Iterator[Symbol]:
    for i in count():
        yield Symbol('%s%d' % (prefix, i))


class DifferentialExtension:
    """
    A container for all the information relating to a differential extension.

    Explanation
    ===========

    The attributes of this object are (see also the docstring of __init__):

    - f: The original (Expr) integrand.
    - x: The variable of integration.
    - T: List of variables in the extension.
    - D: List of derivations in the extension; corresponds to the elements of T.
    - fa: TowerPoly of the numerator of the integrand.
    - fd: TowerPoly of the denominator of the integrand.
    - Tfuncs: Lambda() representations of each element of T (except for x).
      For back-substitution after integration.
    - backsubs: A (possibly empty) list of further substitutions to be made on
      the final integral to make it look more like the integrand.
    - sincos_args: The set of arguments of the sines, cosines, secants and
      cosecants that were rewritten through the tangent of the half angle
      to build the tower; restore_sincos() rewrites rational functions of
      those tangents back through them.
    - exts: The type ('exp', 'log', 'tan' or 'atan') of each extension;
      exts[i] describes T[i + 1] (T[0] == x is not an extension).
    - extargs: The argument of the exp, log, tan or atan of each
      extension, indexed like exts (i.e. T[i + 1] == exts[i](extargs[i])).
    - cases: List of string representations of the cases of T.
    - t: The top level extension variable, as defined by the current level
      (see level below).
    - d: The top level extension derivation, as defined by the current
      derivation (see level below).
    - case: The string representation of the case of self.d.
    (Note that self.T and self.D will always contain the complete extension,
    regardless of the level.  Therefore, you should ALWAYS use DE.t and DE.d
    instead of DE.T[-1] and DE.D[-1].  If you want to have a list of the
    derivations or variables only up to the current level, use
    DE.D[:len(DE.D) + DE.level + 1] and DE.T[:len(DE.T) + DE.level + 1].  Note
    that, in particular, the derivation() function does this.)

    The following are also attributes, but will probably not be useful other
    than in internal use:
    - newf: Expr form of fa/fd.
    - level: The number (between -1 and -len(self.T)) such that
      self.T[self.level] == self.t and self.D[self.level] == self.d.
      Use the methods self.increment_level() and self.decrement_level() to change
      the current level.

    A tower given by hand (for testing) is built by :meth:`manual`.

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.functions import exp, log
    >>> from sympy_extras.integrals.risch.risch import DifferentialExtension
    >>> x = symbols('x')
    >>> DE = DifferentialExtension(log(x) + exp(x), x, handle_first='exp')
    >>> DE.T, DE.D, DE.exts, DE.cases
    ([x, t0, t1], [TowerPoly(1, x), TowerPoly(t0, t0), TowerPoly(1/x, t1)], ['exp', 'log'], ['base', 'exp', 'primitive'])
    >>> DE.fa, DE.fd
    (TowerPoly(t0 + t1, t1), TowerPoly(1, t1))
    """
    f: Expr
    origf: Expr
    x: Symbol
    T: list[Symbol]
    D: list[TowerPoly]
    fa: TowerPoly
    fd: TowerPoly
    Tfuncs: list[TFunc]
    backsubs: Substitutions
    sincos_args: set[Expr]
    exts: list[Optional[Extension]]
    extargs: list[Optional[Expr]]
    cases: list[Case]
    case: Case
    t: Symbol
    d: TowerPoly
    newf: Expr
    level: int
    ts: Iterator[Symbol]
    dummy: bool
    _elements: dict[int, RationalFunction]

    def __init__(self, f: Expr, x: Symbol, handle_first: str = 'log', dummy: bool = False,
            rewrite_complex: Optional[bool] = None) -> None:
        """
        Tries to build a transcendental extension tower from ``f`` with respect to ``x``.

        Explanation
        ===========

        If it is successful, creates a DifferentialExtension object with, among
        others, the attributes fa, fd, D, T, Tfuncs, and backsubs such that
        fa and fd are TowerPolys in T[-1] with rational coefficients in T[:-1],
        fa/fd == f, and D[i] is a TowerPoly in T[i] with rational coefficients in
        T[:i] representing the derivative of T[i] for each i from 1 to len(T).
        Tfuncs is a list of Lambda objects for back replacing the functions
        after integrating.  Lambda() is only used (instead of lambda) to make
        them easier to test and debug. Note that Tfuncs corresponds to the
        elements of T, except for T[0] == x, but they should be back-substituted
        in reverse order.  backsubs is a (possibly empty) back-substitution list
        that should be applied on the completed integral to make it look more
        like the original integrand.

        If it is unsuccessful, it raises NotImplementedError.
        """
        if handle_first not in ('log', 'exp'):
            raise ValueError("handle_first must be 'log' or 'exp', not %s." %
                str(handle_first))

        # f will be the original function, self.f might change if we reset
        # (e.g., we pull out a constant from an exponential); origf always
        # stays the expression the user handed in
        self.f = f
        self.origf = f
        self.x = x
        # setting the default value 'dummy'
        self.dummy = dummy
        self._elements = {}
        # Not cleared by reset(): the restarts below work on the
        # rewritten integrand, where the original functions are gone
        self.sincos_args = set()
        self.reset()
        exp_new_extension: Optional[bool] = True
        log_new_extension = True

        # case of 'automatic' choosing
        if rewrite_complex is None:
            rewrite_complex = self.f.has(I)

        if rewrite_complex:
            # rewrite the trigonometric components
            self.newf = _expr(self.newf.rewrite(Sin, Cos, Cot, Tan, Sinh, Cosh, Coth, Tanh, Exp))
            self.newf = _expr(self.newf.rewrite(Asin, Acos, Acot, Atan, Log))
            self.newf = _cancel_expr(self.newf)
        else:
            if any(i.has(x) for i in self.f.atoms(Sinh, Cosh, Coth, Tanh, Asin, Acos)):
                raise NotImplementedError("Hyperbolic, arc-sine and "
                    "arc-cosine extensions are not supported (yet!).  Try "
                    "rewriting in terms of exp and log, or using "
                    "rewrite_complex=True.")
            self._rewrite_trig()

        exps: list[Expr] = []
        pows: list[Expr] = []
        numpows: list[Expr] = []
        sympows: list[Expr] = []
        logs: list[Expr] = []
        symlogs: list[Expr] = []
        tans: list[Expr] = []
        atans: list[Expr] = []
        tan_new_extension, atan_new_extension = True, True

        while True:
            if _is_rational_function(self.newf, self.T):
                break

            if not (exp_new_extension or log_new_extension or
                    tan_new_extension or atan_new_extension):
                # We couldn't find a new extension on the last pass, so I guess
                # we can't do it.
                raise NotImplementedError("Couldn't find an elementary "
                    "transcendental extension for %s.  Try using a " % str(f) +
                    "manual extension with the extension flag.")

            exps, pows, numpows, sympows, log_new_extension = \
                    self._rewrite_exps_pows(exps, pows, numpows, sympows, log_new_extension)

            logs, symlogs = self._rewrite_logs(logs, symlogs)

            if handle_first == 'exp' or not log_new_extension:
                exp_new_extension = self._exp_part(exps)
                if exp_new_extension is None:
                    # Reset and restart.  reset() clears backsubs, so any
                    # rewrite recorded there -- a branch-constant Dummy
                    # from _log_part(), a user radical's notation -- must
                    # first be folded back into newf, or the Dummy leaks
                    # into the final result as a free symbol and the
                    # original notation is never re-encountered by the
                    # rebuild.
                    self.newf = _subs(self.newf, self.backsubs)
                    self.f = self.newf
                    self.reset()
                    # The fold restores acot from its branch constant
                    self._rewrite_trig()
                    exp_new_extension = True
                    continue

            if handle_first == 'log' or not exp_new_extension:
                log_new_extension = self._log_part(logs)

            tans = update_sets(tans, self._atoms(Tan),
                lambda i: _is_rational_function(_arg(i), self.T) and _has(_arg(i), self.T))
            tan_new_extension = self._tan_part(tans)
            atans = update_sets(atans, self._atoms(Atan),
                lambda i: _is_rational_function(_arg(i), self.T) and _has(_arg(i), self.T))
            atan_new_extension = self._atan_part(atans)

        self.fa, self.fd = frac_in(self.newf, self.t)
        self._auto_attrs()

    @classmethod
    def manual(cls, D: Sequence[TowerPoly], exts: Optional[Sequence[Optional[Extension]]] = None,
               extargs: Optional[Sequence[Optional[Expr]]] = None, Tfuncs: Optional[Sequence[TFunc]] = None,
               backsubs: Optional[Substitutions] = None, x: Optional[Symbol] = None,
               T: Optional[Sequence[Symbol]] = None, fa: Optional[TowerPoly] = None,
               fd: Optional[TowerPoly] = None) -> DifferentialExtension:
        """A tower given by its derivations (SymPy's ``extension={'D': ...}``):
        ``D[i]`` the derivative of the generator ``T[i]`` (by default the
        generator of ``D[i]``). This does not check that the tower is
        valid (transcendental) or even self-consistent; it is meant for
        testing and debugging.

        >>> from sympy_extras.core import symbols
        >>> from sympy_extras.integrals.risch.risch import DifferentialExtension
        >>> from sympy_extras.integrals.risch.tower import TowerPoly
        >>> x, t = symbols('x t')
        >>> DE = DifferentialExtension.manual([TowerPoly.from_expr(1, x), TowerPoly.from_expr(t, t)])
        >>> DE.t, DE.d, DE.case, DE.cases
        (t, TowerPoly(t, t), 'exp', ['base', 'exp'])
        """
        DE = cls.__new__(cls)
        DE.D = list(D)
        DE.T = list(T) if T is not None else [_symbol(d.gen) for d in DE.D]
        DE.x = x if x is not None else DE.T[0]
        DE.exts = list(exts) if exts is not None else []
        DE.extargs = list(extargs) if extargs is not None else []
        DE.Tfuncs = list(Tfuncs) if Tfuncs is not None else []
        DE.backsubs = list(backsubs) if backsubs is not None else []
        DE.f = _ZERO
        DE.origf = _ZERO
        DE.newf = _ZERO
        DE.sincos_args = set()
        DE.dummy = False
        DE.ts = _numbered_symbols('t')
        DE._elements = {}
        DE.fa = fa if fa is not None else TowerPoly.zero(DE.T[-1])
        DE.fd = fd if fd is not None else TowerPoly.one(DE.T[-1])
        DE._auto_attrs()
        return DE

    def _atoms(self, *types: type) -> set[Expr]:
        return {_expr(a) for a in self.newf.atoms(*types)}

    def _rewrite_trig(self) -> None:
        """
        Rewrite the real trigonometric functions of x in terms of tan and
        atan, which are the functions the tower is built from.

        Explanation
        ===========

        sin, cos, sec and csc become rational functions of the tangent of
        the half angle, cot becomes 1/tan, and acot(u) becomes atan(1/u)
        plus an opaque constant that is restored on backsubstitution (the
        difference is locally constant wherever both are defined).
        Functions of constants are left alone.  The arguments of the
        rewritten sines, cosines, secants and cosecants are recorded in
        sincos_args for restore_sincos().
        """
        # xreplace() does not descend into the replacements, so nested
        # functions (e.g. acot(sin(x))) take several passes.
        acots: dict[Expr, Expr] = {}
        while True:
            reps: dict[Basic, Basic] = {}
            for i in self._atoms(Sin, Cos, Sec, Csc, Cot):
                if i.has(self.x):
                    reps[i] = _tan_form(i)
                    if not isinstance(i, Cot):
                        self.sincos_args.add(_arg(i))
            for i in self._atoms(Acot):
                if i.has(self.x):
                    if i not in acots:
                        branch_const = unique_dummy('acot_branch')
                        new = atan(pow(_arg(i), Integer(-1)))
                        self.backsubs.append((branch_const, add(i, mul(Integer(-1), new))))
                        acots[i] = add(new, branch_const)
                    reps[i] = acots[i]
            if not reps:
                return
            self.newf = _xrep(self.newf, reps)

    def _rewrite_exps_pows(self, exps: list[Expr], pows: list[Expr],
            numpows: list[Expr], sympows: list[Expr], log_new_extension: bool
            ) -> tuple[list[Expr], list[Expr], list[Expr], list[Expr], bool]:
        """
        Rewrite exps/pows for better processing.
        """
        from .prde import is_deriv_k

        # Pre-preparsing.
        #################
        # Get all exp arguments, so we can avoid ahead of time doing
        # something like t1 = exp(x), t2 = exp(x/2) == sqrt(t1).

        # Things like sqrt(exp(x)) do not automatically simplify to
        # exp(x/2), so they will be viewed as algebraic.  The easiest way
        # to handle this is to convert all instances of exp(a)**Rational
        # to exp(Rational*a) before doing anything else.  Note that the
        # _exp_part code can generate terms of this form, so we do need to
        # do this at each pass (or else modify it to not do that).

        ratpows_repl: list[tuple[Pow, Expr]] = []
        for i in self._atoms(Pow):
            if not isinstance(i, Pow):
                continue
            base = i.base
            if isinstance(base, Exp) and isinstance(i.exp, Rational):
                ratpows_repl.append((i, exp(mul(i.exp, _arg(base)))))
        # exp(u)**q == exp(q*u) is exact for integer q, but for
        # fractional q the left side is a principal root that differs
        # from exp(q*u) by a locally constant root of unity off the real
        # line.  Radicals of exponentials also arise internally (the
        # radical case of _exp_part substitutes t**(p/n) and restarts);
        # those denote a root the construction is free to choose, so
        # folding them to exp(q*u) is exact, and rewriting the answer
        # back into principal-root notation is what turned correctly
        # integrated results into non-antiderivatives at complex points.
        # A fractional power the user wrote, however, means the
        # principal root specifically: fold it as ratio*exp(q*u) with an
        # opaque locally constant ratio (a root of unity on each
        # component), restored exactly on backsubstitution -- which also
        # keeps integrands mixing exp(u)**q with exp(q*u) itself
        # pointwise correct.
        subs_map: dict[Basic, Basic] = {}
        for i, j in ratpows_repl:
            if isinstance(i.exp, Integer):
                self.backsubs.append((j, i))
                subs_map[i] = j
            elif self.origf.has(i):
                ratio = unique_dummy('exp_branch')
                self.backsubs.append((ratio, mul(i, pow(j, Integer(-1)))))
                subs_map[i] = mul(ratio, j)
            else:
                subs_map[i] = j
        self.newf = _xrep(self.newf, subs_map)

        # To make the process deterministic, the args are sorted
        # so that functions with smaller op-counts are processed first.
        # Ties are broken with the default_sort_key.

        exps = update_sets(exps, self._atoms(Exp),
            lambda i: _is_rational_function(_arg(i), self.T) and _has(_arg(i), self.T))
        pows = update_sets(pows, {p for p in self._atoms(Pow) if isinstance(p, Pow)},
            lambda i: isinstance(i, Pow) and _is_rational_function(i.exp, self.T) and _has(i.exp, self.T))
        numpows = update_sets(numpows, set(pows),
            lambda i: isinstance(i, Pow) and not _has(i.base, self.T))
        sympows = update_sets(sympows, set(pows) - set(numpows),
            lambda i: isinstance(i, Pow) and _is_rational_function(i.base, self.T) and
            not isinstance(i.exp, Integer))

        # The easiest way to deal with non-base E powers is to convert them
        # into base E, integrate, and then convert back.
        for i in [_expr(b) for b in ordered(pows)]:
            if not isinstance(i, Pow):
                continue
            old = i
            new = exp(mul(i.exp, log(i.base)))
            newterm: Expr
            # If exp is ever changed to automatically reduce exp(x*log(2))
            # to 2**x, then this will break.  The solution is to not change
            # exp to do that :)
            if i in sympows:
                if isinstance(i.exp, Rational):
                    raise NotImplementedError("Algebraic extensions are "
                        "not supported (%s)." % str(i))
                # We can add a**b only if log(a) in the extension, because
                # a**b == exp(b*log(a)).
                basea, based = frac_in(i.base, self.t)
                A = is_deriv_k(basea, based, self)
                if A is None:
                    # Nonelementary monomial (so far)

                    # TODO: Would there ever be any benefit from just
                    # adding log(base) as a new monomial?
                    # ANSWER: Yes, otherwise we can't integrate x**x (or
                    # rather prove that it has no elementary integral)
                    # without first manually rewriting it as exp(x*log(x))
                    self.newf = _xrep(self.newf, {old: new})
                    self.backsubs += [(new, old)]
                    log_new_extension = self._log_part([log(i.base)])
                    exps = update_sets(exps, self._atoms(Exp), lambda e:
                        _is_rational_function(_arg(e), self.T) and _has(_arg(e), self.T))
                    continue
                ans, u, const = A
                newterm = exp(mul(i.exp, add(log(const), u)))
                # Under the current implementation, exp kills terms
                # only if they are of the form a*log(x), where a is a
                # Number.  This case should have already been killed by the
                # above tests.  Again, if this changes to kill more than
                # that, this will break, which maybe is a sign that you
                # shouldn't be changing that.  Actually, if anything, this
                # auto-simplification should be removed.  See
                # https://groups.google.com/group/sympy/browse_thread/thread/a61d48235f16867f

                self.newf = _xrep(self.newf, {i: newterm})

            elif i not in numpows:
                continue
            else:
                # i in numpows
                newterm = new
            # TODO: Just put it in self.Tfuncs
            self.backsubs.append((new, old))
            self.newf = _xrep(self.newf, {old: newterm})
            exps.append(newterm)

        return exps, pows, numpows, sympows, log_new_extension

    def _rewrite_logs(self, logs: list[Expr], symlogs: list[Expr]
            ) -> tuple[list[Expr], list[Expr]]:
        """
        Rewrite logs for better processing.
        """
        atoms = self._atoms(Log)
        logs = update_sets(logs, atoms,
            lambda i: _is_rational_function(_arg(i), self.T) and _has(_arg(i), self.T))

        def symbolic_log(i: Expr) -> bool:
            a = _arg(i)
            return (_has(i, self.T) and isinstance(a, Pow) and _is_rational_function(a.base, self.T) and
                    not isinstance(a.exp, Integer))
        symlogs = update_sets(symlogs, atoms, symbolic_log)

        # We can handle things like log(x**y) by converting it to y*log(x)
        # This will fix not only symbolic exponents of the argument, but any
        # non-Integer exponent, like log(sqrt(x)).  The exponent can also
        # depend on x, like log(x**x).
        for i in [_expr(b) for b in ordered(symlogs)]:
            # Unlike in the exponential case above, we do not ever
            # potentially add new monomials (above we had to add log(a)).
            # Therefore, there is no need to run any is_deriv functions
            # here.  Just convert log(a**b) to b*log(a) and let
            # log_new_extension() handle it from there.
            a = _arg(i)
            if not isinstance(a, Pow):
                continue
            lbase = log(a.base)
            logs.append(lbase)
            new = mul(a.exp, lbase)
            self.newf = _xrep(self.newf, {i: new})
            self.backsubs.append((new, i))

        # remove any duplicates
        logs = sorted(set(logs), key=default_sort_key)

        return logs, symlogs

    def _auto_attrs(self) -> None:
        """
        Set attributes that are generated automatically.
        """
        self.cases = [get_case(d, t) for d, t in zip(self.D, self.T)]
        self.level = -1
        self.t = self.T[self.level]
        self.d = self.D[self.level]
        self.case = self.cases[self.level]
        self._elements = {}

    def _darg(self, arga: TowerPoly, argd: TowerPoly) -> RationalFunction:
        """The derivative of ``arga/argd`` as an element."""
        darga = (argd*derivation(arga, self) - arga*derivation(argd, self))
        dargd = argd**2
        return darga.as_element() / dargd.as_element()

    def _exp_part(self, exps: list[Expr]) -> Optional[bool]:
        """
        Try to build an exponential extension.

        Returns
        =======

        Returns True if there was a new extension, False if there was no new
        extension but it was able to rewrite the given exponentials in terms
        of the existing extension, and None if the entire extension building
        process should be restarted.  If the process fails because there is no
        way around an algebraic extension (e.g., exp(log(x)/2)), it will raise
        NotImplementedError.
        """
        from .prde import is_log_deriv_k_t_radical
        new_extension = False
        restart = False
        expargs = [_arg(i) for i in exps]
        ip = integer_powers(expargs)
        for arg, others in ip:
            # Minimize potential problems with algebraic substitution
            others.sort(key=lambda i: _number_key(i[1]))

            arga, argd = frac_in(arg, self.t)
            A = is_log_deriv_k_t_radical(arga, argd, self)

            if A is not None:
                ans, u, n, const = A
                # if n is 1 or -1, it's algebraic, but we can handle it
                if n == -1:
                    # This probably will never happen, because
                    # Rational.as_numer_denom() returns the negative term in
                    # the numerator.  But in case that changes, reduce it to
                    # n == 1.
                    n = 1
                    u = pow(u, Integer(-1))
                    const = mul(Integer(-1), const)
                    ans = [(i, mul(Integer(-1), j)) for i, j in ans]

                if n == 1:
                    # Example: exp(x + x**2) over QQ(x, exp(x), exp(x**2))
                    self.newf = _xrep(self.newf, {exp(arg): mul(exp(const), *[
                        pow(u_, power) for u_, power in ans])})
                    self.newf = _xrep(self.newf, {exp(mul(p, exparg)):
                        mul(exp(mul(const, p)), *[pow(u_, power) for u_, power in ans])
                        for exparg, p in others})
                    # TODO: Add something to backsubs to put exp(const*p)
                    # back together.

                    continue

                else:
                    # Bad news: we have an algebraic radical.  But maybe we
                    # could still avoid it by choosing a different extension.
                    # For example, integer_powers() won't handle exp(x/2 + 1)
                    # over QQ(x, exp(x)), but if we pull out the exp(1), it
                    # will.  Or maybe we have exp(x + x**2/2), over
                    # QQ(x, exp(x), exp(x**2)), which is exp(x)*sqrt(exp(x**2)),
                    # but if we use QQ(x, exp(x), exp(x**2/2)), then they will
                    # all work.
                    #
                    # So here is what we do: If there is a non-zero const, pull
                    # it out and retry.  Also, if len(ans) > 1, then rewrite
                    # exp(arg) as the product of exponentials from ans, and
                    # retry that.  If const == 0 and len(ans) == 1, then we
                    # assume that it would have been handled by either
                    # integer_powers() or n == 1 above if it could be handled,
                    # so we give up at that point.  For example, you can never
                    # handle exp(log(x)/2) because it equals sqrt(x).

                    if const != _ZERO or len(ans) > 1:
                        rad = mul(*[pow(term, mul(power, Rational(1, n))) for term, power in ans])
                        self.newf = _xrep(self.newf, {exp(mul(p, exparg)):
                            mul(exp(mul(const, p)), rad) for exparg, p in others})
                        self.newf = _xrep(self.newf, dict(zip(reversed(self.T),
                            reversed([f(self.x) for f in self.Tfuncs]))))
                        restart = True
                        break
                    else:
                        # TODO: give algebraic dependence in error string
                        raise NotImplementedError("Cannot integrate over "
                            "algebraic extensions.")

            else:
                arga, argd = frac_in(arg, self.t)
                darg = self._darg(arga, argd)
                t_new = next(self.ts)
                # the derivation in the new generator: darg*t
                D = TowerPoly.from_element_strict(darg, t_new) * TowerPoly.monomial(t_new, 1)
                self._push(arg, 'exp', D, t_new, exp)
                self.newf = _xrep(self.newf,
                        {exp(exparg): pow(self.t, p) for exparg, p in others})
                new_extension = True

        if restart:
            return None
        return new_extension

    def _push(self, arg: Expr, ext: Extension, D: TowerPoly, t_new: Symbol, func: Callable[[Expr], Expr]) -> None:
        self.t = t_new
        self.T.append(t_new)
        self.extargs.append(arg)
        self.exts.append(ext)
        self.D.append(D)
        self._elements = {}
        i: Symbol
        if self.dummy:
            i = unique_dummy("i")
        else:
            i = Symbol('i')
        self.Tfuncs += [Lambda(i, func(arg.subs(self.x, i)))]

    def _log_part(self, logs: list[Expr]) -> bool:
        """
        Try to build a logarithmic extension.

        Returns
        =======

        Returns True if there was a new extension and False if there was no new
        extension but it was able to rewrite the given logarithms in terms
        of the existing extension.  Unlike with exponential extensions, there
        is no way that a logarithm is not transcendental over and cannot be
        rewritten in terms of an already existing extension in a non-algebraic
        way, so this function does not ever return None or raise
        NotImplementedError.
        """
        from .prde import is_deriv_k
        new_extension = False
        logargs = [_arg(i) for i in logs]
        for arg in [_expr(b) for b in ordered(logargs)]:
            # The log case is easier, because whenever a logarithm is algebraic
            # over the base field, it is of the form a1*t1 + ... an*tn + c,
            # which is a polynomial, so we can just replace it with that.
            # In other words, we don't have to worry about radicals.
            arga, argd = frac_in(arg, self.t)
            A = is_deriv_k(arga, argd, self)
            if A is not None:
                ans, u, const = A
                # u + log(const) equals log(arg) only up to a locally
                # constant branch term: log(const) is the principal-branch
                # choice, valid where every argument involved is positive
                # (e.g. log(-x**2) == 2*log(x) + I*pi only holds for
                # x > 0).  The one case where it is exact everywhere is a
                # single tower logarithm with coefficient one and a
                # positive constant: log(c*w) == log(c) + log(w) for
                # c > 0 on the whole complex plane.
                if (isinstance(u, Symbol) and u != self.x and u in self.T
                        and const.is_positive is True and
                        self.exts[self.T.index(u) - 1] == 'log'):
                    self.newf = _xrep(self.newf, {log(arg): add(log(const), u)})
                    continue
                # Otherwise the difference log(arg) - u is locally
                # constant wherever the functions are defined, so
                # integrating with an opaque constant in its place and
                # restoring the exact difference on backsubstitution keeps
                # the answer, now expressed through the original
                # logarithm, correct on every connected component.
                branch_const = unique_dummy('log_branch')
                # arg and u may involve tower symbols; express the
                # difference through the concrete functions (and through
                # any originals recorded in backsubs so far, so that
                # e.g. x**x reappears as x**x rather than exp(x*log(x))).
                concrete = list(zip(reversed(self.T),
                    reversed([f(self.x) for f in self.Tfuncs])))
                diff_expr = _subs(add(log(_subs(arg, concrete)),
                    mul(Integer(-1), _subs(u, concrete))), self.backsubs)
                self.backsubs.append((branch_const, diff_expr))
                self.newf = _xrep(self.newf, {log(arg): add(branch_const, u)})
                continue

            else:
                arga, argd = frac_in(arg, self.t)
                darg = self._darg(arga, argd)
                t_new = next(self.ts)
                D = TowerPoly.from_element_strict(darg / _element(arg), t_new)
                self._push(arg, 'log', D, t_new, log)
                self.newf = _xrep(self.newf, {log(arg): self.t})
                new_extension = True

        return new_extension

    def _tan_part(self, tans: list[Expr]) -> bool:
        """
        Try to build a hypertangent extension.

        Returns
        =======

        Returns True if there was a new extension and False if there was no
        new extension but it was able to rewrite the given tangents in terms
        of the existing extension.  If a tangent is algebraic of degree
        greater than one over the existing extension (e.g. tan(atan(x)/2)
        over QQ(x, atan(x))), it raises NotImplementedError.
        """
        from .prde import is_log_deriv_k_t_radical_tan
        new_extension = False
        # tan(n*g + c) is a rational function of tan(g) for any integer n
        # and constant c (multiple-angle and addition formulas), so the
        # arguments are split into a constant part and a part depending on
        # the tower, and the latter are grouped as integer multiples of a
        # common base with integer_powers(), as for exponentials.  Note
        # that, unlike the exponential case, the rewriting through tan(g)
        # is an exact identity wherever the functions are defined, so no
        # branch constants are needed.
        split: dict[Expr, list[tuple[Expr, Expr]]] = {}
        for ti in tans:
            c, g = _arg(ti).as_independent(*self.T, as_Add=True)
            split.setdefault(g, []).append((ti, c))
        for g, others in integer_powers(list(split)):
            ga, gd = frac_in(g, self.t)
            A = is_log_deriv_k_t_radical_tan(ga, gd, self)
            tang: Expr
            if A is None:
                darg = self._darg(ga, gd)
                t_new = next(self.ts)
                D = TowerPoly.from_element_strict(darg, t_new) * \
                    TowerPoly.from_list([1, 0, 1], t_new)
                self._push(g, 'tan', D, t_new, tan)
                tang = self.t
                new_extension = True
            else:
                ans, u, n, const = A
                if n != 1:
                    raise NotImplementedError("Cannot integrate over "
                        "algebraic extensions (tan(%s) is algebraic of "
                        "degree %s over the tower)." % (g, n))
                # tan(g) == tan(u + const) where u == Sum(ri*termi) with
                # integer ri; each termi is either the argument of a
                # tangent generator (tan(termi) is that generator) or an
                # arc-tangent generator (tan(termi) is its argument).
                tanterms: list[tuple[Expr, Expr]] = []
                for term, r in ans:
                    if isinstance(term, Symbol) and term in self.T:
                        found = self.extargs[self.T.index(term) - 1]
                        if found is None:
                            raise NotImplementedError("a tangent of a generator without argument")
                        tanterms.append((found, r))
                    else:
                        tanterms.append((self.T[self.extargs.index(term) + 1], r))
                tang = self._tan_combination(tanterms, const)
            for gi, p in others:
                for tanatom, c in split[gi]:
                    self.newf = _xrep(self.newf,
                        {tanatom: self._tan_combination([(tang, p)], c)})

        return new_extension

    @staticmethod
    def _tan_combination(tanterms: list[tuple[Expr, Expr]], c: Expr) -> Expr:
        """
        tan(Sum(ri*ai) + c) as a rational function of the tan(ai), which
        are given as the list of tuples (tan(ai), ri) with integer ri.
        """
        ys = [unique_dummy('y') for _ in tanterms]
        cd = unique_dummy('c')
        arg = add(*[mul(r, y) for (_, r), y in zip(tanterms, ys)])
        if c != _ZERO:
            arg = add(arg, cd)
        new = expand_trig(tan(arg))
        reps: dict[Basic, Basic] = {tan(y): i for (i, _), y in zip(tanterms, ys)}
        reps[tan(cd)] = tan(c)
        return _xrep(new, reps)

    def _atan_part(self, atans: list[Expr]) -> bool:
        """
        Try to build an arc-tangent extension.

        Returns
        =======

        Returns True if there was a new extension and False if there was no
        new extension but it was able to rewrite the given arc-tangents in
        terms of the existing extension.  Like logarithms, an arc-tangent
        that is not transcendental over the existing extension is in it up
        to an additive constant, so this function never raises
        NotImplementedError.
        """
        from .prde import is_deriv_k_atan
        new_extension = False
        for arg in [_expr(b) for b in ordered({_arg(i) for i in atans})]:
            arga, argd = frac_in(arg, self.t)
            A = is_deriv_k_atan(arga, argd, self)
            if A is not None:
                ans, u = A
                # atan(arg) - u is locally constant wherever the functions
                # are defined, but only piecewise constant as a function
                # (e.g. atan(2*x/(1 - x**2)) - 2*atan(x) jumps at x == 1),
                # so, as for logarithms in _log_part(), integrate with an
                # opaque constant in its place and restore the exact
                # difference on backsubstitution.
                branch_const = unique_dummy('atan_branch')
                concrete = list(zip(reversed(self.T),
                    reversed([f(self.x) for f in self.Tfuncs])))
                diff_expr = _subs(add(atan(_subs(arg, concrete)),
                    mul(Integer(-1), _subs(u, concrete))), self.backsubs)
                self.backsubs.append((branch_const, diff_expr))
                self.newf = _xrep(self.newf, {atan(arg): add(branch_const, u)})
                continue

            darg = self._darg(arga, argd)
            t_new = next(self.ts)
            D = TowerPoly.from_element_strict(darg / (_element(arg)**2 + 1), t_new)
            self._push(arg, 'atan', D, t_new, atan)
            self.newf = _xrep(self.newf, {atan(arg): self.t})
            new_extension = True

        return new_extension

    @property
    def _important_attrs(self) -> tuple[TowerPoly, TowerPoly, list[TowerPoly], list[Symbol],
            list[TFunc], Substitutions, list[Optional[Extension]], list[Optional[Expr]]]:
        """
        Returns some of the more important attributes of self.

        Explanation
        ===========

        Used for testing and debugging purposes.

        The attributes are (fa, fd, D, T, Tfuncs, backsubs,
        exts, extargs).
        """
        return (self.fa, self.fd, self.D, self.T, self.Tfuncs,
            self.backsubs, self.exts, self.extargs)

    def __repr__(self) -> str:
        return self.__class__.__name__ + '(fa=%s, fd=%s, D=%s, T=%s, exts=%s, extargs=%s, level=%s)' % (
            self.fa, self.fd, self.D, self.T, self.exts, self.extargs, self.level)

    # fancy printing of DifferentialExtension object
    def __str__(self) -> str:
        return (self.__class__.__name__ + '({fa=%s, fd=%s, D=%s})' %
                (self.fa, self.fd, self.D))

    def reset(self) -> None:
        """
        Reset self to an initial state.  Used by __init__.
        """
        self.t = self.x
        self.T = [self.x]
        self.D = [TowerPoly.one(self.x)]
        self.level = -1
        self.exts = []
        self.extargs = []
        if self.dummy:
            self.ts = _numbered_dummies('t')
        else:
            # For testing
            self.ts = _numbered_symbols('t')
        # For various things that we change to make things work that we need to
        # change back when we are done.
        self.backsubs = []
        self.Tfuncs = []
        self.newf = self.f
        self._elements = {}

    def restore_sincos(self, expr: Expr, drop_constants: bool = False) -> Expr:
        """
        Rewrite the half-angle tangents that _rewrite_trig() introduced
        back through the sines and cosines of the original angles.

        Explanation
        ===========

        For each hypertangent generator tan(g) of the tower that was not
        written by the user as tan(g) or cot(g), but whose double angle 2*g
        is (up to a constant shift) the argument of a sine, cosine, secant
        or cosecant that _rewrite_trig() rewrote, the rational functions
        of tan(g) in ``expr`` are rewritten as rational functions of
        sin(2*g) and cos(2*g) by _half_angle_to_sincos(), which is exact.
        If ``drop_constants`` is True, ``expr`` is an antiderivative and
        additive terms that do not depend on x are dropped from it.

        ``expr`` must already be back-substituted (functions of x, not
        the tower variables).  Generators are processed from the bottom of
        the tower up, so that the argument of a nested generator (e.g.
        tan(sin(x)/2) for sin(sin(x))) has the same form as in ``expr``
        when its turn comes.  A double angle that the rewriting does not
        reproduce syntactically is not recognized, and its generator is
        left as a tangent.
        """
        if not self.sincos_args:
            return expr
        s = list(zip(reversed(self.T), reversed([f(self.x) for f in self.Tfuncs])))
        usertans = {_expr(a) for a in self.origf.atoms(Tan)} | {tan(_arg(i))
            for i in self.origf.atoms(Cot)}
        done: list[tuple[Expr, Expr]] = []
        for i in self.indices('tan'):
            extarg = self.extargs[i - 1]
            if extarg is None:
                continue
            g = _subs(extarg, s)
            for tang, theta in done:
                g = _half_angle_to_sincos(g, tang, theta)
            tang = tan(g)
            theta = mul(Integer(2), g)
            if tang in usertans or not any(
                    not expand(add(u, mul(Integer(-1), theta))).has(self.x)
                    for u in self.sincos_args):
                continue
            expr = _half_angle_to_sincos(expr, tang, theta,
                self.x if drop_constants else None)
            done.append((tang, theta))
        if drop_constants:
            expr = add(*[i for i in Add.make_args(expr) if i.has(self.x)])
        return expr

    def indices(self, extension: Extension) -> list[int]:
        """
        Parameters
        ==========

        extension : str
            Represents a valid extension type.

        Returns
        =======

        list: A list of indices into T (and D) of the extensions of
            type 'extension'.  Note that self.exts[i] describes the
            extension T[i + 1], since T[0] == x is not an extension.

        Examples
        ========

        >>> from sympy_extras.core import symbols
        >>> from sympy_extras.functions import exp, log
        >>> from sympy_extras.integrals.risch.risch import DifferentialExtension
        >>> x = symbols('x')
        >>> DE = DifferentialExtension(log(x) + exp(x), x, handle_first='exp')
        >>> DE.indices('log')
        [2]
        >>> DE.indices('exp')
        [1]

        """
        return [i for i, ext in enumerate(self.exts, 1) if ext == extension]

    def increment_level(self) -> None:
        """
        Increment the level of self.

        Explanation
        ===========

        This makes the working differential extension larger.  self.level is
        given relative to the end of the list (-1, -2, etc.), so we do not need
        do worry about it when building the extension.
        """
        if self.level >= -1:
            raise ValueError("The level of the differential extension cannot "
                "be incremented any further.")

        self.level += 1
        self.t = self.T[self.level]
        self.d = self.D[self.level]
        self.case = self.cases[self.level]

    def decrement_level(self) -> None:
        """
        Decrease the level of self.

        Explanation
        ===========

        This makes the working differential extension smaller.  self.level is
        given relative to the end of the list (-1, -2, etc.), so we do not need
        do worry about it when building the extension.
        """
        if self.level <= -len(self.T):
            raise ValueError("The level of the differential extension cannot "
                "be decremented any further.")

        self.level -= 1
        self.t = self.T[self.level]
        self.d = self.D[self.level]
        self.case = self.cases[self.level]

    def derivation_of(self, i: int) -> RationalFunction:
        """The derivative of ``T[i]`` as an element of the field."""
        found = self._elements.get(i)
        if found is None:
            found = self.D[i].as_element()
            self._elements[i] = found
        return found

    def backsubstitutions(self) -> Substitutions:
        """The pairs ``(T[i], Tfuncs[i - 1](x))`` from the top of the tower
        down."""
        return list(zip(reversed(self.T), reversed([f(self.x) for f in self.Tfuncs])))


def _symbol(e: Expr) -> Symbol:
    if not isinstance(e, Symbol):
        raise TypeError('a symbol is expected, got %s' % (e,))
    return e


def _number_key(e: Expr) -> Fraction:
    if isinstance(e, Rational):
        return Fraction(e.p, e.q)
    return Fraction(0)


def update_sets(seq: Iterable[Expr], atoms: set[Expr], func: Callable[[Expr], bool]) -> list[Expr]:
    s = set(seq)
    s = atoms.intersection(s)
    new = atoms - s
    s.update(list(filter(func, new)))
    return list(s)


def _cancel_expr(e: Expr) -> Expr:
    """SymPy's ``cancel`` of the integrand: the canonical fraction of its
    kernels, or the native ``cancel`` when the field does not represent it."""
    try:
        return _element(e).as_expr()
    except UnsupportedConstant:
        from sympy_extras.polys.polytools import cancel
        return cancel(e)


class DecrementLevel:
    """
    A context manager for decrementing the level of a DifferentialExtension.
    """
    __slots__ = ('DE',)

    def __init__(self, DE: DifferentialExtension) -> None:
        self.DE: Final[DifferentialExtension] = DE

    def __enter__(self) -> None:
        self.DE.decrement_level()

    def __exit__(self, exc_type: Optional[type[BaseException]],
            exc_value: Optional[BaseException],
            traceback: Optional[TracebackType]) -> None:
        self.DE.increment_level()


class NonElementaryIntegralException(Exception):
    """
    Exception used by subroutines within the Risch algorithm to indicate to one
    another that the function being integrated does not have an elementary
    integral in the given differential field.
    """


def gcdex_diophantine(a: TowerPoly, b: TowerPoly, c: TowerPoly) -> tuple[TowerPoly, TowerPoly]:
    """
    Extended Euclidean Algorithm, Diophantine version.

    Explanation
    ===========

    Given ``a``, ``b`` in K[x] and ``c`` in (a, b), the ideal generated by ``a`` and
    ``b``, return (s, t) such that s*a + t*b == c and either s == 0 or s.degree()
    < b.degree().

    This is the Diophantine version of ``ExtendedEuclidean`` from Section
    1.3 of Bronstein's book.

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.integrals.risch.risch import gcdex_diophantine
    >>> from sympy_extras.integrals.risch.tower import TowerPoly
    >>> x = symbols('x')
    >>> P = lambda e: TowerPoly.from_expr(e, x)
    >>> gcdex_diophantine(P(x**3 + 6*x + 7), P(x**2 + 3*x + 2), P(x + 1))
    (TowerPoly(1/13, x), TowerPoly(3/13 - x/13, x))
    """
    s, g = a.half_gcdex(b)
    s = s * c.exquo(g)  # Inexact division means c is not in (a, b)
    if s and s.degree() >= b.degree():
        _, s = s.div(b)
    t = (c - s*a).exquo(b)
    return (s, t)


def frac_in(f: FractionLike, t: Expr, *, cancel: bool = False) -> tuple[TowerPoly, TowerPoly]:
    """
    Returns the tuple (fa, fd), where fa and fd are TowerPolys in t.

    Explanation
    ===========

    This is a common idiom in the Risch Algorithm functions, so we abstract
    it out here. ``f`` should be a basic expression, an element of the
    field, a TowerPoly, or a tuple (fa, fd), where fa and fd are TowerPolys,
    and f == fa/fd. The fraction is in lowest terms; with ``cancel`` the
    denominator is monic.

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.integrals.risch.risch import frac_in
    >>> x, t = symbols('x t')
    >>> frac_in((x + 1)/x*t, x)
    (TowerPoly(t*x + t, x), TowerPoly(x, x))
    """
    g: RationalFunction
    if isinstance(f, tuple):
        fa, fd = f
        g = fa.as_element() / fd.as_element()
    elif isinstance(f, TowerPoly):
        g = f.as_element()
    elif isinstance(f, RationalFunction):
        g = f
    else:
        try:
            g = _element(f)
        except UnsupportedConstant:
            raise
        if any(v != t and v.has(t) for v in g.gens):
            raise ValueError("Could not turn %s into a fraction in %s." % (f, t))
    fa_poly = TowerPoly.from_element(g.numerator(), t)
    fd_poly = TowerPoly.from_element(g.denominator(), t)
    if fa_poly is None or fd_poly is None:
        raise ValueError("Could not turn %s into a fraction in %s." % (f, t))
    if cancel:
        fa_poly, fd_poly = fa_poly.cancel(fd_poly, include=True)
    return (fa_poly, fd_poly)


class LaurentPolynomial:
    """An element ``sum c_i t**i`` of ``k[t, 1/t]``: the coefficients by
    their exponent (SymPy's ``Poly`` in ``t`` and ``z = 1/t`` of
    :func:`as_poly_1t`)."""

    __slots__ = ('coefficients', 'gen')

    def __init__(self, coefficients: dict[int, RationalFunction], gen: Expr) -> None:
        self.coefficients: Final[dict[int, RationalFunction]] = {i: c for i, c in coefficients.items() if c}
        self.gen: Final[Expr] = gen

    def nth(self, i: int) -> RationalFunction:
        """The coefficient of ``t**i``."""
        return self.coefficients.get(i, RationalFunction.from_int(0))

    @property
    def is_zero(self) -> bool:
        return not self.coefficients

    def degree(self) -> int:
        """The highest exponent (``0`` for zero)."""
        return max(self.coefficients, default=0)

    def low_degree(self) -> int:
        """The lowest exponent (``0`` for zero)."""
        return min(self.coefficients, default=0)

    def as_element(self) -> RationalFunction:
        t = RationalFunction.generator(self.gen)
        total = RationalFunction.from_int(0)
        for i, c in self.coefficients.items():
            total = total + c * t**i
        return total

    def as_expr(self) -> Expr:
        return self.as_element().as_expr()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LaurentPolynomial):
            return NotImplemented
        return self.gen == other.gen and self.coefficients == other.coefficients

    def __hash__(self) -> int:
        return hash((self.gen, tuple(sorted(self.coefficients.items(), key=lambda i: i[0]))))

    def __repr__(self) -> str:
        return 'LaurentPolynomial(%s, %s)' % (self.as_expr(), self.gen)


def as_poly_1t(p: Union[Expr, RationalFunction, TowerPoly], t: Expr, z: Optional[Expr] = None) -> LaurentPolynomial:
    """
    The element ``p`` of K[t, 1/t] as a Laurent polynomial (SymPy's
    ``Poly`` in ``t`` and ``z == 1/t``); ``PolynomialError`` when ``p`` is
    not in K[t, 1/t].

    See issue 5131.

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.integrals.risch.risch import as_poly_1t
    >>> x, t = symbols('x t')
    >>> as_poly_1t(2/t + 3/t**2 + x*t, t).coefficients
    {1: x, -1: 2, -2: 3}
    """
    pa, pd = frac_in(p, t, cancel=True)
    if not pd.is_monomial:
        raise PolynomialError("%s is not an element of K[%s, 1/%s]." % (p, t, t))
    k = pd.degree()
    lc = pd.LC().invert()
    coefficients: dict[int, RationalFunction] = {}
    for i, c in pa.terms():
        coefficients[i - k] = c * lc
    return LaurentPolynomial(coefficients, t)


def derivation_element(a: RationalFunction, DE: DifferentialExtension, coefficientD: bool = False) -> RationalFunction:
    """The derivation ``Da`` of an element of the field: ``sum D(v) da/dv``
    over the generators ``v`` of the tower up to the current level (below it
    with ``coefficientD``, Definition 3.2.2 of Bronstein's book).

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.integrals.risch.risch import DifferentialExtension, derivation_element
    >>> from sympy_extras.integrals.risch.tower import TowerPoly, RationalFunction
    >>> x, t = symbols('x t')
    >>> DE = DifferentialExtension.manual([TowerPoly.from_expr(1, x), TowerPoly.from_expr(1/x, t)])
    >>> derivation_element(RationalFunction.from_expr(t**2/x), DE)
    (-t**2 + 2*t)/x**2
    """
    top = len(DE.T) + DE.level + 1
    if coefficientD:
        top -= 1
    result = RationalFunction.from_int(0)
    for i in range(top):
        v = DE.T[i]
        if not a.has(v):
            continue
        result = result + DE.derivation_of(i) * a.diff(v)
    return result


def derivation(p: TowerPoly, DE: DifferentialExtension, coefficientD: bool = False) -> TowerPoly:
    """
    Computes Dp.

    Explanation
    ===========

    Given the derivation D with D = d/dx and p is a polynomial in t over
    K(x), return Dp.

    If coefficientD is True, it computes the derivation kD
    (kappaD), which is defined as kD(sum(ai*Xi**i, (i, 0, n))) ==
    sum(Dai*Xi**i, (i, 1, n)) (Definition 3.2.2).  X in this case is
    T[-1], so coefficientD computes the derivative just with respect to T[:-1],
    with T[-1] treated as a constant.

    The result is a polynomial in the generator of ``p`` (which needs not be
    a generator of the tower: the coefficients are differentiated). For
    the derivation of an element of the field, see
    :func:`derivation_element` (SymPy's ``basic=True``).

    See Definition 3.2.2 in Section 3.2 of Bronstein's book.

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.integrals.risch.risch import DifferentialExtension, derivation
    >>> from sympy_extras.integrals.risch.tower import TowerPoly
    >>> x, t = symbols('x t')
    >>> DE = DifferentialExtension.manual([TowerPoly.from_expr(1, x), TowerPoly.from_expr(t, t)])
    >>> derivation(TowerPoly.from_expr(x*t**2 + 1, t), DE)
    TowerPoly(t**2*(2*x + 1), t)
    """
    if coefficientD and DE.level <= -len(DE.T):
        # 'base' case, the answer is 0.
        return TowerPoly.zero(p.gen)
    top = len(DE.T) + DE.level + 1
    if coefficientD:
        top -= 1
    gen = p.gen
    rep = [derivation_element(c, DE, coefficientD) if c else c for c in p.rep]
    result = TowerPoly(rep, gen)
    if gen in DE.T[:top] and p.degree() > 0:
        i = DE.T.index(_symbol(gen))
        D = DE.D[i]
        if D.gen != gen:
            D = D.as_poly_strict(gen)
        result = result + p.diff() * D
    return result


def get_case(d: TowerPoly, t: Expr) -> Case:
    """
    Returns the type of the derivation d.

    Returns one of {'exp', 'tan', 'base', 'primitive', 'other_linear',
    'other_nonlinear'}.

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.integrals.risch.risch import get_case
    >>> from sympy_extras.integrals.risch.tower import TowerPoly
    >>> x, t = symbols('x t')
    >>> get_case(TowerPoly.from_expr(t*x, t), t), get_case(TowerPoly.from_expr(t**2 + 1, t), t)
    ('exp', 'tan')
    """
    if not d.has(t):
        if d.is_one:
            return 'base'
        return 'primitive'
    if d.gen != t:
        d = d.as_poly_strict(t)
    if d.rem(TowerPoly.monomial(t, 1)).is_zero:
        return 'exp'
    if d.rem(TowerPoly.from_list([1, 0, 1], t)).is_zero:
        return 'tan'
    if d.degree(t) > 1:
        return 'other_nonlinear'
    return 'other_linear'


def _in_t(p: TowerPoly, DE: DifferentialExtension) -> TowerPoly:
    return p if p.gen == DE.t else p.as_poly_strict(DE.t)


def splitfactor(p: TowerPoly, DE: DifferentialExtension, coefficientD: bool = False,
        z: Optional[Symbol] = None) -> tuple[TowerPoly, TowerPoly]:
    """
    Splitting factorization.

    Explanation
    ===========

    Given a derivation D on k[t] and ``p`` in k[t], return (p_n, p_s) in
    k[t] x k[t] such that p = p_n*p_s, p_s is special, and each square
    factor of p_n is normal.

    This is ``SplitFactor`` from Section 3.5 of Bronstein's book.

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.integrals.risch.risch import DifferentialExtension, splitfactor
    >>> from sympy_extras.integrals.risch.tower import TowerPoly
    >>> x, t = symbols('x t')
    >>> DE = DifferentialExtension.manual([TowerPoly.from_expr(1, x), TowerPoly.from_expr(-t**2 - 3/(2*x)*t + 1/(2*x), t)])
    >>> splitfactor(TowerPoly.from_expr(4*x**4*t**5 + (-4*x**3 - 4*x**4)*t**4 + (-3*x**2 + 2*x**3)*t**3 + (2*x + 7*x**2 + 2*x**3)*t**2 + (1 - 4*x - 4*x**2)*t - 1 + 2*x, t), DE)[1]
    TowerPoly(t**2 + t/x + (1 - 2*x)/(4*x**2), t)
    """
    if p.gen not in DE.T and not p.is_ground:
        # a polynomial in the variable z of a resultant: split in z
        gen = p.gen
    else:
        p = _in_t(p, DE)
        gen = DE.t
    One = TowerPoly.one(gen)
    if p.is_zero:
        return (p, One)

    if p.degree() <= 0:
        # p is a unit
        return (p, One)

    Dp = derivation(p, DE, coefficientD=coefficientD)
    if not Dp.is_zero:
        h = p.gcd(Dp)
        g = p.gcd(p.diff())
        s = h.exquo(g)

        if s.degree() == 0:
            return (p, One)

        q_split = splitfactor(p.exquo(s), DE, coefficientD=coefficientD)

        return (q_split[0], q_split[1]*s)
    else:
        return (p, One)


def splitfactor_sqf(p: TowerPoly, DE: DifferentialExtension, coefficientD: bool = False,
        z: Optional[Symbol] = None) -> tuple[tuple[tuple[TowerPoly, int], ...], tuple[tuple[TowerPoly, int], ...]]:
    """
    Splitting Square-free Factorization.

    Explanation
    ===========

    Given a derivation D on k[t] and ``p`` in k[t], returns (N1, ..., Nm)
    and (S1, ..., Sm) in k[t]^m such that p =
    (N1*N2**2*...*Nm**m)*(S1*S2**2*...*Sm**m) is a splitting
    factorization of ``p`` and the Ni and Si are square-free and coprime.

    The polynomial is in its own generator (``t``, or the variable ``z``
    of the resultants of :func:`residue_reduce`); the derivation is applied
    to its coefficients (and to the generator when it is ``t``).

    This is ``SplitSquarefreeFactor`` from Section 3.5 of Bronstein's book.
    """
    special: list[tuple[TowerPoly, int]] = []
    normal: list[tuple[TowerPoly, int]] = []
    if p.is_zero:
        return (((p, 1),), ())
    p_sqf = p.sqf_list_include()

    for pi, i in p_sqf:
        Si = pi.gcd(derivation(pi, DE, coefficientD=coefficientD))
        Ni = pi.exquo(Si)
        if not Si.is_one:
            special.append((Si, i))
        if not Ni.is_one:
            normal.append((Ni, i))

    return (tuple(normal), tuple(special))


def canonical_representation(a: TowerPoly, d: TowerPoly, DE: DifferentialExtension
        ) -> tuple[TowerPoly, tuple[TowerPoly, TowerPoly], tuple[TowerPoly, TowerPoly]]:
    """
    Canonical Representation.

    Explanation
    ===========

    Given a derivation D on k[t] and f = a/d in k(t), return (f_p, f_s,
    f_n) in k[t] x k(t) x k(t) such that f = f_p + f_s + f_n is the
    canonical representation of f (f_p is a polynomial, f_s is reduced
    (has a special denominator), and f_n is simple (has a normal
    denominator).

    This is ``CanonicalRepresentation`` from Section 3.5 of Bronstein's
    book.
    """
    # Make d monic
    l = d.LC().invert()
    a, d = a.mul_ground(l), d.mul_ground(l)

    q, r = a.div(d)
    dn, ds = splitfactor(d, DE)

    b, c = gcdex_diophantine(_in_t(dn, DE), _in_t(ds, DE), _in_t(r, DE))

    return (q, (b, ds), (c, dn))


def hermite_reduce(a: TowerPoly, d: TowerPoly, DE: DifferentialExtension
        ) -> tuple[tuple[TowerPoly, TowerPoly], tuple[TowerPoly, TowerPoly], tuple[TowerPoly, TowerPoly]]:
    """
    Hermite Reduction - Mack's Linear Version.

    Given a derivation D on k(t) and f = a/d in k(t), returns g, h, r in
    k(t) such that f = Dg + h + r, h is simple, and r is reduced.

    This is ``HermiteReduce`` (Mack's linear version) from Section 5.3 of
    Bronstein's book.
    """
    a, d = _in_t(a, DE), _in_t(d, DE)
    # Make d monic
    l = d.LC().invert()
    a, d = a.mul_ground(l), d.mul_ground(l)

    fp, fs, fn = canonical_representation(a, d, DE)
    a, d = fn
    l = d.LC().invert()
    a, d = a.mul_ground(l), d.mul_ground(l)

    ga = TowerPoly.zero(DE.t)
    gd = TowerPoly.one(DE.t)

    dd = derivation(d, DE)
    dm = d.gcd(dd)
    ds, _ = d.div(dm)

    while dm.degree(DE.t) > 0:

        ddm = derivation(dm, DE)
        dm2 = dm.gcd(ddm)
        dms, _ = dm.div(dm2)
        ds_ddm = ds.mul(ddm)
        ds_ddm_dm, _ = ds_ddm.div(dm)

        b, c = gcdex_diophantine(-ds_ddm_dm, dms, a)

        db = derivation(b, DE)
        ds_dms, _ = ds.div(dms)
        a = c - db.mul(ds_dms)

        ga = ga*dm + b*gd
        gd = gd*dm
        ga, gd = ga.cancel(gd, include=True)
        dm = dm2

    q, r = a.div(ds)
    ga, gd = ga.cancel(gd, include=True)

    r, d = r.cancel(ds, include=True)
    rra = q*fs[1] + fp*fs[1] + fs[0]
    rrd = fs[1]
    rra, rrd = rra.cancel(rrd, include=True)

    return ((ga, gd), (r, d), (rra, rrd))


def polynomial_reduce(p: TowerPoly, DE: DifferentialExtension) -> tuple[TowerPoly, TowerPoly]:
    """
    Polynomial Reduction.

    Explanation
    ===========

    Given a derivation D on k(t) and p in k[t] where t is a nonlinear
    monomial over k, return q, r in k[t] such that p = Dq  + r, and
    deg(r) < deg_t(Dt).

    This is ``PolynomialReduce`` from Section 5.4 of Bronstein's book.
    """
    p = _in_t(p, DE)
    q = TowerPoly.zero(DE.t)
    while p.degree(DE.t) >= DE.d.degree(DE.t):
        m = p.degree(DE.t) - DE.d.degree(DE.t) + 1
        q0 = TowerPoly.monomial(DE.t, m, p.LC() / (DE.d.LC() * m))
        q += q0
        p = p - derivation(q0, DE)

    return (q, p)


def _unique_symbols(name: str, n: int) -> list[Symbol]:
    return [unique_dummy('%s%d' % (name, i)) for i in range(n)]


def laurent_series(a: TowerPoly, d: TowerPoly, F: TowerPoly, n: int, DE: DifferentialExtension
        ) -> tuple[TowerPoly, TowerPoly, list[TowerPoly]]:
    """
    Contribution of ``F`` to the full partial fraction decomposition of A/D.

    Explanation
    ===========

    Given a field K of characteristic 0 and ``A``,``D``,``F`` in K[x] with D monic,
    nonzero, coprime with A, and ``F`` the factor of multiplicity n in the square-
    free factorization of D, return the principal parts of the Laurent series of
    A/D at all the zeros of ``F``.

    Returns
    =======

    (delta_a, delta_d, H) : tuple
        ``delta_a`` and ``delta_d`` are TowerPolys in ``DE.t`` such that
        ``delta_a/delta_d`` is the sum of the principal parts of the
        Laurent series of A/D at the real zeros of ``F``.  ``H`` is the
        list of TowerPolys ``[H_1, ..., H_n]`` from Theorem 2.7.1: at each
        zero ``alpha`` of ``F``, ``H_j(alpha)`` is the coefficient of
        ``1/(t - alpha)**(n - j + 1)`` in the principal part of A/D at
        ``alpha``; in particular, ``H[-1]`` evaluated at ``alpha`` is the
        residue of A/D at ``alpha``.  If ``F`` has degree 0 (and so has
        no zeros), ``(TowerPoly(0, DE.t), TowerPoly(1, DE.t), [])`` is returned.

    This is ``LaurentSeries`` from Section 2.7 of Bronstein's book.  The
    book's version returns only the sum of the principal parts (so the
    degree 0 case returns the expression 0); ``H`` is additionally
    returned here for use by ``recognize_derivative()``.  ``F`` may not
    contain special irreducible factors (p dividing Dp): the
    construction needs D(F) invertible mod F, so such an ``F`` raises
    ``NotImplementedError``.
    """
    a, d, F = _in_t(a, DE), _in_t(d, DE), _in_t(F, DE)
    if F.degree() == 0:
        return (TowerPoly.zero(DE.t), TowerPoly.one(DE.t), [])
    Z: list[Symbol] = _unique_symbols('z', n)
    z = unique_dummy('z')
    Z.insert(0, z)
    zf = RationalFunction.generator(z)
    delta_a = TowerPoly.zero(DE.t)
    delta_d = TowerPoly.one(DE.t)

    E = d.quo(F**n)
    ha, hd = (a, E.mul_ground(zf**n))
    dF = derivation(F, DE)
    if F.gcd(dF).degree(DE.t) > 0:
        # F has special irreducible factors (p divides Dp).  The
        # construction below needs D(F) invertible mod F (the analogue
        # of gcd(F, F') == 1 for squarefree F in K[x]), so the
        # principal parts at special primes cannot be computed this
        # way.
        raise NotImplementedError("laurent_series() cannot compute the "
            "principal parts at special primes (gcd(F, DF) != 1).")
    One = TowerPoly.one(DE.t)
    B, _ = gcdex_diophantine(E, F, One)
    C, _ = gcdex_diophantine(dF, F, One)

    # initialization
    F_store = F
    V: list[RationalFunction] = []
    DE_D_list: list[TowerPoly] = []
    H_list: list[TowerPoly] = []

    for j in range(0, n):
    # jth derivative of z would be substituted with dfnth/(j+1) where dfnth =(d^n)f/(dx)^n
        F_store = derivation(F_store, DE)
        v = F_store.as_element() / (j + 1)
        V.append(v)
        DE_D_list.append(TowerPoly.ground(RationalFunction.generator(Z[j + 1]), Z[j]))

    DE_new = DifferentialExtension.manual(DE_D_list)  # a differential indeterminate
    for j in range(0, n):
        zEha = (E**(j + 1)*ha).mul_ground(zf**(n + j))
        zEhd = hd
        Pa, Pd = zEha.cancel(zEhd, include=True)
        Q = Pa.quo(Pd)
        for i in range(0, j + 1):
            Q = Q.subs(Z[i], V[i])
        # ha/hd == h**(j)/j!; differentiate once more (in both x and the
        # differential indeterminate z) and divide by j + 1, using the
        # quotient rule: D(ha/hd) == (hd*D(ha) - ha*D(hd))/hd**2.
        Dha = (hd*derivation(ha, DE) - ha*derivation(hd, DE)
             + hd*derivation(ha, DE_new) - ha*derivation(hd, DE_new))
        Dhd = (hd**2).mul_ground(RationalFunction.from_int(j + 1))
        ha, hd = Dha, Dhd

        QBC = Q*B**(1 + j)*C**(n + j)
        H_list.append(QBC)

        Ff, _ = F.div(F.gcd(Q))
        F_stara, F_stard = frac_in(Ff, DE.t)
        if F_stara.degree(DE.t) - F_stard.degree(DE.t) > 0:
            # XXX: Only the principal parts at the real zeros of F are
            # included; contributions from complex zeros are dropped (the
            # H_list return value covers all zeros, since it is not
            # evaluated at the roots).
            H = (QBC*F_stard).rem(F_stara)
            from .rde import _rational_roots
            # the rational roots (SymPy's real_roots, without the irrational ones)
            alphas = _rational_roots(F_stara)
            for alpha in alphas:
                # delta += H(alpha)/(t - alpha)**(n - j)
                pa = TowerPoly.from_list([1, -alpha], DE.t)**(n - j)
                delta_a = delta_a*pa + delta_d.mul_ground(H.eval(alpha))
                delta_d = delta_d*pa
    delta_a, delta_d = delta_a.cancel(delta_d, include=True)
    return (delta_a, delta_d, H_list)


def recognize_derivative(a: TowerPoly, d: TowerPoly, DE: DifferentialExtension,
        z: Optional[Symbol] = None) -> bool:
    """
    Compute the squarefree factorization of the denominator of f
    and for each Di the polynomial H in K[x] (see Theorem 2.7.1), using the
    LaurentSeries algorithm. Write Di = GiEi where Gj = gcd(Hn, Di) and
    gcd(Ei,Hn) = 1. Since the residues of f at the roots of Gj are all 0, and
    the residue of f at a root alpha of Ei is Hi(a) != 0, f is the derivative of a
    rational function if and only if Ei = 1 for each i, which is equivalent to
    Di | H[-1] for each i.

    This is the first derivative-recognition criterion from Section 2.9
    of Bronstein's book (stated there for K(x)), via Theorem 2.7.1
    (Section 2.7) and ``LaurentSeries``.  The generalization to monomial
    extensions is only justified at primes with constant roots that are
    not special; poles at other primes raise ``NotImplementedError``.
    """
    flag = True
    a, d = _in_t(a, DE), _in_t(d, DE)
    a, d = a.cancel(d, include=True)
    _, r = a.div(d)
    Np, Sp = splitfactor_sqf(d, DE, coefficientD=True, z=z)

    # Degree-zero (content) factors are not poles
    normal = [(s, n) for s, n in Np if s.degree(DE.t) > 0]
    if any(n == 1 for s, n in normal):
        # A simple pole at a normal prime always has a nonzero residue
        # (nu_p(Dv) == nu_p(v) - 1 <= -2 for any pole of Dv at a normal p),
        # so f is not the derivative of a rational function.
        return False
    undecidable_special: list[TowerPoly] = []
    for s, n in Sp:
        sp = _in_t(s, DE)
        # Special irreducible factors (p divides Dp): the Laurent
        # series machinery cannot compute the principal parts there,
        # so set them aside (for a squarefree s, the special part is
        # exactly gcd(s, Ds)), decide every other factor first (a
        # conclusive False stands), and raise only if none of them
        # settles the question.
        s_special = sp.gcd(derivation(sp, DE))
        if s_special.degree(DE.t) > 0:
            undecidable_special.append(s_special)
            sp = sp.quo(s_special)
            if sp.degree(DE.t) == 0:
                continue
        delta_a, delta_d, H = laurent_series(r, d, sp, n, DE)
        if not H[-1].rem(sp).is_zero:  # Di does not divide H[-1]
            # A conclusive False from a decidable factor stands
            # regardless of any undecidable factors, so check these
            # first.
            flag = False
            break
    else:
        if normal:
            # The Laurent series machinery above is only justified for
            # factors with constant roots (Theorem 2.7.1 is stated for
            # K[x] with the roots algebraic over the constant field K);
            # deciding whether the residues vanish at nonconstant poles
            # of order > 1 is not yet implemented.
            raise NotImplementedError("recognize_derivative() cannot decide "
                "residue vanishing at nonconstant poles of order greater "
                "than 1.")
        if undecidable_special:
            raise NotImplementedError("recognize_derivative() cannot decide "
                "residue vanishing at special primes.")
    return flag


def recognize_log_derivative(a: TowerPoly, d: TowerPoly, DE: DifferentialExtension,
        z: Optional[Symbol] = None) -> bool:
    """
    Necessary conditions for f == a/d to be Dv/v for some v in k(t)*.

    Explanation
    ===========

    If f == Dv/v for v in k(t)*, then the proper part of f is simple
    (its denominator is normal) and all the roots of its
    Rothstein-Trager resultant are integers (Section 5.12 of
    Bronstein's book, the "Recognizing Logarithmic Derivatives"
    subsection; neither edition gives it as named pseudocode).  This
    function checks exactly those conditions, so False is conclusive
    (f is not a logarithmic derivative), while True is not: the
    polynomial part of f is discarded unexamined (deciding it requires
    the recursive and parametric machinery of Section 5.12), which
    makes this a sound cheap filter to run before the full parametric
    logarithmic derivative problem.
    """
    z = z or unique_dummy('z')
    a, d = _in_t(a, DE), _in_t(d, DE)
    a, d = a.cancel(d, include=True)
    _, a = a.div(d)

    pz = RationalFunction.generator(z)
    Dd = derivation(d, DE)
    if d.gcd(Dd).degree(DE.t) > 0:
        # If f == Dv/v, the denominator of the proper part of f is
        # normal (squarefree and coprime with its derivative), so a
        # non-normal denominator is conclusive.
        return False
    q = a - Dd.mul_ground(pz)
    res, _ = d.resultant_prs(q)
    r = TowerPoly.from_element_strict(res, z)
    Np, Sp = splitfactor_sqf(r, DE, coefficientD=True, z=z)

    if any(s.degree() > 0 for s, _ in Np):
        # The normal part of the splitting factorization contains the
        # factors of the resultant whose roots are not constants; such
        # roots cannot be integers, so f is not the logarithmic derivative
        # of a k(t)-radical.
        return False

    from .rde import _rational_roots
    for s, _ in Sp:
        # All roots of the resultant must be integers, which holds iff the
        # rational roots of s, counted with multiplicity, account for its
        # full degree (i.e., s splits into linear factors over QQ) and are
        # all integers.
        found = _rational_roots(s)
        if len(found) != s.degree() or not all(j.is_integer for j in found):
            return False
    return True


def residue_reduce(a: TowerPoly, d: TowerPoly, DE: DifferentialExtension,
        z: Optional[Symbol] = None, invert: bool = True
        ) -> tuple[list[tuple[TowerPoly, TowerPoly]], bool]:
    """
    Lazard-Rioboo-Rothstein-Trager resultant reduction.

    Explanation
    ===========

    Given a derivation ``D`` on k(t) and f in k(t) simple, return g
    elementary over k(t) and a Boolean b in {True, False} such that f -
    Dg in k[t] if b == True or f + h and f + h - Dg do not have an
    elementary integral over k(t) for any h in k<t> (reduced) if b ==
    False.

    Returns (G, b), where G is a tuple of tuples of the form (s_i, S_i),
    such that g = Add(*[RootSum(s_i, lambda z: z*log(S_i(z, t))) for
    S_i, s_i in G]). f - Dg is the remaining integral, which is elementary
    only if b == True, and hence the integral of f is elementary only if
    b == True. The s_i are polynomials in ``z``, the S_i polynomials in
    ``t`` with coefficients in ``k(z)``.

    f - Dg is not calculated in this function because that would require
    explicitly calculating the RootSum.  Use residue_reduce_derivation().

    This is ``ResidueReduce`` (the Lazard-Rioboo-Trager version) from
    Section 5.6 of Bronstein's book.
    """
    z = z or unique_dummy('z')
    a, d = _in_t(a, DE), _in_t(d, DE)
    a, d = a.cancel(d, include=True)
    lc = d.LC().invert()
    a, d = a.mul_ground(lc), d.mul_ground(lc)

    if a.is_zero:
        return ([], True)
    _, a = a.div(d)

    pz = RationalFunction.generator(z)

    Dd = derivation(d, DE)
    q = a - Dd.mul_ground(pz)

    R: list[TowerPoly]
    if Dd.degree(DE.t) <= d.degree(DE.t):
        res, R = d.resultant_prs(q)
    else:
        res, R = q.resultant_prs(d)

    R_map: dict[int, TowerPoly] = {}
    H: list[tuple[TowerPoly, TowerPoly]] = []
    for prs_element in R:
        R_map[prs_element.degree()] = prs_element

    r = TowerPoly.from_element_strict(res, z)
    Np, Sp = splitfactor_sqf(r, DE, coefficientD=True, z=z)

    for s, i in Sp:
        if i == d.degree(DE.t):
            s = s.monic()
            H.append((s, d))
        else:
            found = R_map.get(i)
            if found is None:
                continue
            h = found
            s = s.monic()
            h_lc = TowerPoly.from_element_strict(h.LC(), z)

            h_lc_sqf = h_lc.sqf_list_include()

            for a_, j in h_lc_sqf:
                g = a_.gcd(s)
                h = h.exquo(TowerPoly.ground(g.as_element()**j, DE.t))

            if invert:
                h_lc_z = TowerPoly.from_element_strict(h.LC(), z)
                inv = h_lc_z.invert(s)
                coeffs: list[RationalFunction] = [RationalFunction.from_int(1)]

                for coeff in h.rep[1:]:
                    if not coeff:
                        coeffs.append(coeff)
                        continue
                    coeffs.append(_mod(coeff, s, z, inv).as_element())

                h = TowerPoly(coeffs, DE.t)

            H.append((s, h))

    b = not any(i.has(DE.t, z) for i, _ in Np)

    return (H, b)


def _mod(e: RationalFunction, s: TowerPoly, z: Symbol, factor: Optional[TowerPoly] = None) -> TowerPoly:
    """``factor*e`` reduced modulo ``s`` in ``k[z]/(s)`` (the denominator
    of ``e`` inverted modulo ``s``)."""
    n, d = frac_in(e, z)
    value = n.mul_ground(d.LC().invert()) if d.is_ground else n * d.invert(s)
    if factor is not None:
        value = value * factor
    return value.rem(s)


def _lambda_variable() -> Symbol:
    return Dummy('i')


def residue_reduce_to_basic(H: list[tuple[TowerPoly, TowerPoly]], DE: DifferentialExtension,
        z: Symbol) -> Expr:
    """
    Converts the tuple returned by residue_reduce() into a Basic expression.

    Terms whose residues can all be computed explicitly are rewritten as
    real logarithms and arc-tangents using log_to_real() (Rioboo's
    algorithm from Section 2.8 of Bronstein's book, with the roots of
    each s_i as the residues and S_i in place of the Rothstein-Trager
    resultant's remainder); the remaining terms are returned as RootSums.
    log_to_real() is only valid over a real field, so it is not used when
    I appears in a term or in the extension tower.
    """
    from .rationaltools import log_to_real
    from sympy_extras.polys.rootoftools import root_sum
    i = _lambda_variable()
    s = DE.backsubstitutions()
    real_tower = not any(f(DE.x).has(I) for f in DE.Tfuncs)

    result: Expr = _ZERO
    for sa, h in H:
        real: Optional[Expr] = None
        if real_tower and not (sa.has(I) or h.has(I)):
            real = log_to_real(h.as_expr(), sa.as_expr(), DE.t, z)
        if real is not None:
            result = add(result, _subs(real, s))
        else:
            body = _subs(_xrep(mul(i, log(h.as_expr())), {z: i}), s)
            result = add(result, root_sum(sa.as_expr().subs(z, i), Lambda(i, body), i))

    return result


def residue_reduce_derivation(H: list[tuple[TowerPoly, TowerPoly]], DE: DifferentialExtension,
        z: Symbol) -> Expr:
    """
    Computes the derivation of an expression returned by residue_reduce().

    In general, this is a rational function in t, so this returns an
    as_expr() result.
    """
    from sympy_extras.polys.rootoftools import root_sum
    i = _lambda_variable()
    terms: list[Expr] = []
    for sa, h in H:
        body = mul(i, _xrep(mul(derivation(h, DE).as_expr(), pow(h.as_expr(), Integer(-1))), {z: i}))
        terms.append(root_sum(sa.as_expr().subs(z, i), Lambda(i, body), i))
    return add(*terms)


def _residue_derivation_element(H: list[tuple[TowerPoly, TowerPoly]], DE: DifferentialExtension,
        z: Symbol) -> RationalFunction:
    """:func:`residue_reduce_derivation` as an element of ``k(t)`` (the
    sums over the roots of the ``s_i`` are rational in the coefficients:
    the trace in ``k(z)/(s_i)``)."""
    total = RationalFunction.from_int(0)
    for sa, h in H:
        total = total + _trace(sa, derivation(h, DE), h, z, DE.t)
    return total


def _trace(sa: TowerPoly, num: TowerPoly, den: TowerPoly, z: Symbol, t: Expr) -> RationalFunction:
    """``sum_{s(a) = 0} a*num(a, t)/den(a, t)``, an element of ``k(t)``: the
    trace of ``z*num/den`` in ``k(t)[z]/(s)``, through the inverse of
    ``den`` modulo ``s`` (``den`` is invertible modulo ``s`` since ``s``
    and the resultant are coprime with it)."""
    s = sa.monic()
    n = s.degree()
    if n <= 0:
        return RationalFunction.from_int(0)
    zf = RationalFunction.generator(z)
    s_t = TowerPoly(s.rep, z)
    try:
        value = _mod(num.as_element() * zf / den.as_element(), s_t, z)
    except ValueError:
        raise NotImplementedError('a sum over the roots of %s with a pole' % (s,))
    # the trace of z**k by Newton's identities
    c = [s_t.nth(n - k) for k in range(n + 1)]  # c[0] = 1, s = sum c[k] z**(n - k)
    p: list[RationalFunction] = [RationalFunction.from_int(n)]
    for k in range(1, n):
        acc = c[k] * k
        for j in range(1, k):
            acc = acc + c[j] * p[k - j]
        p.append(-acc)
    total = RationalFunction.from_int(0)
    for k in range(min(n, value.degree() + 1)):
        total = total + value.nth(k) * p[k]
    return total


def integrate_primitive_polynomial(p: TowerPoly, DE: DifferentialExtension
        ) -> tuple[TowerPoly, TowerPoly, bool]:
    """
    Integration of primitive polynomials.

    Explanation
    ===========

    Given a primitive monomial t over k, and ``p`` in k[t], return q in k[t],
    r in k, and a bool b in {True, False} such that r = p - Dq is in k if b is
    True, or r = p - Dq does not have an elementary integral over k(t) if b is
    False.

    This is ``IntegratePrimitivePolynomial`` from Section 5.8 of
    Bronstein's book.
    """
    p = _in_t(p, DE)
    Zero = TowerPoly.zero(DE.t)
    q = TowerPoly.zero(DE.t)

    if not p.has(DE.t):
        return (Zero, p, True)

    from .prde import limited_integrate
    while True:
        if not p.has(DE.t):
            return (q, p, True)

        Dta, Dtb = frac_in(DE.d, DE.T[DE.level - 1])

        with DecrementLevel(DE):  # We had better be integrating the lowest extension (x)
                                  # with ratint().
            a = p.LC()
            aa, ad = frac_in(a, DE.t)

            try:
                rv = limited_integrate(aa, ad, [(Dta, Dtb)], DE)
                if rv is None:
                    raise NonElementaryIntegralException
                (ba, bd), c = rv
            except NonElementaryIntegralException:
                return (q, p, False)

        m = p.degree(DE.t)
        q0 = TowerPoly.monomial(DE.t, m + 1, c[0] / (m + 1)) + \
            TowerPoly.monomial(DE.t, m, ba.as_element() / bd.as_element())

        p = p - derivation(q0, DE)
        q = q + q0


def _back(e: RationalFunction, s: Substitutions) -> Expr:
    return _subs(e.as_expr(), s)


def _fraction_expr(a: TowerPoly, d: TowerPoly) -> Expr:
    """``a.as_expr()/d.as_expr()`` (SymPy's form of a fraction of the
    tower)."""
    if d.is_one:
        return a.as_expr()
    return mul(a.as_expr(), pow(d.as_expr(), Integer(-1)))


def _quotient(a: TowerPoly, d: TowerPoly) -> RationalFunction:
    return a.as_element() / d.as_element()


def _derivative_of_fraction(ga: TowerPoly, gd: TowerPoly, DE: DifferentialExtension) -> RationalFunction:
    """``D(ga/gd)``."""
    return (gd*derivation(ga, DE) - ga*derivation(gd, DE)).as_element() / (gd**2).as_element()


def integrate_primitive(a: TowerPoly, d: TowerPoly, DE: DifferentialExtension,
        z: Optional[Symbol] = None) -> tuple[Expr, Expr, bool]:
    """
    Integration of primitive functions.

    Explanation
    ===========

    Given a primitive monomial t over k and f in k(t), return g elementary over
    k(t), i in k(t), and b in {True, False} such that i = f - Dg is in k if b
    is True or i = f - Dg does not have an elementary integral over k(t) if b
    is False.

    This function returns a Basic expression for the first argument.  If b is
    True, the second argument is Basic expression in k to recursively integrate.
    If b is False, the second argument is an unevaluated Integral, which has
    been proven to be nonelementary.

    This is ``IntegratePrimitive`` from Section 5.8 of Bronstein's book.
    """
    # XXX: a and d must be canceled, or this might return incorrect results
    z = z or unique_dummy("z")
    s = DE.backsubstitutions()

    g1, h, r = hermite_reduce(a, d, DE)
    g2, b = residue_reduce(h[0], h[1], DE, z=z)
    if not b:
        ie = _quotient(a, d) - _derivative_of_fraction(g1[0], g1[1], DE) - \
            _residue_derivation_element(g2, DE, z)
        i: Expr = NonElementaryIntegral(_back(ie, s), DE.x)
        return (add(_subs(_fraction_expr(g1[0], g1[1]), s),
            residue_reduce_to_basic(g2, DE, z)), i, b)

    # h - Dg2 + r
    pe = _quotient(h[0], h[1]) - _residue_derivation_element(g2, DE, z) + _quotient(r[0], r[1])
    p = TowerPoly.from_element_strict(pe, DE.t)

    q, ip, b = integrate_primitive_polynomial(p, DE)

    ret = add(_subs(add(_fraction_expr(g1[0], g1[1]), q.as_expr()), s),
        residue_reduce_to_basic(g2, DE, z))
    if not b:
        # i == p - Dq from integrate_primitive_polynomial(), which has been
        # proven to have no elementary integral over k(t), and ret contains
        # the partial q, so f == D(ret) + i holds for the returned values.
        i = NonElementaryIntegral(_back(ip.as_element(), s), DE.x)
    else:
        i = ip.as_element().as_expr()

    return (ret, i, b)


def integrate_hyperexponential_polynomial(p: LaurentPolynomial, DE: DifferentialExtension,
        z: Symbol) -> tuple[TowerPoly, TowerPoly, bool]:
    """
    Integration of hyperexponential polynomials.

    Explanation
    ===========

    Given a hyperexponential monomial t over k and ``p`` in k[t, 1/t], return q in
    k[t, 1/t] (as a fraction ``qa/qd`` in ``t``) and a bool b in {True, False}
    such that p - Dq in k if b is True, or p - Dq does not have an elementary
    integral over k(t) if b is False.

    This is ``IntegrateHyperexponentialPolynomial`` from Section 5.9 of
    Bronstein's book.
    """
    terms, b = _hyperexponential_terms(p, DE)
    q = RationalFunction.from_int(0)
    t1f = RationalFunction.generator(DE.t)
    for i, v in terms:
        q = q + v * t1f**i
    qa, qd = frac_in(q, DE.t)
    return (qa, qd, b)


def _hyperexponential_terms(p: LaurentPolynomial, DE: DifferentialExtension
        ) -> tuple[list[tuple[int, RationalFunction]], bool]:
    """The terms ``v_i t**i`` of the ``q`` of
    :func:`integrate_hyperexponential_polynomial`, and its ``b``."""
    dtt = DE.d.exquo(TowerPoly.monomial(DE.t, 1))
    terms: list[tuple[int, RationalFunction]] = []
    b = True

    if p.is_zero:
        return (terms, b)

    from .rde import rischDE

    with DecrementLevel(DE):
        for i in range(p.low_degree(), p.degree() + 1):
            if not i:
                continue
            a = p.nth(i)
            aa, ad = frac_in(a, DE.t, cancel=True)
            iDta, iDtd = frac_in(dtt.as_element() * i, DE.t, cancel=True)
            try:
                va, vd = rischDE(iDta, iDtd, aa, ad, DE)
            except NonElementaryIntegralException:
                b = False
            else:
                # q += v*t**i
                v = _quotient(va, vd)
                if v:
                    terms.append((i, v))
    return (terms, b)


def integrate_hyperexponential(a: TowerPoly, d: TowerPoly, DE: DifferentialExtension,
        z: Optional[Symbol] = None, conds: str = 'piecewise') -> tuple[Expr, Expr, bool]:
    """
    Integration of hyperexponential functions.

    Explanation
    ===========

    Given a hyperexponential monomial t over k and f in k(t), return g
    elementary over k(t), i in k(t), and a bool b in {True, False} such that
    i = f - Dg is in k if b is True or i = f - Dg does not have an elementary
    integral over k(t) if b is False.

    This function returns a Basic expression for the first argument.  If b is
    True, the second argument is Basic expression in k to recursively integrate.
    If b is False, the second argument is an unevaluated Integral, which has
    been proven to be nonelementary.

    This is ``IntegrateHyperexponential`` from Section 5.9 of Bronstein's
    book.
    """
    # XXX: a and d must be canceled, or this might return incorrect results
    z = z or unique_dummy("z")
    s = DE.backsubstitutions()

    g1, h, r = hermite_reduce(a, d, DE)
    g2, b = residue_reduce(h[0], h[1], DE, z=z)
    if not b:
        ie = _quotient(a, d) - _derivative_of_fraction(g1[0], g1[1], DE) - \
            _residue_derivation_element(g2, DE, z)
        i: Expr = NonElementaryIntegral(_cancel_expr(_back(ie, s)), DE.x)
        return (add(_subs(_fraction_expr(g1[0], g1[1]), s),
            residue_reduce_to_basic(g2, DE, z)), i, b)

    # p should be a polynomial in t and 1/t, because Sirr == k[t, 1/t]
    # h - Dg2 + r
    pe = _quotient(h[0], h[1]) - _residue_derivation_element(g2, DE, z) + _quotient(r[0], r[1])
    pp = as_poly_1t(pe, DE.t, z)

    terms, b = _hyperexponential_terms(pp, DE)
    q = RationalFunction.from_int(0)
    t1f = RationalFunction.generator(DE.t)
    for k, v in terms:
        q = q + v * t1f**k
    qa, qd = frac_in(q, DE.t)

    i0 = pp.nth(0)

    ret = add(_subs(_fraction_expr(g1[0], g1[1]), s), residue_reduce_to_basic(g2, DE, z))

    # SymPy's form: the Laurent polynomial in t and 1/t over its common
    # denominator, both with the denominators of the coefficients cleared
    qexpr, qd_expr = _laurent_expr(terms, DE.t)
    qexpr = _subs(qexpr, s)
    qds = _subs(qd_expr, s)
    if (conds == 'piecewise' and DE.x not in qds.free_symbols and
            qds.is_zero is not False):
        # We have to be careful if the exponent is S.Zero!

        # XXX: Does qd = 0 always necessarily correspond to the exponential
        # equaling 1?
        inner = _subs((pe - i0).subs(DE.t, RationalFunction.from_int(1)).as_expr(), s)
        ret = add(ret, piecewise(
                (qexpr, ne(qds, 0)),
                (_integrate_in_k(inner, DE.x), True)
            ))
    else:
        ret = add(ret, qexpr)

    if not b:
        ie = pe - _derivative_of_fraction(qa, qd, DE)
        i = NonElementaryIntegral(_back(ie, s), DE.x)
    else:
        i = i0.as_expr()
    return (ret, i, b)


def _laurent_expr(terms: list[tuple[int, RationalFunction]], t: Expr) -> tuple[Expr, Expr]:
    """``(sum v_k t**k, d)`` as SymPy writes ``qa/qd`` (``qa`` in ``t`` and
    ``z == 1/t``): the common denominator ``d`` of the ``v_k`` cleared."""
    from .tower import integral_pair
    if not terms:
        return _ZERO, _ONE
    numerators, denominator = integral_pair([v for _, v in terms], [RationalFunction.from_int(1)])
    d = denominator[0]
    total = add(*[mul(n.as_expr(), pow(t, Integer(k))) for (k, _), n in zip(terms, numerators)])
    if d.is_one:
        return total, _ONE
    return mul(total, pow(d.as_expr(), Integer(-1))), d.as_expr()


def _integrate_in_k(e: Expr, x: Symbol) -> Expr:
    """SymPy's ``integrate`` of the second case of the Piecewise of
    :func:`integrate_hyperexponential` (an element of the lower field at
    an exponent zero): the Risch algorithm again."""
    return risch_integrate(e, x)


def integrate_hypertangent_polynomial(p: TowerPoly, DE: DifferentialExtension
        ) -> tuple[TowerPoly, TowerPoly]:
    """
    Integration of hypertangent polynomials.

    Explanation
    ===========

    Given a differential field k such that sqrt(-1) is not in k, a
    hypertangent monomial t over k, and p in k[t], return q in k[t] and
    c in k such that p - Dq - c*D(t**2 + 1)/(t**2 + 1) is in k and p -
    Dq does not have an elementary integral over k(t) if Dc != 0.

    This is ``IntegrateHypertangentPolynomial`` from Section 5.10 of
    Bronstein's book.
    """
    # XXX: Make sure that sqrt(-1) is not in k.
    q, r = polynomial_reduce(p, DE)
    a = DE.d.exquo(TowerPoly.from_list([1, 0, 1], DE.t))
    c = TowerPoly.ground(r.nth(1) / (a.as_element() * 2), DE.t)
    return (q, c)


def integrate_hypertangent_reduced(pa: TowerPoly, pd: TowerPoly, DE: DifferentialExtension
        ) -> tuple[TowerPoly, TowerPoly, bool]:
    """
    Integration of hypertangent reduced elements.

    Explanation
    ===========

    Given a differential field k such that sqrt(-1) is not in k, a
    hypertangent monomial t over k, and p == pa/pd in k<t> (i.e. pd is a
    power of t**2 + 1 up to a unit), return (qa, qd, b) with q == qa/qd
    in k(t) and b in {True, False} such that p - Dq is in k[t] if b is
    True, or p - Dq does not have an elementary integral over k(t) if b
    is False.

    Each step removes the highest power of t**2 + 1 from the denominator
    by solving the coupled differential system (5.20) of Bronstein's
    book over k for the coefficients c, d of the numerator c*t + d of
    the candidate (c*t + d)/(t**2 + 1)**m.

    This is ``IntegrateHypertangentReduced`` from Section 5.10 of
    Bronstein's book.
    """
    from .cde import coupled_DE_system
    from .rde import order_at
    h2 = TowerPoly.from_list([1, 0, 1], DE.t)
    eta = DE.d.exquo(h2).as_element()
    qa, qd = TowerPoly.zero(DE.t), TowerPoly.one(DE.t)
    pa, pd = _in_t(pa, DE), _in_t(pd, DE)
    pa, pd = pa.cancel(pd, include=True)
    tf = RationalFunction.generator(DE.t)
    while True:
        m = order_at(pd, h2, DE.t)
        if pd.degree(DE.t) != 2*m:
            raise ValueError("%s/%s is not reduced (the denominator must "
                "be a power of t**2 + 1)." % (pa, pd))
        if m <= 0:
            return (qa, qd, True)
        # h == p*(t**2 + 1)**m in k[t], h == (t**2 + 1)*q + a*t + b
        h = (pa*h2**m).exquo(pd)
        r = h.rem(h2)
        a, b = r.nth(1), r.nth(0)
        with DecrementLevel(DE):
            f2 = frac_in(eta * (2*m), DE.t)
            g1 = frac_in(a, DE.t)
            g2 = frac_in(b, DE.t)
            try:
                # Dc - 2*m*eta*d == a, Dd + 2*m*eta*c == b
                (ca, cd), (da, dd) = coupled_DE_system(
                    (TowerPoly.zero(DE.t), TowerPoly.one(DE.t)), f2, g1, g2, DE)
            except NonElementaryIntegralException:
                return (qa, qd, False)
        c = _quotient(ca, cd)
        d = _quotient(da, dd)
        q0a, q0d = frac_in((c*tf + d) / h2.as_element()**m, DE.t, cancel=True)
        # p -= Dq0, q += q0
        dq0a = q0d*derivation(q0a, DE) - q0a*derivation(q0d, DE)
        dq0d = q0d**2
        pa, pd = (pa*dq0d - dq0a*pd).cancel(pd*dq0d, include=True)
        qa, qd = (qa*q0d + q0a*qd).cancel(qd*q0d, include=True)


def integrate_hypertangent(a: TowerPoly, d: TowerPoly, DE: DifferentialExtension,
        z: Optional[Symbol] = None) -> tuple[Expr, Expr, bool]:
    """
    Integration of hypertangent functions.

    Explanation
    ===========

    Given a differential field k such that sqrt(-1) is not in k, a
    hypertangent monomial t over k and f in k(t), return g elementary
    over k(t), i in k(t), and b in {True, False} such that i = f - Dg is
    in k if b is True or i = f - Dg does not have an elementary integral
    over k(t) if b is False.

    This function returns a Basic expression for the first argument.  If b is
    True, the second argument is Basic expression in k to recursively integrate.
    If b is False, the second argument is an unevaluated Integral, which has
    been proven to be nonelementary.

    This is ``IntegrateHypertangent`` from Section 5.10 of Bronstein's
    book.
    """
    # XXX: a and d must be canceled, or this might return incorrect results
    z = z or unique_dummy("z")
    s = DE.backsubstitutions()

    g1, h, r = hermite_reduce(a, d, DE)
    g2, b = residue_reduce(h[0], h[1], DE, z=z)
    if not b:
        ie = _quotient(a, d) - _derivative_of_fraction(g1[0], g1[1], DE) - \
            _residue_derivation_element(g2, DE, z)
        i: Expr = NonElementaryIntegral(_back(ie, s), DE.x)
        return (add(_subs(_fraction_expr(g1[0], g1[1]), s),
            residue_reduce_to_basic(g2, DE, z)), i, b)

    # p == h - Dg2 + r is reduced, since Sirr == {t**2 + 1}
    pe = _quotient(h[0], h[1]) - _residue_derivation_element(g2, DE, z) + _quotient(r[0], r[1])
    pa, pd = frac_in(pe, DE.t, cancel=True)
    q1a, q1d, b = integrate_hypertangent_reduced(pa, pd, DE)

    ret = add(_subs(add(_fraction_expr(g1[0], g1[1]), _fraction_expr(q1a, q1d)), s),
        residue_reduce_to_basic(g2, DE, z))
    pe = pe - _derivative_of_fraction(q1a, q1d, DE)
    if not b:
        i = NonElementaryIntegral(_back(pe, s), DE.x)
        return (ret, i, b)

    # p is now in k[t]
    pp = TowerPoly.from_element_strict(pe, DE.t)
    q2, c = integrate_hypertangent_polynomial(pp, DE)
    ret = add(ret, _subs(q2.as_expr(), s))
    p = pp - derivation(q2, DE)
    if derivation(c, DE).is_zero:
        # p - c*D(t**2 + 1)/(t**2 + 1) == p - 2*c*eta*t is in k
        c_elem = c.as_element()
        eta = DE.d.exquo(TowerPoly.from_list([1, 0, 1], DE.t)).as_element()
        ret = add(ret, mul(c_elem.as_expr(), _subs(log(add(pow(DE.t, Integer(2)), _ONE)), s)))
        i = (p.as_element() - c_elem*eta*RationalFunction.generator(DE.t)*2).as_expr()
        return (ret, i, True)
    i = NonElementaryIntegral(_back(p.as_element(), s), DE.x)
    return (ret, i, False)


def integrate_nonlinear_no_specials(a: TowerPoly, d: TowerPoly, DE: DifferentialExtension,
        z: Optional[Symbol] = None) -> tuple[Expr, bool]:
    """
    Integration of nonlinear monomials with no specials.

    Explanation
    ===========

    Given a nonlinear monomial t over k such that Sirr ({p in k[t] | p is
    special, monic, and irreducible}) is empty, and f in k(t), returns g
    elementary over k(t) and a Boolean b in {True, False} such that f - Dg is
    in k if b == True, or f - Dg does not have an elementary integral over k(t)
    if b == False.

    This function is applicable to all nonlinear extensions, but in the case
    where it returns b == False, it will only have proven that the integral of
    f - Dg is nonelementary if Sirr is empty.

    This function returns a Basic expression.

    This is ``IntegrateNonLinearNoSpecial`` from Section 5.11 of
    Bronstein's book.
    """
    # TODO: Integral from k?
    # TODO: split out nonelementary integral
    # XXX: a and d must be canceled, or this might not return correct results
    z = z or unique_dummy("z")
    s = DE.backsubstitutions()

    g1, h, r = hermite_reduce(a, d, DE)
    g2, b = residue_reduce(h[0], h[1], DE, z=z)
    if not b:
        return (add(_subs(_fraction_expr(g1[0], g1[1]), s),
            residue_reduce_to_basic(g2, DE, z)), b)

    # Because f has no specials, this should be a polynomial in t, or else
    # there is a bug.
    pe = _quotient(h[0], h[1]) - _residue_derivation_element(g2, DE, z) + _quotient(r[0], r[1])
    p = TowerPoly.from_element_strict(pe, DE.t)
    q1, q2 = polynomial_reduce(p, DE)

    if q2.has(DE.t):
        b = False
    else:
        b = True

    ret = add(_back(_quotient(g1[0], g1[1]) + q1.as_element(), s),
        residue_reduce_to_basic(g2, DE, z))
    return (ret, b)


class NonElementaryIntegral(Expr):
    """
    Represents a nonelementary Integral.

    Explanation
    ===========

    If the result of :func:`risch_integrate` contains an instance of this
    class, the integral it stands for has been proven to be nonelementary.
    In some cases, the algorithm can split an integral into an elementary
    and nonelementary part, so that the result will be the sum of an
    elementary expression and a NonElementaryIntegral. The node does not
    evaluate; its derivative in its variable is its function.

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.functions import exp, log
    >>> from sympy_extras.integrals.risch.risch import risch_integrate, NonElementaryIntegral
    >>> x = symbols('x')
    >>> a = risch_integrate(exp(-x**2), x)
    >>> a, isinstance(a, NonElementaryIntegral), a.diff(x)
    (Integral(exp(-x**2), x), True, exp(-x**2))
    >>> risch_integrate((2*log(x)**2 - log(x) - x**2)/(log(x)**3 - x**2*log(x)), x)
    -log(-x + log(x))/2 + log(x + log(x))/2 + Integral(1/log(x), x)
    """

    __slots__ = ('_function', '_variable')

    def __init__(self, function: Expr, variable: Symbol) -> None:
        if not isinstance(function, Expr):
            raise TypeError("an expression is expected, got %s" % (function,))
        if not isinstance(variable, Symbol):
            raise TypeError("a symbol is expected, got %s" % (variable,))
        self._function: Final[Expr] = function
        self._variable: Final[Symbol] = variable
        Basic.__init__(self)

    @property
    def args(self) -> tuple[Basic, ...]:
        return (self._function, self._variable)

    @property
    def function(self) -> Expr:
        return self._function

    @property
    def variable(self) -> Symbol:
        return self._variable

    @property
    def free_symbols(self) -> set[Symbol]:
        # an indefinite integral depends on its variable (SymPy's)
        return self._function.free_symbols | {self._variable}

    @staticmethod
    def _checked(args: tuple[Basic, ...]) -> tuple[Expr, Symbol]:
        f, v = args
        if not (isinstance(f, Expr) and isinstance(v, Symbol)):
            raise TypeError("invalid arguments of an Integral: %s" % (args,))
        return f, v

    def _rebuild(self, args: tuple[Basic, ...]) -> Basic:
        f, v = self._checked(args)
        return NonElementaryIntegral(f, v)

    def _evaluated(self, args: tuple[Basic, ...]) -> Basic:
        return self._rebuild(args)

    def _subs(self, old: Basic, new: Basic) -> Basic:
        # the variable is the one of the integration (SymPy's Integral(f,
        # x).subs(x, y) is Integral(f, (x, y)), which is not kept here)
        if self == old:
            return new
        if old == self._variable:
            return self
        f = self._function._subs(old, new)
        if f is self._function:
            return self
        if not isinstance(f, Expr):
            raise TypeError("the substitution made %s a non-expression" % (self,))
        return NonElementaryIntegral(f, self._variable)

    def _eval_derivative_node(self, s: Symbol) -> Optional[Expr]:
        if s == self._variable:
            return self._function
        if s not in self._function.free_symbols:
            return _ZERO
        return None

    def _sympystr(self, printer: NodePrinter) -> str:
        return 'Integral(%s, %s)' % (printer.doprint(self._function), printer.doprint(self._variable))


def _rationalized(e: Expr) -> Expr:
    """``e`` with its floats replaced by their decimal values."""
    rule: dict[Basic, Basic] = {}
    for a in e.atoms(Float):
        if isinstance(a, Float):
            q = Fraction(str(a))
            rule[a] = Rational(q.numerator, q.denominator)
    return _xrep(e, rule)


def _floated(e: Expr) -> Expr:
    """The numeric coefficients of the terms of ``e`` as floats."""
    terms: list[Expr] = []
    for term in Add.make_args(e):
        c, rest = term.as_coeff_Mul()
        if isinstance(c, Rational):
            terms.append(mul(Float(Fraction(c.p, c.q)), rest))
        else:
            terms.append(term)
    return add(*terms)


def risch_integrate(f: Expr, x: Symbol, extension: Optional[DifferentialExtension] = None,
        handle_first: str = 'log', rewrite_complex: Optional[bool] = None,
        conds: str = 'piecewise') -> Expr:
    r"""
    The Risch Integration Algorithm.

    Explanation
    ===========

    Only transcendental functions are supported: exponentials, logarithms,
    tangents and arc-tangents (sin, cos, sec, csc and cot are rewritten
    through the tangent of the half angle, which the answer is rewritten
    back from, and acot through atan; hyperbolic functions, asin and acos
    need ``rewrite_complex=True``).

    If this function returns a :class:`NonElementaryIntegral` in the
    result, it means that it has proven that integral to be nonelementary.
    Any errors will result in raising NotImplementedError.

    handle_first may be either 'exp' or 'log'.  This changes the order in
    which the extension is built, and may result in a different (but
    equivalent) solution (for an example of this, see issue 5109).  It is also
    possible that the integral may be computed with one but not the other,
    because not all cases have been implemented yet.  It defaults to 'log' so
    that the outer extension is exponential when possible, because more of the
    exponential case has been implemented.

    :func:`risch_integrate_separate` returns the elementary and the
    nonelementary parts apart.

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.functions import exp, log
    >>> from sympy_extras.integrals.risch.risch import risch_integrate
    >>> x = symbols('x')

    First, we try integrating exp(-x**2). Except for a constant factor of
    2/sqrt(pi), this is the famous error function.

    >>> risch_integrate(exp(-x**2), x)
    Integral(exp(-x**2), x)

    The unevaluated Integral in the result means that risch_integrate() has
    proven that exp(-x**2) does not have an elementary anti-derivative.

    >>> risch_integrate(exp(x)*exp(exp(x)), x)
    exp(exp(x))
    >>> risch_integrate(exp(exp(x)), x)
    Integral(exp(exp(x)), x)
    >>> risch_integrate(x*x**x*log(x) + x**x + x*x**x, x)
    x*x**x
    >>> risch_integrate(x**x, x)
    Integral(x**x, x)
    >>> risch_integrate(-1/(x*log(x)*log(log(x))**2), x)
    1/log(log(x))

    This implements the integration procedure outlined in Section 5.2 of
    Bronstein's book (there is no single corresponding pseudocode
    function).
    """
    result, i = risch_integrate_separate(f, x, extension, handle_first, rewrite_complex, conds)
    return add(result, i)


def risch_integrate_separate(f: Expr, x: Symbol, extension: Optional[DifferentialExtension] = None,
        handle_first: str = 'log', rewrite_complex: Optional[bool] = None,
        conds: str = 'piecewise') -> tuple[Expr, Expr]:
    """:func:`risch_integrate` with ``separate_integral=True``: ``(ans, i)``
    with the integral ``ans + i``, ``ans`` elementary and ``i`` a
    :class:`NonElementaryIntegral` or ``0``.

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.functions import log
    >>> from sympy_extras.integrals.risch.risch import risch_integrate_separate
    >>> x = symbols('x')
    >>> risch_integrate_separate((2*log(x)**2 - log(x) - x**2)/(log(x)**3 - x**2*log(x)), x)
    (-log(-x + log(x))/2 + log(x + log(x))/2, Integral(1/log(x), x))
    """
    if extension is None and f.has(Float):
        # SymPy's domain RR: the floats as their decimal rationals, the
        # coefficients of the answer as floats again
        exact, rest = risch_integrate_separate(_rationalized(f), x, None, handle_first, rewrite_complex, conds)
        if isinstance(rest, NonElementaryIntegral):
            rest = NonElementaryIntegral(f, x) if exact == _ZERO else \
                NonElementaryIntegral(_floated(rest.function), x)
        return (_floated(exact), rest)
    DE = extension or DifferentialExtension(f, x, handle_first=handle_first,
            dummy=True, rewrite_complex=rewrite_complex)
    fa, fd = DE.fa, DE.fd

    result: Expr = _ZERO
    for case in reversed(DE.cases):
        if not fa.has(DE.t) and not fd.has(DE.t) and not case == 'base':
            DE.decrement_level()
            fa, fd = frac_in((fa, fd), DE.t)
            continue

        fa, fd = fa.cancel(fd, include=True)
        ans: Expr
        i: Expr
        b: bool
        if case == 'exp':
            ans, i, b = integrate_hyperexponential(fa, fd, DE, conds=conds)
        elif case == 'primitive':
            ans, i, b = integrate_primitive(fa, fd, DE)
        elif case == 'tan':
            ans, i, b = integrate_hypertangent(fa, fd, DE)
        elif case == 'base':
            # XXX: We can't call ratint() directly here because it doesn't
            # handle polynomials correctly.
            from .rationaltools import ratint
            ans = ratint((fa.as_expr(), fd.as_expr()), DE.x)
            b = False
            i = _ZERO
        else:
            raise NotImplementedError("Only exponential, logarithmic, "
            "hypertangent and primitive extensions are currently supported.")

        result = add(result, ans)
        if b:
            DE.decrement_level()
            fa, fd = frac_in(i, DE.t)
        else:
            result = DE.restore_sincos(_subs(result, DE.backsubs),
                drop_constants=True)
            if isinstance(i, NonElementaryIntegral):
                if result == _ZERO:
                    # Nothing was integrated, so i is f itself: show it
                    # as the user wrote it
                    i = NonElementaryIntegral(f, x)
                else:
                    i = NonElementaryIntegral(DE.restore_sincos(
                        _subs(i.function, DE.backsubs)), i.variable)
                return (result, i)
            return (result, _ZERO)
    # Not reached: the cases end with the 'base' case, which returns
    raise NotImplementedError("The differential extension has no base case.")
