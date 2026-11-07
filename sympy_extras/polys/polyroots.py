"""``roots``: the roots of a univariate polynomial in radicals, with
their multiplicities.

Ported from SymPy's ``sympy.polys.polyroots``: the power of the variable
and the content come out, a common scale of the coefficients (``x**2 -
a**2`` is ``a**2 (y**2 - 1)``) and an integer basis are removed
(``preprocess_roots``); then the polynomial is factored (over ``ZZ`` in
all its symbols) and each factor solved by the formulas: linear,
quadratic, binomial (``a x**n + b``, the ``n``-th roots by
``exp(2 k pi I/n)`` expanded), cyclotomic, or after a functional
decomposition ``f = g(h)``. Over a polynomial domain (``x**2 - a``) the
roots are factored.

>>> from sympy_extras.core import Symbol
>>> x, a = Symbol('x'), Symbol('a')
>>> roots(x**2 - 3*x + 2, x), roots(x**2 - a**2, x), roots(x**3 - 1, x)
({2: 1, 1: 1}, {-a: 1, a: 1}, {1: 1, -1/2 - sqrt(3)*I/2: 1, -1/2 + sqrt(3)*I/2: 1})
>>> roots(x**4 - 2*x**2 + 1, x), roots(x**2 + 1, x, multiple=True)
({1: 2, -1: 2}, [I, -I])

Not ported yet: ``roots_cubic`` and ``roots_quartic`` for irreducible
cubics and quartics (``NotImplementedError``), quintics, numerical
coefficients (``nroots``), algebraic coefficients (SymPy's ``EX`` domain
and ``to_rational_coeffs``), the ``trig`` option.
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/polys/polyroots.py (roots, roots_linear, roots_quadratic,
# roots_binomial, roots_cyclotomic, _inv_totient_estimate, preprocess_roots,
# _integer_basis), sympy/polys/densetools.py (dup_decompose), sympy/polys/factortools.py
# (dup_cyclotomic_p), sympy/core/mul.py (expand_2arg), sympy/core/add.py
# (_unevaluated_Add) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: typed; Poly over ZZ, QQ, ZZ[a], QQ[a] only (the fraction field
# ZZ(a) of SymPy's to_field is implicit in the expression arithmetic).
# ---------------------------------------------------------------------
from __future__ import annotations

import math
from fractions import Fraction
from typing import Callable, Optional, Sequence, Union

from sympy_extras.core.add import Add, add
from sympy_extras.core.basic import Basic
from sympy_extras.core.exprtools import factor_terms
from sympy_extras.core.expr import Expr, ExprLike
from sympy_extras.core.function import expand
from sympy_extras.core.intfunc import integer_nthroot
from sympy_extras.core.mul import Mul, mul
from sympy_extras.core.numbers import I, Integer, Number, Rational, pi
from sympy_extras.core.power import Pow, pow
from sympy_extras.core.sorting import compare_key, ordered
from sympy_extras.core.symbol import Dummy, Symbol
from sympy_extras.ntheory.factor_ import factorint
from sympy_extras.ntheory.generate import nextprime
from sympy_extras.ntheory.primetest import isprime

from . import densebasic, factortools, sqfreetools
from .domains import QQ, ZZ
from .domains.rationalfield import RationalField
from .polyerrors import CoercionFailed, GeneratorsNeeded, PolynomialError
from .polytools import Poly, cancel, factor

__all__ = ['roots', 'roots_linear', 'roots_quadratic', 'roots_binomial', 'roots_cyclotomic', 'preprocess_roots']

RootFilter = Callable[[Expr], bool]


def _divisors(n: int) -> list[int]:
    result = [1]
    for p, e in factorint(n).items():
        result = [d * p**k for d in result for k in range(e + 1)]
    return sorted(result)


def _numerical(f: Poly) -> bool:
    """SymPy's ``dom.is_Numerical``: ``ZZ`` or ``QQ`` (a polynomial domain
    is ``Composite``)."""
    return not f._domain.params


# PORTED FROM SYMPY: sympy/core/add.py:_unevaluated_Add
def _unevaluated_add(args: Sequence[Expr]) -> Expr:
    pending = list(args)
    rest: list[Expr] = []
    number: Expr = Integer(0)
    while pending:
        a = pending.pop()
        if isinstance(a, Add):
            pending.extend(a.args)
        elif isinstance(a, Number):
            number = add(number, a)
        else:
            rest.append(a)
    rest.sort(key=compare_key)
    if not (isinstance(number, Number) and number == 0):
        rest.insert(0, number)
    return Add(*rest) if len(rest) != 1 else rest[0]


# PORTED FROM SYMPY: sympy/core/mul.py:expand_2arg
def _expand_2arg(e: Expr) -> Expr:
    """A number times a sum distributed, bottom up, the sum unevaluated."""
    if not e.args:
        return e
    rebuilt = e._evaluated(tuple(_expand_2arg(a) if isinstance(a, Expr) else a for a in e.args))
    if not isinstance(rebuilt, Expr):
        return e
    if isinstance(rebuilt, Mul):
        c, r = rebuilt.as_coeff_Mul()
        if isinstance(r, Add) and not (isinstance(c, Integer) and c.p == 1):
            return _unevaluated_add([mul(c, t) for t in r.args])
    return rebuilt


# PORTED FROM SYMPY: sympy/polys/polyroots.py:roots_linear
def roots_linear(f: Poly) -> list[Expr]:
    r = mul(Integer(-1), f.nth(0), pow(f.nth(1), Integer(-1)))
    if not _numerical(f):
        r = factor(r)
    return [r]


def _sqrt(d: Expr) -> Expr:
    """``sqrt(d)`` with the even powers of the factors taken out."""
    from sympy_extras.functions import sqrt
    taken: list[Expr] = []
    other: list[Expr] = []
    for di in (d.args if isinstance(d, Mul) else (d,)):
        if not isinstance(di, Expr):
            continue
        if isinstance(di, Pow) and isinstance(di.exp, Integer) and di.exp.p % 2 == 0:
            taken.append(pow(di.base, Integer(di.exp.p // 2)))
        else:
            other.append(di)
    if taken:
        return mul(mul(*taken), sqrt(mul(*other)))
    return sqrt(d)


# PORTED FROM SYMPY: sympy/polys/polyroots.py:roots_quadratic
def roots_quadratic(f: Poly) -> list[Expr]:
    a, b, c = f.all_coeffs()
    numerical = _numerical(f)

    def simplified(e: Expr) -> Expr:
        return e if numerical else factor(e)

    if isinstance(c, Integer) and c.p == 0:
        r0: Expr = Integer(0)
        r1 = mul(Integer(-1), b, pow(a, Integer(-1)))
        if not numerical:
            r1 = simplified(r1)
        elif r1.is_negative is True:
            r0, r1 = r1, r0
    elif isinstance(b, Integer) and b.p == 0:
        r = simplified(mul(Integer(-1), c, pow(a, Integer(-1))))
        R = _sqrt(r)
        r0, r1 = mul(Integer(-1), R), R
    else:
        d = add(pow(b, Integer(2)), mul(Integer(-4), a, c))
        A = mul(Integer(2), a)
        B = mul(Integer(-1), b, pow(A, Integer(-1)))
        if not numerical:
            d, B = simplified(d), simplified(B)
        D = factor_terms(mul(_sqrt(d), pow(A, Integer(-1))))
        r0, r1 = add(B, mul(Integer(-1), D)), add(B, D)
        if a.is_negative is True:
            r0, r1 = r1, r0
        elif not numerical:
            r0, r1 = _expand_2arg(r0), _expand_2arg(r1)
    return [r0, r1]


# PORTED FROM SYMPY: sympy/polys/polyroots.py:roots_binomial
def roots_binomial(f: Poly) -> list[Expr]:
    from sympy_extras.functions import exp, root
    n = f.degree()
    a, b = f.nth(n), f.nth(0)
    base = mul(Integer(-1), cancel(mul(b, pow(a, Integer(-1)))))
    alpha = root(base, Integer(n))
    if alpha.is_number:
        alpha = expand(alpha, complex=True)
    neg = base.is_negative is True
    even = n % 2 == 0
    ks: list[int] = []
    imax = n // 2
    if even:
        ks.append(imax)
        imax -= 1
    if not neg:
        ks.append(0)
    for i in range(imax, 0, -1):
        ks.extend([i, -i] if neg else [-i, i])
    if neg:
        ks.append(0)
    result: list[Expr] = []
    d = mul(Integer(2), I, pi, Rational(1, n) if n != 1 else Integer(1))
    for k in ks:
        zeta = expand(exp(mul(Integer(k), d)), complex=True)
        result.append(expand(mul(alpha, zeta), power_base=False))
    return result


# PORTED FROM SYMPY: sympy/polys/polyroots.py:_inv_totient_estimate
def _inv_totient_estimate(m: int) -> tuple[int, int]:
    primes = [d + 1 for d in _divisors(m) if isprime(d + 1)]
    a, b = 1, 1
    for p in primes:
        a *= p
        b *= p - 1
    L = m
    U = int(math.ceil(m * (float(a) / b)))
    P = p = 2
    found: list[int] = []
    while P <= U:
        p = nextprime(p)
        found.append(p)
        P *= p
    P //= p
    b = 1
    for p in found[:-1]:
        b *= p - 1
    U = int(math.ceil(m * (float(P) / b)))
    return L, U


# PORTED FROM SYMPY: sympy/polys/polyroots.py:roots_cyclotomic
def roots_cyclotomic(f: Poly) -> list[Expr]:
    from sympy_extras.functions import exp
    L, U = _inv_totient_estimate(f.degree())
    coefficients = [int(_rational(c)) for c in f.all_coeffs()]
    for n in range(L, U + 1):
        if factortools.dup_zz_cyclotomic_poly(n) == coefficients:
            break
    else:
        raise RuntimeError('failed to find index of a cyclotomic polynomial')
    h = n // 2
    ks = [i for i in range(1, n + 1) if math.gcd(i, n) == 1]
    ks.sort(key=lambda k: (k, -1) if k <= h else (abs(k - n), 1))
    d = mul(Integer(2), I, pi, Rational(1, n) if n != 1 else Integer(1))
    return [expand(exp(mul(Integer(k), d)), complex=True) for k in reversed(ks)]


def _rational(c: Expr) -> Fraction:
    if not isinstance(c, Rational):
        raise CoercionFailed('%s is not rational' % (c,))
    return c.as_fraction()


# PORTED FROM SYMPY: sympy/polys/factortools.py:dup_cyclotomic_p
def _is_cyclotomic(f: Poly) -> bool:
    if f._domain.params or len(f.gens) != 1:
        return False
    fractions = [_rational(c) for c in f.all_coeffs()]
    if any(c.denominator != 1 for c in fractions):
        return False
    return _dup_cyclotomic_p([int(c) for c in fractions])


def _dup_cyclotomic_p(f: list[int]) -> bool:
    if f[0] != 1 or f[-1] not in (-1, 1):
        return False
    coeff, factors = factortools.dup_zz_factor(f, ZZ)
    if coeff != 1 or factors != [(f, 1)]:
        return False
    return _dup_cyclotomic_p_irreducible(f)


def _dup_cyclotomic_p_irreducible(f: list[int]) -> bool:
    from .densearith import dup_mul, dup_neg, dup_sub
    from .densetools import dup_mirror
    n = len(f) - 1
    g = [f[i] for i in range(n % 2, n + 1, 2)]
    h = [f[i] for i in range((n - 1) % 2, n, 2)] if n >= 1 else []
    g = densebasic.dup_strip(list(reversed(list(reversed(g)))), ZZ)
    g2 = dup_mul(g, g, ZZ)
    h2 = dup_mul(h, h, ZZ)
    F = dup_sub(g2, h2 + [0], ZZ)
    if F and F[0] < 0:
        F = dup_neg(F, ZZ)
    if F == f:
        return True
    m = dup_mirror(f, ZZ)
    if m and m[0] < 0:
        m = dup_neg(m, ZZ)
    if F == m and _dup_cyclotomic_p(m):
        return True
    G = sqfreetools.dup_sqf_part(F, ZZ)
    if dup_mul(G, G, ZZ) == F and _dup_cyclotomic_p(G):
        return True
    return False


# PORTED FROM SYMPY: sympy/polys/polyroots.py:_integer_basis
def _integer_basis(f: Poly) -> Optional[int]:
    terms = f.terms()
    monoms = [m[0] for m, _ in terms]
    coeffs = [abs(int(_rational(c))) for _, c in terms]
    if coeffs[0] < coeffs[-1]:
        coeffs = list(reversed(coeffs))
        n = monoms[0]
        monoms = [n - i for i in reversed(monoms)]
    else:
        return None
    monoms, coeffs = monoms[:-1], coeffs[:-1]
    if len(monoms) == 1:
        r, exact = integer_nthroot(coeffs[0], monoms[0])
        return r if exact else None
    g = 0
    for c in coeffs:
        g = math.gcd(g, c)
    for div in reversed(_divisors(g)[1:]):
        if all(c % div**m == 0 for m, c in zip(monoms, coeffs)):
            return div
    return None


# PORTED FROM SYMPY: sympy/polys/polyroots.py:preprocess_roots
def preprocess_roots(f: Poly) -> tuple[Expr, Poly]:
    """``(coeff, g)``: the roots of ``f`` are ``coeff`` times those of
    ``g`` (denominators and content removed, a common power of a parameter
    and an integer basis taken out)."""
    coeff: Expr = Integer(1)
    x = f.gen
    expr = _cleared_primitive(f)
    poly = Poly(expr, x)
    params = list(poly._domain.params)
    if params and all(_is_term(c) for c in poly.coeffs()):
        injected = Poly(poly.as_expr(), x, *params)
        monoms = injected.monoms()
        strips = list(zip(*monoms))
        base, others = strips[0], strips[1:]
        remaining = list(params)
        for gen, strip0 in zip(list(params), others):
            strip = list(strip0)
            reverse = False
            if strip[0] < strip[-1]:
                strip = list(reversed(strip))
                reverse = True
            ratio: Optional[int] = None
            ok = True
            for a, b in zip(base, strip):
                if not a and not b:
                    continue
                if not a or not b or b % a != 0:
                    ok = False
                    break
                current = b // a
                if ratio is None:
                    ratio = current
                elif ratio != current:
                    ok = False
                    break
            if ok and ratio is not None:
                if reverse:
                    ratio = -ratio
                evaluated = injected.eval(gen, Integer(1))
                injected = evaluated if isinstance(evaluated, Poly) else Poly(evaluated, x)
                coeff = mul(coeff, pow(gen, Integer(-ratio)))
                remaining.remove(gen)
        poly = Poly(injected.as_expr(), x)
    if not poly._domain.params and len(poly.gens) == 1 and poly._domain.integral:
        basis = _integer_basis(poly)
        if basis is not None:
            n = poly.degree()
            poly = Poly(add(*[mul(Integer(int(_rational(c)) // basis**(n - m[0])), pow(x, Integer(m[0])))
                              for m, c in poly.terms()]), x)
            coeff = mul(coeff, Integer(basis))
    return coeff, poly


def _is_term(c: Expr) -> bool:
    """SymPy's ``PolyElement.is_term``: a single term."""
    return not isinstance(c, Add)


def _cleared_primitive(f: Poly) -> Expr:
    """``f`` with the denominators cleared and the content divided out
    (scalings which keep the roots)."""
    from math import lcm
    d = 1
    for c in f._rep.terms.values():
        d = lcm(d, c.denominator)
    scaled = {m: c * d for m, c in f._rep.terms.items()}
    g = 0
    for c in scaled.values():
        g = math.gcd(g, int(c))
    g = g or 1
    rep = f._rep.ring.from_dict({m: c / g for m, c in scaled.items()})
    content_expr: Expr = Integer(1)
    if f._domain.params:
        # the content in the parameters (SymPy's primitive over ZZ[a])
        p = Poly._new(rep, f.gens, f._domain)
        content, _ = _parameter_content(p)
        content_expr = content
    expr = Poly._new(rep, f.gens, f._domain).as_expr()
    if not (isinstance(content_expr, Integer) and content_expr.p == 1):
        expr = cancel(mul(expr, pow(content_expr, Integer(-1))))
    return expr


def _parameter_content(p: Poly) -> tuple[Expr, Poly]:
    from .domains.sparsepolynomialring import IntegerPolynomialRing, IteratedPolynomialRing
    n = len(p.gens)
    names = p._rep.ring.names
    R = IteratedPolynomialRing(IntegerPolynomialRing(names[n:]), names[:n])
    element = R.ring.from_dict({m: int(c) for m, c in p._rep.terms.items()})
    content = R.ground_content(element)
    return p._coefficient_expr({m: Fraction(c) for m, c in content.terms.items()}), p


# PORTED FROM SYMPY: sympy/polys/densetools.py:dup_decompose, _dup_right_decompose, _dup_left_decompose
def _decompose(f: Poly) -> list[Poly]:
    """``[g1, ..., gn]`` with ``f = g1(g2(...gn))`` (over ``QQ``; a
    polynomial domain is not decomposed)."""
    if f._domain.params:
        return [f]
    x = f.gen
    dense = [_rational(c) for c in f.all_coeffs()]
    parts: list[list[Fraction]] = []
    while True:
        found = _decompose_once(dense)
        if found is None:
            break
        dense, h = found
        parts.insert(0, h)
    return [Poly(_from_dense(p, x), x) for p in [dense] + parts]


def _from_dense(f: Sequence[Fraction], x: Expr) -> Expr:
    from sympy_extras.core.numbers import number_from_fraction
    n = len(f) - 1
    return add(*[mul(number_from_fraction(c), pow(x, Integer(n - i))) for i, c in enumerate(f) if c])


def _decompose_once(f: list[Fraction]) -> Optional[tuple[list[Fraction], list[Fraction]]]:
    df = len(f) - 1
    for s in range(2, df):
        if df % s:
            continue
        h = _right_decompose(f, s)
        if h is not None:
            g = _left_decompose(f, h)
            if g is not None:
                return g, h
    return None


def _right_decompose(f: list[Fraction], s: int) -> Optional[list[Fraction]]:
    n = len(f) - 1
    lc = f[0]
    raw = {n - i: c for i, c in enumerate(f) if c}
    g: dict[int, Fraction] = {s: Fraction(1)}
    r = n // s
    for i in range(1, s):
        coeff = Fraction(0)
        for j in range(0, i):
            if n + j - i not in raw or s - j not in g:
                continue
            coeff += (i - r * j) * raw[n + j - i] * g[s - j]
        g[s - i] = coeff / (i * r * lc)
    return densebasic.dup_strip([g.get(k, Fraction(0)) for k in range(s, -1, -1)], _QQ())


def _left_decompose(f: list[Fraction], h: list[Fraction]) -> Optional[list[Fraction]]:
    from .densearith import dup_div
    K = _QQ()
    g: dict[int, Fraction] = {}
    i = 0
    while f:
        q, r = dup_div(f, h, K)
        if len(r) - 1 > 0:
            return None
        g[i] = r[0] if r else Fraction(0)
        f, i = q, i + 1
    top = max(g) if g else 0
    return densebasic.dup_strip([g.get(k, Fraction(0)) for k in range(top, -1, -1)], K)


def _QQ() -> RationalField:
    return QQ


def roots_cubic(f: Poly) -> list[Expr]:
    raise NotImplementedError('roots_cubic is not ported yet')


def roots_quartic(f: Poly) -> list[Expr]:
    raise NotImplementedError('roots_quartic is not ported yet')


RootsResult = dict[Expr, int]


# PORTED FROM SYMPY: sympy/polys/polyroots.py:roots
def roots(f: Union[Poly, ExprLike], *gens: ExprLike, cubics: bool = True, quartics: bool = True,
          multiple: bool = False, filter: Optional[str] = None,
          predicate: Optional[RootFilter] = None) -> Union[RootsResult, list[Expr]]:
    """The roots of ``f`` with their multiplicities (a list with each root
    repeated when ``multiple``); ``filter`` is ``'Z'``, ``'Q'``, ``'R'``,
    ``'I'`` or ``'C'``."""
    result = _roots(f, gens, cubics, quartics, filter, predicate)
    if not multiple:
        return result
    listed: list[Expr] = []
    for zero in ordered(list(result)):
        if isinstance(zero, Expr):
            listed.extend([zero] * result[zero])
    return listed


def _roots(f0: Union[Poly, ExprLike], gens: Sequence[ExprLike], cubics: bool, quartics: bool,
           filter: Optional[str], predicate: Optional[RootFilter]) -> RootsResult:
    try:
        f = f0 if isinstance(f0, Poly) and not gens else Poly(f0 if not isinstance(f0, Poly) else f0.as_expr(), *gens)
    except GeneratorsNeeded:
        return {}
    if not isinstance(f0, Poly) and not isinstance(f.gen, Symbol):
        raise PolynomialError('generator must be a Symbol')
    n = f.degree()
    if f.length() == 2 and n > 2:
        rescaled = _rescaled_binomial(f, cubics, quartics, filter, predicate)
        if rescaled is not None:
            return rescaled
    if len(f.gens) > 1:
        raise PolynomialError('multivariate polynomials are not supported')
    # SymPy renames the generator to Dummy('x0') (printed _x0, sorted after
    # the parameters when the polynomial is factored in all its symbols)
    x = Dummy('x0')
    f = Poly(f.as_expr().xreplace({f.gen: x}), x)
    k = min(m[0] for m in f.monoms()) if not f.is_zero else 0
    if k:
        f = Poly(cancel(mul(f.as_expr(), pow(x, Integer(-k)))), x)
    zeros: RootsResult = {Integer(0): k} if k else {}
    coeff, f = preprocess_roots(f)
    result: RootsResult = {}

    def update(r: Expr, multiplicity: int) -> None:
        if isinstance(r, Integer) and r.p == 0:
            zeros[Integer(0)] = zeros.get(Integer(0), 0) + multiplicity
        result[r] = result.get(r, 0) + multiplicity

    if not f.is_ground:
        if f.degree() == 1:
            update(roots_linear(f)[0], 1)
        elif f.length() == 2:
            for r in (roots_quadratic(f) if f.degree() == 2 else roots_binomial(f)):
                update(r, 1)
        else:
            _, factors = Poly(f.as_expr()).factor_list()
            if len(factors) == 1 and f.degree() == 2:
                for r in roots_quadratic(f):
                    update(r, 1)
            elif len(factors) == 1 and factors[0][1] == 1:
                for r in _try_decompose(f, cubics, quartics):
                    update(r, 1)
            else:
                for g, m in factors:
                    for r in _try_heuristics(Poly(g.as_expr(), x), cubics, quartics):
                        update(r, m)
    if not (isinstance(coeff, Integer) and coeff.p == 1):
        result = {mul(coeff, r): m for r, m in result.items()}
    if filter not in (None, 'C'):
        handlers: dict[str, RootFilter] = {
            'Z': lambda r: isinstance(r, Integer),
            'Q': lambda r: isinstance(r, Rational),
            'R': lambda r: all(a.is_real is True for a in r.as_numer_denom()),
            'I': lambda r: r.is_imaginary is True,
        }
        if filter not in handlers:
            raise ValueError('Invalid filter: %s' % filter)
        query = handlers[filter]
        result = {r: m for r, m in result.items() if query(r)}
    if predicate is not None:
        result = {r: m for r, m in result.items() if predicate(r)}
    result.update(zeros)
    return result


def _rescaled_binomial(f: Poly, cubics: bool, quartics: bool, filter: Optional[str],
                       predicate: Optional[RootFilter]) -> Optional[RootsResult]:
    """SymPy's check for ``c*gen**m - foo**n``: integer powers of sums in
    the constant replaced by positive dummies."""
    con, dep = f.as_expr().as_independent(*f.gens)
    fcon = mul(Integer(-1), factor(mul(Integer(-1), con)))
    if fcon == con:
        return None
    bases: list[tuple[Expr, Dummy]] = []
    for i in (fcon.args if isinstance(fcon, Mul) else (fcon,)):
        if isinstance(i, Pow):
            b, e = i.as_base_exp()
            if isinstance(e, Integer) and isinstance(b, Add):
                bases.append((b, Dummy(positive=True)))
    if not bases:
        return None
    rule: dict[Basic, Basic] = {b: d for b, d in bases}
    replaced = add(dep, fcon).xreplace(rule)
    inner = _roots(Poly(replaced, *f.gens), (), cubics, quartics, filter, predicate)
    back: dict[Basic, Basic] = {d: b for b, d in bases}
    return {_as_expr(factor_terms(r._xreplace_evaluated(back))): m for r, m in inner.items()}


def _as_expr(b: Basic) -> Expr:
    if not isinstance(b, Expr):
        raise TypeError('an expression is expected')
    return b


def _try_decompose(f: Poly, cubics: bool, quartics: bool) -> list[Expr]:
    parts = _decompose(f)
    found = _try_heuristics(parts[0], cubics, quartics)
    for g in parts[1:]:
        previous, found = found, []
        for r in previous:
            shifted = add(g.as_expr(), mul(Integer(-1), r))
            try:
                h = Poly(shifted, f.gen)
            except CoercionFailed:
                raise NotImplementedError('roots over the algebraic domain EX are not ported yet')
            found.extend(_try_heuristics(h, cubics, quartics))
    return found


def _try_heuristics(f: Poly, cubics: bool, quartics: bool) -> list[Expr]:
    if f.is_ground:
        return []
    if f.length() == 1:
        return [Integer(0)] * f.degree()
    if f.length() == 2:
        if f.degree() == 1:
            return [cancel(r) for r in roots_linear(f)]
        return roots_binomial(f)
    result: list[Expr] = []
    x = f.gen
    for i in (-1, 1):
        value = f.eval(x, Integer(i))
        if isinstance(value, Expr) and isinstance(value, Integer) and value.p == 0:
            f = f.exquo(Poly(add(x, Integer(-i)), x))
            result.append(Integer(i))
            break
    n = f.degree()
    if n == 1:
        result += [cancel(r) for r in roots_linear(f)]
    elif n == 2:
        result += [cancel(r) for r in roots_quadratic(f)]
    elif _is_cyclotomic(f):
        result += roots_cyclotomic(f)
    elif n == 3 and cubics:
        result += roots_cubic(f)
    elif n == 4 and quartics:
        result += roots_quartic(f)
    return result
