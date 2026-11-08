"""The evaluation of sums (SymPy's ``Sum.doit``, ``summation``,
``eval_sum``): definite sums of few terms term by term, the symbolic
methods (linearity, telescoping, powers of the index by Bernoulli
polynomials and harmonic numbers, geometric sums), then Gosper's
algorithm, the hypergeometric and the residue methods.

Ported from SymPy's ``sympy.concrete.summations`` (Gosper's algorithm
in :mod:`~sympy_extras.concrete.gosper`). The residue method with poles
which SymPy writes as ``CRootOf``, the Kronecker delta sums and the
equation solving of the telescoping go through SymPy for that call (not
ported yet).

Examples
========

>>> from sympy_extras.core import Symbol, oo
>>> from sympy_extras.concrete.summations import summation
>>> i, n, x = Symbol('i', integer=True), Symbol('n', integer=True), Symbol('x')
>>> summation(i**2, (i, 1, n)), summation(x**i, (i, 0, n))
(n**3/3 + n**2/2 + n/6, Piecewise((n + 1, Eq(x, 1)), ((1 - x**(n + 1))/(1 - x), True)))
>>> summation(1/(i*(i + 1)), (i, 1, n)), summation(i, (i, 1, 4))
(1 - 1/(n + 1), 10)
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/concrete/summations.py (Sum.doit, summation,
# telescopic_direct, telescopic, eval_sum, eval_sum_direct,
# eval_sum_symbolic, _dummy_with_inherited_properties_concrete)
# (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: typed; eval_sum_residue,
# deltasummation, solve and the zeta function match go through SymPy.
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Optional, Union

from sympy_extras.core.add import Add, add
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.logic import Boolean, true
from sympy_extras.core.mul import Mul, mul
from sympy_extras.core.numbers import Infinity, Integer, NaN, NegativeInfinity, oo
from sympy_extras.core.symbol import Dummy, Symbol, Wild

from .expr_with_limits import LimitLike, Limits, Sum, sum_

__all__ = ['summation', 'sum_doit', 'eval_sum', 'eval_sum_direct', 'eval_sum_symbolic', 'telescopic']

ZERO = Integer(0)
ONE = Integer(1)


def _e(b: Basic) -> Expr:
    assert isinstance(b, Expr)
    return b


def _is_true(b: Basic) -> bool:
    return b is true


def _two_terms(e: Expr) -> tuple[Expr, Expr]:
    """SymPy's ``as_two_terms``: the first argument and the rest."""
    args = [a for a in e.args if isinstance(a, Expr)]
    rest = add(*args[1:]) if isinstance(e, Add) else mul(*args[1:])
    return args[0], rest


# ---------------------------------------------------------------------
# the steps going through SymPy


def _through(name: str, f: Expr, limits: Limits) -> Optional[Expr]:
    """``sympy.concrete.<module>.<name>(f, limits)``, ``None`` kept."""
    import importlib
    from sympy_extras.core.interop import expr_from_sympy, to_sympy
    module, function = name.rsplit('.', 1)
    fn = getattr(importlib.import_module('sympy.concrete.' + module), function)
    i, a, b = limits
    r = fn(to_sympy(f), (to_sympy(i), to_sympy(a), to_sympy(b)))
    if r is None:
        return None
    return expr_from_sympy(r)


def _solve(e: Expr, v: Symbol) -> list[Expr]:
    """SymPy's ``solve(e, v)`` (through SymPy)."""
    import sympy
    from sympy_extras.core.interop import expr_from_sympy, to_sympy
    return [expr_from_sympy(s) for s in sympy.solve(to_sympy(e), to_sympy(v))]


# PORTED FROM SYMPY: sympy/concrete/summations.py:telescopic_direct
def telescopic_direct(L: Expr, R: Expr, n: int, limits: Limits) -> Expr:
    """The sum of ``L + R`` when ``R`` is ``-L`` shifted by ``n``."""
    i, a, b = limits
    return add(*[_e(L.subs(i, a + m)) + _e(R.subs(i, b - m)) for m in range(n)])


# PORTED FROM SYMPY: sympy/concrete/summations.py:telescopic
def telescopic(L: Expr, R: Expr, limits: Limits) -> Optional[Expr]:
    """The sum of ``L + R`` by telescoping, ``None`` when ``R`` is not
    ``-L`` shifted by an integer."""
    i, a, b = limits
    if isinstance(L, Add) or isinstance(R, Add):
        return None
    k = Wild('k')
    sol = (-R).match(_e(L.subs(i, i + k)))
    s: Optional[Expr] = None
    if sol and k in sol:
        s = _e(sol[k])
        if not (isinstance(s, Integer) and _e(L.subs(i, i + s)) + R == 0):
            s = None
    if s is None:
        m = Dummy('m')
        try:
            found = _solve(_e(L.subs(i, i + m)) + R, m)
        except NotImplementedError:
            return None
        found = [si for si in found if isinstance(si, Integer) and (_e(L.subs(i, i + si)) + R).expand().is_zero]
        if len(found) != 1:
            return None
        s = found[0]
    assert isinstance(s, Integer)
    if s.p < 0:
        return telescopic_direct(R, L, abs(s.p), limits)
    if s.p > 0:
        return telescopic_direct(L, R, s.p, limits)
    return None


# PORTED FROM SYMPY: sympy/concrete/summations.py:eval_sum
def eval_sum(f: Expr, limits: Limits) -> Optional[Expr]:
    """The sum of ``f`` over one limit, ``None`` when no method works."""
    from sympy_extras.functions.elementary.piecewise import Piecewise, piecewise
    from sympy_extras.functions.special.tensor_functions import KroneckerDeltaNode
    i, a, b = limits
    if f.is_zero is True:
        return ZERO
    if i not in f.free_symbols:
        return f * (b - a + 1)
    if a == b:
        return _e(f.subs(i, a))
    if a.is_comparable and b.is_comparable and _is_true(a > b):
        return eval_sum(f, (i, b + ONE, a - ONE))
    if isinstance(f, Piecewise):
        if not any(i in pair.cond.free_symbols for pair in f.args):
            newargs: list[tuple[Expr, Boolean]] = []
            for pair in f.args:
                newexpr = eval_sum(pair.expr, limits)
                if newexpr is None:
                    return None
                newargs.append((newexpr, pair.cond))
            return piecewise(*newargs)
    if f.has(KroneckerDeltaNode):
        return _through('delta.deltasummation', f, limits)
    dif = b - a
    definite = isinstance(dif, Integer)
    if isinstance(dif, Integer) and dif.p < 100:
        return eval_sum_direct(f, (i, a, b))
    if isinstance(f, Piecewise):
        return None
    value = eval_sum_symbolic(f.expand(), (i, a, b))
    if value is not None:
        return value
    if definite:
        return eval_sum_direct(f, (i, a, b))
    return None


# PORTED FROM SYMPY: sympy/concrete/summations.py:eval_sum_direct
def eval_sum_direct(expr: Expr, limits: Limits) -> Optional[Expr]:
    """A definite sum term by term (the factors and terms free of the
    index taken out first)."""
    from sympy_extras.polys.partfrac import apart
    from sympy_extras.polys.polyerrors import PolynomialError
    i, a, b = limits
    dif = b - a
    if isinstance(expr, Mul):
        without_i, with_i = expr.as_independent(i)
        if without_i != 1:
            s = eval_sum_direct(with_i, (i, a, b))
            if s:
                r = without_i * s
                if not isinstance(r, NaN):
                    return r
        else:
            L, R = _two_terms(expr)
            if not L.has(i):
                sR = eval_sum_direct(R, (i, a, b))
                if sR:
                    return L * sR
            if not R.has(i):
                sL = eval_sum_direct(L, (i, a, b))
                if sL:
                    return sL * R
    try:
        expr = apart(expr, i)
    except PolynomialError:
        pass
    if isinstance(expr, Add):
        without_i, with_i = expr.as_independent(i)
        if without_i != 0:
            s = eval_sum_direct(with_i, (i, a, b))
            if s:
                r = without_i * (dif + 1) + s
                if not isinstance(r, NaN):
                    return r
        else:
            L, R = _two_terms(expr)
            lsum = eval_sum_direct(L, (i, a, b))
            rsum = eval_sum_direct(R, (i, a, b))
            if lsum is not None and rsum is not None:
                r = lsum + rsum
                if not isinstance(r, NaN):
                    return r
    assert isinstance(dif, Integer)
    return add(*[_e(expr.subs(i, a + j)) for j in range(dif.p + 1)])


# PORTED FROM SYMPY: sympy/concrete/summations.py:eval_sum_symbolic
def eval_sum_symbolic(f: Expr, limits: Limits) -> Optional[Expr]:
    """A sum by the symbolic methods: linearity, telescoping, powers of
    the index, geometric sums, Gosper's algorithm, the hypergeometric and
    residue methods, the factored summand."""
    from sympy_extras.core.relational import Eq
    from sympy_extras.core.sorting import ordered
    from sympy_extras.functions.combinatorial.numbers import bernoulli, harmonic
    from sympy_extras.functions.elementary.piecewise import piecewise
    from sympy_extras.polys.partfrac import apart
    from sympy_extras.polys.polyerrors import PolynomialError
    from sympy_extras.polys.polytools import factor
    from sympy_extras.polys.rationaltools import together
    from sympy_extras.simplify.powsimp import powsimp
    from sympy_extras.simplify.radsimp import denom
    f_orig = f
    i, a, b = limits
    if not f.has(i):
        return f * (b - a + 1)
    if isinstance(f, Mul):
        without_i, with_i = f.as_independent(i)
        if without_i != 1:
            s = eval_sum_symbolic(with_i, (i, a, b))
            if s:
                r = without_i * s
                if not isinstance(r, NaN):
                    return r
        else:
            L, R = _two_terms(f)
            if not L.has(i):
                sR = eval_sum_symbolic(R, (i, a, b))
                if sR:
                    return L * sR
            if not R.has(i):
                sL = eval_sum_symbolic(L, (i, a, b))
                if sL:
                    return sL * R
    try:
        f = apart(f, i)
    except PolynomialError:
        pass
    if isinstance(f, Add):
        L, R = _two_terms(f)
        lrsum = telescopic(L, R, (i, a, b))
        if lrsum:
            return lrsum
        without_i, with_i = f.as_independent(i)
        if without_i != 0:
            s = eval_sum_symbolic(with_i, (i, a, b))
            if s:
                r = without_i * (b - a + 1) + s
                if not isinstance(r, NaN):
                    return r
        else:
            lsum = eval_sum_symbolic(L, (i, a, b))
            rsum = eval_sum_symbolic(R, (i, a, b))
            if lsum is not None and rsum is not None:
                r = lsum + rsum
                if not isinstance(r, NaN):
                    return r
    nw = Wild('n')
    result = f.match(i**nw)
    if result is not None:
        n = _e(result[nw])
        if isinstance(n, Integer):
            if n.p >= 0:
                if (isinstance(b, Infinity) and not isinstance(a, NegativeInfinity)) or \
                        (isinstance(a, NegativeInfinity) and not isinstance(b, Infinity)):
                    return oo
                return ((bernoulli(n + 1, b + 1) - bernoulli(n + 1, a)) / (n + 1)).expand()
            if isinstance(a, Integer) and a.p >= 1:
                if n.p == -1:
                    return harmonic(b) - harmonic(a - 1)
                return harmonic(b, abs(n.p)) - harmonic(a - 1, abs(n.p))
    if not (a.has(Infinity) or a.has(NegativeInfinity) or b.has(Infinity) or b.has(NegativeInfinity)):
        c1, c2, c3 = Wild('c1', exclude=[i]), Wild('c2', exclude=[i]), Wild('c3', exclude=[i])
        wexp = Wild('wexp')
        e = powsimp(f).match(c1**wexp)
        if e is not None:
            exponent = _e(e.pop(wexp))
            e_exp = exponent.expand().match(c2 * i + c3)
            if e_exp is not None:
                e.update(e_exp)
                p = _e((c1**c3).subs(e))
                q = _e((c1**c2).subs(e))
                r = p * (q**a - q**(b + 1)) / (1 - q)
                l_ = p * (b - a + 1)
                return piecewise((l_, Eq(q, ONE)), (r, True))
        from .gosper import gosper_sum
        gr = gosper_sum(f, (i, a, b))
        if isinstance(gr, (Mul, Add)):
            non_limit = gr.free_symbols - (a.free_symbols | b.free_symbols)
            den = denom(together(gr))
            den_sym = non_limit & den.free_symbols
            args: list[tuple[Expr, Boolean]] = []
            for v in ordered(list(den_sym)):
                assert isinstance(v, Symbol)
                try:
                    sols = _solve(den, v)
                except NotImplementedError:
                    continue
                if sols:
                    cond = Eq(v, sols[0])
                    inner = sum_(_e(f_orig.subs(v, sols[0])), limits)
                    args.append((sum_doit(inner) if isinstance(inner, Sum) else inner, cond))
                break
            args.append((gr, true))
            return piecewise(*args)
        if gr is not None and not isinstance(gr, NaN):
            return gr
    h = eval_sum_hyper(f_orig, (i, a, b))
    if h is not None:
        return h
    rr = eval_sum_residue(f_orig, (i, a, b))
    if rr is not None:
        return rr
    factored = factor(f_orig)
    if factored != f_orig:
        return eval_sum_symbolic(factored, (i, a, b))
    return None


Condition = Union[bool, Boolean]


def _cond(c: Condition) -> Boolean:
    from sympy_extras.core.logic import false
    if isinstance(c, bool):
        return true if c else false
    return c


# PORTED FROM SYMPY: sympy/concrete/summations.py:_eval_sum_hyper
def _eval_sum_hyper(f: Expr, i: Symbol, a: Expr) -> Optional[tuple[Expr, Condition]]:
    """The sum of ``f`` from ``a`` to ``oo`` as a hypergeometric function
    expanded, with its condition of convergence; ``None`` when ``f`` is not
    a hypergeometric term of the needed form."""
    from sympy_extras.core.numbers import Float
    from sympy_extras.functions.special.hyper import Hyper, hyper
    from sympy_extras.polys.polytools import factor
    from sympy_extras.simplify.gammasimp import combsimp
    from sympy_extras.simplify.hyperexpand import hyperexpand
    from sympy_extras.simplify.radsimp import fraction
    from sympy_extras.simplify.simplify import nsimplify, simplify
    from .gosper import hypersimp, _UPoly
    if a != 0:
        return _eval_sum_hyper(_e(f.subs(i, i + a)), i, ZERO)
    if _e(f.subs(i, ZERO)) == 0:
        if simplify(_e(f.subs(i, Dummy('i', integer=True, positive=True)))) == 0:
            return ZERO, True
        return _eval_sum_hyper(_e(f.subs(i, i + 1)), i, ZERO)
    hs = hypersimp(f, i)
    if hs is None:
        return None
    if isinstance(hs, Float):
        hs = nsimplify(hs)
    numer, denom = fraction(factor(hs))
    top, topl = _coeff_mul(numer, i)
    bot, botl = _coeff_mul(denom, i)
    ab: list[Expr] = [top, bot]
    params: list[list[Expr]] = [[], []]
    for k, facs in enumerate((topl, botl)):
        for fac in facs:
            multiplicity = 1
            if fac.is_Pow:
                exponent = _e(fac.args[1])
                if not isinstance(exponent, Integer):
                    return None
                multiplicity = exponent.p
                fac = _e(fac.args[0])
            try:
                p = _UPoly.of(fac, i)
            except ValueError:
                return None
            if p.degree() != 1:
                return None
            m, n = p.nth(1), p.nth(0)
            ab[k] = ab[k] * m**multiplicity
            params[k] += [n / m] * multiplicity
    ap = params[0] + [ONE]
    bq = params[1]
    x = ab[0] / ab[1]
    h = hyper(ap, bq, x)
    statement: Condition = h.convergence_statement if isinstance(h, Hyper) else True
    f = combsimp(f)
    return _e(f.subs(i, ZERO)) * hyperexpand(h), statement


def _coeff_mul(e: Expr, x: Symbol) -> tuple[Expr, list[Expr]]:
    """SymPy's ``e.as_coeff_mul(x)``: the factors free of ``x`` and the
    others."""
    factors = [a for a in (e.args if isinstance(e, Mul) else (e,)) if isinstance(a, Expr)]
    if not isinstance(e, Mul):
        return (ONE, [e]) if e.has(x) else (e, [])
    return mul(*[a for a in factors if not a.has(x)]), [a for a in factors if a.has(x)]


def _count_illegal(e: Expr) -> int:
    """SymPy's count of ``oo``, ``-oo``, ``zoo`` and ``nan`` in ``e``."""
    from sympy_extras.core.basic import preorder_traversal
    from sympy_extras.core.numbers import ComplexInfinity
    return sum(1 for node in preorder_traversal(e)
               if isinstance(node, (Infinity, NegativeInfinity, ComplexInfinity, NaN)))


# PORTED FROM SYMPY: sympy/concrete/summations.py:eval_sum_hyper
def eval_sum_hyper(f: Expr, limits: Limits) -> Optional[Expr]:
    """An infinite (or symbolic) sum of a hypergeometric term by
    hyperexpand, as a ``Piecewise`` on the condition of convergence."""
    from sympy_extras.core.logic import and_, false
    from sympy_extras.functions.elementary.piecewise import piecewise
    i, a, b = limits
    if f.is_hypergeometric(i) is False:
        return None
    if isinstance(b - a, Integer):
        return None
    old_sum = sum_(f, (i, a, b))
    if not isinstance(b, Infinity):
        if isinstance(a, NegativeInfinity):
            res = _eval_sum_hyper(_e(f.subs(i, -i)), i, -b)
            if res is not None:
                return piecewise((res[0], _cond(res[1])), (old_sum, true))
            return None
        had = _count_illegal(f)
        res1 = _eval_sum_hyper(f, i, a)
        if res1 is None or _count_illegal(res1[0]) > had:
            return None
        res2 = _eval_sum_hyper(f, i, b + 1)
        if res2 is None or _count_illegal(res2[0]) > had:
            return None
        cond = and_(_cond(res1[1]), _cond(res2[1]))
        if cond is false:
            return None
        return piecewise((res1[0] - res2[0], cond), (old_sum, true))
    if isinstance(a, NegativeInfinity):
        r1 = _eval_sum_hyper(_e(f.subs(i, -i)), i, ONE)
        r2 = _eval_sum_hyper(f, i, ZERO)
        if r1 is None or r2 is None:
            return None
        cond = and_(_cond(r1[1]), _cond(r2[1]))
        if cond is false or _as_set_is_empty(cond):
            return None
        return piecewise((r1[0] + r2[0], cond), (old_sum, true))
    res = _eval_sum_hyper(f, i, a)
    if res is None:
        return None
    r, c = res
    if c is False or c is false:
        if r.is_number:
            shifted = _e(f.subs(i, Dummy('i', integer=True, positive=True) + a))
            if shifted.is_positive is True or shifted.is_zero is True:
                return oo
            if shifted.is_negative is True:
                return -oo
        return None
    return piecewise((r, _cond(c)), (old_sum, true))


def _as_set_is_empty(cond: Boolean) -> bool:
    """SymPy's ``cond.as_set() == S.EmptySet`` (through SymPy)."""
    import sympy
    from sympy_extras.core.interop import to_sympy
    try:
        sympy_cond = to_sympy(cond)
        assert isinstance(sympy_cond, sympy.logic.boolalg.Boolean)
        return bool(sympy_cond.as_set() == sympy.S.EmptySet)
    except NotImplementedError:
        return False


class _NotPorted(Exception):
    """A case of eval_sum_residue the native code does not take (SymPy's
    CRootOf poles, a coefficient domain our Poly lacks)."""


# PORTED FROM SYMPY: sympy/concrete/summations.py:eval_sum_residue
def eval_sum_residue(f: Expr, limits: Limits) -> Optional[Expr]:
    """An infinite sum of a rational function (or ``(-1)**i`` times one) by
    the residues of ``f*pi*cot(pi*z)`` (``csc`` when alternating) at its
    poles; ``None`` when the method does not apply."""
    try:
        return _eval_sum_residue(f, limits)
    except _NotPorted:
        return _through('summations.eval_sum_residue', f, limits)


def _eval_sum_residue(f: Expr, limits: Limits) -> Optional[Expr]:
    from sympy_extras.core.numbers import pi
    from sympy_extras.functions.elementary.trigonometric import cot, csc
    from sympy_extras.polys.polyerrors import CoercionFailed, PolynomialError
    from sympy_extras.polys.polyroots import roots
    from sympy_extras.polys.polytools import Poly
    from sympy_extras.series.residues import residue
    i, a, b = limits
    if a.is_comparable and b.is_comparable and _is_true(a > b):
        return _eval_sum_residue(f, (i, b + ONE, a - ONE))

    def poly(e: Expr) -> Poly:
        try:
            return Poly(e, i)
        except (CoercionFailed, NotImplementedError) as exc:
            raise _NotPorted() from exc

    def is_even_function(numer: Poly, denom: Poly) -> bool:
        nm = [m[0] for m in numer.monoms()]
        dm = [m[0] for m in denom.monoms()]
        return (all(k % 2 == 0 for k in nm) and all(k % 2 == 0 for k in dm)) or \
            (all(k % 2 == 1 for k in nm) and all(k % 2 == 1 for k in dm))

    def match_rational(g: Expr) -> Optional[tuple[Poly, Poly]]:
        numer, denom = g.as_numer_denom()
        try:
            return poly(numer), poly(denom)
        except PolynomialError:
            return None

    def get_poles(denom: Poly) -> Optional[tuple[list[Expr], list[Expr]]]:
        # SymPy's denom.sqf_part().all_roots(): closed forms for the factors
        # of degree at most 2 and the binomials, CRootOf otherwise
        _, factors = denom.factor_list()
        found: list[Expr] = []
        for fac, _ in factors:
            if fac.degree() > 2 and fac.length() != 2:
                raise _NotPorted()
            rs = roots(fac.as_expr(), i)
            if not isinstance(rs, dict) or sum(rs.values()) != fac.degree():
                raise _NotPorted()
            found.extend(rs)
        if any(r.is_integer is None for r in found):
            return None
        return [r for r in found if r.is_integer is True], [r for r in found if r.is_integer is False]

    def shifted(p: Poly, by: Expr) -> Poly:
        moved = p.as_expr().subs(i, i + by)
        assert isinstance(moved, Expr)
        return poly(moved.expand())

    z = Dummy('z')

    def residue_factor(numer: Poly, denom: Poly, alternating: bool) -> Expr:
        g = _e((numer.as_expr() / denom.as_expr()).subs(i, z))
        return g * (csc(pi * z) if alternating else cot(pi * z))

    if f.free_symbols - {i}:
        return None
    if not (isinstance(a, (Integer, Infinity, NegativeInfinity)) and isinstance(b, (Integer, Infinity, NegativeInfinity))):
        return None
    if not isinstance(a, NegativeInfinity) and not isinstance(b, Infinity):
        return None
    match = match_rational(f)
    alternating = False
    if match is None:
        match = match_rational(f / Integer(-1)**i)
        if match is None:
            return None
        alternating = True
    numer, denom = match
    if denom.degree() - numer.degree() < 2:
        return None
    if isinstance(a, NegativeInfinity) and isinstance(b, Infinity):
        poles = get_poles(denom)
        if poles is None:
            return None
        int_roots, nonint_roots = poles
        if int_roots:
            return None
        factor_ = residue_factor(numer, denom, alternating)
        return -pi * add(*[residue(factor_, z, root) for root in nonint_roots])
    if not (a.is_finite is True and isinstance(b, Infinity)):
        return None
    if not is_even_function(numer, denom):
        n = denom.degree()
        lead = denom.coeff_monomial((n,))
        nxt = denom.coeff_monomial((n - 1,))
        shift = -nxt / lead / n
        if not isinstance(shift, Integer) or shift == 0:
            return None
        numer, denom = shifted(numer, shift), shifted(denom, shift)
        if not is_even_function(numer, denom):
            return None
        if alternating:
            g = Integer(-1)**i * (Integer(-1)**shift * numer.as_expr() / denom.as_expr())
        else:
            g = numer.as_expr() / denom.as_expr()
        return _eval_sum_residue(g, (i, a - shift, b - shift))
    poles = get_poles(denom)
    if poles is None:
        return None
    int_roots, nonint_roots = poles
    ints = [r.p for r in int_roots if isinstance(r, Integer)]
    assert isinstance(a, Integer)
    if ints:
        if len(ints) != max(ints) - min(ints) + 1:
            return None
        if a.p <= max(ints):
            return None
    factor_ = residue_factor(numer, denom, alternating)
    full_sum = -pi * add(*[residue(factor_, z, root) for root in list(int_roots) + list(nonint_roots)])
    if not ints:
        # SymPy's xreplace re-evaluates: the values of the terms
        half_sum = (full_sum + _e(f.subs(i, ZERO))) / 2
        extraneous_neg = [_e(f.subs(i, Integer(i0))) for i0 in range(a.p, 0)]
        extraneous_pos = [_e(f.subs(i, Integer(i0))) for i0 in range(0, a.p)]
        return half_sum + add(*extraneous_neg) - add(*extraneous_pos)
    half_sum = full_sum / 2
    extraneous = [_e(f.subs(i, Integer(i0))) for i0 in range(max(ints) + 1, a.p)]
    return half_sum - add(*extraneous)


_ASSUMPTIONS = ['extended_nonnegative', 'nonnegative', 'extended_nonpositive', 'nonpositive', 'extended_positive',
                'positive', 'extended_negative', 'negative', 'integer', 'rational', 'finite', 'zero', 'real',
                'extended_real']


# PORTED FROM SYMPY: sympy/concrete/summations.py:_dummy_with_inherited_properties_concrete
def _dummy_with_inherited_properties(limits: Limits) -> Optional[Dummy]:
    """A dummy with the facts of the variable and those shared by both
    bounds, ``None`` when the bounds add nothing."""
    from sympy_extras.core.assumptions import facts_of
    x, a, b = limits
    keep: dict[str, bool] = {}
    added: dict[str, bool] = {}
    facts = facts_of(x)
    for name in _ASSUMPTIONS:
        if facts.get(name) is True:
            keep[name] = True
        elif all(facts_of(e).get(name) is True for e in (a, b)):
            added[name] = True
    if added:
        keep.update(added)
        return Dummy('d', **keep)
    return None


def _zeta_match(f: Expr, limits: Limits) -> bool:
    """Whether SymPy's ``eval_zeta_function`` applies (``(w*i + y)**(-z)``
    summed to ``oo``)."""
    i, a, b = limits
    if a.is_comparable and b.is_comparable and _is_true(a > b):
        return _zeta_match(f, (i, b + ONE, a - ONE))
    if not isinstance(b, Infinity):
        return False
    w, y, z = Wild('w', exclude=[i]), Wild('y', exclude=[i]), Wild('z', exclude=[i])
    return bool(f.match((w * i + y)**(-z)))


# PORTED FROM SYMPY: sympy/concrete/summations.py:Sum.doit
def sum_doit(s: Sum, deep: bool = True) -> Expr:
    """The sum evaluated: each limit summed by :func:`eval_sum`, the sum
    kept where no method works."""
    from sympy_extras.functions.elementary.piecewise import Piecewise
    f = s.function.doit() if deep else s.function
    reps: dict[Basic, Basic] = {}
    for xab in s.limits:
        d = _dummy_with_inherited_properties(xab)
        if d is not None:
            reps[xab[0]] = d
    if reps:
        undo = {v: k for k, v in reps.items()}
        replaced = s.xreplace(reps)
        assert isinstance(replaced, Sum)
        return _e(sum_doit(replaced, deep).xreplace(undo))
    for n, limit in enumerate(s.limits):
        i, a, b = limit
        dif = b - a
        if dif == -1:
            return ZERO
        if dif.is_integer is True and dif.is_negative is True:
            a, b = b + 1, a - 1
            f = -f
        newf = eval_sum(f, (i, a, b))
        if newf is None:
            if f == s.function:
                if _zeta_match(f, (i, a, b)):
                    from sympy_extras.core.interop import expr_through_sympy
                    return expr_through_sympy(s, lambda t: t.doit(deep=False))
                return s
            return sum_(f, *s.limits[n:])
        f = newf
    if deep and not isinstance(f, Piecewise):
        return f.doit()
    return f


def summation(f: ExprLike, *limits: LimitLike) -> Expr:
    """The sum of ``f`` over the limits (SymPy's ``summation``:
    ``Sum(f, *limits).doit(deep=False)``)."""
    s = sum_(to_expr(f), *limits)
    if not isinstance(s, Sum):
        return s
    return sum_doit(s, deep=False)

