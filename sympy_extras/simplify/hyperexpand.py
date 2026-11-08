"""Expanding hypergeometric functions and Meijer G-functions into named
special functions (SymPy's ``hyperexpand``).

Ported from SymPy's ``sympy.simplify.hyperexpand``. A function is
reduced, by shifting its parameters by integers, to one of a table of
known formulas: each formula is a basis ``B`` of functions with
``z*d/dz B = M*B`` and the closed form ``C*B``; the shifts are
operators on the basis.

This module is ported in parts: the formula tables, the parameter
buckets and the instantiation of a formula for a function are here;
the operators, the plans and the expansion follow.

Examples
========

>>> from sympy_extras.simplify.hyperexpand import FormulaCollection, HyperFunction
>>> from sympy_extras.core import Rational
>>> f = FormulaCollection().lookup_origin(HyperFunction([Rational(1, 2), 1], [Rational(3, 2)]))
>>> f.func, f.closed_form
(HyperFunction((1/2, 1), (3/2,)), HyperRep_atanh(_z))
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/simplify/hyperexpand.py (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: typed; Hyper_Function and G_Function are plain immutable
# classes (HyperFunction, GFunction); the matrices are ExprMatrix; the
# comparisons SymPy makes with bool(relational) go through _truth, which
# raises TypeError as SymPy does when they cannot be decided; the linear
# equations of find_instantiations are solved directly.
# ---------------------------------------------------------------------
from __future__ import annotations

from functools import reduce
from itertools import product
from typing import Callable, Optional, Sequence, Union

from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.function import expand
from sympy_extras.core.logic import Boolean, false, true
from sympy_extras.core.mod import Mod, mod
from sympy_extras.core.numbers import (ComplexInfinity, EulerGamma, I, Infinity, Integer, NaN, NegativeInfinity,
                                       Number, Rational, pi)
from sympy_extras.core.power import root, sqrt
from sympy_extras.core.sorting import SortKey, default_sort_key
from sympy_extras.core.symbol import Dummy, Symbol
from sympy_extras.functions.special.hyperrep import (HyperRep_asin1, HyperRep_asin2, HyperRep_atanh,
                                                     HyperRep_cosasin, HyperRep_log1, HyperRep_log2,
                                                     HyperRep_power1, HyperRep_power2, HyperRep_sinasin,
                                                     HyperRep_sqrts1, HyperRep_sqrts2)
from sympy_extras.matrices.dense import ExprMatrix

__all__ = ['HyperFunction', 'GFunction', 'Formula', 'FormulaCollection', 'MeijerFormula',
           'MeijerFormulaCollection', 'make_simp']

HALF = Rational(1, 2)
ZERO = Integer(0)
ONE = Integer(1)
Repl = dict[Basic, Basic]


def _truth(b: Boolean) -> bool:
    """The truth of a comparison, as SymPy's ``bool(relational)``: a
    ``TypeError`` when it is not decided."""
    if b is true:
        return True
    if b is false:
        return False
    raise TypeError('cannot determine truth value of %s' % (b,))


def _is_zero(e: Expr) -> bool:
    """SymPy's ``not e`` (only a zero number is false)."""
    return isinstance(e, Number) and e == 0


class _Ordered:
    """A sort key comparing expressions as SymPy's ``list.sort`` does
    (``a < b``, a ``TypeError`` when undecided)."""

    __slots__ = ('e',)

    def __init__(self, e: Expr) -> None:
        self.e = e

    def __lt__(self, other: _Ordered) -> bool:
        return _truth(self.e < other.e)


def _min(values: Sequence[Expr]) -> Expr:
    return reduce(lambda m, v: v if _truth(v < m) else m, values[1:], values[0])


def _max(values: Sequence[Expr]) -> Expr:
    return reduce(lambda m, v: v if _truth(m < v) else m, values[1:], values[0])


def _expr(b: Basic) -> Expr:
    assert isinstance(b, Expr)
    return b


# PORTED FROM SYMPY: sympy/simplify/hyperexpand.py:_mod1
def _mod1(x: Expr) -> Expr:
    """``x`` modulo 1 for its numeric part (``Mod(c, 1) + rest``)."""
    if isinstance(x, Number):
        return mod(x, ONE)
    c, rest = x.as_coeff_Add()
    return mod(c, ONE) + rest


def _sift(params: Sequence[Expr], key: Callable[[Expr], Expr]) -> dict[Expr, list[Expr]]:
    """SymPy's ``sift``: the parameters by key, in the order of appearance."""
    buckets: dict[Expr, list[Expr]] = {}
    for p in params:
        buckets.setdefault(key(p), []).append(p)
    return buckets


def _keys_union(a: dict[Expr, list[Expr]], b: dict[Expr, list[Expr]]) -> list[Expr]:
    """The keys of both (SymPy iterates over a set; sorted here)."""
    return sorted(set(a) | set(b), key=default_sort_key)


def _xreplace(e: Expr, repl: Repl) -> Expr:
    return _expr(e.xreplace(repl))


# PORTED FROM SYMPY: sympy/simplify/hyperexpand.py:make_simp
def make_simp(z: Symbol) -> Callable[[Expr], Expr]:
    """A simplification for a rational function in ``z``: the numerator
    expanded, the common factors cancelled."""
    from sympy_extras.polys.polyerrors import CoercionFailed
    from sympy_extras.polys.polytools import Poly

    def cancel_in_z(numer: Expr, denom: Expr) -> Expr:
        c, n, d = Poly(numer, z).cancel(Poly(denom, z))
        return c * n.as_expr() / d.as_expr()

    def simp(e: Expr) -> Expr:
        numer, denom = e.as_numer_denom()
        numer = expand(numer)
        try:
            return cancel_in_z(numer, denom)
        except CoercionFailed:
            pass
        # SymPy's composite domain (ZZ[pi], ZZ[gamma(a)], ...): the
        # coefficient atoms our Poly does not take become generators
        rep = _coefficient_generators([numer, expand(denom)], z)
        back: Repl = {v: k for k, v in rep.items()}
        return _xreplace(cancel_in_z(_xreplace(numer, rep), _xreplace(denom, rep)), back)
    return simp


def _coefficient_generators(exprs: Sequence[Expr], z: Symbol) -> Repl:
    """Dummies for the factors free of ``z`` which are neither numbers nor
    symbols (``pi``, ``gamma(a)``, ``sqrt(2)``, ``I``), bases of integer
    powers or whole powers."""
    from sympy_extras.core.add import Add
    from sympy_extras.core.mul import Mul
    found: list[Expr] = []
    for e in exprs:
        for t in (e.args if isinstance(e, Add) else (e,)):
            assert isinstance(t, Expr)
            for f in (t.args if isinstance(t, Mul) else (t,)):
                assert isinstance(f, Expr)
                if f.has(z) or isinstance(f, (Number, Symbol)):
                    continue
                b, ex = f.as_base_exp()
                g = b if isinstance(ex, Integer) and not isinstance(b, (Number, Symbol)) else f
                if isinstance(g, (Number, Symbol)) or g.has(z):
                    continue
                if g not in found:
                    found.append(g)
    found.sort(key=default_sort_key)
    return {g: Dummy('c') for g in found}


# PORTED FROM SYMPY: sympy/simplify/hyperexpand.py:Hyper_Function
class HyperFunction:
    """A generalised hypergeometric function without its argument: the
    parameters ``ap`` and ``bq``, expanded."""

    __slots__ = ('ap', 'bq')

    def __init__(self, ap: Sequence[ExprLike], bq: Sequence[ExprLike]) -> None:
        self.ap: tuple[Expr, ...] = tuple(expand(to_expr(a)) for a in ap)
        self.bq: tuple[Expr, ...] = tuple(expand(to_expr(b)) for b in bq)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, HyperFunction) and self.ap == other.ap and self.bq == other.bq

    def __hash__(self) -> int:
        return hash(('HyperFunction', self.ap, self.bq))

    def __repr__(self) -> str:
        def tup(t: tuple[Expr, ...]) -> str:
            return '(%s,)' % t[0] if len(t) == 1 else '(%s)' % ', '.join(str(e) for e in t)
        return 'HyperFunction(%s, %s)' % (tup(self.ap), tup(self.bq))

    __str__ = __repr__

    @property
    def sizes(self) -> tuple[int, int]:
        return (len(self.ap), len(self.bq))

    @property
    def gamma(self) -> int:
        """The number of negative integer upper parameters."""
        return sum(1 for x in self.ap if x.is_integer is True and x.is_negative is True)

    def __call__(self, arg: ExprLike) -> Expr:
        from sympy_extras.functions.special.hyper import hyper
        return hyper(self.ap, self.bq, arg)

    def has(self, *patterns: Union[Basic, type]) -> bool:
        return any(p.has(*patterns) for p in self.ap + self.bq)

    def xreplace(self, repl: Repl) -> HyperFunction:
        return HyperFunction([_xreplace(a, repl) for a in self.ap], [_xreplace(b, repl) for b in self.bq])

    # PORTED FROM SYMPY: sympy/simplify/hyperexpand.py:Hyper_Function.build_invariants
    def build_invariants(self) -> tuple[int, tuple[tuple[Expr, int], ...], tuple[tuple[Expr, int], ...]]:
        """The invariants of the parameters under integer shifts: the
        ``gamma``, and for ``ap`` and ``bq`` the residues modulo 1 with
        their counts."""
        def tr(bucket: dict[Expr, list[Expr]]) -> tuple[tuple[Expr, int], ...]:
            items = list(bucket.items())
            if not any(isinstance(k, Mod) for k, _ in items):
                items.sort(key=lambda kv: default_sort_key(kv[0]))
            return tuple((k, len(v)) for k, v in items if v)
        return (self.gamma, tr(_sift(self.ap, _mod1)), tr(_sift(self.bq, _mod1)))

    # PORTED FROM SYMPY: sympy/simplify/hyperexpand.py:Hyper_Function.difficulty
    def difficulty(self, func: HyperFunction) -> Expr:
        """The total shift from ``self`` to ``func``, ``-1`` when they have
        different invariants."""
        from sympy_extras.functions.elementary.complexes import abs_
        if self.gamma != func.gamma:
            return Integer(-1)
        oabuckets, obbuckets, abuckets, bbuckets = [_sift(params, _mod1) for params in
                                                    (self.ap, self.bq, func.ap, func.bq)]
        diff: Expr = ZERO
        for bucket, obucket in [(abuckets, oabuckets), (bbuckets, obbuckets)]:
            for m in _keys_union(bucket, obucket):
                if m not in bucket or m not in obucket or len(bucket[m]) != len(obucket[m]):
                    return Integer(-1)
                l1 = sorted(bucket[m], key=_Ordered)
                l2 = sorted(obucket[m], key=_Ordered)
                for i, j in zip(l1, l2):
                    diff = diff + abs_(i - j)
        return diff

    # PORTED FROM SYMPY: sympy/simplify/hyperexpand.py:Hyper_Function._is_suitable_origin
    def _is_suitable_origin(self) -> bool:
        """Whether the function can be an origin of shifts: no ``a - b``
        a nonnegative integer, no zero ``a``, no nonpositive integer ``b``."""
        for a in self.ap:
            for b in self.bq:
                if (a - b).is_integer is True and (a - b).is_negative is False:
                    return False
        for a in self.ap:
            if a == 0:
                return False
        for b in self.bq:
            if b.is_integer is True and b.is_nonpositive is True:
                return False
        return True


Buckets = dict[Expr, list[Expr]]


# PORTED FROM SYMPY: sympy/simplify/hyperexpand.py:G_Function
class GFunction:
    """A Meijer G-function without its argument: the four parameter
    lists, expanded."""

    __slots__ = ('an', 'ap', 'bm', 'bq')

    def __init__(self, an: Sequence[ExprLike], ap: Sequence[ExprLike], bm: Sequence[ExprLike],
                 bq: Sequence[ExprLike]) -> None:
        self.an: tuple[Expr, ...] = tuple(expand(to_expr(a)) for a in an)
        self.ap: tuple[Expr, ...] = tuple(expand(to_expr(a)) for a in ap)
        self.bm: tuple[Expr, ...] = tuple(expand(to_expr(b)) for b in bm)
        self.bq: tuple[Expr, ...] = tuple(expand(to_expr(b)) for b in bq)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, GFunction) and self.lists == other.lists

    def __hash__(self) -> int:
        return hash(('GFunction',) + self.lists)

    def __repr__(self) -> str:
        return 'GFunction(%s)' % ', '.join('(%s)' % ', '.join(str(e) for e in t) for t in self.lists)

    __str__ = __repr__

    @property
    def lists(self) -> tuple[tuple[Expr, ...], tuple[Expr, ...], tuple[Expr, ...], tuple[Expr, ...]]:
        return (self.an, self.ap, self.bm, self.bq)

    def __call__(self, z: ExprLike) -> Expr:
        from sympy_extras.functions.special.hyper import meijerg
        return meijerg(self.an, self.ap, self.bm, self.bq, z)

    # PORTED FROM SYMPY: sympy/simplify/hyperexpand.py:G_Function.compute_buckets
    def compute_buckets(self) -> tuple[Buckets, Buckets, Buckets, Buckets]:
        """The four lists by residue modulo 1, each bucket sorted by the
        difference to its first element (``an`` and ``bq`` decreasing)."""
        dicts: list[Buckets] = [{}, {}, {}, {}]
        for dic, lis in zip(dicts, self.lists):
            for x in lis:
                dic.setdefault(_mod1(x), []).append(x)
        for dic, flip in zip(dicts, (True, False, False, True)):
            for m, items in dic.items():
                x0 = items[0]
                dic[m] = sorted(items, key=lambda x: _Ordered(x - x0), reverse=flip)
        return (dicts[0], dicts[1], dicts[2], dicts[3])

    @property
    def signature(self) -> tuple[int, int, int, int]:
        return (len(self.an), len(self.ap), len(self.bm), len(self.bq))


_x = Dummy('x')


def _solve_linear(e: Expr, n: Symbol) -> Expr:
    """The root of ``e``, linear in ``n`` (SymPy's ``n0, = solve(e, n)``,
    a ``ValueError`` without exactly one)."""
    e = expand(e)
    c1 = expand(e.diff(n))
    if c1 == 0 or c1.has(n):
        raise ValueError('not one solution of %s = 0 in %s' % (e, n))
    c0 = _expr(e.subs(n, ZERO))
    return -c0 / c1


# PORTED FROM SYMPY: sympy/simplify/hyperexpand.py:Formula
class Formula:
    """A hypergeometric function ``func`` in ``z`` with a basis ``B``,
    ``C*B`` its closed form and ``z*d/dz B = M*B``; ``symbols`` are the
    free parameters (from the ``res`` given, ``B``, ``C`` and ``M`` are
    computed from the differential equation)."""

    __slots__ = ('func', 'z', 'symbols', 'B', 'C', 'M')

    def __init__(self, func: HyperFunction, z: Symbol, res: Optional[Expr], symbols: Sequence[Symbol],
                 B: Optional[ExprMatrix] = None, C: Optional[ExprMatrix] = None,
                 M: Optional[ExprMatrix] = None) -> None:
        self.func = func
        self.z = z
        self.symbols: list[Symbol] = [x for x in symbols if func.has(x)]
        if res is not None:
            B, C, M = self._compute_basis(res)
        assert B is not None and C is not None and M is not None
        self.B: ExprMatrix = B
        self.C: ExprMatrix = C
        self.M: ExprMatrix = M

    # PORTED FROM SYMPY: sympy/simplify/hyperexpand.py:Formula._compute_basis
    def _compute_basis(self, closed_form: Expr) -> tuple[ExprMatrix, ExprMatrix, ExprMatrix]:
        """The basis of ``closed_form`` and its derivatives ``(z*d/dz)**k``,
        with the companion matrix of the hypergeometric equation."""
        from sympy_extras.core.mul import mul
        from sympy_extras.polys.polytools import Poly
        afactors = [_x + a for a in self.func.ap]
        bfactors = [_x + b - 1 for b in self.func.bq]
        e = _x * mul(*bfactors) - self.z * mul(*afactors)
        poly = Poly(e, _x)
        n = poly.degree() - 1
        b = [closed_form]
        for _ in range(n):
            b.append(self.z * b[-1].diff(self.z))
        B = ExprMatrix.column(b)
        C = ExprMatrix.from_rows([[ONE] + [ZERO] * n])
        m = ExprMatrix.eye(n).col_insert(0, ExprMatrix.zeros(n, 1))
        coeffs = poly.all_coeffs()
        last = list(reversed(coeffs[1:]))
        M = m.row_insert(n, -ExprMatrix.from_rows([last]) / coeffs[0])
        return B, C, M

    @property
    def closed_form(self) -> Expr:
        return _closed_form(self.C, self.B)

    # PORTED FROM SYMPY: sympy/simplify/hyperexpand.py:Formula.find_instantiations
    def find_instantiations(self, func: HyperFunction) -> list[Repl]:
        """The values of the free parameters which give a function with
        the invariants of ``func``, within the integer shifts which reach
        its parameters."""
        from sympy_extras.functions.elementary.integers import ceiling, floor
        ap, bq = func.ap, func.bq
        if len(ap) != len(self.func.ap) or len(bq) != len(self.func.bq):
            raise TypeError('Cannot instantiate other number of parameters')
        symbol_values: list[tuple[Expr, ...]] = []
        for a in self.symbols:
            if a in self.func.ap:
                symbol_values.append(ap)
            elif a in self.func.bq:
                symbol_values.append(bq)
            else:
                raise ValueError('At least one of the parameters of the formula must be equal to %s' % (a,))
        base_repl: list[Repl] = [dict(zip(self.symbols, values)) for values in product(*symbol_values)]
        abuckets, bbuckets = _sift(ap, _mod1), _sift(bq, _mod1)
        critical_values: list[list[Expr]] = [[ZERO] for _ in self.symbols]
        result: list[Repl] = []
        n_ = Dummy()
        for repl in base_repl:
            def key(x: Expr) -> Expr:
                return _mod1(_xreplace(x, repl))
            symb_a, symb_b = _sift(self.func.ap, key), _sift(self.func.bq, key)
            for bucket, obucket in [(abuckets, symb_a), (bbuckets, symb_b)]:
                for m in _keys_union(bucket, obucket):
                    if m not in bucket or m not in obucket or len(bucket[m]) != len(obucket[m]):
                        break
                    for a, vals in zip(self.symbols, critical_values):
                        if repl[a].free_symbols:
                            continue
                        exprs = [e for e in obucket[m] if e.has(a)]
                        repl0 = dict(repl)
                        repl0[a] = _expr(repl[a]) + n_
                        for e in exprs:
                            for target in bucket[m]:
                                n0 = _solve_linear(_xreplace(e, repl0) - target, n_)
                                if n0.free_symbols:
                                    raise ValueError('Value should not be true')
                                vals.append(n0)
            # SymPy's for ... else: the loop over the two buckets never
            # breaks, the instantiations are always added
            values: list[list[Expr]] = []
            for a, vals in zip(self.symbols, critical_values):
                a0 = _expr(repl[a])
                lo, hi = floor(_min(vals)), ceiling(_max(vals))
                assert isinstance(lo, Integer) and isinstance(hi, Integer)
                values.append([a0 + k for k in range(lo.p, hi.p + 1)])
            result.extend(dict(zip(self.symbols, choice)) for choice in product(*values))
        return result


def _closed_form(C: ExprMatrix, B: ExprMatrix) -> Expr:
    """``C*B`` as an expression (SymPy's ``reduce`` over the entries)."""
    total: Expr = ZERO
    for c, b in zip(C, B):
        total = total + c * b
    return total


def _has_infinities(m: ExprMatrix) -> bool:
    return any(e.has(NaN) or e.has(Infinity) or e.has(NegativeInfinity) or e.has(ComplexInfinity) for e in m)


# PORTED FROM SYMPY: sympy/simplify/hyperexpand.py:FormulaCollection
class FormulaCollection:
    """The formulas of the table: the concrete ones by their invariants,
    the ones with free parameters by their sizes."""

    def __init__(self) -> None:
        self.symbolic_formulae: dict[tuple[int, int], list[Formula]] = {}
        self.concrete_formulae: dict[tuple[int, int], dict[object, Formula]] = {}
        self.formulae: list[Formula] = []
        add_formulae(self.formulae)
        for f in self.formulae:
            sizes = f.func.sizes
            if f.symbols:
                self.symbolic_formulae.setdefault(sizes, []).append(f)
            else:
                self.concrete_formulae.setdefault(sizes, {})[f.func.build_invariants()] = f

    # PORTED FROM SYMPY: sympy/simplify/hyperexpand.py:FormulaCollection.lookup_origin
    def lookup_origin(self, func: HyperFunction) -> Optional[Formula]:
        """A formula with the invariants of ``func`` (the nearest
        instantiation of a symbolic one), ``None`` when there is none."""
        inv = func.build_invariants()
        sizes = func.sizes
        if sizes in self.concrete_formulae and inv in self.concrete_formulae[sizes]:
            return self.concrete_formulae[sizes][inv]
        if sizes not in self.symbolic_formulae:
            return None
        possible: list[tuple[Expr, Repl, Formula, HyperFunction]] = []
        for f in self.symbolic_formulae[sizes]:
            for repl in f.find_instantiations(func):
                func2 = f.func.xreplace(repl)
                if not func2._is_suitable_origin():
                    continue
                diff = func2.difficulty(func)
                if diff == -1:
                    continue
                possible.append((diff, repl, f, func2))
        possible.sort(key=lambda p: _Ordered(p[0]))
        for _, repl, f, func2 in possible:
            f2 = Formula(func2, f.z, None, [], f.B.subs(repl), f.C.subs(repl), f.M.subs(repl))
            if not any(_has_infinities(m) for m in (f2.B, f2.M, f2.C)):
                return f2
        return None


Matcher = Callable[['GFunction'], Optional[tuple[Repl, 'GFunction']]]


# PORTED FROM SYMPY: sympy/simplify/hyperexpand.py:MeijerFormula
class MeijerFormula:
    """A Meijer G-function in ``z`` with a basis, as :class:`Formula`; the
    ``matcher`` gives the values of the free parameters and the function
    of the formula which matches a given one."""

    __slots__ = ('func', 'z', 'symbols', '_matcher', 'B', 'C', 'M')

    def __init__(self, an: Sequence[ExprLike], ap: Sequence[ExprLike], bm: Sequence[ExprLike],
                 bq: Sequence[ExprLike], z: Symbol, symbols: Sequence[Symbol], B: ExprMatrix, C: ExprMatrix,
                 M: ExprMatrix, matcher: Optional[Matcher]) -> None:
        self.func = GFunction(an, ap, bm, bq)
        self.z = z
        self.symbols = list(symbols)
        self._matcher = matcher
        self.B = B
        self.C = C
        self.M = M

    @property
    def closed_form(self) -> Expr:
        return _closed_form(self.C, self.B)

    # PORTED FROM SYMPY: sympy/simplify/hyperexpand.py:MeijerFormula.try_instantiate
    def try_instantiate(self, func: GFunction) -> Optional[MeijerFormula]:
        """The formula instantiated for a function of the same signature
        which its matcher accepts."""
        if func.signature != self.func.signature or self._matcher is None:
            return None
        res = self._matcher(func)
        if res is None:
            return None
        subs, newfunc = res
        return MeijerFormula(newfunc.an, newfunc.ap, newfunc.bm, newfunc.bq, self.z, [], self.B.subs(subs),
                             self.C.subs(subs), self.M.subs(subs), None)


# PORTED FROM SYMPY: sympy/simplify/hyperexpand.py:MeijerFormulaCollection
class MeijerFormulaCollection:
    """The Meijer G formulas by signature."""

    def __init__(self) -> None:
        formulae: list[MeijerFormula] = []
        add_meijerg_formulae(formulae)
        self.formulae: dict[tuple[int, int, int, int], list[MeijerFormula]] = {}
        for formula in formulae:
            self.formulae.setdefault(formula.func.signature, []).append(formula)

    def lookup_origin(self, func: GFunction) -> Optional[MeijerFormula]:
        for formula in self.formulae.get(func.signature, []):
            res = formula.try_instantiate(func)
            if res is not None:
                return res
        return None


def _rows(*rows: Sequence[ExprLike]) -> ExprMatrix:
    return ExprMatrix.from_rows(rows)


def _col(*entries: ExprLike) -> ExprMatrix:
    return ExprMatrix.column(entries)


# PORTED FROM SYMPY: sympy/simplify/hyperexpand.py:add_formulae
def add_formulae(formulae: list[Formula]) -> None:
    """The table of hypergeometric formulas (Luke; Prudnikov, Brychkov
    and Marichev; functions.wolfram.com)."""
    from sympy_extras.functions.elementary.complexes import polar_lift
    from sympy_extras.functions.elementary.exponential import exp, exp_polar, log
    from sympy_extras.functions.elementary.hyperbolic import cosh, sinh
    from sympy_extras.functions.elementary.trigonometric import sin
    from sympy_extras.functions.special.bessel import besseli, besselj
    from sympy_extras.functions.special.elliptic_integrals import elliptic_e, elliptic_k
    from sympy_extras.functions.special.error_functions import Chi, Ei, Shi, erf, expint, fresnelc, fresnels
    from sympy_extras.functions.special.gamma_functions import gamma, lowergamma, uppergamma
    a, b, c, z = Dummy('a'), Dummy('b'), Dummy('c'), Dummy('z')
    abc = (a, b, c)

    def add(ap: Sequence[ExprLike], bq: Sequence[ExprLike], res: Expr) -> None:
        formulae.append(Formula(HyperFunction(ap, bq), z, res, abc))

    def addb(ap: Sequence[ExprLike], bq: Sequence[ExprLike], B: ExprMatrix, C: ExprMatrix, M: ExprMatrix) -> None:
        formulae.append(Formula(HyperFunction(ap, bq), z, None, abc, B, C, M))

    # Luke, Y. L. (1969), The Special Functions and Their Approximations,
    # Volume 1, section 6.2
    # 0F0
    add((), (), exp(z))
    # 1F0
    add((a,), (), HyperRep_power1(-a, z))
    # 2F1
    addb((a, a - HALF), (2*a,),
         _col(HyperRep_power2(a, z), HyperRep_power2(a + HALF, z)/2),
         _rows([1, 0]),
         _rows([(a - HALF)*z/(1 - z), (HALF - a)*z/(1 - z)],
               [a/(1 - z), a*(z - 2)/(1 - z)]))
    addb((1, 1), (2,),
         _col(HyperRep_log1(z), 1), _rows([-1/z, 0]),
         _rows([0, z/(z - 1)], [0, 0]))
    addb((HALF, 1), (Rational(3, 2),),
         _col(HyperRep_atanh(z), 1),
         _rows([1, 0]),
         _rows([Rational(-1, 2), 1/(1 - z)/2], [0, 0]))
    addb((HALF, HALF), (Rational(3, 2),),
         _col(HyperRep_asin1(z), HyperRep_power1(Rational(-1, 2), z)),
         _rows([1, 0]),
         _rows([Rational(-1, 2), HALF], [0, z/(1 - z)/2]))
    addb((a, HALF + a), (HALF,),
         _col(HyperRep_sqrts1(-a, z), -HyperRep_sqrts2(-a - HALF, z)),
         _rows([1, 0]),
         _rows([0, -a],
               [z*(-2*a - 1)/2/(1 - z), HALF - z*(-2*a - 1)/(1 - z)]))
    # A. P. Prudnikov, Yu. A. Brychkov and O. I. Marichev (1990).
    # Integrals and Series: More Special Functions, Vol. 3,.
    # Gordon and Breach Science Publisher
    addb([a, -a], [HALF],
         _col(HyperRep_cosasin(a, z), HyperRep_sinasin(a, z)),
         _rows([1, 0]),
         _rows([0, -a], [a*z/(1 - z), 1/(1 - z)/2]))
    addb([1, 1], [3*HALF],
         _col(HyperRep_asin2(z), 1), _rows([1, 0]),
         _rows([(z - HALF)/(1 - z), 1/(1 - z)/2], [0, 0]))
    # Complete elliptic integrals K(z) and E(z), both a 2F1 function
    addb([HALF, HALF], [ONE],
         _col(elliptic_k(z), elliptic_e(z)),
         _rows([2/pi, 0]),
         _rows([Rational(-1, 2), -1/(2*z - 2)],
               [Rational(-1, 2), HALF]))
    addb([Rational(-1, 2), HALF], [ONE],
         _col(elliptic_k(z), elliptic_e(z)),
         _rows([0, 2/pi]),
         _rows([Rational(-1, 2), -1/(2*z - 2)],
               [Rational(-1, 2), HALF]))
    # 3F2
    addb([Rational(-1, 2), 1, 1], [HALF, 2],
         _col(z*HyperRep_atanh(z), HyperRep_log1(z), 1),
         _rows([Rational(-2, 3), -ONE/(3*z), Rational(2, 3)]),
         _rows([HALF, 0, z/(1 - z)/2],
               [0, 0, z/(z - 1)],
               [0, 0, 0]))
    # actually the formula for 3/2 is much nicer ...
    addb([Rational(-1, 2), 1, 1], [2, 2],
         _col(HyperRep_power1(HALF, z), HyperRep_log2(z), 1),
         _rows([Rational(4, 9) - 16/(9*z), 4/(3*z), 16/(9*z)]),
         _rows([z/2/(z - 1), 0, 0], [1/(2*(z - 1)), 0, HALF], [0, 0, 0]))
    # 1F1
    addb([1], [b], _col(z**(1 - b) * exp(z) * lowergamma(b - 1, z), 1),
         _rows([b - 1, 0]), _rows([1 - b + z, 1], [0, 0]))
    addb([a], [2*a],
         _col(z**(HALF - a)*exp(z/2)*besseli(a - HALF, z/2) * gamma(a + HALF)/4**(HALF - a),
              z**(HALF - a)*exp(z/2)*besseli(a + HALF, z/2) * gamma(a + HALF)/4**(HALF - a)),
         _rows([1, 0]),
         _rows([z/2, z/2], [z/2, (z/2 - 2*a)]))
    mz = polar_lift(Integer(-1))*z
    addb([a], [a + 1],
         _col(mz**(-a)*a*lowergamma(a, mz), a*exp(z)),
         _rows([1, 0]),
         _rows([-a, 1], [0, z]))
    # This one is redundant.
    add([Rational(-1, 2)], [HALF], exp(z) - sqrt(pi*z)*(-I)*erf(I*sqrt(z)))
    # Added to get nice results for Laplace transform of Fresnel functions
    # https://functions.wolfram.com/07.22.03.6437.01
    # Manually tuned rule
    addb([1], [Rational(3, 4), Rational(5, 4)],
         _col(sqrt(pi)*(I*sinh(2*sqrt(z))*fresnels(2*root(z, 4)*exp(I*pi/4)/sqrt(pi))
                        + cosh(2*sqrt(z))*fresnelc(2*root(z, 4)*exp(I*pi/4)/sqrt(pi)))
              * exp(-I*pi/4)/(2*root(z, 4)),
              sqrt(pi)*root(z, 4)*(sinh(2*sqrt(z))*fresnelc(2*root(z, 4)*exp(I*pi/4)/sqrt(pi))
                                   + I*cosh(2*sqrt(z))*fresnels(2*root(z, 4)*exp(I*pi/4)/sqrt(pi)))
              * exp(-I*pi/4)/2,
              1),
         _rows([1, 0, 0]),
         _rows([Rational(-1, 4), 1, Rational(1, 4)],
               [z, Rational(1, 4), 0],
               [0, 0, 0]))
    # 2F2
    addb([HALF, a], [Rational(3, 2), a + 1],
         _col(a/(2*a - 1)*(-I)*sqrt(pi/z)*erf(I*sqrt(z)),
              a/(2*a - 1)*(polar_lift(Integer(-1))*z)**(-a) * lowergamma(a, polar_lift(Integer(-1))*z),
              a/(2*a - 1)*exp(z)),
         _rows([1, -1, 0]),
         _rows([Rational(-1, 2), 0, 1], [0, -a, 1], [0, 0, z]))
    # We make a "basis" of four functions instead of three, and give EulerGamma
    # an extra slot (it could just be a coefficient to 1). The advantage is
    # that this way Polys will not see multivariate polynomials (it treats
    # EulerGamma as an indeterminate), which is *way* faster.
    addb([1, 1], [2, 2],
         _col(Ei(z) - log(z), exp(z), 1, EulerGamma),
         _rows([1/z, 0, 0, -1/z]),
         _rows([0, 1, -1, 0], [0, z, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]))
    # 0F1
    add((), (HALF,), cosh(2*sqrt(z)))
    addb([], [b],
         _col(gamma(b)*z**((1 - b)/2)*besseli(b - 1, 2*sqrt(z)),
              gamma(b)*z**(1 - b/2)*besseli(b, 2*sqrt(z))),
         _rows([1, 0]), _rows([0, 1], [z, (1 - b)]))
    # 0F3
    x = 4*z**Rational(1, 4)

    def fp(order: Expr) -> Expr:
        return besseli(order, x) + besselj(order, x)

    def fm(order: Expr) -> Expr:
        return besseli(order, x) - besselj(order, x)

    # TODO branching
    addb([], [HALF, a, a + HALF],
         _col(fp(2*a - 1), fm(2*a)*z**Rational(1, 4), fm(2*a - 1)*sqrt(z), fp(2*a)*z**Rational(3, 4))
         * (2**(-2*a)*gamma(2*a)*z**((1 - 2*a)/4)),
         _rows([1, 0, 0, 0]),
         _rows([0, 1, 0, 0],
               [0, HALF - a, 1, 0],
               [0, 0, HALF, 1],
               [z, 0, 0, 1 - a]))
    x = 2*(4*z)**Rational(1, 4)*exp_polar(I*pi/4)
    addb([], [a, a + HALF, 2*a],
         _col(besselj(2*a - 1, x)*besseli(2*a - 1, x),
              x*(besseli(2*a, x)*besselj(2*a - 1, x) - besseli(2*a - 1, x)*besselj(2*a, x)),
              x**2*besseli(2*a, x)*besselj(2*a, x),
              x**3*(besseli(2*a, x)*besselj(2*a - 1, x) + besseli(2*a - 1, x)*besselj(2*a, x)))
         * ((2*sqrt(polar_lift(Integer(-1))*z))**(1 - 2*a)*gamma(2*a)**2),
         _rows([1, 0, 0, 0]),
         _rows([0, Rational(1, 4), 0, 0],
               [0, (1 - 2*a)/2, Rational(-1, 2), 0],
               [0, 0, 1 - 2*a, Rational(1, 4)],
               [-32*z, 0, 0, 1 - a]))
    # 1F2
    addb([a], [a - HALF, 2*a],
         _col(z**(HALF - a)*besseli(a - HALF, sqrt(z))**2,
              z**(1 - a)*besseli(a - HALF, sqrt(z)) * besseli(a - Rational(3, 2), sqrt(z)),
              z**(Rational(3, 2) - a)*besseli(a - Rational(3, 2), sqrt(z))**2),
         _rows([-gamma(a + HALF)**2/4**(HALF - a), 2*gamma(a - HALF)*gamma(a + HALF)/4**(1 - a), 0]),
         _rows([1 - 2*a, 1, 0], [z/2, HALF - a, HALF], [0, z, 0]))
    addb([HALF], [b, 2 - b],
         _col(besseli(1 - b, sqrt(z))*besseli(b - 1, sqrt(z)),
              sqrt(z)*(besseli(-b, sqrt(z))*besseli(b - 1, sqrt(z)) + besseli(1 - b, sqrt(z))*besseli(b, sqrt(z))),
              besseli(-b, sqrt(z))*besseli(b, sqrt(z)))
         * (pi*(1 - b)/sin(pi*b)),
         _rows([1, 0, 0]),
         _rows([b - 1, HALF, 0],
               [z, 0, z],
               [0, HALF, -b]))
    addb([HALF], [Rational(3, 2), Rational(3, 2)],
         _col(Shi(2*sqrt(z))/2/sqrt(z), sinh(2*sqrt(z))/2/sqrt(z), cosh(2*sqrt(z))),
         _rows([1, 0, 0]),
         _rows([Rational(-1, 2), HALF, 0], [0, Rational(-1, 2), HALF], [0, 2*z, 0]))
    # FresnelS, manually tuned rule
    addb([Rational(3, 4)], [Rational(3, 2), Rational(7, 4)],
         _col(fresnels(exp(pi*I/4)*root(z, 4)*2/sqrt(pi)) / (pi * (exp(pi*I/4)*root(z, 4)*2/sqrt(pi))**3),
              sinh(2*sqrt(z))/sqrt(z),
              cosh(2*sqrt(z))),
         _rows([6, 0, 0]),
         _rows([Rational(-3, 4), Rational(1, 16), 0],
               [0, Rational(-1, 2), 1],
               [0, z, 0]))
    # FresnelC, manually tuned rule
    addb([Rational(1, 4)], [HALF, Rational(5, 4)],
         _col(sqrt(pi)*exp(-I*pi/4)*fresnelc(2*root(z, 4)*exp(I*pi/4)/sqrt(pi))/(2*root(z, 4)),
              cosh(2*sqrt(z)),
              sinh(2*sqrt(z))*sqrt(z)),
         _rows([1, 0, 0]),
         _rows([Rational(-1, 4), Rational(1, 4), 0],
               [0, 0, 1],
               [0, z, HALF]))
    # 2F3
    # XXX with this five-parameter formula is pretty slow with the current
    #     Formula.find_instantiations (creates 2!*3!*3**(2+3) ~ 3000
    #     instantiations ... But it's not too bad.
    addb([a, a + HALF], [2*a, b, 2*a - b + 1],
         _col(besseli(b - 1, sqrt(z))*besseli(2*a - b, sqrt(z)),
              sqrt(z)*besseli(b, sqrt(z))*besseli(2*a - b, sqrt(z)),
              sqrt(z)*besseli(b - 1, sqrt(z))*besseli(2*a - b + 1, sqrt(z)),
              besseli(b, sqrt(z))*besseli(2*a - b + 1, sqrt(z)))
         * (gamma(b)*gamma(2*a - b + 1) * (sqrt(z)/2)**(1 - 2*a)),
         _rows([1, 0, 0, 0]),
         _rows([0, HALF, HALF, 0],
               [z/2, 1 - b, 0, z/2],
               [z/2, 0, b - 2*a, z/2],
               [0, HALF, HALF, -2*a]))
    # (C/f above comment about eulergamma in the basis).
    addb([1, 1], [2, 2, Rational(3, 2)],
         _col(Chi(2*sqrt(z)) - log(2*sqrt(z)), cosh(2*sqrt(z)), sqrt(z)*sinh(2*sqrt(z)), 1, EulerGamma),
         _rows([1/z, 0, 0, 0, -1/z]),
         _rows([0, HALF, 0, Rational(-1, 2), 0],
               [0, 0, 1, 0, 0],
               [0, z, HALF, 0, 0],
               [0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0]))
    # 3F3
    # This is rule: https://functions.wolfram.com/07.31.03.0134.01
    # Initial reason to add it was a nice solution for
    # integrate(erf(a*z)/z**2, z) and same for erfc and erfi.
    # Manually tuned rule
    addb([1, 1, a], [2, 2, a + 1],
         _col(a*(log(-z) + expint(1, -z) + EulerGamma)/(z*(a**2 - 2*a + 1)),
              a*(-z)**(-a)*(gamma(a) - uppergamma(a, -z))/(a - 1)**2,
              a*exp(z)/(a**2 - 2*a + 1),
              a/(z*(a**2 - 2*a + 1))),
         _rows([1 - a, 1, -1/z, 1]),
         _rows([-1, 0, -1/z, 1],
               [0, -a, 1, 0],
               [0, 0, z, 0],
               [0, 0, 0, -1]))


# PORTED FROM SYMPY: sympy/simplify/hyperexpand.py:add_meijerg_formulae
def add_meijerg_formulae(formulae: list[MeijerFormula]) -> None:
    """The table of Meijer G formulas (the upper incomplete gamma
    function, sine and cosine integrals)."""
    from sympy_extras.functions.elementary.exponential import exp
    from sympy_extras.functions.elementary.trigonometric import cos, sin
    from sympy_extras.functions.special.error_functions import Ci, Si
    from sympy_extras.functions.special.gamma_functions import gamma, uppergamma
    from .simplify import simplify
    a, b, c, z = Dummy('a'), Dummy('b'), Dummy('c'), Dummy('z')
    rho = Dummy('rho')

    def add(an: Sequence[ExprLike], ap: Sequence[ExprLike], bm: Sequence[ExprLike], bq: Sequence[ExprLike],
            B: ExprMatrix, C: ExprMatrix, M: ExprMatrix, matcher: Matcher) -> None:
        formulae.append(MeijerFormula(an, ap, bm, bq, z, [a, b, c, rho], B, C, M, matcher))

    def detect_uppergamma(func: GFunction) -> Optional[tuple[Repl, GFunction]]:
        x = func.an[0]
        y, w = func.bm
        swapped = False
        if _is_zero(_mod1(simplify(x - y))):
            swapped = True
            (y, w) = (w, y)
        if not _is_zero(_mod1(simplify(x - w))) or _truth(x - w > 0):
            return None
        bm = [y, x]
        if swapped:
            bm = [x, y]
        return {rho: y, a: x - y}, GFunction([x], [], bm, [])

    add([a + rho], [], [rho, a + rho], [],
        _col(gamma(1 - a)*z**rho*exp(z)*uppergamma(a, z), gamma(1 - a)*z**(a + rho)),
        _rows([1, 0]),
        _rows([rho + z, -1], [0, a + rho]),
        detect_uppergamma)

    def detect_3113(func: GFunction) -> Optional[tuple[Repl, GFunction]]:
        """https://functions.wolfram.com/07.34.03.0984.01"""
        x = func.an[0]
        u, v, w = func.bm
        sig: tuple[Expr, Expr, Expr]
        if _mod1(simplify(u - v)) == 0:
            if _mod1(simplify(v - w)) == 0:
                return None
            sig = (HALF, HALF, ZERO)
            x1, x2, y = u, v, w
        else:
            if _mod1(simplify(x - u)) == 0:
                sig = (HALF, ZERO, HALF)
                x1, y, x2 = u, v, w
            else:
                sig = (ZERO, HALF, HALF)
                y, x1, x2 = u, v, w
        if (_mod1(simplify(x - x1)) != 0 or _mod1(simplify(x - x2)) != 0 or _mod1(simplify(x - y)) != HALF
                or _truth(x - x1 > 0) or _truth(x - x2 > 0)):
            return None
        return {a: x}, GFunction([x], [], [x - HALF + t for t in sig], [])

    s = sin(2*sqrt(z))
    c_ = cos(2*sqrt(z))
    S_ = Si(2*sqrt(z)) - pi/2
    C = Ci(2*sqrt(z))
    add([a], [], [a, a, a - HALF], [],
        _col(sqrt(pi)*z**(a - HALF)*(c_*S_ - s*C), sqrt(pi)*z**a*(s*S_ + c_*C), sqrt(pi)*z**a),
        _rows([-2, 0, 0]),
        _rows([a - HALF, -1, 0], [z, a, HALF], [0, 0, a]),
        detect_3113)


# ---------------------------------------------------------------------
# the operators: polynomials in z*d/dz acting on the basis


def _poly_coeffs(e: Expr, x: Symbol) -> list[Expr]:
    """The coefficients of ``e``, a polynomial in ``x``, the leading one
    first (SymPy's ``Poly(e, x).all_coeffs()``), each cancelled as an
    element of a fraction field of the parameters is."""
    from sympy_extras.core.add import Add, add
    from sympy_extras.polys.polyerrors import PolynomialError
    from sympy_extras.polys.polytools import cancel
    by_degree: dict[int, list[Expr]] = {}
    for t in (expand(e).args if isinstance(expand(e), Add) else (expand(e),)):
        assert isinstance(t, Expr)
        c, k = t.as_coeff_exponent(x)
        if not isinstance(k, Integer) or k.p < 0 or c.has(x):
            raise PolynomialError('%s is not a polynomial in %s' % (e, x))
        by_degree.setdefault(k.p, []).append(c)
    degree = max(by_degree) if by_degree else 0
    return [cancel(add(*by_degree.get(k, []))) for k in range(degree, -1, -1)]


def _from_coeffs(coeffs: Sequence[Expr], x: Expr) -> Expr:
    """``Poly(coeffs, x).as_expr()`` evaluated at ``x``."""
    from sympy_extras.core.add import add
    n = len(coeffs) - 1
    return add(*[c * x**(n - k) for k, c in enumerate(coeffs)])


# PORTED FROM SYMPY: sympy/simplify/hyperexpand.py:Operator
class Operator:
    """A differential operator ``sum c_k*D**k`` with ``D = z*d/dz``, as the
    coefficients of the polynomial in ``D``, the leading one first."""

    __slots__ = ('coeffs',)

    def __init__(self, coeffs: Sequence[Expr]) -> None:
        self.coeffs: tuple[Expr, ...] = tuple(coeffs)

    # PORTED FROM SYMPY: sympy/simplify/hyperexpand.py:Operator.apply
    def apply(self, obj: Expr, op: Callable[[Expr], Expr]) -> Expr:
        """The operator applied to ``obj``, ``op`` being ``D``."""
        coeffs = list(reversed(self.coeffs))
        diffs = [obj]
        for _ in coeffs[1:]:
            diffs.append(op(diffs[-1]))
        r = coeffs[0] * diffs[0]
        for c, d in zip(coeffs[1:], diffs[1:]):
            r = r + c * d
        return r

    def apply_matrix(self, obj: ExprMatrix, op: Callable[[ExprMatrix], ExprMatrix]) -> ExprMatrix:
        """:meth:`apply` for a matrix of functions."""
        coeffs = list(reversed(self.coeffs))
        diffs = [obj]
        for _ in coeffs[1:]:
            diffs.append(op(diffs[-1]))
        r = diffs[0] * coeffs[0]
        for c, d in zip(coeffs[1:], diffs[1:]):
            r = r + d * c
        return r


def _op_of(e: Expr) -> list[Expr]:
    return _poly_coeffs(e, _x)


# PORTED FROM SYMPY: sympy/simplify/hyperexpand.py:MultOperator
class MultOperator(Operator):
    """The operator of a polynomial in ``D``."""

    __slots__ = ()

    def __init__(self, p: Expr) -> None:
        Operator.__init__(self, _op_of(p))


# PORTED FROM SYMPY: sympy/simplify/hyperexpand.py:ShiftA
class ShiftA(Operator):
    """Increment an upper index ``ai``: ``D/ai + 1``."""

    __slots__ = ()

    def __init__(self, ai: Expr) -> None:
        if ai == 0:
            raise ValueError('Cannot increment zero upper index.')
        Operator.__init__(self, _op_of(_x/ai + 1))

    def __str__(self) -> str:
        return '<Increment upper %s.>' % (1/self.coeffs[0])


# PORTED FROM SYMPY: sympy/simplify/hyperexpand.py:ShiftB
class ShiftB(Operator):
    """Decrement a lower index ``bi``: ``D/(bi - 1) + 1``."""

    __slots__ = ()

    def __init__(self, bi: Expr) -> None:
        if bi == 1:
            raise ValueError('Cannot decrement unit lower index.')
        Operator.__init__(self, _op_of(_x/(bi - 1) + 1))

    def __str__(self) -> str:
        return '<Decrement lower %s.>' % (1/self.coeffs[0] + 1)


def _unshift(m: Expr, n: Expr, A: Symbol, value: Expr, b0_sign: int, minus_n: bool) -> list[Expr]:
    """The common end of the unshift operators: ``n`` (a polynomial in
    ``A``) without its constant term divided by ``A`` and evaluated at
    ``value``; the operator ``(n - m)/b0`` (``(m - n)/b0`` when
    ``minus_n``), ``b0`` being the constant term times ``b0_sign``."""
    coeffs = _poly_coeffs(n, A)
    b0 = b0_sign * coeffs[-1]
    if b0 == 0:
        raise ValueError('cancels')
    nx = _from_coeffs(coeffs[:-1], value) if len(coeffs) > 1 else ZERO
    return _op_of(((m - nx) if minus_n else (nx - m)) / b0)


# PORTED FROM SYMPY: sympy/simplify/hyperexpand.py:UnShiftA
class UnShiftA(Operator):
    """Decrement the upper index ``#i``."""

    __slots__ = ('_ap', '_bq', '_i')

    def __init__(self, ap: Sequence[Expr], bq: Sequence[Expr], i: int, z: Expr) -> None:
        from sympy_extras.core.mul import mul
        self._ap, self._bq, self._i = tuple(ap), tuple(bq), i
        aps = list(ap)
        ai = aps.pop(i) - 1
        if ai == 0:
            raise ValueError('Cannot decrement unit upper index.')
        m = z * ai * mul(*[_x + a for a in aps])
        A = Dummy('A')
        D = ai*A - ai
        n = D * mul(*[D + (b - 1) for b in bq])
        try:
            coeffs = _unshift(m, n, A, _x/ai + 1, -1, False)
        except ValueError:
            raise ValueError('Cannot decrement upper index: cancels with lower')
        Operator.__init__(self, coeffs)

    def __str__(self) -> str:
        return '<Decrement upper index #%s of %s, %s.>' % (self._i, self._ap, self._bq)


# PORTED FROM SYMPY: sympy/simplify/hyperexpand.py:UnShiftB
class UnShiftB(Operator):
    """Increment the lower index ``#i``."""

    __slots__ = ('_ap', '_bq', '_i')

    def __init__(self, ap: Sequence[Expr], bq: Sequence[Expr], i: int, z: Expr) -> None:
        from sympy_extras.core.mul import mul
        self._ap, self._bq, self._i = tuple(ap), tuple(bq), i
        bqs = list(bq)
        bi = bqs.pop(i) + 1
        if bi == 0:
            raise ValueError('Cannot increment -1 lower index.')
        m = _x * (bi - 1) * mul(*[_x + b - 1 for b in bqs])
        B = Dummy('B')
        D = (bi - 1)*B - bi + 1
        n = z * mul(*[D + a for a in ap])
        try:
            coeffs = _unshift(m, n, B, _x/(bi - 1) + 1, 1, True)
        except ValueError:
            raise ValueError('Cannot increment index: cancels with upper')
        Operator.__init__(self, coeffs)

    def __str__(self) -> str:
        return '<Increment lower index #%s of %s, %s.>' % (self._i, self._ap, self._bq)


# PORTED FROM SYMPY: sympy/simplify/hyperexpand.py:MeijerShiftA
class MeijerShiftA(Operator):
    """Increment an upper ``b``: ``bi - D``."""

    __slots__ = ()

    def __init__(self, bi: Expr) -> None:
        Operator.__init__(self, _op_of(bi - _x))

    def __str__(self) -> str:
        return '<Increment upper b=%s.>' % (self.coeffs[1],)


# PORTED FROM SYMPY: sympy/simplify/hyperexpand.py:MeijerShiftB
class MeijerShiftB(Operator):
    """Decrement an upper ``a``: ``1 - bi + D``."""

    __slots__ = ()

    def __init__(self, bi: Expr) -> None:
        Operator.__init__(self, _op_of(1 - bi + _x))

    def __str__(self) -> str:
        return '<Decrement upper a=%s.>' % (1 - self.coeffs[1],)


# PORTED FROM SYMPY: sympy/simplify/hyperexpand.py:MeijerShiftC
class MeijerShiftC(Operator):
    """Increment a lower ``b``: ``D - bi``."""

    __slots__ = ()

    def __init__(self, bi: Expr) -> None:
        Operator.__init__(self, _op_of(-bi + _x))

    def __str__(self) -> str:
        return '<Increment lower b=%s.>' % (-self.coeffs[1],)


# PORTED FROM SYMPY: sympy/simplify/hyperexpand.py:MeijerShiftD
class MeijerShiftD(Operator):
    """Decrement a lower ``a``: ``bi - 1 - D``."""

    __slots__ = ()

    def __init__(self, bi: Expr) -> None:
        Operator.__init__(self, _op_of(bi - 1 - _x))

    def __str__(self) -> str:
        return '<Decrement lower a=%s.>' % (self.coeffs[1] + 1,)


class _MeijerUnShift(Operator):
    __slots__ = ('_an', '_ap', '_bm', '_bq', '_i')

    def _keep(self, an: Sequence[Expr], ap: Sequence[Expr], bm: Sequence[Expr], bq: Sequence[Expr], i: int) -> None:
        self._an, self._ap, self._bm, self._bq, self._i = tuple(an), tuple(ap), tuple(bm), tuple(bq), i

    def _describe(self, what: str) -> str:
        return '<%s #%s of %s, %s, %s, %s.>' % (what, self._i, self._an, self._ap, self._bm, self._bq)


# PORTED FROM SYMPY: sympy/simplify/hyperexpand.py:MeijerUnShiftA
class MeijerUnShiftA(_MeijerUnShift):
    """Decrement the upper ``b`` index ``#i``."""

    __slots__ = ()

    def __init__(self, an: Sequence[Expr], ap: Sequence[Expr], bm: Sequence[Expr], bq: Sequence[Expr], i: int,
                 z: Expr) -> None:
        from sympy_extras.core.mul import mul
        self._keep(an, ap, bm, bq, i)
        bms = list(bm)
        bi = bms.pop(i) - 1
        m = mul(*[b - _x for b in bms]) * mul(*[_x - b for b in bq])
        A = Dummy('A')
        D = bi - A
        n = z * mul(*[D + 1 - a for a in an]) * mul(*[-D + a - 1 for a in ap])
        try:
            coeffs = _unshift(m, n, A, bi - _x, 1, True)
        except ValueError:
            raise ValueError('Cannot decrement upper b index (cancels)')
        Operator.__init__(self, coeffs)

    def __str__(self) -> str:
        return self._describe('Decrement upper b index')


# PORTED FROM SYMPY: sympy/simplify/hyperexpand.py:MeijerUnShiftB
class MeijerUnShiftB(_MeijerUnShift):
    """Increment the upper ``a`` index ``#i``."""

    __slots__ = ()

    def __init__(self, an: Sequence[Expr], ap: Sequence[Expr], bm: Sequence[Expr], bq: Sequence[Expr], i: int,
                 z: Expr) -> None:
        from sympy_extras.core.mul import mul
        self._keep(an, ap, bm, bq, i)
        ans = list(an)
        ai = ans.pop(i) + 1
        m = z * mul(*[1 - a + _x for a in ans]) * mul(*[a - 1 - _x for a in ap])
        B = Dummy('B')
        D = B + ai - 1
        n = mul(*[-D + b for b in bm]) * mul(*[D - b for b in bq])
        try:
            coeffs = _unshift(m, n, B, 1 - ai + _x, 1, True)
        except ValueError:
            raise ValueError('Cannot increment upper a index (cancels)')
        Operator.__init__(self, coeffs)

    def __str__(self) -> str:
        return self._describe('Increment upper a index')


# PORTED FROM SYMPY: sympy/simplify/hyperexpand.py:MeijerUnShiftC
class MeijerUnShiftC(_MeijerUnShift):
    """Decrement the lower ``b`` index ``#i``."""

    __slots__ = ()

    def __init__(self, an: Sequence[Expr], ap: Sequence[Expr], bm: Sequence[Expr], bq: Sequence[Expr], i: int,
                 z: Expr) -> None:
        from sympy_extras.core.mul import mul
        self._keep(an, ap, bm, bq, i)
        bqs = list(bq)
        bi = bqs.pop(i) - 1
        m = mul(*[b - _x for b in bm]) * mul(*[_x - b for b in bqs])
        C = Dummy('C')
        D = bi + C
        n = z * mul(*[D + 1 - a for a in an]) * mul(*[-D + a - 1 for a in ap])
        try:
            coeffs = _unshift(m, n, C, _x - bi, 1, True)
        except ValueError:
            raise ValueError('Cannot decrement lower b index (cancels)')
        Operator.__init__(self, coeffs)

    def __str__(self) -> str:
        return self._describe('Decrement lower b index')


# PORTED FROM SYMPY: sympy/simplify/hyperexpand.py:MeijerUnShiftD
class MeijerUnShiftD(_MeijerUnShift):
    """Increment the lower ``a`` index ``#i``."""

    __slots__ = ()

    def __init__(self, an: Sequence[Expr], ap: Sequence[Expr], bm: Sequence[Expr], bq: Sequence[Expr], i: int,
                 z: Expr) -> None:
        from sympy_extras.core.mul import mul
        self._keep(an, ap, bm, bq, i)
        aps = list(ap)
        ai = aps.pop(i) + 1
        m = z * mul(*[1 - a + _x for a in an]) * mul(*[a - 1 - _x for a in aps])
        B = Dummy('B')
        D = ai - 1 - B
        n = mul(*[-D + b for b in bm]) * mul(*[D - b for b in bq])
        try:
            coeffs = _unshift(m, n, B, ai - 1 - _x, 1, True)
        except ValueError:
            raise ValueError('Cannot increment lower a index (cancels)')
        Operator.__init__(self, coeffs)

    def __str__(self) -> str:
        return self._describe('Increment lower a index')


# PORTED FROM SYMPY: sympy/simplify/hyperexpand.py:ReduceOrder
class ReduceOrder(Operator):
    """Cancel an upper ``ai`` with a lower ``bj``, ``ai - bj`` a
    nonnegative integer (SymPy's constructor returns ``None`` otherwise:
    :meth:`make`)."""

    __slots__ = ('_a', '_b')

    def __init__(self, coeffs: Sequence[Expr], a: Expr, b: Expr) -> None:
        Operator.__init__(self, coeffs)
        self._a = a
        self._b = b

    @classmethod
    def make(cls, ai: Expr, bj: Expr) -> Optional[ReduceOrder]:
        n = ai - bj
        if not isinstance(n, Integer) or n.p < 0:
            return None
        if bj.is_integer is True and bj.is_nonpositive is True:
            return None
        p: Expr = ONE
        for k in range(n.p):
            p = p * (_x + bj + k)/(bj + k)
        return cls(_op_of(p), ai, bj)

    # PORTED FROM SYMPY: sympy/simplify/hyperexpand.py:ReduceOrder._meijer
    @classmethod
    def _meijer(cls, b: Expr, a: Expr, sign: int) -> Optional[ReduceOrder]:
        n = b - a
        if n.is_negative is True or not isinstance(n, Integer):
            return None
        p: Expr = ONE
        for k in range(n.p):
            p = p * (sign*_x + a + k)
        # SymPy keeps 1 + (a - 1) unevaluated for the description when
        # sign is 1; the values are b and a in both cases
        return cls(_op_of(p), b, a)

    @classmethod
    def meijer_minus(cls, b: Expr, a: Expr) -> Optional[ReduceOrder]:
        return cls._meijer(b, a, -1)

    @classmethod
    def meijer_plus(cls, a: Expr, b: Expr) -> Optional[ReduceOrder]:
        return cls._meijer(1 - a, 1 - b, 1)

    def __str__(self) -> str:
        return '<Reduce order by cancelling upper %s with lower %s.>' % (self._a, self._b)


# PORTED FROM SYMPY: sympy/simplify/hyperexpand.py:_reduce_order
def _reduce_order(ap: Sequence[Expr], bq: Sequence[Expr], gen: Callable[[Expr, Expr], Optional[ReduceOrder]],
                  key: Callable[[Expr], SortKey]) -> tuple[list[Expr], list[Expr], list[ReduceOrder]]:
    """The parameters left after cancelling every ``a`` with the first
    ``b`` which ``gen`` accepts, with the operators."""
    aps = sorted(ap, key=key)
    bqs = sorted(bq, key=key)
    nap: list[Expr] = []
    operators: list[ReduceOrder] = []
    for a in aps:
        op: Optional[ReduceOrder] = None
        for i in range(len(bqs)):
            op = gen(a, bqs[i])
            if op is not None:
                bqs.pop(i)
                break
        if op is None:
            nap.append(a)
        else:
            operators.append(op)
    return nap, bqs, operators


# PORTED FROM SYMPY: sympy/simplify/hyperexpand.py:reduce_order
def reduce_order(func: HyperFunction) -> tuple[HyperFunction, list[ReduceOrder]]:
    """The function with the parameters ``a``, ``b`` which differ by a
    nonnegative integer cancelled, with the operators which give the
    original from it.

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.simplify.hyperexpand import reduce_order, HyperFunction
    >>> a, b = Symbol('a'), Symbol('b')
    >>> reduce_order(HyperFunction((a + 1, 2, 3), (a, 1)))[0]
    HyperFunction((3,), ())
    """
    nap, nbq, operators = _reduce_order(func.ap, func.bq, ReduceOrder.make, default_sort_key)
    return HyperFunction(nap, nbq), operators


# PORTED FROM SYMPY: sympy/simplify/hyperexpand.py:reduce_order_meijer
def reduce_order_meijer(func: GFunction) -> tuple[GFunction, list[ReduceOrder]]:
    """The Meijer G-function with its cancelling parameters removed, with
    the operators."""
    nan, nbq, ops1 = _reduce_order(func.an, func.bq, ReduceOrder.meijer_plus, lambda x: default_sort_key(-x))
    nbm, nap, ops2 = _reduce_order(func.bm, func.ap, ReduceOrder.meijer_minus, default_sort_key)
    return GFunction(nan, nap, nbm, nbq), ops1 + ops2


# PORTED FROM SYMPY: sympy/simplify/hyperexpand.py:make_derivative_operator
def make_derivative_operator(M: ExprMatrix, z: Symbol) -> Callable[[ExprMatrix], ExprMatrix]:
    """``D`` on the coefficient row ``C``: ``z*dC/dz + C*M``, simplified."""
    simp = make_simp(z)

    def doit(C: ExprMatrix) -> ExprMatrix:
        return (C.diff(z) * z + C * M).applyfunc(simp)
    return doit


# PORTED FROM SYMPY: sympy/simplify/hyperexpand.py:apply_operators
def apply_operators(obj: Expr, ops: Sequence[Operator], op: Callable[[Expr], Expr]) -> Expr:
    """The operators applied to ``obj``, the last first."""
    res = obj
    for o in reversed(ops):
        res = o.apply(res, op)
    return res


def apply_operators_matrix(obj: ExprMatrix, ops: Sequence[Operator],
                           op: Callable[[ExprMatrix], ExprMatrix]) -> ExprMatrix:
    """:func:`apply_operators` on a matrix."""
    res = obj
    for o in reversed(ops):
        res = o.apply_matrix(res, op)
    return res


# ---------------------------------------------------------------------
# the plans and the expansion of hypergeometric functions


def _others(dic: dict[Expr, list[Expr]], key: Expr) -> list[Expr]:
    out: list[Expr] = []
    for k, v in dic.items():
        if k != key:
            out.extend(v)
    return out


# PORTED FROM SYMPY: sympy/simplify/hyperexpand.py:devise_plan
def devise_plan(target: HyperFunction, origin: HyperFunction, z: Expr) -> list[Operator]:
    """The operators which turn ``origin`` into ``target`` (parameters
    congruent modulo 1), to be applied in the order of the list.

    >>> from sympy_extras.core import Rational, Symbol
    >>> from sympy_extras.simplify.hyperexpand import HyperFunction, devise_plan
    >>> z = Symbol('z')
    >>> [str(op) for op in devise_plan(HyperFunction((3, 2), ()), HyperFunction((1, 2), ()), z)]
    ['<Increment upper 2.>', '<Increment upper 1.>']
    """
    abuckets, bbuckets, nabuckets, nbbuckets = [_sift(params, _mod1) for params in
                                                (target.ap, target.bq, origin.ap, origin.bq)]
    if len(abuckets) != len(nabuckets) or len(bbuckets) != len(nbbuckets):
        raise ValueError('%s not reachable from %s' % (target, origin))
    ops: list[Operator] = []

    def do_shifts(fro: list[Expr], to: list[Expr], inc: Callable[[list[Expr], int], Operator],
                  dec: Callable[[list[Expr], int], Operator]) -> list[Operator]:
        shifts: list[Operator] = []
        for i in range(len(fro)):
            if _truth(to[i] - fro[i] > 0):
                sh, ch = inc, 1
            else:
                sh, ch = dec, -1
            while to[i] != fro[i]:
                shifts.append(sh(fro, i))
                fro[i] = fro[i] + ch
        return shifts

    def do_shifts_a(nal: list[Expr], nbk: list[Expr], al: list[Expr], aother: list[Expr],
                    bother: list[Expr]) -> list[Operator]:
        return do_shifts(nal, al, lambda p, i: ShiftA(p[i]), lambda p, i: UnShiftA(p + aother, nbk + bother, i, z))

    def do_shifts_b(nal: list[Expr], nbk: list[Expr], bk: list[Expr], aother: list[Expr],
                    bother: list[Expr]) -> list[Operator]:
        return do_shifts(nbk, bk, lambda p, i: UnShiftB(nal + aother, p + bother, i, z), lambda p, i: ShiftB(p[i]))

    for r in sorted(list(abuckets) + list(bbuckets), key=default_sort_key):
        al = sorted(abuckets.get(r, []), key=default_sort_key)
        nal = sorted(nabuckets.get(r, []), key=default_sort_key) if r in abuckets else []
        bk = sorted(bbuckets.get(r, []), key=default_sort_key)
        nbk = sorted(nbbuckets.get(r, []), key=default_sort_key) if r in bbuckets else []
        if len(al) != len(nal) or len(bk) != len(nbk):
            raise ValueError('%s not reachable from %s' % (target, origin))
        aother = _others(nabuckets, r)
        bother = _others(nbbuckets, r)
        if not al:
            ops += do_shifts_b([], nbk, bk, aother, bother)
        elif not bk:
            ops += do_shifts_a(nal, [], al, aother, bother)
        else:
            namax, amax = nal[-1], al[-1]
            if _truth(nbk[0] - namax <= 0) or _truth(bk[0] - amax <= 0):
                raise ValueError('Non-suitable parameters.')
            if _truth(namax - amax > 0):
                ops += do_shifts_a(nal, nbk, al, aother, bother)
                ops += do_shifts_b(al, nbk, bk, aother, bother)
            else:
                ops += do_shifts_b(nal, nbk, bk, aother, bother)
                ops += do_shifts_a(nal, bk, al, aother, bother)
        nabuckets[r] = al
        nbbuckets[r] = bk
    ops.reverse()
    return ops


def _int(e: Expr) -> int:
    assert isinstance(e, Integer), e
    return e.p


# PORTED FROM SYMPY: sympy/simplify/hyperexpand.py:try_shifted_sum
def try_shifted_sum(func: HyperFunction, z: Expr) -> Optional[tuple[HyperFunction, list[Operator], Expr]]:
    """A function with one positive integer upper and one positive
    integer lower parameter as a shifted series of a simpler one: the
    function, the operators and the polynomial to add."""
    from sympy_extras.core.mul import mul
    from sympy_extras.functions.combinatorial.factorials import factorial, rf
    abuckets, bbuckets = _sift(func.ap, _mod1), _sift(func.bq, _mod1)
    if len(abuckets.get(ZERO, [])) != 1:
        return None
    r = abuckets[ZERO][0]
    if _truth(r <= 0):
        return None
    if ZERO not in bbuckets:
        return None
    k = sorted(bbuckets[ZERO], key=_Ordered)[0]
    if _truth(k <= 0):
        return None
    nap = list(func.ap)
    nap.remove(r)
    nbq = list(func.bq)
    nbq.remove(k)
    k = k - 1
    nap = [x - k for x in nap]
    nbq = [x - k for x in nbq]
    ops: list[Operator] = [ShiftA(Integer(n + 1)) for n in range(_int(r) - 1)]
    ops.reverse()
    fac = factorial(k) / z**k
    fac = fac * mul(*[rf(b, k) for b in nbq])
    fac = fac / mul(*[rf(a, k) for a in nap])
    ops.append(MultOperator(fac))
    p: Expr = ZERO
    for n in range(_int(k)):
        m = z**n / factorial(n)
        m = m * mul(*[rf(a, n) for a in nap])
        m = m / mul(*[rf(b, n) for b in nbq])
        p = p + m
    return HyperFunction(nap, nbq), ops, -p


# PORTED FROM SYMPY: sympy/simplify/hyperexpand.py:try_polynomial
def try_polynomial(func: HyperFunction, z: Expr) -> Optional[Expr]:
    """The function as a polynomial when an upper parameter is a
    nonpositive integer (``oo`` when a lower one ends the series first)."""
    from sympy_extras.core.mul import mul
    from sympy_extras.core.numbers import oo
    abuckets, bbuckets = _sift(func.ap, _mod1), _sift(func.bq, _mod1)
    a0 = sorted(abuckets.get(ZERO, []), key=_Ordered)
    b0 = sorted(bbuckets.get(ZERO, []), key=_Ordered)
    al0 = [x for x in a0 if _truth(x <= 0)]
    bl0 = [x for x in b0 if _truth(x <= 0)]
    if bl0 and all(_truth(a < bl0[-1]) for a in al0):
        return oo
    if not al0:
        return None
    a = al0[-1]
    fac: Expr = ONE
    res: Expr = ONE
    for n in range(-_int(a)):
        fac = fac * z
        fac = fac / (n + 1)
        fac = fac * mul(*[x + n for x in func.ap])
        fac = fac / mul(*[b + n for b in func.bq])
        res = res + fac
    return res


# PORTED FROM SYMPY: sympy/simplify/hyperexpand.py:try_lerchphi
def try_lerchphi(func: HyperFunction) -> Optional[Formula]:
    """A formula in terms of Lerch transcendents, for the functions whose
    series coefficients are a rational function of the index."""
    from sympy_extras.core.add import Add, add
    from sympy_extras.core.function import expand_func
    from sympy_extras.functions.combinatorial.factorials import rf
    from sympy_extras.functions.special.zeta_functions import lerchphi
    from sympy_extras.polys.partfrac import apart
    from sympy_extras.polys.polytools import Poly
    abuckets, bbuckets = _sift(func.ap, _mod1), _sift(func.bq, _mod1)
    paired: dict[Expr, tuple[list[Expr], list[Expr]]] = {}
    for key, value in abuckets.items():
        if key != 0 and key not in bbuckets:
            return None
        paired[key] = (list(value), list(bbuckets.get(key, [])))
        bbuckets.pop(key, None)
    if bbuckets:
        return None
    if ZERO not in abuckets:
        return None
    aints, bints = paired[ZERO]
    paired[ZERO] = (aints, bints + [ONE])
    t = Dummy('t')
    numer: Expr = ONE
    denom: Expr = ONE
    for avalue, bvalue in paired.values():
        if len(avalue) != len(bvalue):
            return None
        for a, b in zip(avalue, bvalue):
            if (a - b).is_positive is True:
                k = a - b
                numer = numer * rf(b + t, k)
                denom = denom * rf(b, k)
            else:
                k = b - a
                numer = numer * rf(a, k)
                denom = denom * rf(a + t, k)
    part = apart(numer / denom, t)
    monomials: list[tuple[Expr, int]] = []
    terms: dict[Expr, list[tuple[Expr, int]]] = {}
    for arg in (part.args if isinstance(part, Add) else (part,)):
        assert isinstance(arg, Expr)
        num, den = arg.as_numer_denom()
        if not den.has(t):
            p = Poly(num, t)
            if p.length() > 1:
                raise TypeError('p should be monomial')
            (b_exp,), lc = p.LT()
            monomials.append((lc / den, b_exp))
            continue
        if num.has(t):
            raise NotImplementedError('Need partial fraction decomposition with linear denominators')
        indep, dep = den.as_independent(t, as_Add=False)
        if len(dep.args) > 1 and dep.is_Mul:
            raise ValueError('more than one factor depends on %s' % (t,))
        n = 1
        if dep.is_Pow:
            n = _int(_expr(dep.args[1]))
            dep = _expr(dep.args[0])
        shift: Expr
        if dep == t:
            shift = ZERO
        elif isinstance(dep, Add):
            shift, tmp = dep.as_independent(t, as_Add=True)
            scale: Expr = ONE
            if tmp != t:
                scale, _ = tmp.as_independent(t, as_Add=False)
            if dep != scale*t + shift:
                raise NotImplementedError('unrecognised form %s' % dep)
            shift = shift / scale
            indep = indep * scale**n
        else:
            raise NotImplementedError('unrecognised form of partial fraction')
        terms.setdefault(shift, []).append((num / indep, n))
    deriv: dict[Expr, list[tuple[Expr, Expr]]] = {}
    coeffs: dict[Expr, list[Expr]] = {}
    z = Dummy('z')
    monomials.sort(key=lambda m: m[1])
    mon: dict[int, Expr] = {0: 1 / (1 - z)}
    if monomials:
        for k2 in range(monomials[-1][1]):
            mon[k2 + 1] = z * mon[k2].diff(z)
    for coeff, n in monomials:
        coeffs.setdefault(ONE, []).append(coeff * mon[n])
    for shift, lst in terms.items():
        for c, k3 in lst:
            coeffs.setdefault(lerchphi(z, k3, shift), []).append(c)
        lst.sort(key=lambda ck: ck[1])
        for k4 in range(2, lst[-1][1] + 1):
            deriv[lerchphi(z, k4, shift)] = [(-shift, lerchphi(z, k4, shift)), (ONE, lerchphi(z, k4 - 1, shift))]
        deriv[lerchphi(z, 1, shift)] = [(-shift, lerchphi(z, 1, shift)), (1 / (1 - z), ONE)]
    trans: dict[Expr, int] = {}
    for idx, bb in enumerate([ONE] + list(deriv.keys())):
        trans[bb] = idx
    basis = [expand_func(bb) for bb, _ in sorted(trans.items(), key=lambda kv: kv[1])]
    size = len(basis)
    C_entries: list[Expr] = [ZERO] * size
    for bb, cs in coeffs.items():
        C_entries[trans[bb]] = add(*cs)
    M_entries: list[Expr] = [ZERO] * (size * size)
    for bb, lst2 in deriv.items():
        for c, b2 in lst2:
            M_entries[trans[bb] * size + trans[b2]] = c
    return Formula(func, z, None, [], ExprMatrix.column(basis), ExprMatrix(1, size, C_entries),
                   ExprMatrix(size, size, M_entries))


# PORTED FROM SYMPY: sympy/simplify/hyperexpand.py:build_hypergeometric_formula
def build_hypergeometric_formula(func: HyperFunction) -> Formula:
    """A formula for ``func`` whose basis is the function and shifted
    copies of it (always available; the answer stays hypergeometric)."""
    from sympy_extras.core.mul import mul
    from sympy_extras.functions.special.hyper import hyper
    from sympy_extras.polys.polytools import Poly
    z = Dummy('z')
    if func.ap:
        afactors = [_x + a for a in func.ap]
        bfactors = [_x + b - 1 for b in func.bq]
        poly = Poly(_x * mul(*bfactors) - z * mul(*afactors), _x)
        n = poly.degree()
        basis: list[Expr] = []
        M = ExprMatrix.zeros(n, n)
        for k in range(n):
            a = func.ap[0] + k
            basis.append(hyper([a] + list(func.ap[1:]), func.bq, z))
            if k < n - 1:
                M = M.with_entry(k, k, -a).with_entry(k, k + 1, a)
        B = ExprMatrix.column(basis)
        C = ExprMatrix.from_rows([[ONE] + [ZERO] * (n - 1)])
        derivs = [ExprMatrix.eye(n)]
        for k in range(n):
            derivs.append(M * derivs[k])
        coeffs = poly.all_coeffs()
        res: list[Expr] = [ZERO] * n
        for k, c in enumerate(reversed(coeffs)):
            for r, d in enumerate(C * derivs[k]):
                res[r] = res[r] + c * d
        for k, c in enumerate(res):
            M = M.with_entry(n - 1, k, -c / derivs[n - 1][0, n - 1] / coeffs[0])
        return Formula(func, z, None, [], B, C, M)
    basis = []
    bq = list(func.bq)
    for i in range(len(bq)):
        basis.append(hyper([], bq, z))
        bq[i] = bq[i] + 1
    basis.append(hyper([], bq, z))
    B = ExprMatrix.column(basis)
    n = len(B)
    C = ExprMatrix.from_rows([[ONE] + [ZERO] * (n - 1)])
    M = ExprMatrix.zeros(n, n).with_entry(0, n - 1, z / mul(*func.bq))
    for k in range(1, n):
        M = M.with_entry(k, k - 1, func.bq[k - 1]).with_entry(k, k, -func.bq[k - 1])
    return Formula(func, z, None, [], B, C, M)


# PORTED FROM SYMPY: sympy/simplify/hyperexpand.py:hyperexpand_special
def hyperexpand_special(ap: Sequence[Expr], bq: Sequence[Expr], z: Expr) -> Expr:
    """The special values of 2F1 at ``z = 1`` (Gauss) and ``z = -1``
    (Kummer); ``hyper(ap, bq, z)`` otherwise."""
    from sympy_extras.functions.elementary.complexes import unpolarify
    from sympy_extras.functions.elementary.trigonometric import cos
    from sympy_extras.functions.special.gamma_functions import gamma
    from sympy_extras.functions.special.hyper import hyper
    from .simplify import simplify
    p, q = len(ap), len(bq)
    z_ = z
    z = unpolarify(z)
    if z == 0:
        return ONE
    if p == 2 and q == 1:
        a, b, c = list(ap) + list(bq)
        if z == 1:
            return gamma(c - a - b)*gamma(c)/gamma(c - a)/gamma(c - b)
        if z == -1 and simplify(b - a + c) == 1:
            b, a = a, b
        if z == -1 and simplify(a - b + c) == 1:
            # Kummer
            if b.is_integer is True and b.is_negative is True:
                return 2*cos(pi*b/2)*gamma(-b)*gamma(b - a + 1)/gamma(-b/2)/gamma(b/2 - a + 1)
            return gamma(b/2 + 1)*gamma(b - a + 1)/gamma(b + 1)/gamma(b/2 - a + 1)
    return hyper(ap, bq, z_)


def _replace_hyper_special(e: Expr) -> Expr:
    """``e.replace(hyper, hyperexpand_special)``."""
    from sympy_extras.core.containers import Tuple
    from sympy_extras.functions.special.hyper import Hyper

    def value(*args: Basic) -> Basic:
        ap, bq, z = args
        assert isinstance(ap, Tuple) and isinstance(bq, Tuple) and isinstance(z, Expr)
        return hyperexpand_special([_expr(a) for a in ap.args], [_expr(b) for b in bq.args], z)
    return _expr(e.replace(Hyper, value))


Rewrite = Optional[str]

_collection: Optional[FormulaCollection] = None


def _formula_collection() -> FormulaCollection:
    global _collection
    if _collection is None:
        _collection = FormulaCollection()
    return _collection


def _rewrite(e: Expr, rewrite: Rewrite) -> Expr:
    """SymPy's ``e.rewrite(rewrite)``: only the ``HyperRep`` functions
    have the rewrites 'nonrep' and 'nonrepsmall'."""
    from sympy_extras.functions.special.hyperrep import rewrite_nonrep
    if rewrite == 'nonrepsmall':
        return rewrite_nonrep(e, small=True)
    if rewrite == 'nonrep':
        return rewrite_nonrep(e, small=False)
    return e


_z0 = Dummy('z0')


# PORTED FROM SYMPY: sympy/simplify/hyperexpand.py:_hyperexpand
def _hyperexpand(func: HyperFunction, z: Expr, ops0: Sequence[Operator] = (), z0: Symbol = _z0,
                 premult: Expr = ONE, prem: Expr = ZERO, rewrite: Rewrite = 'default') -> Expr:
    """The expansion of ``func`` at ``z``: ``premult`` times the function,
    with ``ops0`` applied (their derivative operator shifted by
    ``prem``); the closed forms rewritten by ``rewrite``."""
    from sympy_extras.functions.elementary.complexes import polarify, unpolarify
    from .powsimp import powdenest
    from .simplify import simplify
    if z.is_zero is True:
        return ONE
    z = polarify(z, subs=False)[0]
    if rewrite == 'default':
        rewrite = 'nonrepsmall'

    def carryout_plan(f: Formula, ops: Sequence[Operator]) -> Expr:
        M0 = f.M.subs(f.z, z0)
        C = apply_operators_matrix(f.C.subs(f.z, z0), ops, make_derivative_operator(M0, z0))
        C = apply_operators_matrix(C, ops0, make_derivative_operator(M0 + ExprMatrix.eye(f.M.rows) * prem, z0))
        if premult == 1:
            C = C.applyfunc(make_simp(z0))
        r = _closed_form(C, f.B.subs(f.z, z0)) * premult
        res = _expr(r.subs(z0, z))
        return _rewrite(res, rewrite)

    def d(f: Expr) -> Expr:
        return z0 * f.diff(z0)

    collection = _formula_collection()
    func, reduce_ops = reduce_order(func)
    ops: list[Operator] = list(reduce_ops)
    res = try_polynomial(func, z0)
    if res is not None:
        p = apply_operators(res, ops, d)
        p = apply_operators(p * premult, ops0, d)
        return unpolarify(_expr(simplify(p).subs(z0, z)))
    p = ZERO
    shifted = try_shifted_sum(func, z0)
    if shifted is not None:
        func, nops, p = shifted
        ops += nops
    p = apply_operators(p, ops, d)
    p = apply_operators(p * premult, ops0, d)
    p = _expr(simplify(p).subs(z0, z))
    if unpolarify(z) in (ONE, Integer(-1)) and (len(func.ap), len(func.bq)) == (2, 1):
        from sympy_extras.functions.special.hyper import Hyper
        f = build_hypergeometric_formula(func)
        r = _replace_hyper_special(carryout_plan(f, ops))
        if not r.has(Hyper):
            return r + p
    formula = collection.lookup_origin(func)
    if formula is None:
        formula = try_lerchphi(func)
    if formula is None:
        formula = build_hypergeometric_formula(func)
    ops += devise_plan(func, formula.func, z0)
    r = carryout_plan(formula, ops) + p
    return _replace_hyper_special(powdenest(r, polar=True))


# ---------------------------------------------------------------------
# Meijer G-functions


# PORTED FROM SYMPY: sympy/simplify/hyperexpand.py:devise_plan_meijer
def devise_plan_meijer(fro: GFunction, to: GFunction, z: Expr) -> list[Operator]:
    """The operators which turn ``fro`` into ``to`` (parameters congruent
    modulo 1), to be applied in the order of the list; a
    ``NotImplementedError`` when the shifts cannot be found."""
    fan, fap, fbm, fbq = list(fro.an), list(fro.ap), list(fro.bm), list(fro.bq)

    def try_shift(f: list[Expr], t: Sequence[Expr], shifter: Callable[[int], Operator], diff: int,
                  counter: list[Expr]) -> Optional[Operator]:
        """Shift one ``f[i]`` by ``diff`` towards ``t[i]``, when it is not
        in ``counter``."""
        for idx, (a, b) in enumerate(zip(f, t)):
            if (a - b).is_integer is True and _truth((b - a) / diff > 0) and all(a != x for x in counter):
                sh = shifter(idx)
                f[idx] = f[idx] + diff
                return sh
        return None

    ops: list[Operator] = []
    change = True
    while change:
        change = False
        attempts: list[tuple[list[Expr], tuple[Expr, ...], Callable[[int], Operator], int, list[Expr]]] = [
            (fan, to.an, lambda i: MeijerUnShiftB(fan, fap, fbm, fbq, i, z), 1, fbm + fbq),
            (fap, to.ap, lambda i: MeijerUnShiftD(fan, fap, fbm, fbq, i, z), 1, fbm + fbq),
            (fbm, to.bm, lambda i: MeijerUnShiftA(fan, fap, fbm, fbq, i, z), -1, fan + fap),
            (fbq, to.bq, lambda i: MeijerUnShiftC(fan, fap, fbm, fbq, i, z), -1, fan + fap),
            (fan, to.an, lambda i: MeijerShiftB(fan[i]), -1, []),
            (fap, to.ap, lambda i: MeijerShiftD(fap[i]), -1, []),
            (fbm, to.bm, lambda i: MeijerShiftA(fbm[i]), 1, []),
            (fbq, to.bq, lambda i: MeijerShiftC(fbq[i]), 1, [])]
        # the counters are SymPy's at the call: an attempt which changes a
        # list ends the round
        for f, t, shifter, diff, counter in attempts:
            op = try_shift(f, t, shifter, diff, counter)
            if op is not None:
                ops.append(op)
                change = True
                break
    if fan != list(to.an) or fap != list(to.ap) or fbm != list(to.bm) or fbq != list(to.bq):
        raise NotImplementedError('Could not devise plan.')
    ops.reverse()
    return ops


_meijercollection: Optional[MeijerFormulaCollection] = None

Condition = Union[bool, Boolean]


def _meijer_collection() -> MeijerFormulaCollection:
    global _meijercollection
    if _meijercollection is None:
        _meijercollection = MeijerFormulaCollection()
    return _meijercollection


def _count(e: Expr, cls: type) -> int:
    """SymPy's ``e.count(cls)``: the number of nodes of the class."""
    from sympy_extras.core.basic import preorder_traversal
    return sum(1 for node in preorder_traversal(e) if isinstance(node, cls))


def _subs_condition(c: Condition, old: Expr, new: Expr) -> Condition:
    if isinstance(c, bool):
        return c
    r = c.subs(old, new)
    assert isinstance(r, Boolean)
    return r


# PORTED FROM SYMPY: sympy/simplify/hyperexpand.py:_meijergexpand
def _meijergexpand(func: GFunction, z0: Expr, allow_hyper: bool = False, rewrite: Rewrite = 'default',
                   place: Optional[Expr] = None) -> Expr:
    """The expansion of the Meijer G-function ``func`` at ``z0``: a formula
    of the table, or Slater's theorem (sums of hypergeometric functions,
    around 0 and around infinity) with the conditions of their validity
    as a ``Piecewise``; the function itself when nothing works."""
    from sympy_extras.core.function import expand_func
    from sympy_extras.core.numbers import oo, zoo
    from sympy_extras.functions.combinatorial.factorials import factorial, rf
    from sympy_extras.functions.elementary.complexes import abs_, polar_lift
    from sympy_extras.functions.elementary.piecewise import piecewise
    from sympy_extras.functions.special.gamma_functions import gamma
    from sympy_extras.functions.special.hyper import Hyper
    from sympy_extras.series.residues import residue
    from .powsimp import powdenest
    collection = _meijer_collection()
    if rewrite == 'default':
        rewrite = None
    func0 = func
    z = Dummy('z')
    func, reduce_ops = reduce_order_meijer(func)
    ops: list[Operator] = list(reduce_ops)
    f = collection.lookup_origin(func)
    if f is not None:
        ops += devise_plan_meijer(f.func, func, z)
        C = apply_operators_matrix(f.C.subs(f.z, z), ops, make_derivative_operator(f.M.subs(f.z, z), z))
        C = C.applyfunc(make_simp(z))
        r = (C * f.B.subs(f.z, z))[0, 0]
        return powdenest(_expr(r.subs(z, z0)), polar=True)

    def can_do(pbm: Buckets, pap: Buckets) -> bool:
        """Whether no residue is of too high an order (Slater's theorem)."""
        for i in pbm:
            if len(pbm[i]) > 1:
                n = len(pap[i]) if i in pap else 0
                if n + 1 < len(pbm[i]):
                    return False
        return True

    t = Dummy('t')

    def do_slater(an: Sequence[Expr], bm: Sequence[Expr], ap: Sequence[Expr], bq: Sequence[Expr], z: Symbol,
                  zfinal: Expr, ops: Sequence[Operator]) -> tuple[Expr, Condition]:
        g = GFunction(an, bm, ap, bq)
        _, pbm, pap, _ = g.compute_buckets()
        if not can_do(pbm, pap):
            return ZERO, False
        cond: Condition = len(an) + len(ap) < len(bm) + len(bq)
        if len(an) + len(ap) == len(bm) + len(bq):
            cond = abs_(z) < 1
        if cond is False:
            return ZERO, False
        res: Expr = ZERO
        for m in pbm:
            if len(pbm[m]) == 1:
                bh = pbm[m][0]
                fac: Expr = ONE
                bo = list(bm)
                bo.remove(bh)
                for bj in bo:
                    fac = fac * gamma(bj - bh)
                for aj in an:
                    fac = fac * gamma(1 + bh - aj)
                for bj in bq:
                    fac = fac / gamma(1 + bh - bj)
                for aj in ap:
                    fac = fac / gamma(aj - bh)
                nap = [1 + bh - a for a in list(an) + list(ap)]
                nbq = [1 + bh - b for b in list(bo) + list(bq)]
                k = polar_lift(Integer(-1)**(len(ap) - len(bm)))
                harg = k * zfinal
                premult = (t / k)**bh
                hyp = _hyperexpand(HyperFunction(nap, nbq), harg, ops, t, premult, bh, rewrite=None)
                res = res + fac * hyp
            else:
                b_ = pbm[m][0]
                ki = [bi - b_ for bi in pbm[m][1:]]
                u = len(ki)
                li = [ai - b_ for ai in pap[m][:u + 1]]
                bo = list(bm)
                for b in pbm[m]:
                    bo.remove(b)
                ao = list(ap)
                for a in pap[m][:u]:
                    ao.remove(a)
                lu = li[-1]
                di = [l_ - k_ for (l_, k_) in zip(li, ki)]
                s = Dummy('s')
                integrand: Expr = z**s
                for b in bm:
                    if isinstance(b, Number) and _is_zero(mod(b, ONE)):
                        b = Integer(round(float(b)))
                    integrand = integrand * gamma(b - s)
                for a in an:
                    integrand = integrand * gamma(1 - a + s)
                for b in bq:
                    integrand = integrand / gamma(1 - b + s)
                for a in ap:
                    integrand = integrand / gamma(a - s)
                integrand = expand_func(integrand)
                for r in range(int(round(float(lu)))):
                    resid = residue(integrand, s, b_ + r)
                    resid = apply_operators(resid, ops, lambda e: z * e.diff(z))
                    res = res - resid
                au = b_ + lu
                k = polar_lift(Integer(-1)**(len(ao) + len(bo) + 1))
                harg = k * zfinal
                premult = (t / k)**au
                nap = [1 + au - a for a in list(an) + list(ap)] + [ONE]
                nbq = [1 + au - b for b in list(bm) + list(bq)]
                hyp = _hyperexpand(HyperFunction(nap, nbq), harg, ops, t, premult, au, rewrite=None)
                C: Expr = Integer(-1)**lu / factorial(lu)
                for i in range(u):
                    C = C * Integer(-1)**di[i] / rf(lu - li[i] + 1, di[i])
                for a in an:
                    C = C * gamma(1 - a + au)
                for b in bo:
                    C = C * gamma(b - au)
                for a in ao:
                    C = C / gamma(a - au)
                for b in bq:
                    C = C / gamma(1 - b + au)
                res = res + C * hyp
        return res, cond

    slater1, cond1 = do_slater(func.an, func.bm, func.ap, func.bq, z, z0, ops)

    def tr(params: Sequence[Expr]) -> list[Expr]:
        return [1 - x for x in params]

    # SymPy changes the operators in place: D -> -D, z -> 1/t
    ops2: list[Operator] = [Operator(_op_of(_xreplace(_expr(_from_coeffs(op.coeffs, _x).subs(z, 1 / t)), {_x: -_x})))
                            for op in ops]
    slater2, cond2 = do_slater(tr(func.bm), tr(func.an), tr(func.bq), tr(func.ap), t, 1 / z0, ops2)
    slater1 = powdenest(_expr(slater1.subs(z, z0)), polar=True)
    slater2 = powdenest(_expr(slater2.subs(t, 1 / z0)), polar=True)
    cond2 = _subs_condition(cond2, t, 1 / z)
    # m = func(z): its delta and nu (SymPy's meijerg.delta and meijerg.nu)
    full_ap, full_bq = list(func.an) + list(func.ap), list(func.bm) + list(func.bq)
    delta = Integer(len(func.bm) + len(func.an)) - Rational(len(full_ap) + len(full_bq), 2)
    # SymPy's (re(m.nu) < -1) is not False holds always (a relational is
    # never the Python False)
    if _truth(delta > 0) or (delta == 0 and len(full_ap) == len(full_bq) and polar_lift(z0) == polar_lift(ONE)):
        if cond1 is not False:
            cond1 = True
        if cond2 is not False:
            cond2 = True
    slater1 = _rewrite(slater1, rewrite or ('nonrep' if cond1 is True else 'nonrepsmall'))
    slater2 = _rewrite(slater2, rewrite or ('nonrep' if cond2 is True else 'nonrepsmall'))
    if cond1 is not False and cond2 is not False:
        if place is not None and place == 0:
            cond2 = False
        if place is not None and place == zoo:
            cond1 = False
    cond1 = _subs_condition(cond1, z, z0)
    cond2 = _subs_condition(cond2, z, z0)

    def weight(e: Expr, cond: Condition) -> tuple[int, int, int]:
        from sympy_extras.core.numbers import nan
        c0 = 0 if cond is True else 1 if cond is False else 2
        if e.has(oo) or e.has(zoo) or e.has(-oo) or e.has(nan):
            c0 = 3
        ops_count = e.count_ops()
        assert isinstance(ops_count, Integer)
        return (c0, _count(e, Hyper), ops_count.p)

    w1 = weight(slater1, cond1)
    w2 = weight(slater2, cond2)
    best = min(w1, w2)
    # SymPy's min(w1, w2) <= (0, 1, oo)
    if best[0] < 0 or (best[0] == 0 and best[1] <= 1):
        return slater1 if w1 < w2 else slater2
    if max(w1[0], w2[0]) <= 1 and max(w1[1], w2[1]) <= 1:
        return piecewise((slater1, cond1), (slater2, cond2), (func0(z0), True))
    r = piecewise((slater1, cond1), (slater2, cond2), (func0(z0), True))
    if not r.has(Hyper) or allow_hyper:
        return r
    return func0(z0)


# PORTED FROM SYMPY: sympy/simplify/hyperexpand.py:hyperexpand
def hyperexpand(f: ExprLike, allow_hyper: bool = False, rewrite: Rewrite = 'default',
                place: Optional[Expr] = None) -> Expr:
    """The hypergeometric functions and Meijer G-functions of ``f``
    expanded into named special functions, where the tables, the shifts
    and Slater's theorem find expressions; ``allow_hyper`` accepts results
    in terms of simpler hypergeometric functions, ``place`` (``0`` or
    ``zoo``) the point around which a Meijer G-function is expanded.

    Examples
    ========

    >>> from sympy_extras.core import Symbol, Rational
    >>> from sympy_extras.functions.special.hyper import hyper, meijerg
    >>> from sympy_extras.simplify.hyperexpand import hyperexpand
    >>> z = Symbol('z')
    >>> hyperexpand(hyper([], [], z)), hyperexpand(hyper([1, 1], [2], z))
    (exp(z), -log(1 - z)/z)
    >>> hyperexpand(meijerg([1], [], [], [], z))
    exp(-1/z)
    """
    from sympy_extras.core.containers import Tuple
    from sympy_extras.core.numbers import nan, oo, zoo
    from sympy_extras.functions.special.hyper import Hyper, MeijerG, hyper, meijerg
    e = to_expr(f)

    def tup(b: Basic) -> list[Expr]:
        assert isinstance(b, Tuple)
        return [_expr(x) for x in b.args]

    def do_replace(*args: Basic) -> Basic:
        ap, bq, z = args
        return _hyperexpand(HyperFunction(tup(ap), tup(bq)), _expr(z), rewrite=rewrite)

    def do_meijer(*args: Basic) -> Basic:
        ap, bq, z = args
        assert isinstance(ap, Tuple) and isinstance(bq, Tuple)
        an, aother, bm, bother = tup(ap.args[0]), tup(ap.args[1]), tup(bq.args[0]), tup(bq.args[1])
        r = _meijergexpand(GFunction(an, aother, bm, bother), _expr(z), allow_hyper, rewrite=rewrite, place=place)
        if not (r.has(nan) or r.has(zoo) or r.has(oo) or r.has(-oo)):
            return r
        return meijerg(an, aother, bm, bother, _expr(z))   # SymPy's None: the node stays

    return _expr(_expr(e.replace(Hyper, do_replace)).replace(MeijerG, do_meijer))
