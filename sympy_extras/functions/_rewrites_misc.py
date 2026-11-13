"""The ``rewrite`` rules of the complex, integer, piecewise and
discontinuous functions (``Abs``, ``sign``, ``arg``, ``atan2``, ``re``,
``im``, ``floor``, ``ceiling``, ``frac``, ``Max``, ``Min``, ``Mod``,
``Heaviside``, ``DiracDelta``, ``SingularityFunction``,
``KroneckerDelta``, ``Piecewise``) and of the nodes which are not
functions (the logical connectives, ``Eq``, the sums and products), as
typed tables for :func:`sympy_extras.core.rewriting.rewrite`. Ported from
the ``_eval_rewrite_as_<name>`` methods of SymPy's classes (the modules
are listed below).

Left out: ``Piecewise._eval_rewrite_as_ITE`` and
``ITE._eval_rewrite_as_Piecewise`` (they need a piecewise function with
boolean values, which the native ``Piecewise``, an expression, cannot
hold), ``FiniteSet._eval_rewrite_as_PowerSet`` (there is no native
``PowerSet``).
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/functions/elementary/complexes.py, sympy/functions/elementary/trigonometric.py
# (atan2), sympy/functions/elementary/integers.py, sympy/functions/elementary/miscellaneous.py,
# sympy/core/mod.py, sympy/functions/elementary/piecewise.py, sympy/functions/special/delta_functions.py,
# sympy/functions/special/singularity_functions.py, sympy/functions/special/tensor_functions.py,
# sympy/logic/boolalg.py, sympy/core/relational.py, sympy/concrete/summations.py,
# sympy/concrete/products.py (the _eval_rewrite_as_<name> methods) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: typed; tables keyed by (class, target) in place of methods; the
# rules of the nodes which are not expressions take and give any node.
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Optional

from sympy_extras.core.add import add
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr
from sympy_extras.core.logic import Boolean, true
from sympy_extras.core.mul import mul, neg
from sympy_extras.core.numbers import I, Integer, Rational
from sympy_extras.core.rewriting import NodeRule, RewriteHints, Rule, RuleKey
from sympy_extras.core.symbol import Symbol

__all__ = ['rules', 'node_rules']

ZERO = Integer(0)
ONE = Integer(1)
HALF = Rational(1, 2)


def _sub(a: Expr, b: Expr) -> Expr:
    return add(a, neg(b))


def _div(a: Expr, b: Expr) -> Expr:
    from sympy_extras.core.power import pow
    return mul(a, pow(b, Integer(-1)))


# -- complexes

# PORTED FROM SYMPY: sympy/functions/elementary/complexes.py:re._eval_rewrite_as_im
def _re_im(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.functions.elementary.complexes import im
    a = node.args[0]
    assert isinstance(a, Expr)
    return _sub(a, mul(I, im(a)))


# PORTED FROM SYMPY: sympy/functions/elementary/complexes.py:im._eval_rewrite_as_re
def _im_re(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.functions.elementary.complexes import re
    a = node.args[0]
    assert isinstance(a, Expr)
    return mul(neg(I), _sub(a, re(a)))


# PORTED FROM SYMPY: sympy/functions/elementary/complexes.py:sign._eval_rewrite_as_Piecewise
def _sign_piecewise(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.core.relational import gt, lt
    from sympy_extras.functions.elementary.piecewise import piecewise
    a = args[0]
    if a.is_extended_real is True:
        return piecewise((ONE, gt(a, ZERO)), (Integer(-1), lt(a, ZERO)), (ZERO, True))
    return None


# PORTED FROM SYMPY: sympy/functions/elementary/complexes.py:sign._eval_rewrite_as_Heaviside
def _sign_heaviside(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.functions.special.delta_functions import heaviside
    a = args[0]
    if a.is_extended_real is True:
        return add(mul(heaviside(a), Integer(2)), Integer(-1))
    return None


# PORTED FROM SYMPY: sympy/functions/elementary/complexes.py:Abs._eval_rewrite_as_Heaviside
def _abs_heaviside(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.functions.special.delta_functions import heaviside
    a = args[0]
    if a.is_extended_real is True:
        return mul(a, _sub(heaviside(a), heaviside(neg(a))))
    return None


# PORTED FROM SYMPY: sympy/functions/elementary/complexes.py:Abs._eval_rewrite_as_Piecewise
def _abs_piecewise(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.core.relational import ge
    from sympy_extras.functions.elementary.piecewise import piecewise
    a = args[0]
    if a.is_extended_real is True:
        return piecewise((a, ge(a, ZERO)), (neg(a), True))
    if a.is_imaginary is True:
        ia = mul(I, a)
        return piecewise((ia, ge(ia, ZERO)), (neg(ia), True))
    return None


# PORTED FROM SYMPY: sympy/functions/elementary/complexes.py:Abs._eval_rewrite_as_sign
def _abs_sign(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.functions.elementary.complexes import sign
    return _div(args[0], sign(args[0]))


# PORTED FROM SYMPY: sympy/functions/elementary/complexes.py:Abs._eval_rewrite_as_conjugate
def _abs_conjugate(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.core.power import sqrt
    from sympy_extras.functions.elementary.complexes import conjugate
    return sqrt(mul(args[0], conjugate(args[0])))


# PORTED FROM SYMPY: sympy/functions/elementary/complexes.py:arg._eval_rewrite_as_atan2
def _arg_atan2(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.functions.elementary.complexes import atan2
    a = node.args[0]
    assert isinstance(a, Expr)
    x, y = a.as_real_imag()
    return atan2(y, x)


# PORTED FROM SYMPY: sympy/functions/elementary/trigonometric.py:atan2._eval_rewrite_as_log
def _atan2_log(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.core.power import sqrt
    from sympy_extras.functions.elementary.exponential import log
    y, x = args
    return mul(neg(I), log(_div(add(x, mul(I, y)), sqrt(add(mul(x, x), mul(y, y))))))


# PORTED FROM SYMPY: sympy/functions/elementary/trigonometric.py:atan2._eval_rewrite_as_atan
def _atan2_atan(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.core.numbers import nan, pi
    from sympy_extras.core.power import sqrt
    from sympy_extras.core.relational import lt, ne
    from sympy_extras.functions.elementary.complexes import re
    from sympy_extras.functions.elementary.piecewise import piecewise
    from sympy_extras.functions.elementary.trigonometric import atan
    y, x = args
    return piecewise((mul(Integer(2), atan(_div(y, add(x, sqrt(add(mul(x, x), mul(y, y))))))), ne(y, ZERO)),
                     (pi, lt(re(x), ZERO)), (ZERO, ne(x, ZERO)), (nan, True))


# PORTED FROM SYMPY: sympy/functions/elementary/trigonometric.py:atan2._eval_rewrite_as_arg
def _atan2_arg(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.core.power import sqrt
    from sympy_extras.functions.elementary.complexes import abs_, arg
    from sympy_extras.functions.elementary.exponential import log
    y, x = args
    if x.is_extended_real is True and y.is_extended_real is True:
        return arg(add(x, mul(y, I)))
    n = add(x, mul(I, y))
    d = add(mul(x, x), mul(y, y))
    return _sub(arg(_div(n, sqrt(d))), mul(I, log(_div(abs_(n), sqrt(abs_(d))))))


# -- integers

# PORTED FROM SYMPY: sympy/functions/elementary/integers.py:floor._eval_rewrite_as_ceiling
def _floor_ceiling(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.functions.elementary.integers import ceiling
    return neg(ceiling(neg(args[0])))


# PORTED FROM SYMPY: sympy/functions/elementary/integers.py:floor._eval_rewrite_as_frac
def _floor_frac(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.functions.elementary.integers import frac
    return _sub(args[0], frac(args[0]))


# PORTED FROM SYMPY: sympy/functions/elementary/integers.py:ceiling._eval_rewrite_as_floor
def _ceiling_floor(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.functions.elementary.integers import floor
    return neg(floor(neg(args[0])))


# PORTED FROM SYMPY: sympy/functions/elementary/integers.py:ceiling._eval_rewrite_as_frac
def _ceiling_frac(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.functions.elementary.integers import frac
    return add(args[0], frac(neg(args[0])))


# PORTED FROM SYMPY: sympy/functions/elementary/integers.py:frac._eval_rewrite_as_floor
def _frac_floor(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.functions.elementary.integers import floor
    return _sub(args[0], floor(args[0]))


# PORTED FROM SYMPY: sympy/functions/elementary/integers.py:frac._eval_rewrite_as_ceiling
def _frac_ceiling(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.functions.elementary.integers import ceiling
    return add(args[0], ceiling(neg(args[0])))


# PORTED FROM SYMPY: sympy/core/mod.py:Mod._eval_rewrite_as_floor
def _mod_floor(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.functions.elementary.integers import floor
    a, b = args
    return _sub(a, mul(b, floor(_div(a, b))))


# -- Max and Min

def _lattice(node: Expr, args: tuple[Expr, ...]) -> Expr:
    """SymPy's ``self.func(*args)``: the maximum or minimum, evaluated."""
    from sympy_extras.functions.elementary.miscellaneous import Max, max_, min_
    return max_(*args) if isinstance(node, Max) else min_(*args)


# PORTED FROM SYMPY: sympy/functions/elementary/miscellaneous.py:MinMaxBase._eval_rewrite_as_Abs
def _minmax_abs(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.core.rewriting import rewrite
    from sympy_extras.functions.elementary.complexes import Abs, abs_
    from sympy_extras.functions.elementary.miscellaneous import Max
    rest = _lattice(node, args[1:])
    s = mul(add(args[0], rest), HALF)
    d = mul(abs_(_sub(args[0], rest)), HALF)
    result = add(s, d) if isinstance(node, Max) else _sub(s, d)
    rewritten = rewrite(result, (), Abs, hints)
    assert isinstance(rewritten, Expr)
    return rewritten


# PORTED FROM SYMPY: sympy/functions/elementary/miscellaneous.py:Max._eval_rewrite_as_Heaviside,
# Min._eval_rewrite_as_Heaviside
def _minmax_heaviside(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.functions.elementary.miscellaneous import Max
    from sympy_extras.functions.special.delta_functions import heaviside
    is_max = isinstance(node, Max)
    return add(*[mul(j, mul(*[heaviside(_sub(j, i) if is_max else _sub(i, j)) for i in args if i != j]))
                 for j in args])


# PORTED FROM SYMPY: sympy/functions/elementary/miscellaneous.py:_minmax_as_Piecewise (Max and Min
# ._eval_rewrite_as_Piecewise)
def _minmax_piecewise(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.core.logic import and_
    from sympy_extras.core.relational import ge, le
    from sympy_extras.functions.elementary.miscellaneous import Max
    from sympy_extras.functions.elementary.piecewise import PairLike, piecewise
    relation = ge if isinstance(node, Max) else le
    pairs: list[PairLike] = []
    for i, a in enumerate(args):
        pairs.append((a, and_(*[relation(a, args[j]) for j in range(i + 1, len(args))])))
    return piecewise(*pairs)


# -- delta functions

# PORTED FROM SYMPY: sympy/functions/special/delta_functions.py:DiracDelta._eval_rewrite_as_Piecewise
def _dirac_piecewise(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.core.relational import eq
    from sympy_extras.functions.elementary.piecewise import piecewise
    from sympy_extras.functions.special.delta_functions import DiracDelta
    if len(args) == 1:
        return piecewise((DiracDelta(ZERO), eq(args[0], ZERO)), (ZERO, True))
    return None


def _single_root(e: Expr, x: Symbol) -> Expr:
    """SymPy's ``solve(e, x)[0]``."""
    from sympy_extras.solvers.solvers import solve_equation
    roots = solve_equation(e, x)
    if not roots:
        raise IndexError('list index out of range')
    return roots[0]


def _only_symbol(node: Expr, function: str) -> Symbol:
    free = node.free_symbols
    if len(free) != 1:
        raise TypeError('rewrite(SingularityFunction) does not support arguments with more that one variable.')
    x = next(iter(free))
    assert isinstance(x, Symbol)
    return x


# PORTED FROM SYMPY: sympy/functions/special/delta_functions.py:DiracDelta._eval_rewrite_as_SingularityFunction
def _dirac_singularity(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.functions.special.delta_functions import DiracDelta
    from sympy_extras.functions.special.singularity_functions import SingularityFunction
    if node == DiracDelta(ZERO):
        return SingularityFunction(ZERO, ZERO, Integer(-1))
    if node == DiracDelta(ZERO, ONE):
        return SingularityFunction(ZERO, ZERO, Integer(-2))
    x = _only_symbol(node, 'DiracDelta')
    if len(args) == 1:
        return SingularityFunction(x, _single_root(args[0], x), Integer(-1))
    return SingularityFunction(x, _single_root(args[0], x), add(neg(args[1]), Integer(-1)))


def _h0(args: tuple[Expr, ...]) -> Expr:
    return args[1] if len(args) > 1 else HALF


# PORTED FROM SYMPY: sympy/functions/special/delta_functions.py:Heaviside._eval_rewrite_as_Piecewise
def _heaviside_piecewise(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.core.relational import eq, le, lt
    from sympy_extras.functions.elementary.piecewise import piecewise
    a, h0 = args[0], _h0(args)
    if h0 == ZERO:
        return piecewise((ZERO, le(a, ZERO)), (ONE, True))
    if h0 == ONE:
        return piecewise((ZERO, lt(a, ZERO)), (ONE, True))
    return piecewise((ZERO, lt(a, ZERO)), (h0, eq(a, ZERO)), (ONE, True))


# PORTED FROM SYMPY: sympy/functions/special/delta_functions.py:Heaviside._eval_rewrite_as_sign
def _heaviside_sign(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.core.relational import eq, ne
    from sympy_extras.functions.elementary.complexes import sign
    from sympy_extras.functions.elementary.piecewise import piecewise
    from sympy_extras.functions.special.delta_functions import heaviside
    a, h0 = args[0], _h0(args)
    if a.is_extended_real is True:
        half_sign = mul(add(sign(a), ONE), HALF)
        pw1 = piecewise((half_sign, ne(a, ZERO)), (heaviside(ZERO, h0), True))
        return piecewise((half_sign, eq(heaviside(ZERO, h0), HALF)), (pw1, True))
    return None


# PORTED FROM SYMPY: sympy/functions/special/delta_functions.py:Heaviside._eval_rewrite_as_SingularityFunction
def _heaviside_singularity(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    # SymPy's first case, self == Heaviside(0), never holds: Heaviside(0) is
    # evaluated (to H0)
    from sympy_extras.functions.special.singularity_functions import SingularityFunction
    x = _only_symbol(node, 'Heaviside')
    return SingularityFunction(x, _single_root(args[0], x), ZERO)


# PORTED FROM SYMPY: sympy/functions/special/singularity_functions.py:SingularityFunction._eval_rewrite_as_Piecewise
def _singularity_piecewise(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.core.numbers import oo
    from sympy_extras.core.power import pow
    from sympy_extras.core.relational import eq, ge
    from sympy_extras.functions.elementary.piecewise import piecewise
    x, a, n = [t for t in node.args if isinstance(t, Expr)]
    if n in (Integer(-1), Integer(-2), Integer(-3), Integer(-4)):
        return piecewise((oo, eq(_sub(x, a), ZERO)), (ZERO, True))
    if n.is_nonnegative is True:
        return piecewise((pow(_sub(x, a), n), ge(_sub(x, a), ZERO)), (ZERO, True))
    return None


# PORTED FROM SYMPY: sympy/functions/special/singularity_functions.py:SingularityFunction._eval_rewrite_as_Heaviside
# (also _eval_rewrite_as_DiracDelta and _eval_rewrite_as_HeavisideDiracDelta)
def _singularity_heaviside(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.core.power import pow
    from sympy_extras.core.sorting import default_sort_key
    from sympy_extras.functions.special.delta_functions import heaviside
    x, a, n = [t for t in node.args if isinstance(t, Expr)]
    for order in (4, 3, 2, 1):
        if n == Integer(-order):
            # SymPy differentiates in x.free_symbols.pop() (an arbitrary one
            # of several): the first in the default order here
            symbols = sorted(x.free_symbols, key=default_sort_key)
            if not symbols:
                raise KeyError('pop from an empty set')
            v = symbols[0]
            assert isinstance(v, Symbol)
            return heaviside(_sub(x, a)).diff(v, order)
    if n.is_nonnegative is True:
        return mul(pow(_sub(x, a), n), heaviside(_sub(x, a), ONE))
    return None


# PORTED FROM SYMPY: sympy/functions/special/tensor_functions.py:KroneckerDelta._eval_rewrite_as_Piecewise
def _kronecker_piecewise(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.core.relational import ne
    from sympy_extras.functions.elementary.piecewise import piecewise
    i, j = args[0], args[1]
    return piecewise((ZERO, ne(i, j)), (ONE, True))


def rules() -> dict[RuleKey, Rule]:
    """The rules of expressions keyed by (the class of the node, the
    target class or SymPy name)."""
    from sympy_extras.core.mod import Mod
    from sympy_extras.functions.elementary import complexes as c, integers as n, miscellaneous as m
    from sympy_extras.functions.special.delta_functions import DiracDelta, Heaviside
    from sympy_extras.functions.special.singularity_functions import SingularityFunctionNode
    from sympy_extras.functions.special.tensor_functions import KroneckerDeltaNode
    table: dict[RuleKey, Rule] = {
        (c.Re, c.Im): _re_im, (c.Im, c.Re): _im_re,
        (c.Sign, 'Piecewise'): _sign_piecewise, (c.Sign, Heaviside): _sign_heaviside,
        (c.Abs, Heaviside): _abs_heaviside, (c.Abs, 'Piecewise'): _abs_piecewise, (c.Abs, c.Sign): _abs_sign,
        (c.Abs, c.Conjugate): _abs_conjugate, (c.Arg, c.Atan2): _arg_atan2,
        (c.Atan2, 'log'): _atan2_log, (c.Atan2, 'atan'): _atan2_atan, (c.Atan2, c.Arg): _atan2_arg,
        (n.Floor, n.Ceiling): _floor_ceiling, (n.Floor, n.Frac): _floor_frac,
        (n.Ceiling, n.Floor): _ceiling_floor, (n.Ceiling, n.Frac): _ceiling_frac,
        (n.Frac, n.Floor): _frac_floor, (n.Frac, n.Ceiling): _frac_ceiling, (Mod, n.Floor): _mod_floor,
        (m.MinMaxBase, c.Abs): _minmax_abs, (m.MinMaxBase, Heaviside): _minmax_heaviside,
        (m.MinMaxBase, 'Piecewise'): _minmax_piecewise,
        (DiracDelta, 'Piecewise'): _dirac_piecewise, (DiracDelta, SingularityFunctionNode): _dirac_singularity,
        (Heaviside, 'Piecewise'): _heaviside_piecewise, (Heaviside, c.Sign): _heaviside_sign,
        (Heaviside, SingularityFunctionNode): _heaviside_singularity,
        (SingularityFunctionNode, 'Piecewise'): _singularity_piecewise,
        (SingularityFunctionNode, Heaviside): _singularity_heaviside,
        (SingularityFunctionNode, DiracDelta): _singularity_heaviside,
        (SingularityFunctionNode, 'HeavisideDiracDelta'): _singularity_heaviside,
        (KroneckerDeltaNode, 'Piecewise'): _kronecker_piecewise}
    return table


# -- the nodes which are not expressions, or whose result is not one

# PORTED FROM SYMPY: sympy/functions/elementary/piecewise.py:Piecewise._eval_rewrite_as_KroneckerDelta
def _piecewise_kronecker(node: Basic, args: tuple[Basic, ...], hints: RewriteHints) -> Optional[Basic]:
    from sympy_extras.core.logic import And, Not, Or
    from sympy_extras.core.relational import Eq, Ne
    from sympy_extras.functions.elementary.piecewise import ExprCondPair
    from sympy_extras.functions.special.tensor_functions import KroneckerDelta

    def as_delta(cond: Boolean) -> Optional[Expr]:
        if isinstance(cond, Eq):
            return KroneckerDelta(cond.lhs, cond.rhs)
        if isinstance(cond, Ne):
            return _sub(ONE, KroneckerDelta(cond.lhs, cond.rhs))
        if not isinstance(cond, (And, Or, Not)):
            return None  # SymPy's UnrecognizedCondition
        parts: list[Expr] = []
        for c in cond.args:
            if not isinstance(c, Boolean):
                return None
            k = as_delta(c)
            if k is None:
                return None
            parts.append(k)
        if isinstance(cond, And):
            return mul(*parts)
        product = mul(*[_sub(ONE, k) for k in parts])
        return _sub(ONE, product) if isinstance(cond, Or) else product

    conditions: list[tuple[Expr, Boolean]] = []
    true_value: Optional[Expr] = None
    for pair in args:
        if not isinstance(pair, ExprCondPair):
            return None
        value, cond = pair.expr, pair.cond
        if isinstance(cond, (And, Or, Not, Eq, Ne)) and type(cond) in (And, Or, Not, Eq, Ne):
            conditions.append((value, cond))
        elif cond == true:
            if true_value is None:
                true_value = value
        else:
            return None
    if true_value is None:
        return None
    result = true_value
    for value, cond in conditions[::-1]:
        k = as_delta(cond)
        if k is None:
            return None
        result = add(mul(k, value), mul(_sub(ONE, k), result))
    return result


# PORTED FROM SYMPY: sympy/logic/boolalg.py:And._eval_rewrite_as_Nor
def _and_nor(node: Basic, args: tuple[Basic, ...], hints: RewriteHints) -> Optional[Basic]:
    from sympy_extras.core.logic import Nor, not_
    return Nor(*[not_(a) for a in node.args if isinstance(a, Boolean)])


# PORTED FROM SYMPY: sympy/logic/boolalg.py:Or._eval_rewrite_as_Nand
def _or_nand(node: Basic, args: tuple[Basic, ...], hints: RewriteHints) -> Optional[Basic]:
    from sympy_extras.core.logic import Nand, not_
    return Nand(*[not_(a) for a in node.args if isinstance(a, Boolean)])


def _parity_terms(n: int, odd: bool) -> list[list[int]]:
    """SymPy's ``_get_odd_parity_terms`` and ``_get_even_parity_terms``:
    the binary words of length ``n`` (most significant digit first) with
    an odd or an even number of ones."""
    words = [[(i >> (n - 1 - k)) & 1 for k in range(n)] for i in range(2**n)]
    return [w for w in words if sum(w) % 2 == (1 if odd else 0)]


# PORTED FROM SYMPY: sympy/logic/boolalg.py:Xor._eval_rewrite_as_Or, Xor._eval_rewrite_as_And
# (with _convert_to_varsSOP, _convert_to_varsPOS)
def _xor_normal_form(node: Basic, disjunctive: bool) -> Optional[Basic]:
    from sympy_extras.core.logic import and_, not_, or_
    variables = [a for a in node.args if isinstance(a, Boolean)]
    if len(variables) != len(node.args):
        return None
    if disjunctive:
        return or_(*[and_(*[v if bit == 1 else not_(v) for v, bit in zip(variables, word)])
                     for word in _parity_terms(len(variables), True)])
    return and_(*[or_(*[v if bit == 0 else not_(v) for v, bit in zip(variables, word)])
                  for word in _parity_terms(len(variables), False)])


def _xor_or(node: Basic, args: tuple[Basic, ...], hints: RewriteHints) -> Optional[Basic]:
    return _xor_normal_form(node, True)


def _xor_and(node: Basic, args: tuple[Basic, ...], hints: RewriteHints) -> Optional[Basic]:
    return _xor_normal_form(node, False)


# PORTED FROM SYMPY: sympy/core/relational.py:Equality._eval_rewrite_as_Add (evaluate=True; deprecated in SymPy 1.13)
def _eq_add(node: Basic, args: tuple[Basic, ...], hints: RewriteHints) -> Optional[Basic]:
    if len(args) != 2 or not isinstance(args[0], Expr) or not isinstance(args[1], Expr):
        return None
    lhs, rhs = args[0], args[1]
    if lhs == ZERO:
        return rhs
    if rhs == ZERO:
        return lhs
    return _sub(lhs, rhs)


# PORTED FROM SYMPY: sympy/concrete/summations.py:Sum._eval_rewrite_as_Product
def _sum_product(node: Basic, args: tuple[Basic, ...], hints: RewriteHints) -> Optional[Basic]:
    from sympy_extras.concrete.expr_with_limits import Sum, product_
    from sympy_extras.functions.elementary.exponential import exp, log
    if not isinstance(node, Sum):
        return None
    if node.function.is_extended_real is True:
        return log(product_(exp(node.function), *node.limits))
    return None


# PORTED FROM SYMPY: sympy/concrete/products.py:Product._eval_rewrite_as_Sum
def _product_sum(node: Basic, args: tuple[Basic, ...], hints: RewriteHints) -> Optional[Basic]:
    from sympy_extras.concrete.expr_with_limits import Product, sum_
    from sympy_extras.functions.elementary.exponential import exp, log
    if not isinstance(node, Product):
        return None
    return exp(sum_(log(node.function), *node.limits))


def node_rules() -> dict[RuleKey, NodeRule]:
    """The rules of the nodes which are not expressions, or whose result
    is not one, keyed by (the class of the node, the target)."""
    from sympy_extras.concrete.expr_with_limits import Product, Sum
    from sympy_extras.core.logic import And, Or, Xor
    from sympy_extras.core.relational import Eq
    from sympy_extras.functions.elementary.piecewise import Piecewise
    from sympy_extras.functions.special.tensor_functions import KroneckerDeltaNode
    table: dict[RuleKey, NodeRule] = {
        (Piecewise, KroneckerDeltaNode): _piecewise_kronecker, (And, 'Nor'): _and_nor, (Or, 'Nand'): _or_nand,
        (Xor, Or): _xor_or, (Xor, And): _xor_and, (Eq, 'Add'): _eq_add, (Sum, Product): _sum_product,
        (Product, Sum): _product_sum}
    return table

