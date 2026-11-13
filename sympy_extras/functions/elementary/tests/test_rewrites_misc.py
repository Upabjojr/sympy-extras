"""The native rewrites of ``functions._rewrites_misc`` (complex parts,
``sign``, ``Abs``, ``arg``, ``atan2``, the integer parts, ``Max``,
``Min``, ``Mod``, the delta and singularity functions, ``KroneckerDelta``,
``Piecewise``, the logical connectives, ``Eq``, sums and products), checked
by the values: each result agrees numerically with the node it comes from
at sample points (reliable values), or by truth tables for the
connectives. They went to SymPy through the bridge before."""
from __future__ import annotations

from itertools import product as cartesian
from typing import Callable, Optional, Union

from sympy_extras._numeric import reliable_complex
from sympy_extras.core import Integer, Rational, Symbol
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr
from sympy_extras.core.logic import And, Boolean, Not, Or, Xor, false, true
from sympy_extras.core.mod import Mod
from sympy_extras.core.numbers import I, pi
from sympy_extras.core.relational import Eq, Ne
from sympy_extras.functions.elementary import complexes as c, integers as n
from sympy_extras.functions.elementary.miscellaneous import Max, Min
from sympy_extras.functions.elementary.piecewise import Piecewise
from sympy_extras.functions.special.delta_functions import DiracDelta, Heaviside
from sympy_extras.functions.special.singularity_functions import SingularityFunctionNode
from sympy_extras.functions.special.tensor_functions import KroneckerDeltaNode

x, y = Symbol('x'), Symbol('y')
r, t = Symbol('r', real=True), Symbol('t', real=True)
k = Symbol('k', integer=True)
REAL: list[Expr] = [Rational(7, 10), Rational(-13, 10), Rational(23, 10), Rational(-1, 3)]
COMPLEX: list[Expr] = REAL + [Rational(3, 10) + Rational(2, 5) * I, Rational(-1, 3) - Rational(6, 5) * I]
Target = Union[type, str]


def _value(e: Expr, values: dict[Symbol, Expr]) -> Optional[complex]:
    return reliable_complex(e, 15, values)


def _agree(before: Expr, after: Basic, symbols: list[Symbol], points: list[Expr]) -> None:
    assert isinstance(after, Expr), after
    assert after != before, before
    checked = 0
    for j in range(len(points)):
        values: dict[Symbol, Expr] = {s: points[(j + i) % len(points)] for i, s in enumerate(symbols)}
        v0, v1 = _value(before, values), _value(after, values)
        if v0 is None or v1 is None:
            continue
        checked += 1
        assert abs(v0 - v1) <= 1e-10 * max(1.0, abs(v0)), (before, after, values, v0, v1)
    assert checked, (before, after)


def test_complex_parts_sign_abs_arg_atan2_agree_numerically() -> None:
    cases: list[tuple[Expr, Target, list[Symbol], list[Expr]]] = [
        (c.Re(x + I * y), c.Im, [x, y], COMPLEX), (c.Im(2 * x + 1), c.Re, [x], COMPLEX),
        (c.Sign(2 * r + 1), 'Piecewise', [r], REAL), (c.Sign(r), Heaviside, [r], REAL),
        (c.Abs(r - 1), Heaviside, [r], REAL), (c.Abs(r), 'Piecewise', [r], REAL),
        (c.Abs(x), c.Sign, [x], COMPLEX), (c.Abs(x + 2), c.Conjugate, [x], COMPLEX),
        (c.Arg(x), c.Atan2, [x], COMPLEX), (c.Atan2(y, x), 'log', [y, x], REAL),
        (c.Atan2(y, x), 'atan', [y, x], REAL), (c.Atan2(r, t), c.Arg, [r, t], REAL),
        (c.Atan2(y, x), c.Arg, [y, x], COMPLEX)]
    for node, target, symbols, points in cases:
        _agree(node, node.rewrite(target), symbols, points)
    # an imaginary argument of Abs, as Piecewise
    v = Symbol('v', imaginary=True)
    rewritten = c.Abs(v).rewrite('Piecewise')
    assert isinstance(rewritten, Piecewise)
    assert rewritten.subs(v, 3 * I) == 3 and rewritten.subs(v, -2 * I) == 2
    # SymPy's rules apply to real arguments only: the node is rebuilt
    assert c.Sign(x).rewrite(Heaviside) == c.sign(x)


def test_integer_parts_and_mod_agree_numerically() -> None:
    pairs: list[tuple[Callable[[Expr], Expr], list[Target]]] = [
        (n.Floor, [n.Ceiling, n.Frac]), (n.Ceiling, [n.Floor, n.Frac]), (n.Frac, [n.Floor, n.Ceiling])]
    for cls, targets in pairs:
        for target in targets:
            for a in (r, 2 * r + Rational(1, 3)):
                node = cls(a)
                _agree(node, node.rewrite(target), [r], REAL)
    node = Mod(r, Rational(5, 2))
    _agree(node, node.rewrite(n.Floor), [r], REAL)


def test_max_min_agree_numerically() -> None:
    for cls in (Max, Min):
        for args in ((r, t), (r, Integer(2)), (r, t, Integer(1))):
            node = cls(*args)
            for target in (c.Abs, Heaviside, 'Piecewise'):
                _agree(node, node.rewrite(target), [r, t], REAL)


def test_delta_and_singularity_functions_agree_numerically() -> None:
    a = Symbol('a', real=True)
    for h0 in (Integer(0), Integer(1), Rational(1, 2)):
        node = Heaviside(a - 1, h0)
        _agree(node, node.rewrite('Piecewise'), [a], REAL + [Integer(1)])
        _agree(node, node.rewrite(c.Sign), [a], REAL + [Integer(1)])
    node = Heaviside(2 * a - 1, Rational(1, 2))
    _agree(node, node.rewrite(SingularityFunctionNode), [a], REAL)
    for order in (Integer(0), Integer(1), Integer(3)):
        sf = SingularityFunctionNode(a, Integer(2), order)
        _agree(sf, sf.rewrite('Piecewise'), [a], REAL + [Rational(5, 2)])
        _agree(sf, sf.rewrite(Heaviside), [a], REAL + [Rational(5, 2)])
    # the impulses: Piecewise with oo at the point, derivatives of Heaviside
    assert SingularityFunctionNode(a, Integer(2), Integer(-1)).rewrite(DiracDelta) == DiracDelta(a - 2)
    assert DiracDelta(a - 5).rewrite(SingularityFunctionNode) == SingularityFunctionNode(a, Integer(5), Integer(-1))
    # DiracDelta(0) as SingularityFunction(0, 0, -1), which is oo (as in SymPy)
    from sympy_extras.functions.special.singularity_functions import SingularityFunction
    assert DiracDelta(Integer(0)).rewrite(SingularityFunctionNode) == \
        SingularityFunction(Integer(0), Integer(0), Integer(-1))
    pw = DiracDelta(a - 5).rewrite('Piecewise')
    assert isinstance(pw, Piecewise)
    assert pw.subs(a, 3) == 0 and pw.subs(a, 5) == DiracDelta(Integer(0))


def test_kronecker_delta_and_piecewise_agree_at_integers() -> None:
    m = Symbol('m', integer=True)
    integers: list[Expr] = [Integer(i) for i in (1, 2, 3, -1)]
    _agree(KroneckerDeltaNode(k, m), KroneckerDeltaNode(k, m).rewrite('Piecewise'), [k, m], integers)
    from sympy_extras.functions.elementary.piecewise import piecewise
    cases: list[Expr] = [piecewise((k, Eq(k, 1)), (Integer(2), True)), piecewise((k, Ne(k, m)), (m**2, True)),
                         piecewise((Integer(1), And(Eq(k, 1), Eq(m, 2))), (Integer(0), True)),
                         piecewise((k, Or(Eq(k, 1), Eq(m, 2))), (Integer(3), Eq(k, 3)), (Integer(0), True)),
                         piecewise((k, Not(Eq(k, 1))), (Integer(0), True))]
    for pw in cases:
        rewritten = pw.rewrite(KroneckerDeltaNode)
        assert isinstance(rewritten, Expr) and not rewritten.has(Piecewise), rewritten
        for i, j in cartesian((1, 2, 3), (1, 2)):
            assert rewritten.subs({k: i, m: j}) == pw.subs({k: i, m: j}), (pw, i, j)
    # a condition which is not an equality: the node is kept
    pw = piecewise((k, k > 1), (Integer(0), True))
    assert pw.rewrite(KroneckerDeltaNode) == pw


def _truth(b: Basic, values: dict[Symbol, Boolean]) -> Basic:
    result = b.subs(values)
    assert result in (true, false), result
    return result


def test_connectives_keep_their_truth_tables() -> None:
    p, q, s, u = Symbol('p'), Symbol('q'), Symbol('s'), Symbol('u')
    from sympy_extras.core.logic import and_, or_, xor
    cases: list[tuple[Basic, Target, list[Symbol]]] = [
        (and_(p, q, s), 'Nor', [p, q, s]), (or_(p, q), 'Nand', [p, q]),
        (Xor(p, q, s), Or, [p, q, s]), (Xor(p, q, s, u), And, [p, q, s, u]), (Xor(p, q), And, [p, q])]
    for node, target, symbols in cases:
        rewritten = node.rewrite(target)
        assert rewritten != node
        for bits in cartesian((true, false), repeat=len(symbols)):
            values: dict[Symbol, Boolean] = dict(zip(symbols, bits))
            expected = _truth(node, values) if not isinstance(node, Xor) else xor(*bits)
            assert _truth(rewritten, values) == expected, (node, target, bits)


def test_eq_sum_and_product_rewrites() -> None:
    from sympy_extras.concrete.expr_with_limits import Product, Sum
    assert Eq(x + y, x - y).rewrite('Add') == 2 * y
    assert Eq(x, Integer(0)).rewrite('Add') == x
    # a positive summation variable: 1/j**2 is real (not at j = 0)
    j = Symbol('j', integer=True, positive=True)
    s = Sum(1 / j**2, (j, Integer(1), Integer(4)))
    as_product = s.rewrite(Product)
    assert isinstance(as_product, Expr) and as_product.has(Product)
    v0, v1 = _value(s.doit(), {}), _value(as_product.doit(), {})
    assert v0 is not None and v1 is not None and abs(v0 - v1) < 1e-12
    pr = Product(j + 1, (j, Integer(1), Integer(4)))
    as_sum = pr.rewrite(Sum)
    assert isinstance(as_sum, Expr) and as_sum.has(Sum)
    v0, v1 = _value(pr.doit(), {}), _value(as_sum.doit(), {})
    assert v0 is not None and v1 is not None and abs(v0 - v1) < 1e-9
    # a complex summand: SymPy's rule does not apply
    assert Sum(I * j, (j, Integer(1), Integer(3))).rewrite(Product) == Sum(I * j, (j, Integer(1), Integer(3)))
    assert pi.rewrite(Sum) == pi
