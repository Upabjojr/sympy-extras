from __future__ import annotations

from sympy import (And, Or, Not, Implies, Equivalent, Xor, ITE, Dummy, latex,
    sstr, srepr, true, false)
from sympy.testing.pytest import raises

from sympy_extras._testing import untyped
from sympy.abc import x, y, z

from sympy_extras.assumptions.quantifiers_sympy import ForAll, Exists, Quantifier, prenex


def test_construction() -> None:
    f = ForAll(x, x > 0)
    assert isinstance(f, Quantifier) and f.quantifier == 'forall'
    assert f.variables == (x,) and f.formula == (x > 0)
    assert f.free_symbols == set()
    assert f.bound_symbols == [x]
    g = ForAll([x, y], x > y)
    assert isinstance(g, ForAll) and g.variables == (x, y)
    assert ForAll((x, y), x > y) == ForAll([x, y], x > y)
    assert ForAll(x, x > y).free_symbols == {y}
    e = Exists(x, x > y)
    assert isinstance(e, Exists) and e.quantifier == 'exists'
    # the constructor does not evaluate the trivial cases, simplify does
    assert isinstance(ForAll(x, True), ForAll)
    assert ForAll(x, True).simplify() is true
    assert Exists(x, false).simplify() is false
    assert ForAll(x, y > 0).simplify() == (y > 0)
    assert ForAll([x, z], y > x).simplify() == ForAll(x, x < y)
    assert ForAll(x, (x > 0) & true).simplify() == ForAll(x, x > 0)
    raises(ValueError, lambda: ForAll([], x > 0))
    raises(TypeError, lambda: untyped(ForAll)(2, x > 0))
    raises(TypeError, lambda: ForAll(x, x + 1))
    raises(ValueError, lambda: ForAll([x, x], x > 0))
    # Boolean operations
    assert (ForAll(x, x > 0) & (y > 0)).func is And
    assert (ForAll(x, x > 0) | (y > 0)).func is Or
    assert isinstance(~ForAll(x, x > 0), Not)
    assert ForAll(x, x > 0) != Exists(x, x > 0)
    assert ForAll(x, x > 0) == ForAll(x, x > 0)
    # substitution respects bound variables
    f = ForAll(x, x > y)
    assert f.subs(y, 2) == ForAll(x, x > 2)
    assert f.subs(x, 2) == f
    assert f.binary_symbols == set()
    # printing
    assert sstr(f) == "ForAll(x, x > y)"
    assert srepr(ForAll([x, y], x > y)).startswith("ForAll((Symbol('x'), Symbol('y'))")
    assert latex(f) == r"\forall x \, x > y"
    assert latex(Exists([x, y], x > y)) == r"\exists x, y \, x > y"


def test_prenex() -> None:
    assert prenex(x > 0) == ([], x > 0)
    assert prenex(True) == ([], true)
    assert prenex(ForAll(x, True)) == ([], true)
    assert prenex(ForAll([x, z], y > x)) == ([('forall', x)], y > x)
    assert prenex(ForAll(x, x > 0)) == ([('forall', x)], x > 0)
    assert prenex(ForAll([x, y], x > y)) == ([('forall', x), ('forall', y)], x > y)
    assert prenex(ForAll(x, Exists(y, x < y))) == ([('forall', x), ('exists', y)], x < y)
    assert prenex(~ForAll(x, Exists(y, x < y))) == ([('exists', x), ('forall', y)], x >= y)
    assert prenex(ForAll(x, x > 0) & (y > 0)) == ([('forall', x)], And(x > 0, y > 0))
    prefix, matrix = prenex(ForAll(x, x > 0) & Exists(y, y < 0))
    assert sorted(prefix, key=str) == [('exists', y), ('forall', x)]
    assert matrix == And(x > 0, y < 0)
    # clashing bound variables are renamed
    prefix, matrix = prenex(ForAll(x, x > 0) & Exists(x, x < 0))
    assert len(prefix) == 2 and {k for k, _ in prefix} == {'forall', 'exists'}
    variables = [v for _, v in prefix]
    assert x in variables
    [d] = [v for v in variables if v != x]
    assert isinstance(d, Dummy) and d.name == 'x'
    assert matrix.free_symbols == {x, d}
    # a bound variable clashing with a free variable of another argument
    prefix, matrix = prenex(Exists(x, x > y) & (x > 0))
    [(kind, d)] = prefix
    assert kind == 'exists' and isinstance(d, Dummy)
    assert matrix == And(d > y, x > 0)
    # implications and the like
    assert prenex(Implies(ForAll(x, x > 0), y > 0)) == ([('exists', x)], Or(x <= 0, y > 0))
    prefix, matrix = prenex(Equivalent(Exists(x, x > y), y < 0))
    assert [k for k, _ in prefix] == ['forall', 'exists']
    prefix, matrix = prenex(Xor(Exists(x, x > y), y < 0))
    assert sorted(k for k, _ in prefix) == ['exists', 'forall']
    prefix, matrix = prenex(ITE(Exists(x, x > y), y < 0, y > 0))
    assert sorted(k for k, _ in prefix) == ['exists', 'forall']
    # quantifiers under Not under And
    prefix, matrix = prenex(And(Not(ForAll(x, x > y)), z > 0))
    assert prefix == [('exists', x)] and matrix == And(x <= y, z > 0)


def test_native_quantifiers() -> None:
    # the nodes of the core: unevaluating classes, evaluating functions
    from sympy_extras.core import Symbol as CSymbol, Dummy as CDummy, true as ctrue, And as CAnd, Not as CNot
    from sympy_extras.assumptions import quantifiers as nq
    a, b, c = CSymbol('x'), CSymbol('y'), CSymbol('z')
    f = nq.ForAll(a, a > b)
    assert isinstance(f, nq.Quantifier) and f.quantifier == 'forall'
    assert f.variables == (a,) and f.formula == (a > b) and f.free_symbols == {b} and f.bound_symbols == (a,)
    assert str(f) == 'ForAll(x, x > y)' and str(nq.Exists([a, b], a > b)) == 'Exists((x, y), x > y)'
    assert nq.ForAll((a, b), a > b) == nq.ForAll([a, b], a > b) != nq.Exists([a, b], a > b)
    assert isinstance(nq.ForAll(a, True), nq.ForAll)
    assert nq.forall(a, True) == ctrue and nq.forall(a, b > 0) == (b > 0)
    assert nq.forall([a, c], a > b) == nq.ForAll(a, a > b)
    assert nq.exists(a, a > 0) == nq.Exists(a, a > 0)
    assert f.subs(b, 2) == nq.ForAll(a, a > 2) and f.subs(a, 2) == f
    assert f.doit() == f and nq.ForAll([a, c], a > b).doit() == f
    assert isinstance(f & (b > 0), CAnd) and isinstance(~f, CNot)
    raises(ValueError, lambda: nq.ForAll([], a > 0))
    raises(ValueError, lambda: nq.ForAll([a, a], a > 0))
    raises(TypeError, lambda: untyped(nq.ForAll)(2, a > 0))
    # prenex: the same results as the SymPy face
    assert nq.prenex(~nq.ForAll(a, nq.Exists(b, a < b))) == ([('exists', a), ('forall', b)], a >= b)
    prefix, matrix = nq.prenex(nq.ForAll(a, a > 0) & nq.Exists(a, a < 0))
    [d] = [v for _, v in prefix if v != a]
    assert isinstance(d, CDummy) and d.name == 'x' and matrix.free_symbols == {a, d}
    assert nq.prenex(nq.ForAll(a, True)) == ([], ctrue)


def test_quantifiers_convert_both_ways() -> None:
    from sympy_extras.interop import from_sympy, to_sympy
    from sympy_extras.assumptions import quantifiers as nq
    for q in [ForAll(x, x > y), Exists([x, y], (x > y) | (z < 0)), ForAll(x, Exists(y, x < y))]:
        n = from_sympy(q)
        assert isinstance(n, nq.Quantifier) and str(n) == str(q)
        assert to_sympy(n) == q
    assert isinstance(to_sympy(from_sympy(ForAll(x, x > 0) & (y > 0))), And)
