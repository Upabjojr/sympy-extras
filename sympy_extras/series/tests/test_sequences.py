"""RecursiveSeq and SeqBase.find_linear_recurrence on the native core
(sympy_extras.series.sequences); test_series_interop.py compares them
with SymPy."""
from __future__ import annotations

import pytest

from sympy_extras.core import Dummy, Eq, Function, Integer, Symbol, Tuple, as_expr, oo, sqrt
from sympy_extras.core.function import expand
from sympy_extras.core.printing import srepr
from sympy_extras.functions.combinatorial.numbers import fibonacci
from sympy_extras.polys.polytools import cancel
from sympy_extras.series.sequences import RecursiveSeq, SeqFormula, SeqPer, sequence

y, n, x = Function('y'), Symbol('n'), Symbol('x')


def test_recursive_seq_fibonacci() -> None:
    fib = RecursiveSeq(y(n - 1) + y(n - 2), y(n), n, [0, 1])
    assert fib[:10] == [fibonacci(k) for k in range(10)]
    assert fib.coeff(20) == 6765
    assert fib.degree == 2
    assert fib.recurrence == Eq(y(n), y(n - 1) + y(n - 2))
    assert fib.y == y and fib.yn == y(n) and fib.n == n
    assert fib.initial == Tuple(0, 1)
    assert fib.start == 0 and fib.stop == oo and fib.interval == (Integer(0), oo)
    assert str(fib) == 'RecursiveSeq(y(n - 2) + y(n - 1), y(n), n, (0, 1), 0)'
    assert srepr(fib).startswith("RecursiveSeq(Add(")
    assert [t for _, t in zip(range(5), fib)] == [0, 1, 1, 2, 3]
    assert fib == RecursiveSeq(y(n - 1) + y(n - 2), y(n), n, [0, 1])


def test_recursive_seq_cache() -> None:
    fib = RecursiveSeq(y(n - 1) + y(n - 2), y(n), n, [0, 1])
    assert fib.cache == {y(0): 0, y(1): 1}
    fib.coeff(4)
    assert fib.cache == {y(0): 0, y(1): 1, y(2): 1, y(3): 2, y(4): 3}


def test_recursive_seq_start() -> None:
    # SymPy 1.14 gets these wrong (it indexes its cache from 0)
    s = RecursiveSeq(y(n - 1) + y(n - 2), y(n), n, [0, 1], start=2)
    assert s[:6] == [0, 1, 1, 2, 3, 5]
    assert s.coeff(4) == 1 and s.coeff(7) == 5
    assert s.cache[y(7)] == 5
    with pytest.raises(IndexError):
        s.coeff(1)
    t = RecursiveSeq(n*y(n - 1) + y(n - 2)**2, y(n), n, [1, 2], start=-3)
    assert t[:5] == [1, 2, -1, 4, 5]


def test_recursive_seq_symbolic_initial() -> None:
    s = RecursiveSeq(y(n - 1) + y(n - 3), y(n), n)
    c = [as_expr(e) for e in s.initial.args]
    assert len(c) == 3 and all(isinstance(e, Dummy) for e in c)
    assert [expand(t) for t in s[:6]] == [c[0], c[1], c[2], c[0] + c[2], c[0] + c[1] + c[2],
                                           c[0] + c[1] + 2*c[2]]
    a, b = Symbol('a'), Symbol('b')
    u = RecursiveSeq(a*y(n - 1) - b*y(n - 2), y(n), n, [1, a])
    assert expand(u.coeff(3)) == a**3 - 2*a*b
    assert u.free_symbols == {n, a, b}
    v = u.subs(a, 2)
    assert isinstance(v, RecursiveSeq) and v.coeff(2) == 4 - b


def test_recursive_seq_errors() -> None:
    with pytest.raises(TypeError):
        RecursiveSeq(y(n - 1), n, n, [0])
    with pytest.raises(TypeError):
        RecursiveSeq(y(n - 1), y(n), y(n), [0])
    with pytest.raises(TypeError):
        RecursiveSeq(y(n - 1), y(n + 1), n, [0])
    with pytest.raises(TypeError):
        RecursiveSeq(y(n + 1), y(n), n, [0])
    with pytest.raises(TypeError):
        RecursiveSeq(y(2*n), y(n), n, [0])
    with pytest.raises(ValueError):
        RecursiveSeq(y(n - 1) + y(n - 2), y(n), n, [0])


def test_find_linear_recurrence() -> None:
    assert sequence(n**2).find_linear_recurrence(10, 2) == []
    assert sequence(n**2).find_linear_recurrence(10) == [3, -3, 1]
    assert sequence(2**n).find_linear_recurrence(10) == [2]
    assert sequence(23*n**4 + 91*n**2).find_linear_recurrence(10) == [5, -10, 10, -5, 1]
    phi = (1 + sqrt(5))/2
    assert sequence(sqrt(5)*(phi**n - (-phi)**(-n))/5).find_linear_recurrence(10) == [1, 1]
    assert SeqPer((1, 2, 3), (n, 0, oo)).find_linear_recurrence(12) == [0, 0, 1]
    fib = RecursiveSeq(y(n - 1) + y(n - 2), y(n), n, [0, 1])
    assert fib.find_linear_recurrence(10) == [1, 1]


def test_find_linear_recurrence_gfvar() -> None:
    coeffs, gf = sequence(3*5**n + 12).find_linear_recurrence(20, gfvar=x)
    assert coeffs == [6, -5]
    assert gf is not None and cancel(gf * (x - 1)*(5*x - 1)) == 15 - 63*x
    assert SeqFormula(n**2, (n, 0, oo)).find_linear_recurrence(10, 2, x) == ([], None)
    coeffs, gf = sequence(2**n).find_linear_recurrence(10, gfvar=x)
    assert coeffs == [2] and gf is not None and cancel(gf*(1 - 2*x)) == 1
