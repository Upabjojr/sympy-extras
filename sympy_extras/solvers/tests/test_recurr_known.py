"""The recurrence solvers against independent sources: the sequences of
the literature (Graham, Knuth, Patashnik, *Concrete Mathematics*;
Petkovšek, Wilf, Zeilberger, *A = B*; the documented examples of
Mathematica's ``RSolve``) with their published closed forms and values,
and, for every equation, the values obtained by iterating the recurrence
from the initial values, which the closed form must reproduce. The solvers
are native: the equations are written with SymPy, whose ``simplify`` and
number functions are the oracles, and converted at the call."""
from __future__ import annotations

from typing import Optional, Sequence, Union

from sympy import (Eq, Expr, Function, Integer, Rational, S, Sum, Symbol, binomial, catalan, factorial, fibonacci,
    harmonic, lucas, simplify, sqrt, symbols, sympify)
from sympy.core.function import AppliedUndef

from sympy_extras._typing import as_expr
from sympy_extras.core.expr import Expr as CoreExpr
from sympy_extras.core.function import AppliedUndef as CoreAppliedUndef
from sympy_extras.core.relational import Eq as CoreEq
from sympy_extras.core.symbol import Symbol as CoreSymbol
from sympy_extras.interop import expr_from_sympy, from_sympy, sympy_expr, to_sympy
from sympy_extras.solvers import recurr

n = Symbol('n')
y = Function('y')
Init = dict[AppliedUndef, Union[Expr, int]]


def rsolve_linear(equation: Union[Expr, Eq], init: Optional[Init] = None) -> Optional[Expr]:
    """The native ``rsolve_linear`` on the SymPy equation in ``y(n)``."""
    core = from_sympy(equation)
    function = from_sympy(y(n))
    assert isinstance(core, (CoreExpr, CoreEq)) and isinstance(function, CoreAppliedUndef)
    core_init: Optional[dict[CoreAppliedUndef, Union[CoreExpr, int]]] = None
    if init is not None:
        core_init = {}
        for point, value in init.items():
            core_point = from_sympy(point)
            assert isinstance(core_point, CoreAppliedUndef)
            core_init[core_point] = expr_from_sympy(sympify(value))
    solution = recurr.rsolve_linear(core, function, core_init)
    return None if solution is None else sympy_expr(to_sympy(solution))


def core_n() -> CoreSymbol:
    symbol = from_sympy(n)
    assert isinstance(symbol, CoreSymbol)
    return symbol


def rsolve_hypergeometric(coefficients: Sequence[Union[Expr, int]]) -> list[Expr]:
    return [sympy_expr(to_sympy(s)) for s in
            recurr.rsolve_hypergeometric([expr_from_sympy(sympify(c)) for c in coefficients], core_n())]


def rsolve_dalembertian(coefficients: Sequence[Union[Expr, int]]) -> list[Expr]:
    return [sympy_expr(to_sympy(s)) for s in
            recurr.rsolve_dalembertian([expr_from_sympy(sympify(c)) for c in coefficients], core_n())]


def solved(equation: Union[Expr, Eq], init: Optional[Init] = None) -> Expr:
    solution = rsolve_linear(equation, init)
    assert solution is not None, equation
    return solution


def iterated(equation: Expr, init: Init, count: int) -> list[Expr]:
    """The values y(0), ..., y(count - 1) by the recurrence itself."""
    values = {int(as_expr(point.args[0])): as_expr(sympify(v)) for point, v in init.items()}
    shifts = [int(as_expr(t.args[0]) - n) for t in equation.atoms(AppliedUndef)]
    order = max(shifts) - min(shifts)
    least = min(shifts)
    for m in range(order, count):
        # the equation at the index which involves y(m) as its highest term
        at = m - order - least
        unknown = Symbol('u')
        instance = equation.subs(n, at)
        instance = instance.subs({y(i): values[i] for i in range(m)}).subs(y(m), unknown)
        values[m] = as_expr(simplify(-instance.subs(unknown, 0)/(instance.coeff(unknown))))
    return [values[i] for i in range(count)]


def agrees(solution: Optional[Expr], values: list[Expr], start: int = 0) -> bool:
    assert solution is not None
    return all(simplify(solution.subs(n, i).doit() - values[i]) == 0 for i in range(start, len(values)))


def test_known_closed_forms() -> None:
    C0, C1 = symbols('C0 C1')
    # Tower of Hanoi (GKP 1.1, Mathematica's RSolve documentation)
    hanoi = y(n + 1) - 2*y(n) - 1
    assert solved(hanoi, {y(0): 0}) == 2**n - 1
    # the factorial and the Fibonacci numbers (A = B, chapter 8; RSolve documentation)
    assert solved(y(n + 1) - (n + 1)*y(n), {y(0): 1}) == factorial(n)
    fib = solved(y(n + 2) - y(n + 1) - y(n), {y(0): 0, y(1): 1})
    assert all(simplify(fib.subs(n, i) - fibonacci(i)) == 0 for i in range(10))
    luc = solved(y(n + 2) - y(n + 1) - y(n), {y(0): 2, y(1): 1})
    assert all(simplify(luc.subs(n, i) - lucas(i)) == 0 for i in range(10))
    # the Catalan numbers (GKP 7.5) and the central binomial coefficients
    cat = solved(Eq((n + 2)*y(n + 1), (4*n + 2)*y(n)), {y(0): 1})
    assert all(cat.subs(n, i) == catalan(i) for i in range(10))
    central = solved(Eq((n + 1)*y(n + 1), (4*n + 2)*y(n)), {y(0): 1})
    assert all(central.subs(n, i) == binomial(2*i, i) for i in range(10))
    # the harmonic numbers, from their second order recurrence (GKP 6.3)
    harm = solved(Eq((n + 2)*y(n + 2), (2*n + 3)*y(n + 1) - (n + 1)*y(n)), {y(0): 0, y(1): 1})
    assert all(harm.subs(n, i).doit() == harmonic(i) for i in range(10))
    # the derangement numbers, D(n + 1) = n (D(n) + D(n - 1)) (GKP 5.13: n! sum (-1)^k/k!)
    der = solved(y(n + 1) - n*(y(n) + y(n - 1)), {y(0): 1, y(1): 0})
    assert [der.subs(n, i).doit() for i in range(8)] == [1, 0, 1, 2, 9, 44, 265, 1854]
    assert all(der.subs(n, i).doit() == factorial(i)*Sum((-1)**Symbol('k')/factorial(Symbol('k')),
                                                          (Symbol('k'), 0, i)).doit() for i in range(8))
    # D(n + 1) = (n + 1) D(n) + (-1)^(n + 1), the first order form (GKP 5.13): a hypergeometric right-hand side
    der2 = solved(y(n + 1) - (n + 1)*y(n) - (-1)**(n + 1), {y(0): 1})
    assert [der2.subs(n, i).doit() for i in range(8)] == [1, 0, 1, 2, 9, 44, 265, 1854]
    # the number of subfactorial-like sums: y(n + 1) = (n + 1) y(n) + 1 gives n! sum 1/k! (A = B, section 8.5 style)
    s = solved(y(n + 1) - (n + 1)*y(n) - 1, {y(0): 1})
    assert [s.subs(n, i).doit() for i in range(7)] == [1, 2, 5, 16, 65, 326, 1957]
    # arithmetic-geometric: y(n + 1) = 2 y(n) + n has 2^n C - n - 1 (RSolve documentation)
    assert solved(y(n + 1) - 2*y(n) - n) == 2**n*C0 - n - 1
    # a recurrence with a double characteristic root: C0 2^n + C1 n 2^n
    double = solved(y(n + 2) - 4*y(n + 1) + 4*y(n))
    assert simplify(double - (C0*2**n + C1*n*2**n)) == 0 or simplify(double - (C1*2**n + C0*n*2**n)) == 0


def test_known_absence_of_solutions() -> None:
    # the involution numbers y(n + 2) = y(n + 1) + (n + 1) y(n) have no
    # hypergeometric term solution (Petkovšek's thesis; A = B, section 8.8)
    assert rsolve_hypergeometric([-(n + 1), -1, 1]) == []
    assert rsolve_dalembertian([-(n + 1), -1, 1]) == []
    # the Motzkin numbers, (n + 3) M(n + 1) = (2 n + 3) M(n) + 3 n M(n - 1): none either
    assert rsolve_hypergeometric([3*n, -(2*n + 3), n + 3]) == []
    # the Bessel-like recurrence y(n + 2) = (n + 1) y(n + 1) - y(n): none
    assert rsolve_dalembertian([1, -(n + 1), 1]) == []


def test_closed_forms_reproduce_the_iterated_recurrences() -> None:
    equations: list[tuple[Expr, Init]] = [
        (y(n + 1) - 2*y(n) - 1, {y(0): 0}),
        (y(n + 2) - y(n + 1) - y(n), {y(0): 0, y(1): 1}),
        ((n + 2)*y(n + 2) - (2*n + 3)*y(n + 1) + (n + 1)*y(n), {y(0): 0, y(1): 1}),
        (y(n + 1) - (n + 1)*y(n) - 1, {y(0): 1}),
        (y(n + 2) - (n + 3)*y(n + 1) + (n + 1)*y(n), {y(0): 1, y(1): 2}),
        (y(n + 3) - 3*y(n + 2) + 3*y(n + 1) - y(n), {y(0): 1, y(1): 3, y(2): 7}),
        (y(n + 1) - y(n) - n - 2**n, {y(0): 0}),
        (y(n + 2) - 2*y(n + 1) + y(n) - n**2, {y(0): 0, y(1): 0}),
        (2*(n + 1)*y(n + 1) - (2*n + 1)*y(n), {y(0): 1}),
        (y(n + 1) - (n + Rational(1, 2))*y(n) - factorial(n), {y(0): 1}),
        ((n + 1)*y(n + 1) - (n**2 + 1)*y(n) - 1, {y(0): 1}),
    ]
    for equation, init in equations:
        values = iterated(equation, init, 9)
        solution = solved(equation, init)
        assert agrees(solution, values), (equation, solution, values)


def test_values_of_the_general_solutions() -> None:
    """Every basis element solves its recurrence for ten consecutive
    values (the solvers check six)."""
    cases = [
        [n + 1, -(2*n + 3), n + 2], [n + 1, -(n + 3), 1], [-1, -1, 1], [n**2 + 1, -(n + 1)],
        [1, -3, 3, -1], [-(2*n + 1), 2*(n + 1)], [n, -(n + 2), 1], [-(n + Rational(1, 2)), 1],
    ]
    for coefficients in cases:
        for solution in rsolve_dalembertian(coefficients):
            start = 1 if solution.subs(n, 0).doit().is_finite is False else 0
            for at in range(start, start + 10):
                value = sum(S(c).subs(n, at)*solution.subs(n, at + i).doit() for i, c in enumerate(coefficients))
                assert simplify(value) == 0, (coefficients, solution, at)
    assert Integer(1) == 1 and sqrt(4) == 2  # keeps the imports honest
