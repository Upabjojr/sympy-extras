from __future__ import annotations

from sympy import And, Eq, Equivalent, FiniteSet, Integer, Ne, Or, Rational, S, sqrt
from sympy.abc import a, b, c, x, y

from sympy_extras._typing import as_expr, free_symbols, sorted_symbols
from sympy_extras.assumptions import ForAll, resolve
from sympy_extras.polys.regularchains import triangularize
from sympy_extras.solvers.parametric import parametric_cases


def test_the_cases_of_a_linear_and_of_a_quadratic_equation() -> None:
    linear = parametric_cases([a*x - b], [x])
    assert [(case.condition, case.solutions) for case in linear] == [
        (Ne(a, 0), FiniteSet((b/a,))), (And(Eq(a, 0), Eq(b, 0)), FiniteSet((x,)))]
    quadratic = parametric_cases([a*x**2 + b*x + c], [x])
    assert [case.condition for case in quadratic] == [
        Ne(a, 0), And(Eq(a, 0), Ne(b, 0)), And(Eq(a, 0), Eq(b, 0), Eq(c, 0))]
    root = sqrt(b**2 - 4*a*c)
    assert quadratic[0].solutions == FiniteSet((-b/(2*a) - root/(2*a),), (-b/(2*a) + root/(2*a),))
    assert quadratic[1].solutions == FiniteSet((-c/b,)) and quadratic[2].solutions == FiniteSet((x,))


def test_an_inequation_in_the_parameters_stays_when_the_chain_implies_it() -> None:
    # the chain (a + 1)*y - 1, (a + 1)*x - 1 has no zero with a = -1, so that
    # a + 1 != 0 is implied by its equations, and was not written: but the
    # solution 1/(a + 1) was written for every a
    cases = parametric_cases([a*x + y - 1, x + a*y - 1], [x, y])
    assert [(case.condition, case.solutions) for case in cases] == [
        (Eq(a, 1), FiniteSet((1 - y, y))), (Ne(a + 1, 0), FiniteSet((1/(a + 1), 1/(a + 1))))]
    assert parametric_cases([a*x*y - 1, x - y], [x, y])[0].condition == Ne(a, 0)
    # one in the unknowns which is implied is not written: y != 0 with a*y**2 = 1
    [case] = parametric_cases([x**2 - a, x*y - 1], [x, y])
    assert case.condition == Ne(a, 0)
    assert case.solutions == FiniteSet((-sqrt(a), -1/sqrt(a)), (sqrt(a), 1/sqrt(a)))


def test_inequations_and_higher_degrees() -> None:
    [case] = parametric_cases([a*x - b], [x], inequations=[b])
    assert case.condition == And(Ne(a, 0), Ne(b, 0)) and case.solutions == FiniteSet((b/a,))
    [cubic] = parametric_cases([x**3 - a], [x])
    assert cubic.solutions is None and cubic.equations == [x**3 - a]


def test_the_cases_are_the_system() -> None:
    # by the quantifier elimination over the complex numbers, which works
    # with comprehensive Groebner systems
    for system, unknowns, excluded in [
            ([a*x - b], [x], []), ([a*x**2 + b*x + c], [x], []), ([a*x + y - 1, x + a*y - 1], [x, y], []),
            ([x*y - a, x + y - b], [x, y], []), ([a*x - 1, b*x - 1], [x], []), ([a*x - b], [x], [b]),
            ([a*x + b*y - 1, b*x + a*y - 1], [x, y], []), ([x**2 - a, x*y - 1], [x, y], [x - 1])]:
        cases = parametric_cases(system, unknowns, inequations=excluded)
        ours = Or(*[case.as_formula() for case in cases])
        given = And(*[Eq(e, 0) for e in system], *[Ne(h, 0) for h in excluded])
        names = sorted_symbols(free_symbols(given) | free_symbols(ours))
        assert resolve(ForAll(names, Equivalent(given, ours)), domain=S.Complexes) == S.true, system


def test_the_solutions_at_values_of_the_parameters() -> None:
    # the points of the cases whose condition holds are the solutions of the
    # system with the values put in
    system = [x*y - a, x + y - b]
    cases = parametric_cases(system, [x, y])
    for values in ({a: Rational(3, 2), b: Integer(2)}, {a: Integer(0), b: Integer(1)},
                   {a: Integer(1), b: Integer(2)}, {a: Integer(0), b: Integer(0)}):
        expected: list[tuple[complex, complex]] = []
        for chain in triangularize([e.subs(values) for e in system], x, y):
            expected.extend((complex(s[x]), complex(s[y])) for s in chain.numerical_solutions(20))
        found: list[tuple[complex, complex]] = []
        for case in cases:
            assert case.solutions is not None
            for point in case.solutions.args:
                try:
                    first, second = as_expr(point.args[0]).subs(values), as_expr(point.args[1]).subs(values)
                    holds = case.condition.subs(values).subs({x: first, y: second})
                except ZeroDivisionError:
                    continue
                if holds == S.true and first.is_finite and second.is_finite:
                    found.append((complex(first), complex(second)))
        assert all(any(abs(p[0] - q[0]) + abs(p[1] - q[1]) < 1e-9 for q in found) for p in expected), values
        assert all(any(abs(p[0] - q[0]) + abs(p[1] - q[1]) < 1e-9 for q in expected) for p in found), values
