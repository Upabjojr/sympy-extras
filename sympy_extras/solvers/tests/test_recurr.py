"""Tests of the recurrence solvers: polynomial, hypergeometric and
d'Alembertian solutions. Every solution is also checked by substitution
(the solvers do that themselves; the tests check the count, the form and
the independence). The solvers are native; SymPy's ``Matrix.rank`` is the
oracle of the independence."""
from __future__ import annotations

import random
from fractions import Fraction

import sympy
from sympy.testing.pytest import raises

from sympy_extras.core import Eq, Expr, Function, I, Integer, Rational, Symbol, expand, sqrt
from sympy_extras.concrete.expr_with_limits import Product, Sum
from sympy_extras.functions import RisingFactorial, factorial, harmonic
from sympy_extras.interop import to_sympy
from sympy_extras.polys.densetools import dup_shift
from sympy_extras.polys.domains import QQ
from sympy_extras.polys.polytools import cancel, lcm
from sympy_extras.simplify.radsimp import fraction
from sympy_extras.solvers.recurr import (_value, gosper_form, is_recurrence_solution, rsolve_dalembertian,
    rsolve_hypergeometric, rsolve_linear, rsolve_polynomial)

n = Symbol('n')
y = Function('y')
ZERO, ONE = Integer(0), Integer(1)


def compose(ratio: Expr, coefficients: list[Expr]) -> list[Expr]:
    """The coefficients of ``(N - ratio)`` applied to the operator with
    the given coefficients, cleared of denominators."""
    d = len(coefficients) - 1
    out: list[Expr] = []
    for i in range(d + 2):
        previous = coefficients[i - 1].subs(n, n + 1) if 1 <= i <= d + 1 else ZERO
        current = coefficients[i] if i <= d else ZERO
        out.append(cancel(previous - ratio*current))
    common: Expr = ONE
    for c in out:
        common = lcm(common, fraction(cancel(c))[1])
    return [expand(cancel(c*common)) for c in out]


def independent(solutions: list[Expr], points: range) -> bool:
    # the sums written out (the summation evaluates the inner sums
    # symbolically first and leaves hypergeometric functions at 1)
    values = sympy.Matrix([[to_sympy(_value(s, n, i)) for s in solutions] for i in points])
    return bool(values.rank(simplify=True) == len(solutions))


def test_polynomial_solutions() -> None:
    assert rsolve_polynomial([1, -2, 1], n) == [1, n]
    assert rsolve_polynomial([1, -3, 3, -1], n) == [1, n, n**2]
    assert rsolve_polynomial([n + 1, -n], n) == [n]
    assert rsolve_polynomial([(n + 1)**2, -n**2 - 1], n) == []
    # y(n + 2) - (n + 2) y(n + 1) + n y(n) has no polynomial solution
    assert rsolve_polynomial([n, -(n + 2), 1], n) == []
    # a degree found only through the second polynomial b_1(e): the
    # difference operator lowers the degree
    assert rsolve_polynomial([-1, 1], n) == [1]
    raises(ValueError, lambda: rsolve_polynomial([0, 0], n))


def test_gosper_form() -> None:
    z, a, b, c = gosper_form((n + 2)/n, n)
    assert (z, a.as_expr(), b.as_expr(), c.as_expr()) == (1, 1, 1, n**2 + n)
    z, a, b, c = gosper_form(2*(n + 1)/(n + 3), n)
    assert (z, a.as_expr(), b.as_expr(), c.as_expr()) == (2, n + 1, n + 3, 1)
    # the ratio of n!/(n + 1): n + 2 is not a nonnegative shift of n + 1,
    # so that nothing moves into c
    z, a, b, c = gosper_form(cancel((n + 1)**2/(n + 2)), n)
    assert z == 1 and a.as_expr() == n**2 + 2*n + 1 and b.as_expr() == n + 2 and c.as_expr() == 1
    # the ratio of (n + 1) n!: n + 2 = (n + 1) shifted by one
    z, a, b, c = gosper_form(cancel((n + 2)*(n + 1)/(n + 1)), n)
    assert z == 1 and a.as_expr() == n + 2 and b.as_expr() == 1 and c.as_expr() == 1
    # n + 2 is n + 3 shifted by one: the pair moves into c
    z, a, b, c = gosper_form((n + 3)*(n + 1)/(n + 2), n)
    assert z == 1 and a.as_expr() == n + 1 and b.as_expr() == 1 and c.as_expr() == n + 2
    raises(ValueError, lambda: gosper_form(ZERO, n))


def test_hypergeometric_solutions() -> None:
    assert rsolve_hypergeometric([-(n + 1), 1], n) == [factorial(n)]
    assert rsolve_hypergeometric([-2, 1], n) == [2**n]
    assert set(rsolve_hypergeometric([-1, -1, 1], n)) == {((1 - sqrt(5))/2)**n, ((1 + sqrt(5))/2)**n}
    # two hypergeometric solutions with polynomial parts
    solutions = rsolve_hypergeometric(compose(2*(n + 1)/n, compose(n + 2, [ONE])), n)
    assert len(solutions) == 2 and independent(solutions, range(1, 5))
    # a solution with an irreducible quadratic factor stays a Product
    [solution] = rsolve_hypergeometric([n**2 + 1, -(n + 1)], n)
    assert solution.has(Product) and is_recurrence_solution([n**2 + 1, -(n + 1)], n, solution)
    # the constant solution of the harmonic recurrence, and no other
    assert rsolve_hypergeometric([n + 1, -(2*n + 3), n + 2], n) == [1]
    # no hypergeometric solution
    assert rsolve_hypergeometric([1, -(n + 1), 1], n) == []
    # the trailing coefficient zero: the recurrence in y(n + 1), y(n + 2)
    assert rsolve_hypergeometric([0, -(n + 2), 1], n) == [factorial(n)]
    # coefficients which are rational functions of n
    assert rsolve_hypergeometric([-(n + 1)/(n + 2), 1], n) == [1/(n + 1)]


def test_hypergeometric_solutions_over_extensions() -> None:
    """The roots outside the field of the coefficients: number fields over
    QQ and extensions of QQ(a) (the outputs of the SymPy-based version of the
    module, checked by substitution by the solver)."""
    a = Symbol('a')
    assert rsolve_hypergeometric([-a, 0, 1], n) == [(-sqrt(a))**n, a**(n/2)]
    assert rsolve_hypergeometric([1, 0, 1], n) == [(-I)**n, I**n]
    assert rsolve_hypergeometric([-sqrt(2), 1], n) == [2**(n/2)]
    assert len(rsolve_dalembertian([6, 0, -5, 0, 1], n)) == 4
    # (z**2 - a)**2: the polynomial parts n of the double roots over QQ(a)(sqrt(a))
    assert rsolve_dalembertian([a**2, 0, -2*a, 0, 1], n) == [(-sqrt(a))**n, n*(-sqrt(a))**n, a**(n/2), n*a**(n/2)]


def test_dalembertian_reduction_over_a_number_field() -> None:
    # (N - (n + 1)) (N**2 - N - 1): the third solution comes from the
    # reduction by ((1 - sqrt(5))/2)**n, over QQ(sqrt(5))
    solutions = rsolve_dalembertian([n + 1, n, -(n + 2), 1], n)
    assert len(solutions) == 3 and any(s.count(Sum) == 2 for s in solutions)
    assert independent(solutions, range(0, 4))
    for s in solutions:
        assert is_recurrence_solution([n + 1, n, -(n + 2), 1], n, s, count=8)


def test_dalembertian_solutions() -> None:
    k = Symbol('k', integer=True)
    assert rsolve_dalembertian([n + 1, -(2*n + 3), n + 2], n) == [1, harmonic(n)]
    assert rsolve_dalembertian([n + 1, -(n + 3), 1], n) == [
        factorial(n), factorial(n)*Sum(1/factorial(k + 1), (k, 0, n - 1))]
    assert rsolve_dalembertian([1, -2, 1], n) == [1, n]
    assert rsolve_dalembertian([-1, 3, -3, 1], n) == [1, n, n**2]
    assert rsolve_dalembertian([1, -(n + 1), 1], n) == []
    # a hypergeometric solution without a second d'Alembertian one: the
    # involution numbers y(n + 2) = y(n + 1) + (n + 1) y(n) have none, the
    # recurrence of n! times the involution numbers neither
    assert rsolve_dalembertian([-(n + 1), -1, 1], n) == []
    # nested sums: a third order operator with the factors N - 1, N - 1, N - (n + 1)
    coefficients = compose(ONE, compose(ONE, compose(n + 1, [ONE])))
    solutions = rsolve_dalembertian(coefficients, n)
    assert len(solutions) == 3 and independent(solutions, range(0, 5))
    assert any(s.count(Sum) == 2 for s in solutions)


def test_dalembertian_solutions_of_products_of_first_order_factors() -> None:
    """Random operators (N - r_3)(N - r_2)(N - r_1): exactly d independent
    solutions, each verified by the solver."""
    rng = random.Random(3)
    linear: list[Expr] = [n + 1, n + 2, 2*n + 1, n + Rational(1, 2), n**2 + 1]
    zs: list[Expr] = [ONE, Integer(2), Integer(-1), Rational(1, 2)]
    for _ in range(12):
        d = rng.randint(1, 2)
        coefficients: list[Expr] = [ONE]
        for _ in range(d):
            a: Expr = ONE
            for f in rng.sample(linear, rng.randint(0, 1)):
                a = a*f
            b: Expr = ONE
            for f in rng.sample(linear, rng.randint(0, 1)):
                b = b*f
            coefficients = compose(cancel(rng.choice(zs)*a/b), coefficients)
        solutions = rsolve_dalembertian(coefficients, n)
        assert len(solutions) == d, (coefficients, solutions)
        if d > 1:
            start = 1 if any(_value(s, n, 0).is_finite is False for s in solutions) else 0
            assert independent(solutions, range(start, start + d + 2))


def test_rsolve_linear() -> None:
    k = Symbol('k', integer=True)
    C0, C1 = Symbol('C0'), Symbol('C1')
    assert rsolve_linear(y(n + 1) - 2*y(n), y(n)) == C0*2**n
    assert rsolve_linear(y(n + 1) - 2*y(n), y(n), {y(0): 3}) == 3*2**n
    assert rsolve_linear(y(n + 1) - (n + 1)*y(n) - 1, y(n)) == \
        C0*factorial(n) + factorial(n)*Sum(1/factorial(k + 1), (k, 0, n - 1))
    assert rsolve_linear(Eq((n + 2)*y(n + 2), (2*n + 3)*y(n + 1) - (n + 1)*y(n)), y(n), {y(0): 0, y(1): 1}) == \
        harmonic(n)
    # the general solution of the Fibonacci recurrence, and the Fibonacci numbers
    general = rsolve_linear(y(n + 2) - y(n + 1) - y(n), y(n))
    assert general is not None and general.has(C0, C1) and general.subs({C0: 0, C1: 0}) == 0
    fibonacci = rsolve_linear(y(n + 2) - y(n + 1) - y(n), y(n), {y(0): 0, y(1): 1})
    assert fibonacci is not None
    assert [fibonacci.subs(n, i).expand() for i in range(8)] == [0, 1, 1, 2, 3, 5, 8, 13]
    # a polynomial right-hand side with a closed form found by the summation
    assert rsolve_linear(y(n + 1) - 2*y(n) - n, y(n)) == 2**n*C0 - n - 1
    # a sum of hypergeometric terms on the right
    assert rsolve_linear(y(n + 1) - y(n) - n - 2**n, y(n), {y(0): 0}) == 2**n + n**2/2 - n/2 - 1
    # incomplete solution spaces give None
    assert rsolve_linear(y(n + 2) - (n + 1)*y(n + 1) + y(n), y(n)) is None
    # a negative shift and rational coefficients
    assert rsolve_linear(y(n) - n*y(n - 1) - 1, y(n)) == \
        C0*factorial(n) + factorial(n)*Sum(1/factorial(k + 1), (k, 0, n - 1))
    assert rsolve_linear(y(n + 1) - y(n)/(n + 1) - 1/(n + 1), y(n)) == \
        C0/factorial(n) + Sum(factorial(k), (k, 0, n - 1))/factorial(n)
    # errors
    raises(ValueError, lambda: rsolve_linear(y(n + 1)*y(n) - 1, y(n)))
    raises(ValueError, lambda: rsolve_linear(y(n**2) - y(n), y(n)))
    raises(ValueError, lambda: rsolve_linear(n**2 - 1, y(n)))
    raises(ValueError, lambda: rsolve_linear(y(n + 1) - 2*y(n), y(n), {y(0): 1, y(1): 3}))


def test_rsolve_linear_with_parameters() -> None:
    """Parameters in the coefficients: the field QQ(a) (the outputs of the
    SymPy-based version of the module)."""
    a, C0, C1 = Symbol('a'), Symbol('C0'), Symbol('C1')
    assert rsolve_linear(y(n + 1) - a*y(n) - 1, y(n)) == C0*a**n + 1/(1 - a)
    assert rsolve_linear(y(n + 2) - 2*y(n + 1) + y(n) - a, y(n)) == C0 + C1*n + a*n**2/2
    assert rsolve_linear(y(n + 1) - (n + a)*y(n), y(n)) == C0*RisingFactorial(a, n)
    # a parameter with assumptions keeps them (the field renames the parameters)
    p = Symbol('p', positive=True)
    [solution] = rsolve_hypergeometric([-(p + 1), 1], n)
    assert solution == (p + 1)**n and solution.free_symbols == {n, p}


def test_is_recurrence_solution() -> None:
    assert is_recurrence_solution([-(n + 1), 1], n, factorial(n))
    assert is_recurrence_solution([n + 1, -(2*n + 3), n + 2], n, harmonic(n))
    assert not is_recurrence_solution([-2, 1], n, 3**n)
    assert is_recurrence_solution([-2, 1], n, 2**n + 1, rhs=-1)
    # a solution valid from n = 1 only
    assert is_recurrence_solution([n, -(n + 2), 1], n, factorial(n - 1), start=1)


def test_rising_factorials_are_written_as_factorials() -> None:
    [solution] = rsolve_hypergeometric([-(n + 3), 1], n)
    assert solution == factorial(n + 2)/2
    [solution] = rsolve_hypergeometric([-(2*n + 1), 2*(n + 1)], n)
    assert solution == RisingFactorial(Rational(1, 2), n)/factorial(n)
    # the shift used throughout
    assert dup_shift([Fraction(1), Fraction(0)], Fraction(1), QQ) == [Fraction(1), Fraction(1)]


def test_zero_test_on_roots_of_a_cubic() -> None:
    from sympy_extras._timeout import attempt
    from sympy_extras.solvers.recurr import _is_zero
    # the tribonacci recurrence: the zero test of the linear system for
    # the constants asked for the minimal polynomial (by resultants) of
    # every entry, combinations of Cardano's formulas, and
    # rsolve_linear with initial values did not return; a number which
    # is not small at 120 digits is now known not to be zero
    from sympy_extras.polys.polyroots import roots
    x = Symbol('x')
    found = roots(x**3 - x**2 - x - 1, x)
    r0, r1, r2 = list(found) if isinstance(found, dict) else list(found)
    assert attempt(lambda: _is_zero(r0 - r1), 10, native=True) is False
    assert attempt(lambda: _is_zero(r0*r1 - r2**2), 10, native=True) is False
    # the zeros are still proved: the sum of the roots is 1, their product 1
    assert attempt(lambda: _is_zero(r0 + r1 + r2 - 1), 10, native=True) is True
    assert attempt(lambda: _is_zero(r0*r1*r2 - 1), 10, native=True) is True
