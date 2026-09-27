"""The isolated values of the parameters at which the general solution of
an ordinary differential equation fails.

A general solution found for generic values of the parameters of an ODE
fails at isolated values of them in two ways (see
:mod:`sympy_extras._special_values` for the mechanism shared with the
integrals and the sums):

* **it is undefined there**: ``y' - k*y = exp(x)`` has the general
  solution ``C1*exp(k*x) - exp(x)/(k - 1)``, which divides by zero at
  ``k = 1`` (resonance: the right-hand side solves the homogeneous
  equation, and the solution there is ``(C1 + x)*exp(x)``). These
  points are the zeros of the denominators, found as for the integrals;
* **it is defined there but is no longer general**: ``y'' - (a + b)*y' +
  a*b*y = 0`` has the fundamental system ``exp(a*x), exp(b*x)``, which
  collapses to one function at ``a = b`` (a repeated characteristic
  root), where the solutions are ``(C1 + C2*x)*exp(b*x)``; nothing is
  divided by zero. The family ``y(x, C1, ..., Cm)`` is a general solution
  of the order ``m`` part of the equation as long as the Jacobian
  determinant of ``(y, y', ..., y^(m-1))`` with respect to ``(C1, ...,
  Cm)`` does not vanish identically: for a solution linear in the
  constants it is the Wronskian of the fundamental system, which for
  constant coefficients is a product of the differences of the
  characteristic roots (the square root of the discriminant), and for an
  Euler equation the same for the indicial roots. Its factors free of
  the variable and of the constants give the points, and a point is
  kept when the family taken there is the degenerate one. Computing it
  from the returned solution rather than from a characteristic
  polynomial covers every solver alike, the equations with variable
  coefficients and the nonlinear ones (the Riccati solution ``-(u1' +
  C1*u2')/(a*(u1 + C1*u2))`` loses its constant where ``u1, u2`` become
  dependent).

At each point kept the equation is **solved again** with the point
substituted, by the same solver (which gives the cases of the equation at
the point in turn), and the solution there is **checked** before it is
listed: it must satisfy the equation at the point (the residual
simplified to zero, or zero at two sample points by
:func:`~sympy_extras._numeric.reliable_value`), carry the same constants
and be general (its Jacobian determinant not zero at a sample point).
A point where the equation is not defined (``nan``, ``zoo``, ``0**e``)
or where its order drops (``k*y'' + y = 0`` at ``k = 0``, a singular
perturbation whose solutions are no specialisation of the general
family) gets no case, and neither does a point whose solution is not
found or not checked within the budget of
:mod:`sympy_extras._special_values` (a tenth of the time limit).

Examples
========

>>> from sympy import Function, symbols, exp
>>> from sympy_extras.solvers._special_values import general_solution_cases
>>> x, k, C1 = symbols('x k C1')
>>> y = Function('y')(x)
>>> equation = y.diff(x) - k*y - exp(x)
>>> general_solution_cases(C1*exp(k*x) - exp(x)/(k - 1), equation, y,
...                        lambda e: C1*exp(x) + x*exp(x))
Piecewise((C1*exp(x) + x*exp(x), Eq(k, 1)), (C1*exp(k*x) - exp(x)/(k - 1), True))
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, Sequence

from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.function import AppliedUndef, Derivative
from sympy.core.numbers import Rational
from sympy.core.relational import Eq
from sympy.core.symbol import Dummy, Symbol
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.piecewise import Piecewise, piecewise_fold
from sympy.logic.boolalg import Boolean, Or, simplify_logic, true
from sympy.matrices.dense import Matrix
from sympy.polys.polytools import cancel
from sympy.simplify.radsimp import fraction
from sympy.simplify.simplify import simplify
from sympy.solvers.deutils import ode_order

from sympy_extras._numeric import reliable_value
from sympy_extras._timeout import attempt
from sympy_extras._typing import as_boolean, as_expr, free_symbols, sorted_symbols

if TYPE_CHECKING:
    # (imported when used: sympy_extras._special_values imports the
    # assumptions, which import the solvers)
    from sympy_extras._special_values import Point
    from sympy_extras.assumptions.ask import Assumptions

__all__ = ['general_solution_cases', 'basis_cases', 'solution_jacobian', 'explicit_solution', 'solution_cases',
           'as_zero']

#: the seconds given to the symbolic check of a solution at a point
_CHECK_SECONDS = 2.0
#: the most equations in a condition which is simplified
_MAX_LOGIC_ATOMS = 6
#: the digits of the numerical checks
_DIGITS = 30
#: the sample values of the variable and of the constants in the
#: numerical checks (positive, away from the usual singular points)
_SAMPLES: tuple[tuple[Rational, ...], ...] = (
    (Rational(7, 5), Rational(3, 7), Rational(5, 11), Rational(-2, 3), Rational(9, 13)),
    (Rational(23, 9), Rational(-4, 5), Rational(7, 3), Rational(3, 8), Rational(-5, 7)),
)


def _variable(f: AppliedUndef) -> Symbol:
    x = f.args[0]
    if not isinstance(x, Symbol):
        raise ValueError("the function must depend on a symbol")
    return x


def solution_jacobian(solution: Expr, constants: Sequence[Symbol], x: Symbol) -> Expr:
    """The Jacobian determinant of ``(y, y', ..., y^(m-1))`` with respect
    to the ``m`` ``constants`` for the family ``y = solution``: zero
    identically exactly when the family has fewer than ``m`` essential
    constants (for a solution linear in the constants, the Wronskian of
    the functions multiplying them).

    Examples
    ========

    >>> from sympy import symbols, exp, factor
    >>> from sympy_extras.solvers._special_values import solution_jacobian
    >>> x, a, b, C1, C2 = symbols('x a b C1 C2')
    >>> factor(solution_jacobian(C1*exp(a*x) + C2*exp(b*x), [C1, C2], x))
    -(a - b)*exp(a*x)*exp(b*x)
    """
    columns = [as_expr(solution.diff(c)) for c in constants]
    rows = [[as_expr(column.diff(x, i)) if i else column for column in columns] for i in range(len(constants))]
    return as_expr(Matrix(rows).det(method='berkowitz'))


def _degenerate_parts(constants: Sequence[Symbol], x: Symbol) -> Callable[[Expr], list[Expr]]:
    """The expressions whose zeros are the points where a branch of a
    general solution is not general: the numerator of its Jacobian
    determinant (the denominator only makes it undefined, and its zeros
    are found as such)."""
    def parts(branch: Expr) -> list[Expr]:
        present = [c for c in constants if branch.has(c)]
        if len(present) != len(constants) or not constants:
            return []
        numerator, _ = fraction(as_expr(cancel(solution_jacobian(branch, constants, x))))
        return [as_expr(numerator)]
    return parts


def _generic(value: Expr) -> Expr:
    """The generic branch of a value with cases (checked by this module;
    the cases before it were checked when they were made)."""
    if isinstance(value, Piecewise):
        return as_expr(value.args[-1].args[0])
    return value


def _values(x: Symbol, symbols: Sequence[Symbol], sample: tuple[Rational, ...]) -> dict[Symbol, Expr]:
    """The sample values of ``x`` and of the other ``symbols`` (the
    constants and the remaining parameters)."""
    values: dict[Symbol, Expr] = {x: sample[0]}
    for i, c in enumerate(symbols):
        values[c] = sample[1 + i % (len(sample) - 1)] + i // (len(sample) - 1)
    return values


def _zero_at_samples(e: Expr, x: Symbol) -> Optional[bool]:
    """Whether ``e`` is zero at the sample points of its symbols: ``True``
    when it is negligible at all of them, ``False`` when it is not at
    one, ``None`` when nothing was checked."""
    tested = 0
    others = sorted_symbols(free_symbols(e) - {x})
    for sample in _SAMPLES:
        number = reliable_value(e, _DIGITS, _values(x, others, sample))
        if number is None:
            continue
        magnitude = as_expr(Abs(number))
        if not magnitude.is_comparable:
            continue
        if magnitude > Rational(1, 10**20):
            return False
        tested += 1
    return True if tested == len(_SAMPLES) else None


def _solves(equation: Expr, f: AppliedUndef, solution: Expr) -> bool:
    """Whether ``f = solution`` satisfies ``equation = 0``: its residual
    simplified to zero, or negligible at the sample points."""
    x = _variable(f)
    residual = as_expr(equation.subs(f, solution).doit())
    if residual.has(Derivative):
        return False
    simplified = attempt(lambda: as_expr(simplify(residual)), _CHECK_SECONDS)
    if simplified is not None and simplified == 0:
        return True
    return _zero_at_samples(residual, x) is True


def _is_general(solution: Expr, constants: Sequence[Symbol], x: Symbol) -> bool:
    """Whether the family has all its constants: its Jacobian determinant
    not zero at a sample point."""
    if not constants:
        return True
    return _zero_at_samples(solution_jacobian(solution, constants, x), x) is False


def general_solution_cases(general: Expr, equation: Expr, f: AppliedUndef,
                           solve_at: Callable[[Expr], Optional[Expr]]) -> Expr:
    """``general``, the general solution of ``equation = 0`` for ``f`` (its
    constants are the symbols not in the equation), with a case for each
    isolated value of the parameters of the equation at which it is
    undefined or not general (see the module documentation); the value
    at a point is ``solve_at`` of the equation with the point
    substituted, checked, and must carry the same constants.

    Examples
    ========

    >>> from sympy import Function, symbols, exp
    >>> from sympy_extras.solvers._special_values import general_solution_cases
    >>> x, a, b, C1, C2 = symbols('x a b C1 C2')
    >>> y = Function('y')(x)
    >>> equation = y.diff(x, 2) - (a + b)*y.diff(x) + a*b*y
    >>> general_solution_cases(C1*exp(a*x) + C2*exp(b*x), equation, y,
    ...                        lambda e: (C1 + C2*x)*exp(b*x))
    Piecewise(((C1 + C2*x)*exp(b*x), Eq(a, b)), (C1*exp(a*x) + C2*exp(b*x), True))
    """
    from sympy_extras._special_values import defined_problem, with_special_values
    x = _variable(f)
    parameters = sorted_symbols(free_symbols(equation) - {x})
    if not parameters:
        return general
    constants = sorted_symbols(free_symbols(general) - free_symbols(equation))
    order = ode_order(equation, f)

    def defined(point: Point) -> bool:
        at = as_expr(equation.xreplace(point))
        return defined_problem(at, [x]) and ode_order(at, f) == order

    def compute(point: Point, assumptions: Assumptions) -> Optional[Expr]:
        at = as_expr(equation.xreplace(point))
        value = solve_at(at)
        if value is None:
            return None
        generic = _generic(value)
        if set(sorted_symbols(free_symbols(generic) - free_symbols(at))) != set(constants):
            # other constants, or fewer: no specialisation of the family
            return None
        if not _solves(at, f, generic) or not _is_general(generic, constants, x):
            return None
        return value

    return with_special_values(general, parameters, None, compute, defined,
                               degenerate=_degenerate_parts(constants, x))


def basis_cases(basis: list[Expr], equation: Expr, f: AppliedUndef,
                solve_at: Callable[[Expr], Optional[list[Expr]]]) -> list[Expr]:
    """The independent solutions ``basis`` of a homogeneous linear
    ``equation = 0``, each with the cases of :func:`general_solution_cases`
    for the family ``C1*basis[0] + C2*basis[1] + ...``: the ``i``-th
    function becomes ``Piecewise((i-th function at the point, Eq(p,
    v)), ..., (basis[i], True))``, the functions at a point being
    ``solve_at`` of the equation there (as many as in ``basis``).

    Examples
    ========

    >>> from sympy import Function, symbols, exp
    >>> from sympy_extras.solvers._special_values import basis_cases
    >>> x, a, b = symbols('x a b')
    >>> y = Function('y')(x)
    >>> equation = y.diff(x, 2) - (a + b)*y.diff(x) + a*b*y
    >>> basis_cases([exp(a*x), exp(b*x)], equation, y, lambda e: [exp(b*x), x*exp(b*x)])
    [Piecewise((exp(b*x), Eq(a, b)), (exp(a*x), True)), Piecewise((x*exp(b*x), Eq(a, b)), (exp(b*x), True))]
    """
    if not basis:
        return basis
    constants = [Dummy('C%d' % (i + 1)) for i in range(len(basis))]

    def combination(functions: list[Expr]) -> Expr:
        # (the functions at a point may have cases of their own, gathered
        # into one Piecewise of combinations)
        return as_expr(piecewise_fold(sum((c*g for c, g in zip(constants, functions)), as_expr(Rational(0)))))

    def combined_at(at: Expr) -> Optional[Expr]:
        functions = solve_at(at)
        if functions is None or len(functions) != len(basis):
            return None
        return combination(functions)

    value = general_solution_cases(combination(basis), equation, f, combined_at)
    if not isinstance(value, Piecewise):
        return basis
    return [_function_cases([(as_expr(pair.args[0]).diff(c), as_boolean(pair.args[1])) for pair in value.args])
            for c in constants]


def _function_cases(cases: list[tuple[Expr, Boolean]]) -> Expr:
    """The ``Piecewise`` of the cases of one function, the conditions of
    consecutive cases with the same value joined and simplified
    (``Eq(a, 1) & Eq(b, 1)`` followed by ``Eq(a, 1)`` is ``Eq(a, 1)``)."""
    joined: list[tuple[Expr, Boolean]] = []
    for e, condition in cases:
        if joined and joined[-1][0] == e:
            condition = as_boolean(Or(joined[-1][1], condition))
            if condition is not true and len(condition.atoms(Eq)) <= _MAX_LOGIC_ATOMS:
                condition = as_boolean(simplify_logic(condition))
            joined[-1] = (e, condition)
        else:
            joined.append((e, condition))
    return as_expr(Piecewise(*joined))


def as_zero(equation: Basic) -> Expr:
    """The expression which is zero: ``lhs - rhs`` for an ``Eq``."""
    if isinstance(equation, Eq):
        return as_expr(equation.lhs - equation.rhs)
    return as_expr(equation)


def explicit_solution(solutions: Sequence[Basic], f: AppliedUndef) -> Optional[Expr]:
    """The right-hand side of the only solution, when there is one and it
    is explicit, ``Eq(f, rhs)`` with ``rhs`` free of ``f``; ``None``
    otherwise (several branches, implicit or parametric solutions get no
    cases).

    >>> from sympy import Function, Eq, symbols, exp
    >>> from sympy_extras.solvers._special_values import explicit_solution
    >>> x, C1 = symbols('x C1')
    >>> y = Function('y')(x)
    >>> explicit_solution([Eq(y, C1*exp(x))], y), explicit_solution([Eq(y**2, C1 + x)], y)
    (C1*exp(x), None)
    """
    if len(solutions) != 1:
        return None
    solution = solutions[0]
    if isinstance(solution, Eq) and solution.lhs == f and not solution.rhs.has(f):
        return as_expr(solution.rhs)
    return None


def solution_cases(solutions: Sequence[Basic], equation: Basic, f: AppliedUndef,
                   solve: Callable[[Expr], Sequence[Basic]]) -> Optional[Eq]:
    """The only solution ``Eq(f, rhs)`` of ``equation`` with the cases of
    :func:`general_solution_cases`, the equation at a point solved by
    ``solve`` (whose only solution must be explicit too); ``None`` when
    the solutions are not a single explicit one, or nothing changes."""
    general = explicit_solution(solutions, f)
    if general is None:
        return None

    def solve_at(at: Expr) -> Optional[Expr]:
        return explicit_solution(solve(at), f)

    value = general_solution_cases(general, as_zero(equation), f, solve_at)
    return None if value == general else Eq(f, value)
