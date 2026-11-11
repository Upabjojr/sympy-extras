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

The module works on the objects of the core and imports no SymPy:
:mod:`sympy_extras.solvers._special_values_sympy` is its face on SymPy
objects. The candidate points and their bookkeeping are those of the
native :func:`sympy_extras._special_values.with_special_values`; one step
is not ported and goes through SymPy at run time: SymPy's
``simplify_logic`` of the joined conditions
(:func:`~sympy_extras.polys.virtual_substitution_sympy.simplify_logic_of_core`).
:func:`ode_order`, the determinants of the Jacobians and the
simplification SymPy applies to them (:func:`dotprodsimp`) are ported
from SymPy (``sympy.solvers.deutils``, ``sympy.matrices.determinant``,
``sympy.simplify.simplify``).

Examples
========

>>> from sympy_extras.core import Function, symbols
>>> from sympy_extras.functions import exp
>>> from sympy_extras.solvers._special_values import general_solution_cases
>>> x, k, C1 = symbols('x k C1')
>>> y = Function('y')(x)
>>> equation = y.diff(x) - k*y - exp(x)
>>> general_solution_cases(C1*exp(k*x) - exp(x)/(k - 1), equation, y,
...                        lambda e: C1*exp(x) + x*exp(x))
Piecewise((C1*exp(x) + x*exp(x), Eq(k, 1)), (C1*exp(k*x) - exp(x)/(k - 1), True))
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/solvers/deutils.py, sympy/matrices/determinant.py,
#     sympy/simplify/simplify.py (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: ode_order, the determinant (explicit formulas up to order 3 with
# dotprodsimp, Berkowitz's algorithm beyond) on the typed core, strict typing.
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Callable, Optional, Sequence

from sympy_extras.core.add import Add
from sympy_extras.core.basic import Basic, preorder_traversal
from sympy_extras.core.expr import Expr
from sympy_extras.core.function import AppliedUndef, Derivative, expand
from sympy_extras.core.mul import Mul
from sympy_extras.core.logic import Boolean, BooleanTrue, or_
from sympy_extras.core.numbers import Float, Integer, Rational, nan, zoo
from sympy_extras.core.power import Pow
from sympy_extras.core.relational import Eq
from sympy_extras.core.symbol import Dummy, Symbol
from sympy_extras.functions.elementary.complexes import abs_
from sympy_extras.functions.elementary.piecewise import Piecewise, piecewise, piecewise_fold
from sympy_extras.functions.special.delta_functions import DiracDelta
from sympy_extras.polys.polytools import cancel
from sympy_extras.polys.rationaltools import together
from sympy_extras.simplify.radsimp import fraction
from sympy_extras.simplify.simplify import simplify
from sympy_extras._special_values import with_special_values
from sympy_extras._timeout import attempt
from sympy_extras.assumptions.ask import Assumptions

__all__ = ['general_solution_cases', 'basis_cases', 'solution_jacobian', 'explicit_solution', 'solution_cases',
           'as_zero', 'ode_order', 'determinant', 'dotprodsimp', 'defined_equation', 'Point', 'SolveAt', 'BasisAt']

#: the values of some parameters, ``{a: b, b: 0}``
Point = dict[Symbol, Expr]
#: the general solution of an equation (with a point substituted), or
#: ``None`` when it is not found
SolveAt = Callable[[Expr], Optional[Expr]]
#: the independent solutions of a linear equation (with a point
#: substituted), or ``None`` when they are not found
BasisAt = Callable[[Expr], Optional[list[Expr]]]

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
#: the threshold under which a sample value counts as zero
_NEGLIGIBLE = Rational(1, 10**20)


def _variable(f: AppliedUndef) -> Symbol:
    x = f.args[0]
    if not isinstance(x, Symbol):
        raise ValueError("the function must depend on a symbol")
    return x


def _replaced(e: Expr, rule: dict[Basic, Basic]) -> Expr:
    """``e`` with the keys of ``rule`` replaced, evaluated (SymPy's
    ``xreplace``)."""
    value = e._xreplace_evaluated(rule)
    if not isinstance(value, Expr):
        raise TypeError("%s is not an expression" % (value,))
    return value


def _symbols(e: Basic) -> set[Symbol]:
    return {s for s in e.free_symbols if isinstance(s, Symbol)}


def _sorted(symbols: set[Symbol]) -> list[Symbol]:
    return sorted(symbols, key=lambda s: s.name)


# PORTED FROM SYMPY: sympy/solvers/deutils.py:ode_order
def ode_order(expr: Basic, func: AppliedUndef) -> int:
    """The order of the highest derivative of ``func`` in ``expr``
    (SymPy's ``ode_order``).

    >>> from sympy_extras.core import Function, Symbol
    >>> from sympy_extras.solvers._special_values import ode_order
    >>> x = Symbol('x')
    >>> f = Function('f')(x)
    >>> ode_order(f.diff(x, 2) + f.diff(x)**3 + f, f), ode_order(x + 1, f)
    (2, 0)
    """
    if not expr.has(func):
        return 0
    if isinstance(expr, Derivative):
        if expr.args[0] == func:
            return len(expr.variables)
        inner = expr.args[0].args
        order = len(expr.variables)
        if inner:
            order += max(ode_order(a, func) for a in inner)
        return order
    return max(ode_order(a, func) for a in expr.args) if expr.args else 0


# PORTED FROM SYMPY: sympy/matrices/determinant.py:_det
def determinant(rows: Sequence[Sequence[Expr]]) -> Expr:
    """The determinant of a square matrix of expressions, as SymPy's
    ``det(method='berkowitz')`` computes it: the explicit formulas up to
    the order 3, Berkowitz's division free algorithm beyond.

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.solvers._special_values import determinant
    >>> a, b, c, d = symbols('a b c d')
    >>> determinant([[a, b], [c, d]])
    a*d - b*c
    """
    n = len(rows)
    if any(len(row) != n for row in rows):
        raise ValueError("the matrix is not square")
    if n == 0:
        return Integer(1)
    if n == 1:
        return rows[0][0]
    if n == 2:
        return dotprodsimp(rows[0][0]*rows[1][1] - rows[0][1]*rows[1][0])
    if n == 3:
        return dotprodsimp(rows[0][0]*rows[1][1]*rows[2][2] + rows[0][1]*rows[1][2]*rows[2][0]
                           + rows[0][2]*rows[1][0]*rows[2][1] - rows[0][2]*rows[1][1]*rows[2][0]
                           - rows[0][0]*rows[1][2]*rows[2][1] - rows[0][1]*rows[1][0]*rows[2][2])
    vector = _berkowitz_vector([list(row) for row in rows])
    return vector[-1] if n % 2 == 0 else -vector[-1]


# PORTED FROM SYMPY: sympy/simplify/simplify.py:dotprodsimp
def dotprodsimp(expr: Expr) -> Expr:
    """SymPy's ``dotprodsimp``, the simplification its determinants of
    order 2 and 3 apply to the explicit formula: the products expanded,
    and an expression with 6 operations or more which may be a rational
    function cancelled (its non-algebraic parts replaced by dummies
    first), ``x/(x - 1) - 1/(x - 1)`` put together.

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.functions import exp
    >>> from sympy_extras.solvers._special_values import dotprodsimp
    >>> b, x = symbols('b x')
    >>> dotprodsimp(-b*x*exp(2*b*x) + (b*x*exp(b*x) + exp(b*x))*exp(b*x))
    exp(2*b*x)
    """
    if not isinstance(expr, (Add, Mul, Pow)):
        return expr
    expr = expand(expr, deep=True, power_base=False, power_exp=False, mul=True, log=False, multinomial=True)
    exprops, ratfunc = _count_ops_alg(expr)
    if exprops >= 6:   # SymPy's empirical cutoff for the expensive simplification
        if ratfunc:
            dummies: dict[Basic, Basic] = {}
            expr2 = _nonalgebraic_dummies(expr, dummies)
            if expr2 is expr or _count_ops_alg(expr2)[0] >= 6:
                expr3 = cancel(expr2)
                if expr3 != expr2:
                    restored = expr3.subs([(d, e) for e, d in dummies.items()])
                    if not isinstance(restored, Expr):
                        raise TypeError("%s is not an expression" % (restored,))
                    expr = restored
    elif (exprops == 5 and isinstance(expr, Add) and isinstance(expr.args[0], Mul)
          and isinstance(expr.args[1], Mul) and isinstance(expr.args[0].args[-1], Pow)
          and isinstance(expr.args[1].args[-1], Pow) and expr.args[0].args[-1].exp == -1
          and expr.args[1].args[-1].exp == -1):
        # the very special case x/(x - 1) - 1/(x - 1) = 1
        expr2 = together(expr)
        if _count_ops_alg(expr2)[0] < exprops:
            expr = expr2
    return expr


# PORTED FROM SYMPY: sympy/simplify/simplify.py:dotprodsimp
def _count_ops_alg(expr: Expr) -> tuple[int, bool]:
    """The algebraic operations of ``expr`` without recursion into its
    non-algebraic parts, and whether it may be a rational function which
    ``cancel`` may shorten (negative exponents, non-number fractions)."""
    ops = 0
    pending: list[Expr] = [expr]
    ratfunc = False
    while pending:
        a = pending.pop()
        if isinstance(a, Rational):
            if a != 1:   # -1/3 = NEG + DIV
                ops += int(a.p < 0) + int(a.q != 1)
        elif isinstance(a, Mul):
            m: Expr = a
            if m.could_extract_minus_sign():
                ops += 1
                if m.args[0] == -1:
                    m = m.as_two_terms()[1]
                else:
                    m = -m
            n, d = fraction(m)
            if isinstance(n, Integer):
                ops += 1 + int(n.p < 0)
                pending.append(d)   # won't be -Mul but could be Add
            elif d != 1:
                if not isinstance(d, Integer):
                    pending.append(d)
                    ratfunc = True
                ops += 1
                pending.append(n)   # could be -Mul
            else:
                ops += len(m.args) - 1
                pending.extend(e for e in m.args if isinstance(e, Expr))
        elif isinstance(a, Add):
            count = len(a.args)
            negatives = 0
            for term in a.args:
                if not isinstance(term, Expr):
                    continue
                if term.could_extract_minus_sign():
                    negatives += 1
                    term = -term
                pending.append(term)
            ops += count - int(negatives != count)   # -x - y = NEG + SUB
        elif isinstance(a, Pow):
            ops += 1
            pending.append(a.base)
            if not ratfunc:
                ratfunc = a.exp.is_negative is not False
    return ops, ratfunc


# PORTED FROM SYMPY: sympy/simplify/simplify.py:dotprodsimp
def _nonalgebraic_dummies(expr: Expr, dummies: dict[Basic, Basic]) -> Expr:
    """``expr`` with dummies for its non-algebraic parts, so that
    ``cancel`` does not evaluate them."""
    if not expr.args:
        return expr
    if isinstance(expr, (Add, Mul, Pow)):
        args: Optional[list[Basic]] = None
        for i, a in enumerate(expr.args):
            if not isinstance(a, Expr):
                continue
            c = _nonalgebraic_dummies(a, dummies)
            if c is a:
                continue
            if args is None:
                args = list(expr.args)
            args[i] = c
        if args is None:
            return expr
        rebuilt = expr._evaluated(tuple(args))
        if not isinstance(rebuilt, Expr):
            raise TypeError("%s is not an expression" % (rebuilt,))
        return rebuilt
    if expr not in dummies:
        dummies[expr] = Dummy()
    replaced = dummies[expr]
    if not isinstance(replaced, Expr):
        raise TypeError("%s is not an expression" % (replaced,))
    return replaced


# PORTED FROM SYMPY: sympy/matrices/determinant.py:_berkowitz_vector
def _berkowitz_vector(M: list[list[Expr]]) -> list[Expr]:
    """The coefficients of the characteristic polynomial of ``M`` (the
    last one is ``(-1)**n det M``), by Berkowitz's recursion on the
    trailing principal submatrix."""
    n = len(M)
    if n == 0:
        return [Integer(1)]
    if n == 1:
        return [Integer(1), -M[0][0]]
    a = M[0][0]
    R = M[0][1:]
    C = [row[0] for row in M[1:]]
    A = [row[1:] for row in M[1:]]
    # the first column of the Toeplitz matrix: 1, -a, -R C, -R A C, ...
    diags = [C]
    for _ in range(n - 2):
        previous = diags[-1]
        diags.append([_dot(row, previous) for row in A])
    column: list[Expr] = [Integer(1), -a] + [-_dot(R, d) for d in diags]
    sub = _berkowitz_vector(A)
    # the Toeplitz matrix (n + 1) x n times the vector of the submatrix
    result: list[Expr] = []
    for i in range(n + 1):
        terms = [column[i - j]*sub[j] for j in range(min(i + 1, len(sub)))]
        total: Expr = Integer(0)
        for t in terms:
            total = total + t
        result.append(total)
    return result


def _dot(u: Sequence[Expr], v: Sequence[Expr]) -> Expr:
    total: Expr = Integer(0)
    for a, b in zip(u, v):
        total = total + a*b
    return total


def solution_jacobian(solution: Expr, constants: Sequence[Symbol], x: Symbol) -> Expr:
    """The Jacobian determinant of ``(y, y', ..., y^(m-1))`` with respect
    to the ``m`` ``constants`` for the family ``y = solution``: zero
    identically exactly when the family has fewer than ``m`` essential
    constants (for a solution linear in the constants, the Wronskian of
    the functions multiplying them).

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.functions import exp
    >>> from sympy_extras.polys.polytools import factor
    >>> from sympy_extras.solvers._special_values import solution_jacobian
    >>> x, a, b, C1, C2 = symbols('x a b C1 C2')
    >>> factor(solution_jacobian(C1*exp(a*x) + C2*exp(b*x), [C1, C2], x))
    -(a - b)*exp(a*x)*exp(b*x)
    """
    columns = [solution.diff(c) for c in constants]
    rows = [[column.diff(x, i) if i else column for column in columns] for i in range(len(constants))]
    return determinant(rows)


def _degenerate_parts(constants: Sequence[Symbol], x: Symbol) -> Callable[[Expr], list[Expr]]:
    """The expressions whose zeros are the points where a branch of a
    general solution is not general: the numerator of its Jacobian
    determinant (the denominator only makes it undefined, and its zeros
    are found as such)."""
    def parts(branch: Expr) -> list[Expr]:
        present = [c for c in constants if branch.has(c)]
        if len(present) != len(constants) or not constants:
            return []
        numerator, _ = fraction(cancel(solution_jacobian(branch, constants, x)))
        return [numerator]
    return parts


def _generic(value: Expr) -> Expr:
    """The generic branch of a value with cases (checked by this module;
    the cases before it were checked when they were made)."""
    if isinstance(value, Piecewise):
        return value.args[-1].expr
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
    from sympy_extras._numeric import reliable_value
    tested = 0
    others = _sorted(_symbols(e) - {x})
    for sample in _SAMPLES:
        number = reliable_value(e, _DIGITS, _values(x, others, sample))
        if number is None:
            continue
        magnitude = abs_(number)
        if not isinstance(magnitude, (Float, Rational)):
            continue
        if isinstance(magnitude > _NEGLIGIBLE, BooleanTrue):
            return False
        tested += 1
    return True if tested == len(_SAMPLES) else None


def _solves(equation: Expr, f: AppliedUndef, solution: Expr) -> bool:
    """Whether ``f = solution`` satisfies ``equation = 0``: its residual
    simplified to zero, or negligible at the sample points."""
    x = _variable(f)
    substituted = equation.subs(f, solution)
    if not isinstance(substituted, Expr):
        return False
    residual = substituted.doit()
    if residual.has(Derivative):
        return False
    simplified = attempt(lambda: simplify(residual), _CHECK_SECONDS, native=True)
    if simplified is not None and simplified == 0:
        return True
    return _zero_at_samples(residual, x) is True


def _is_general(solution: Expr, constants: Sequence[Symbol], x: Symbol) -> bool:
    """Whether the family has all its constants: its Jacobian determinant
    not zero at a sample point."""
    if not constants:
        return True
    return _zero_at_samples(solution_jacobian(solution, constants, x), x) is False


def defined_equation(equation: Expr, x: Symbol) -> bool:
    """Whether an equation with a point of its parameters substituted is
    still defined as a function of ``x`` (the rule of
    :func:`sympy_extras._special_values.defined_problem`): no ``nan`` or
    ``zoo``, no unevaluated power of ``0``, no ``DiracDelta`` of an
    expression free of ``x``.

    >>> from sympy_extras.core import symbols, Integer
    >>> from sympy_extras.solvers._special_values import defined_equation
    >>> a, b, x = symbols('a b x')
    >>> defined_equation((a**(b*x)/x).subs(a, 0), x), defined_equation((a**(b*x)/x).subs(a, 1), x)
    (False, True)
    """
    if equation.has(nan, zoo):
        return False
    for node in preorder_traversal(equation):
        if isinstance(node, Pow) and node.base.is_zero:
            return False
        if isinstance(node, DiracDelta) and x not in node.args[0].free_symbols:
            return False
    return True


def general_solution_cases(general: Expr, equation: Expr, f: AppliedUndef, solve_at: SolveAt) -> Expr:
    """``general``, the general solution of ``equation = 0`` for ``f`` (its
    constants are the symbols not in the equation), with a case for each
    isolated value of the parameters of the equation at which it is
    undefined or not general (see the module documentation); the value
    at a point is ``solve_at`` of the equation with the point
    substituted, checked, and must carry the same constants.

    Examples
    ========

    >>> from sympy_extras.core import Function, symbols
    >>> from sympy_extras.functions import exp
    >>> from sympy_extras.solvers._special_values import general_solution_cases
    >>> x, a, b, C1, C2 = symbols('x a b C1 C2')
    >>> y = Function('y')(x)
    >>> equation = y.diff(x, 2) - (a + b)*y.diff(x) + a*b*y
    >>> general_solution_cases(C1*exp(a*x) + C2*exp(b*x), equation, y,
    ...                        lambda e: (C1 + C2*x)*exp(b*x))
    Piecewise(((C1 + C2*x)*exp(b*x), Eq(a, b)), (C1*exp(a*x) + C2*exp(b*x), True))
    """
    x = _variable(f)
    parameters = _sorted(_symbols(equation) - {x})
    if not parameters:
        return general
    constants = _sorted(_symbols(general) - _symbols(equation))
    order = ode_order(equation, f)

    def defined(point: Point) -> bool:
        at = _replaced(equation, {s: v for s, v in point.items()})
        return defined_equation(at, x) and ode_order(at, f) == order

    def compute(point: Point) -> Optional[Expr]:
        at = _replaced(equation, {s: v for s, v in point.items()})
        value = solve_at(at)
        if value is None:
            return None
        generic = _generic(value)
        if set(_sorted(_symbols(generic) - _symbols(at))) != set(constants):
            # other constants, or fewer: no specialisation of the family
            return None
        if not _solves(at, f, generic) or not _is_general(generic, constants, x):
            return None
        return value

    def at_point(point: Point, assumptions: Assumptions) -> Optional[Expr]:
        return compute(point)

    return with_special_values(general, parameters, None, at_point, defined, degenerate=_degenerate_parts(constants, x))


def basis_cases(basis: list[Expr], equation: Expr, f: AppliedUndef, solve_at: BasisAt) -> list[Expr]:
    """The independent solutions ``basis`` of a homogeneous linear
    ``equation = 0``, each with the cases of :func:`general_solution_cases`
    for the family ``C1*basis[0] + C2*basis[1] + ...``: the ``i``-th
    function becomes ``Piecewise((i-th function at the point, Eq(p,
    v)), ..., (basis[i], True))``, the functions at a point being
    ``solve_at`` of the equation there (as many as in ``basis``).

    Examples
    ========

    >>> from sympy_extras.core import Function, symbols
    >>> from sympy_extras.functions import exp
    >>> from sympy_extras.solvers._special_values import basis_cases
    >>> x, a, b = symbols('x a b')
    >>> y = Function('y')(x)
    >>> equation = y.diff(x, 2) - (a + b)*y.diff(x) + a*b*y
    >>> basis_cases([exp(a*x), exp(b*x)], equation, y, lambda e: [exp(b*x), x*exp(b*x)])
    [Piecewise((exp(b*x), Eq(a, b)), (exp(a*x), True)), Piecewise((x*exp(b*x), Eq(a, b)), (exp(b*x), True))]
    """
    if not basis:
        return basis
    constants: list[Symbol] = [Dummy('C%d' % (i + 1)) for i in range(len(basis))]

    def combination(functions: list[Expr]) -> Expr:
        # (the functions at a point may have cases of their own, gathered
        # into one Piecewise of combinations)
        total: Expr = Integer(0)
        for c, g in zip(constants, functions):
            total = total + c*g
        return piecewise_fold(total)

    def combined_at(at: Expr) -> Optional[Expr]:
        functions = solve_at(at)
        if functions is None or len(functions) != len(basis):
            return None
        return combination(functions)

    value = general_solution_cases(combination(basis), equation, f, combined_at)
    if not isinstance(value, Piecewise):
        return basis
    return [_function_cases([(pair.expr.diff(c), pair.cond) for pair in value.args]) for c in constants]


def _function_cases(cases: list[tuple[Expr, Boolean]]) -> Expr:
    """The ``Piecewise`` of the cases of one function, the conditions of
    consecutive cases with the same value joined and simplified
    (``Eq(a, 1) & Eq(b, 1)`` followed by ``Eq(a, 1)`` is ``Eq(a, 1)``)."""
    from sympy_extras.polys.virtual_substitution_sympy import simplify_logic_of_core
    joined: list[tuple[Expr, Boolean]] = []
    for e, condition in cases:
        if joined and joined[-1][0] == e:
            condition = or_(joined[-1][1], condition)
            if not isinstance(condition, BooleanTrue) and len(condition.atoms(Eq)) <= _MAX_LOGIC_ATOMS:
                condition = simplify_logic_of_core(condition)
            joined[-1] = (e, condition)
        else:
            joined.append((e, condition))
    return piecewise(*joined)


def as_zero(equation: Basic) -> Expr:
    """The expression which is zero: ``lhs - rhs`` for an ``Eq``.

    >>> from sympy_extras.core import Eq, symbols
    >>> from sympy_extras.solvers._special_values import as_zero
    >>> x = symbols('x')
    >>> as_zero(Eq(x**2, 1))
    x**2 - 1
    """
    if isinstance(equation, Eq):
        return equation.lhs - equation.rhs
    if not isinstance(equation, Expr):
        raise TypeError("%s is not an equation" % (equation,))
    return equation


def explicit_solution(solutions: Sequence[Basic], f: AppliedUndef) -> Optional[Expr]:
    """The right-hand side of the only solution, when there is one and it
    is explicit, ``Eq(f, rhs)`` with ``rhs`` free of ``f``; ``None``
    otherwise (several branches, implicit or parametric solutions get no
    cases).

    >>> from sympy_extras.core import Function, Eq, symbols
    >>> from sympy_extras.functions import exp
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
        return solution.rhs
    return None


def solution_cases(solutions: Sequence[Basic], equation: Basic, f: AppliedUndef,
                   solve: Callable[[Expr], Sequence[Basic]]) -> Optional[Eq]:
    """The only solution ``Eq(f, rhs)`` of ``equation`` with the cases of
    :func:`general_solution_cases`, the equation at a point solved by
    ``solve`` (whose only solution must be explicit too); ``None`` when
    the solutions are not a single explicit one, or nothing changes.

    >>> from sympy_extras.core import Function, Eq, symbols
    >>> from sympy_extras.functions import exp
    >>> from sympy_extras.solvers._special_values import solution_cases
    >>> x, k, C1 = symbols('x k C1')
    >>> y = Function('y')(x)
    >>> solution_cases([Eq(y, C1*exp(k*x) - exp(x)/(k - 1))], y.diff(x) - k*y - exp(x), y,
    ...                lambda e: [Eq(y, (C1 + x)*exp(x))])
    Eq(y(x), Piecewise(((C1 + x)*exp(x), Eq(k, 1)), (C1*exp(k*x) - exp(x)/(k - 1), True)))
    """
    general = explicit_solution(solutions, f)
    if general is None:
        return None

    def solve_at(at: Expr) -> Optional[Expr]:
        return explicit_solution(solve(at), f)

    value = general_solution_cases(general, as_zero(equation), f, solve_at)
    return None if value == general else Eq(f, value)
