r"""Elimination of variables from systems of equations, and the implicit
equations of parametrised curves and surfaces.

Given equations in the variables $t_1, \dots, t_k$ and other quantities,
:func:`eliminate` returns polynomial relations between the other
quantities which follow from the equations: generators of the
*elimination ideal* [1]_. For polynomial equations $f_1 = \dots = f_m =
0$ this is $\langle f_1, \dots, f_m \rangle \cap \mathbb{Q}[x]$, read
off a Gröbner basis for a monomial order in which the $t_i$ are larger
than every monomial in the $x_j$
(:meth:`sympy_extras.polys.ideals.Ideal.eliminate`). By the closure
theorem its zero set is the smallest variety containing the projection
of the solutions.

The equations need not be polynomial:

* a rational function contributes its numerator, and the ideal is
  saturated by the denominators (a variable $z$ with $1 - z\, d = 0$,
  eliminated too), so that no relation is produced by a vanishing
  denominator;
* $\sin u$, $\cos u$ (and $\tan u$, $\dots$, the hyperbolic functions)
  become two indeterminates with $\sin^2 u + \cos^2 u = 1$, after the
  multiple angles and the sums of angles have been expanded; a
  fractional power $b^{p/q}$ becomes an indeterminate $\rho$ with
  $\rho^q = b$ (as in :mod:`sympy_extras.simplify.normalform`);
* the exponentials $e^{r w}$ with rational $r$ are powers of one
  indeterminate for each $w$;
* any other function of the eliminated variables ($\log t$, $e^{t^2}$
  next to $e^t$) is one more indeterminate, eliminated as such. The
  relations returned still follow from the equations, but relations
  which depend on the properties of that function are not found.

The indeterminates which contain an eliminated variable are eliminated;
the others ($\sin a$ for a parameter $a$) stay in the result.

:func:`implicitize` applies this to a parametrisation $x_i = f_i(t)$:
the implicit equation of a curve or a surface.

Examples
========

>>> from sympy_extras.core import Eq, Symbol
>>> from sympy_extras.functions import cos, sin, sqrt
>>> from sympy_extras.solvers.elimination import eliminate, implicitize
>>> t, x, y, z, a, b = (Symbol(s) for s in 'txyzab')
>>> eliminate([Eq(x, a*cos(t)), Eq(y, b*sin(t))], [t])                # the ellipse
[a**2*b**2 - a**2*y**2 - b**2*x**2]
>>> implicitize([(1 - t**2)/(1 + t**2), 2*t/(1 + t**2)], [t], [x, y])
[x**2 + y**2 - 1]
>>> implicitize([t**2, t**3], [t], [x, y]), implicitize([cos(t), sin(2*t)], [t], [x, y])
([x**3 - y**2], [4*x**4 - 4*x**2 + y**2])
>>> eliminate([x**2 + y**2 - 1, x - y + z], [x])
[2*y**2 - 2*y*z + z**2 - 1]
>>> eliminate([Eq(y, sqrt(t)), Eq(x, t + 1)], [t])
[-x + y**2 + 1]

References
==========

.. [1] D. Cox, J. Little, D. O'Shea, *Ideals, Varieties, and Algorithms*,
       4th ed., Springer, 2015, Chapter 3 (the elimination and extension
       theorems, the closure theorem, implicitisation of polynomial and
       rational parametrisations).
"""
from __future__ import annotations

from fractions import Fraction
from math import gcd
from typing import Sequence, Union

from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.function import expand_trig
from sympy_extras.core.mul import mul
from sympy_extras.core.numbers import ImaginaryUnit, Integer, Rational, number_from_fraction
from sympy_extras.core.power import pow
from sympy_extras.core.relational import Eq
from sympy_extras.core.sorting import default_sort_key
from sympy_extras.core.symbol import Dummy, Symbol
from sympy_extras.functions import exp
from sympy_extras.functions.elementary.exponential import Exp
from sympy_extras.matrices.linalg import _collect_kernels
from sympy_extras.polys.domains import QQ
from sympy_extras.polys.ideals import Ideal
from sympy_extras.simplify.normalform import _Abstraction, _numer_denom, normal_form

__all__ = ['eliminate', 'implicitize']

Equation = Union[Eq, ExprLike]


def _expression(equation: Equation) -> Expr:
    if isinstance(equation, Eq):
        lhs, rhs = equation.lhs, equation.rhs
        if not isinstance(lhs, Expr) or not isinstance(rhs, Expr):
            raise TypeError('%s is not an equation between expressions' % (equation,))
        return lhs - rhs
    return to_expr(equation)


def _exponentials(expressions: Sequence[Expr], back: dict[Symbol, Expr]) -> list[Expr]:
    """The expressions with ``exp(r*w)``, ``r`` rational, written as
    integer powers of one indeterminate standing for ``exp(g*w)`` (``g``
    the greatest common divisor of the ``r`` met for that ``w``), which is
    recorded in ``back``. The core turns ``exp(t)**2`` into ``exp(2*t)``:
    without this the two would be unrelated kernels."""
    groups: dict[Expr, list[tuple[Basic, Fraction]]] = {}
    for expression in expressions:
        for node in expression.atoms(Exp):
            argument = node.args[0]
            if not isinstance(argument, Expr):
                continue
            coefficient, rest = argument.as_coeff_Mul()
            if isinstance(coefficient, Rational):
                groups.setdefault(rest, []).append((node, Fraction(coefficient.p, coefficient.q)))
    rule: dict[Basic, Basic] = {}
    for rest, nodes in groups.items():
        numerator, denominator = 0, 1
        for _, r in nodes:
            numerator = gcd(numerator*r.denominator, r.numerator*denominator)
            denominator *= r.denominator
        step = Fraction(numerator, denominator)
        placeholder: Symbol = Dummy('E')
        back[placeholder] = exp(number_from_fraction(step)*rest)
        for node, r in nodes:
            power = r/step
            rule[node] = pow(placeholder, Integer(power.numerator))
    return [e.xreplace(rule) for e in expressions] if rule else list(expressions)


def eliminate(equations: Sequence[Equation], variables: Sequence[Symbol]) -> list[Expr]:
    """Polynomial relations, each equal to zero, between the quantities
    of ``equations`` other than ``variables`` (equations ``Eq(lhs, rhs)``
    or expressions which are zero): a Gröbner basis of the elimination
    ideal. The list is empty when the equations imply no relation, and
    ``[1]`` when they have no solution.

    >>> from sympy_extras.core import Eq, Symbol
    >>> from sympy_extras.solvers.elimination import eliminate
    >>> P, V, T, U, n, R, c = (Symbol(s) for s in 'PVTUnRc')
    >>> eliminate([Eq(P*V, n*R*T), Eq(U, c*n*T)], [T])           # the energy of an ideal gas from P and V
    [P*V*c - R*U]
    >>> eliminate([Symbol('x') - Symbol('y')], [Symbol('z')]), eliminate([Symbol('x'), Symbol('x') - 1], [Symbol('x')])
    ([x - y], [1])
    """
    eliminated = set(variables)
    abstraction = _Abstraction()
    back: dict[Symbol, Expr] = {}
    expanded = [expand_trig(_expression(equation)) for equation in equations]
    if any(e.has(ImaginaryUnit) for e in expanded):
        raise NotImplementedError('elimination with the imaginary unit in the equations')
    numerators: list[Expr] = []
    denominators: list[Expr] = []
    for expression in _exponentials(expanded, back):
        numerator, denominator = _numer_denom(abstraction.abstract(expression))
        if numerator != 0:
            numerators.append(numerator)
        if not isinstance(denominator, Rational):
            denominators.append(denominator)
    relations = list(abstraction.relations)
    # the kernels which are left (exp(t), log(a), f(x)) become indeterminates of their own
    found: dict[Expr, int] = {}
    for e in numerators + denominators + relations:
        _collect_kernels(e, found)
    back.update(abstraction.back)
    rule: dict[Basic, Basic] = {}
    for kernel in found:
        if not isinstance(kernel, Symbol):
            placeholder: Symbol = Dummy('k')
            rule[kernel] = placeholder
            back[placeholder] = kernel
    if rule:
        numerators = [e.xreplace(rule) for e in numerators]
        denominators = [e.xreplace(rule) for e in denominators]
        relations = [e.xreplace(rule) for e in relations]
    generators = numerators + relations
    symbols: set[Symbol] = set()
    for e in generators + denominators:
        symbols |= {s for s in e.free_symbols if isinstance(s, Symbol)}
    removed: set[Symbol] = {s for s in symbols if s in eliminated}
    for placeholder, original in back.items():
        if placeholder in symbols and original.free_symbols & eliminated:
            removed.add(placeholder)
    if denominators:
        saturation: Symbol = Dummy('z')
        generators.append(1 - saturation*mul(*denominators))
        symbols.add(saturation)
        removed.add(saturation)
    if not generators:
        return []
    if not symbols:
        # equations between numbers: a nonzero one cannot hold
        return [Integer(1)]
    kept = sorted(symbols - removed, key=default_sort_key)
    dropped = sorted(removed, key=default_sort_key)
    whole = Ideal(generators, *(dropped + kept), domain=QQ)
    if not kept:
        # nothing is left: the equations are consistent or not
        return [Integer(1)] if whole.is_whole_ring() else []
    ideal = whole.eliminate(dropped) if dropped else whole
    results: list[Expr] = []
    for g in ideal.groebner_basis():
        polynomial = _numer_denom(ideal.to_expr(g))[0]
        for placeholder, original in back.items():
            if polynomial.has(placeholder):
                polynomial = polynomial.subs(placeholder, original)
        # the relations between the kernels which stay (sin(a)**2 + cos(a)**2 - 1) are identities, not results
        if normal_form(polynomial) != 0:
            results.append(polynomial)
    return sorted(results, key=default_sort_key)


def implicitize(parametrization: Sequence[ExprLike], parameters: Sequence[Symbol], coordinates: Sequence[Symbol]) -> list[Expr]:
    """The implicit equations of the curve or surface ``coordinates[i] =
    parametrization[i]``: polynomials in the coordinates (and in the
    other symbols of the parametrisation) which vanish on it, generating
    the ideal of the smallest variety containing it.

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.functions import cos, sin
    >>> from sympy_extras.solvers.elimination import implicitize
    >>> s, t, x, y, z = (Symbol(n) for n in 'stxyz')
    >>> implicitize([s*t, s*t**2, s**2], [s, t], [x, y, z])              # the Whitney umbrella
    [x**4 - y**2*z]
    >>> implicitize([sin(s)*cos(t), sin(s)*sin(t), cos(s)], [s, t], [x, y, z])
    [x**2 + y**2 + z**2 - 1]
    """
    if len(parametrization) != len(coordinates):
        raise ValueError('%d coordinates for %d functions' % (len(coordinates), len(parametrization)))
    if set(parameters) & set(coordinates):
        raise ValueError('a parameter cannot be a coordinate')
    return eliminate([x - to_expr(f) for x, f in zip(coordinates, parametrization)], parameters)
