"""Quantifier elimination by virtual substitution for variables occurring
linearly (Loos and Weispfenning).

For a formula `\\varphi` which is a Boolean combination of relations
`a x + b \\;\\rho\\; 0` linear in the quantified variable `x` (with `a` and
`b` polynomials in the other variables), `\\exists x\\, \\varphi` is
equivalent to the disjunction of `\\varphi` with `x` replaced by finitely
many *test points*: `-\\infty`, the roots `-b/a` of the atoms which are
equations or weak inequalities, and the roots shifted by an infinitesimal
`-b/a + \\varepsilon` for the strict inequalities and inequations. The
substitution is *virtual*: instead of the non-polynomial values, each atom
is replaced by the polynomial condition which describes its truth at the
test point (for instance `c x + d < 0` at `-\\infty` becomes
`c > 0 \\vee (c = 0 \\wedge d < 0)`). The result is again a Boolean
combination of polynomial relations in the remaining variables, and no
decomposition of the space is needed.

A universal quantifier is treated as `\\neg \\exists \\neg`. Variables
occurring with higher degree are left to the cylindrical algebraic
decomposition (see :func:`sympy_extras.assumptions.resolve`).

References
==========

.. [LoosWeispfenning] R. Loos, V. Weispfenning, Applying linear
   quantifier elimination, The Computer Journal 36 (1993).
.. [Weispfenning] V. Weispfenning, The complexity of linear problems in
   fields, Journal of Symbolic Computation 5 (1988).
"""
from __future__ import annotations

from typing import Optional

from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.relational import Relational, Eq, Ne, Lt, Le, Gt
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.logic.boolalg import (Boolean, BooleanTrue, BooleanFalse, And, Or, Not,
    Implies, Equivalent, Xor, ITE)
from sympy.polys.polyerrors import PolynomialError
from sympy.polys.polytools import Poly

from sympy_extras._typing import QuantifierPrefix, as_boolean, as_expr

__all__ = ['is_linear_in', 'eliminate_linear', 'linear_quantifier_elimination', 'with_sides']


def _atoms(formula: Boolean) -> list[Relational]:
    return [a for a in formula.atoms(Relational)]


def _linear_coefficients(atom: Relational, x: Symbol) -> Optional[tuple[Expr, Expr]]:
    """``(a, b)`` with ``atom.lhs - atom.rhs == a*x + b``, or ``None``."""
    p = as_expr(atom.lhs - atom.rhs)
    if not p.has(x):
        return S.Zero, p
    try:
        poly = Poly(p, x)
    except PolynomialError:
        return None
    if poly.degree() != 1:
        return None
    a, b = poly.all_coeffs()
    return as_expr(a), as_expr(b)


def is_linear_in(formula: Boolean, x: Symbol) -> bool:
    """Whether every relation of the formula is linear in ``x``.

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy_extras.polys.virtual_substitution import is_linear_in
    >>> is_linear_in((2*x + y > 0) & (x*y < 1), x)
    True
    >>> is_linear_in(x**2 + y > 0, x)
    False
    """
    return all(_linear_coefficients(a, x) is not None for a in _atoms(formula))


def _kind(atom: Relational) -> type:
    return type(atom)


def _relation(kind: type, lhs: Expr) -> Boolean:
    """``lhs rel 0`` of the given kind, evaluated when constant."""
    return as_boolean(kind(lhs, S.Zero))


def _at_root(atom: Relational, a: Expr, b: Expr, c: Expr, d: Expr) -> Boolean:
    """The atom ``c x + d rel 0`` at ``x = -b/a`` (``a != 0``): the sign of
    ``(a d - b c)/a``, decided by the sign of ``a (a d - b c)``."""
    numerator = as_expr(a*d - b*c)
    kind = _kind(atom)
    if kind in (Eq, Ne):
        return _relation(kind, numerator)
    return _relation(kind, as_expr(a*numerator))


def _at_root_plus_epsilon(atom: Relational, a: Expr, b: Expr, c: Expr, d: Expr) -> Boolean:
    """The atom ``c x + d rel 0`` at ``x = -b/a + epsilon``: the value is
    ``(a d - b c)/a + c epsilon``, so its sign is that of the first term
    unless it vanishes, and then that of ``c``."""
    numerator = as_expr(a*d - b*c)
    value_sign = as_expr(a*numerator)  # the sign of (a d - b c)/a
    kind = _kind(atom)
    if kind is Eq:
        return And(Eq(numerator, 0), Eq(c, 0))
    if kind is Ne:
        return Or(Ne(numerator, 0), Ne(c, 0))
    if kind is Lt:
        return Or(value_sign < 0, And(Eq(numerator, 0), c < 0))
    if kind is Le:
        return Or(value_sign < 0, And(Eq(numerator, 0), c <= 0))
    if kind is Gt:
        return Or(value_sign > 0, And(Eq(numerator, 0), c > 0))
    return Or(value_sign > 0, And(Eq(numerator, 0), c >= 0))


def _at_minus_infinity(atom: Relational, c: Expr, d: Expr) -> Boolean:
    """The atom ``c x + d rel 0`` at ``x = -oo``."""
    kind = _kind(atom)
    if kind is Eq:
        return And(Eq(c, 0), Eq(d, 0))
    if kind is Ne:
        return Or(Ne(c, 0), Ne(d, 0))
    if kind is Lt:
        return Or(c > 0, And(Eq(c, 0), d < 0))
    if kind is Le:
        return Or(c > 0, And(Eq(c, 0), d <= 0))
    if kind is Gt:
        return Or(c < 0, And(Eq(c, 0), d > 0))
    return Or(c < 0, And(Eq(c, 0), d >= 0))


def _substitute(formula: Boolean, replacements: dict[Relational, Boolean]) -> Boolean:
    if isinstance(formula, Relational):
        return replacements.get(formula, formula)
    if isinstance(formula, (And, Or, Not, Implies, Equivalent, Xor, ITE)):
        return as_boolean(formula.func(*[_substitute(as_boolean(a), replacements) for a in formula.args]))
    return formula


def eliminate_linear(formula: Boolean, x: Symbol) -> Boolean:
    """`\\exists x\\, \\varphi` for a formula linear in ``x``, as a
    quantifier-free formula in the other variables.

    Examples
    ========

    >>> from sympy import Eq
    >>> from sympy.abc import x, y, a
    >>> from sympy_extras.polys.virtual_substitution import eliminate_linear
    >>> eliminate_linear((x > y) & (x < 1), x)
    y < 1
    >>> eliminate_linear(Eq(a*x, 1), x)
    Ne(a, 0)
    >>> eliminate_linear((x > 0) & (x < 0), x)
    False
    """
    # the test points depend on the polarity of the atoms: negations are
    # pushed down to the relations first (a negated strict inequality is a
    # weak one, whose root is a test point)
    formula = as_boolean(as_boolean(formula).to_nnf(simplify=False))
    if not formula.has(x):
        return formula
    atoms = _atoms(formula)
    coefficients: dict[Relational, tuple[Expr, Expr]] = {}
    for atom in atoms:
        pair = _linear_coefficients(atom, x)
        if pair is None:
            raise ValueError("%s is not linear in %s" % (atom, x))
        coefficients[atom] = pair
    disjuncts: list[Boolean] = []
    # the test point -oo
    disjuncts.append(_substitute(formula, {atom: _at_minus_infinity(atom, c, d)
                                           for atom, (c, d) in coefficients.items()}))
    # the roots of the atoms, plain for equations and weak inequalities,
    # shifted by an infinitesimal for the strict ones
    for atom, (a, b) in coefficients.items():
        if a == 0:
            continue
        shifted = _kind(atom) in (Lt, Gt, Ne)
        replacements: dict[Relational, Boolean] = {}
        for other, (c, d) in coefficients.items():
            if shifted:
                replacements[other] = _at_root_plus_epsilon(other, a, b, c, d)
            else:
                replacements[other] = _at_root(other, a, b, c, d)
        disjuncts.append(And(Ne(a, 0), _substitute(formula, replacements)))
    return _clean(Or(*disjuncts))


def _sides(atom: Relational) -> Boolean:
    """``p - q rel 0`` written as ``p rel q`` with the negative terms moved
    to the right-hand side."""
    if atom.rhs != 0:
        return atom
    lhs = as_expr(atom.lhs)
    positive: list[Expr] = []
    negative: list[Expr] = []
    for term in Add.make_args(lhs):
        if term.could_extract_minus_sign():
            negative.append(as_expr(-term))
        else:
            positive.append(term)
    constants = [t for t in positive if t.is_Number]
    if not negative and len(positive) > 1 and constants:
        # ``x + 1 < 0`` as ``x < -1``
        positive.remove(constants[0])
        negative.append(as_expr(-constants[0]))
    if not negative or not positive:
        return atom
    return as_boolean(atom.func(Add(*positive), Add(*negative)))


def _clean(formula: Boolean) -> Boolean:
    """Light simplification of the result."""
    from sympy.logic.boolalg import simplify_logic
    from sympy_extras.assumptions.facts import normalize
    result = normalize(formula)
    if isinstance(result, (BooleanTrue, BooleanFalse)):
        return result
    try:
        simplified = as_boolean(simplify_logic(result, deep=False))
    except (ValueError, TypeError):
        simplified = result
    return with_sides(simplified)


def with_sides(formula: Boolean) -> Boolean:
    """Every relation ``p - q rel 0`` of the formula written as ``p rel q``.

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy_extras.polys.virtual_substitution import with_sides
    >>> with_sides((x - y > 0) & (x + 1 < 0))
    (x > y) & (x < -1)
    """
    formula = as_boolean(formula)
    return _substitute(formula, {a: _sides(a) for a in _atoms(formula)})


def linear_quantifier_elimination(formula: Boolean, prefix: QuantifierPrefix
                                  ) -> tuple[Boolean, QuantifierPrefix]:
    """Eliminate, from the innermost outwards, the quantified variables in
    which the formula is linear; the remaining prefix (outermost
    quantifiers whose variables occur nonlinearly) is returned with the
    partially eliminated formula.

    Examples
    ========

    >>> from sympy.abc import x, y, z
    >>> from sympy_extras.polys.virtual_substitution import linear_quantifier_elimination
    >>> linear_quantifier_elimination((x > y) & (y > z), [('exists', x)])
    (y > z, [])
    >>> linear_quantifier_elimination((x > y) & (y > z), [('forall', z), ('exists', x)])
    (False, [])
    >>> linear_quantifier_elimination((x**2 > y) & (y > z), [('exists', x), ('exists', z)])
    (x**2 > y, [('exists', x)])
    """
    remaining = list(prefix)
    current = as_boolean(formula)
    while remaining:
        kind, v = remaining[-1]
        if not is_linear_in(current, v):
            break
        remaining.pop()
        if kind == 'exists':
            current = eliminate_linear(current, v)
        else:
            current = _clean(Not(eliminate_linear(Not(current), v)))
    return current, remaining

