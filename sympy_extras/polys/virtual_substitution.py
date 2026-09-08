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

A universal quantifier is treated as `\\neg \\exists \\neg`.

For a variable of degree at most two (Weispfenning's quadratic case) the
test points are the roots `(-b \\pm \\sqrt{b^2 - 4ac})/(2a)` of the
quadratic atoms, guarded by `a \\ne 0` and `b^2 - 4ac \\ge 0`, the roots
`-c/b` of the atoms which are linear when `a = 0`, and `-\\infty`; a value
`(p + q\\sqrt{r})/s` is substituted virtually into `g(x) \\rho 0` by
writing `s^2 g` as `A + B\\sqrt{r}` with polynomials `A, B` and deciding
the sign of `A + B\\sqrt{r}` through the signs of `A`, `B` and
`A^2 - B^2 r`; the infinitesimal shifts use the derivatives of `g`.
Variables of higher degree are left to the cylindrical algebraic
decomposition (see :func:`sympy_extras.assumptions.resolve`).

References
==========

.. [LoosWeispfenning] R. Loos, V. Weispfenning, Applying linear
   quantifier elimination, The Computer Journal 36 (1993).
.. [Weispfenning] V. Weispfenning, The complexity of linear problems in
   fields, Journal of Symbolic Computation 5 (1988).
.. [Weispfenning97] V. Weispfenning, Quantifier elimination for real
   algebra — the quadratic case and beyond, AAECC 8 (1997).
"""
from __future__ import annotations

from typing import Optional

from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.relational import Relational, Eq, Ne, Lt, Le, Gt, Ge
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.logic.boolalg import (Boolean, BooleanTrue, BooleanFalse, And, Or, Not,
    Implies, Equivalent, Xor, ITE)
from sympy.polys.polyerrors import PolynomialError
from sympy.polys.polytools import Poly

from sympy_extras._typing import QuantifierPrefix, as_boolean, as_expr

__all__ = ['is_linear_in', 'is_quadratic_in', 'eliminate_linear', 'eliminate_quadratic',
           'linear_quantifier_elimination', 'virtual_substitution_elimination', 'with_sides']


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



# ---------------------------------------------------------------------------
# the quadratic case

def _quadratic_coefficients(atom: Relational, x: Symbol) -> Optional[tuple[Expr, Expr, Expr]]:
    """``(a, b, c)`` with ``atom.lhs - atom.rhs == a*x**2 + b*x + c``."""
    p = as_expr(atom.lhs - atom.rhs)
    if not p.has(x):
        return S.Zero, S.Zero, p
    try:
        poly = Poly(p, x)
    except PolynomialError:
        return None
    if poly.degree() > 2:
        return None
    coefficients = [as_expr(c) for c in poly.all_coeffs()]
    while len(coefficients) < 3:
        coefficients.insert(0, S.Zero)
    a, b, c = coefficients
    return a, b, c


def is_quadratic_in(formula: Boolean, x: Symbol) -> bool:
    """Whether every relation of the formula has degree at most two in
    ``x``.

    >>> from sympy.abc import x, y
    >>> from sympy_extras.polys.virtual_substitution import is_quadratic_in
    >>> is_quadratic_in((x**2 + y*x > 0) & (x < 1), x)
    True
    >>> is_quadratic_in(x**3 > y, x)
    False
    """
    return all(_quadratic_coefficients(a, x) is not None for a in _atoms(formula))


class _Root:
    """A test point ``(p + q*sqrt(r))/s`` with the guard under which it
    is a real root."""

    def __init__(self, p: Expr, q: Expr, r: Expr, s: Expr, guard: Boolean) -> None:
        self.p, self.q, self.r, self.s, self.guard = p, q, r, s, guard


def _parts(a2: Expr, a1: Expr, a0: Expr, root: _Root) -> tuple[Expr, Expr]:
    """``A, B`` with ``s**2 * g((p + q sqrt(r))/s) == A + B sqrt(r)`` for
    ``g = a2 x**2 + a1 x + a0``."""
    p, q, r, s = root.p, root.q, root.r, root.s
    A = as_expr(a2*(p**2 + q**2*r) + a1*s*p + a0*s**2)
    B = as_expr(2*a2*p*q + a1*s*q)
    return as_expr(A.expand()), as_expr(B.expand())


def _sign_with_sqrt(kind: type, A: Expr, B: Expr, r: Expr) -> Boolean:
    """``A + B sqrt(r) rel 0`` (``r >= 0``) as a formula in ``A, B, r``."""
    if B == 0:
        return _relation(kind, A)
    D = as_expr((A**2 - B**2*r).expand())
    if kind is Eq:
        return Or(And(Eq(A, 0), Eq(B, 0)), And(A*B <= 0, Eq(D, 0)))
    if kind is Ne:
        return Not(_sign_with_sqrt(Eq, A, B, r))
    if kind is Lt:
        return Or(And(B <= 0, Or(A < 0, D < 0)), And(B > 0, A < 0, D > 0))
    if kind is Le:
        return Or(And(B <= 0, Or(A <= 0, D <= 0)), And(B > 0, A <= 0, D >= 0))
    if kind is Gt:
        return _sign_with_sqrt(Lt, as_expr(-A), as_expr(-B), r)
    return _sign_with_sqrt(Le, as_expr(-A), as_expr(-B), r)


def _at_algebraic(atom: Relational, coefficients: tuple[Expr, Expr, Expr], root: _Root) -> Boolean:
    """The atom at the root (its sign is that of ``s**2 g`` since ``s != 0``)."""
    a2, a1, a0 = coefficients
    A, B = _parts(a2, a1, a0, root)
    return _sign_with_sqrt(_kind(atom), A, B, root.r)


def _at_algebraic_plus_epsilon(atom: Relational, coefficients: tuple[Expr, Expr, Expr], root: _Root) -> Boolean:
    """The atom at the root shifted by an infinitesimal: the sign of the
    first nonvanishing derivative ``g, g', g''``."""
    a2, a1, a0 = coefficients
    kind = _kind(atom)
    derivatives = [(a2, a1, a0), (S.Zero, 2*a2, a1), (S.Zero, S.Zero, 2*a2)]
    values = [_parts(*d, root) for d in derivatives]
    zero = [_sign_with_sqrt(Eq, A, B, root.r) for A, B in values]
    if kind is Eq:
        return And(*zero)
    if kind is Ne:
        return Not(And(*zero))
    strict = Lt if kind in (Lt, Le) else Gt
    cases: list[Boolean] = []
    for i, (A, B) in enumerate(values):
        cases.append(And(*zero[:i], _sign_with_sqrt(strict, A, B, root.r)))
    if kind in (Le, Ge):
        cases.append(And(*zero))
    return Or(*cases)


def _at_minus_infinity_quadratic(atom: Relational, coefficients: tuple[Expr, Expr, Expr]) -> Boolean:
    """The atom ``a x**2 + b x + c rel 0`` at ``x = -oo``: the sign of the
    leading coefficient, ``-b`` when ``a = 0``."""
    a, b, c = coefficients
    kind = _kind(atom)
    if kind is Eq:
        return And(Eq(a, 0), Eq(b, 0), Eq(c, 0))
    if kind is Ne:
        return Or(Ne(a, 0), Ne(b, 0), Ne(c, 0))
    strict = Lt if kind in (Lt, Le) else Gt
    reverse = Gt if strict is Lt else Lt
    result = Or(_relation(strict, a), And(Eq(a, 0), _relation(reverse, b)),
                And(Eq(a, 0), Eq(b, 0), _relation(kind, c)))
    return as_boolean(result)


def eliminate_quadratic(formula: Boolean, x: Symbol) -> Boolean:
    """`\\exists x\\, \\varphi` for a formula of degree at most two in ``x``.

    Examples
    ========

    >>> from sympy import Eq
    >>> from sympy.abc import x, a, b, c
    >>> from sympy_extras.polys.virtual_substitution import eliminate_quadratic
    >>> eliminate_quadratic(Eq(x**2 + b*x + c, 0), x)
    b**2 >= 4*c
    >>> eliminate_quadratic((x**2 < a) & (x > 1), x)
    a > 1
    """
    formula = as_boolean(as_boolean(formula).to_nnf(simplify=False))
    if not formula.has(x):
        return formula
    atoms = _atoms(formula)
    coefficients: dict[Relational, tuple[Expr, Expr, Expr]] = {}
    for atom in atoms:
        triple = _quadratic_coefficients(atom, x)
        if triple is None:
            raise ValueError("%s has degree more than 2 in %s" % (atom, x))
        coefficients[atom] = triple
    disjuncts: list[Boolean] = [
        _substitute(formula, {atom: _at_minus_infinity_quadratic(atom, triple)
                              for atom, triple in coefficients.items()})]
    for atom, (a, b, c) in coefficients.items():
        if a == 0 and b == 0:
            continue
        shifted = _kind(atom) in (Lt, Gt, Ne)
        roots: list[_Root] = []
        if b != 0 or a != 0:
            # the linear root when a vanishes
            roots.append(_Root(as_expr(-c), S.Zero, S.Zero, b, And(Eq(a, 0), Ne(b, 0))))
        if a != 0:
            r = as_expr((b**2 - 4*a*c).expand())
            for sign in (S.One, S.NegativeOne):
                roots.append(_Root(as_expr(-b), sign, r, as_expr(2*a), And(Ne(a, 0), r >= 0)))
        for root in roots:
            replacements: dict[Relational, Boolean] = {}
            for other, triple in coefficients.items():
                if shifted:
                    replacements[other] = _at_algebraic_plus_epsilon(other, triple, root)
                else:
                    replacements[other] = _at_algebraic(other, triple, root)
            disjuncts.append(And(root.guard, _substitute(formula, replacements)))
    return _clean(Or(*disjuncts))


def virtual_substitution_elimination(formula: Boolean, prefix: QuantifierPrefix, max_atoms: int = 12,
                                     max_free: int = 2) -> tuple[Boolean, QuantifierPrefix]:
    """Eliminate, innermost first, the quantified variables in which the
    formula is linear or quadratic; the remaining prefix is returned with
    the partially eliminated formula. The quadratic case is used with at
    most ``max_atoms`` relations and ``max_free`` free variables (it
    squares the degrees and its results are only simplified afterwards by
    the decomposition, which handles two free variables).

    >>> from sympy import Eq
    >>> from sympy.abc import x, y, b, c
    >>> from sympy_extras.polys.virtual_substitution import virtual_substitution_elimination
    >>> virtual_substitution_elimination(Eq(x**2 + b*x + c, 0), [('exists', x)])
    (b**2 >= 4*c, [])
    >>> virtual_substitution_elimination(x**2 + b*x + c > 0, [('forall', x)])
    (b**2 < 4*c, [])
    """
    remaining = list(prefix)
    current = as_boolean(formula)
    bound = {v for _, v in prefix}
    free = len([s for s in current.free_symbols if s not in bound])
    while remaining:
        kind, v = remaining[-1]
        if is_linear_in(current, v):
            eliminate = eliminate_linear
        elif is_quadratic_in(current, v) and len(_atoms(current)) <= max_atoms and free <= max_free:
            eliminate = eliminate_quadratic
        else:
            break
        remaining.pop()
        if kind == 'exists':
            current = eliminate(current, v)
        else:
            current = _clean(Not(eliminate(Not(current), v)))
    return current, remaining
