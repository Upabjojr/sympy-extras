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
decomposition (see :func:`sympy_extras.assumptions.resolve.resolve`).

The formulas, the expressions and the symbols are objects of the core
(:mod:`sympy_extras.core`): the module imports no SymPy, and
:mod:`sympy_extras.polys.virtual_substitution_sympy` is its face on SymPy
objects. The coefficients in the eliminated variable come from the
native :class:`~sympy_extras.polys.polytools.Poly`, the normal form of
the relations from the native :func:`sympy_extras.assumptions.facts.normalize`.
The results are simplified by the native port of SymPy's
``simplify_logic`` (:func:`sympy_extras.logic.boolalg.simplify_logic`). The negation normal form
:func:`to_nnf` is a port of SymPy's ``to_nnf`` (``sympy/logic/boolalg.py``).

References
==========

.. [LoosWeispfenning] R. Loos, V. Weispfenning, Applying linear
   quantifier elimination, The Computer Journal 36 (1993).
.. [Weispfenning] V. Weispfenning, The complexity of linear problems in
   fields, Journal of Symbolic Computation 5 (1988).
.. [Weispfenning97] V. Weispfenning, Quantifier elimination for real
   algebra — the quadratic case and beyond, AAECC 8 (1997).
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/logic/boolalg.py (to_nnf and the to_nnf methods
# of Not, Implies, Equivalent, Xor, ITE) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: typed, on the Booleans of the core, simplify=False only.
# ---------------------------------------------------------------------
from __future__ import annotations

from itertools import combinations
from typing import Callable, Final, Optional

from sympy_extras.core.add import Add, add
from sympy_extras.core.expr import Expr, as_expr
from sympy_extras.core.function import expand
from sympy_extras.core.logic import (And, Boolean, BooleanAtom, BooleanFunction, Equivalent, ITE, Implies, Not, Or, Xor, and_,
    not_, or_, to_boolean)
from sympy_extras.core.numbers import Integer
from sympy_extras.core.relational import Eq, Ge, Gt, Le, Lt, Ne, Relational, eq, ge, gt, le, lt, ne
from sympy_extras.core.symbol import Symbol
from sympy_extras.assumptions.facts import normalize
from sympy_extras.assumptions.quantifiers import Prefix

from .polyerrors import CoercionFailed, PolynomialError
from .polytools import Poly, _decompose_with_gens

__all__ = ['is_linear_in', 'is_quadratic_in', 'eliminate_linear', 'eliminate_quadratic',
           'linear_quantifier_elimination', 'virtual_substitution_elimination', 'with_sides', 'to_nnf']

#: the kind of a relation (``Eq``, ``Ne``, ``Lt``, ``Le``, ``Gt``, ``Ge``)
Kind = type[Relational]

_RELATIONS: dict[Kind, Callable[[Expr, Expr], Boolean]] = {Eq: eq, Ne: ne, Lt: lt, Le: le, Gt: gt, Ge: ge}

_ZERO = Integer(0)


def _atoms(formula: Boolean) -> list[Relational]:
    return [a for a in formula.atoms(Relational) if isinstance(a, Relational)]


def _relation(kind: Kind, lhs: Expr) -> Boolean:
    """``lhs rel 0`` of the given kind, evaluated when constant."""
    return _RELATIONS[kind](lhs, _ZERO)


def _coefficients(atom: Relational, x: Symbol) -> Optional[list[Expr]]:
    """The coefficients of ``atom.lhs - atom.rhs`` as a polynomial in
    ``x``, highest degree first (SymPy's ``Poly(p, x).all_coeffs()``:
    ``[0]`` for the zero polynomial), ``None`` when it is not one."""
    p = as_expr(atom.lhs - atom.rhs)
    if not p.has(x):
        return [p]
    try:
        return Poly(p, x).all_coeffs()
    except CoercionFailed:
        # a coefficient domain not ported to the native Poly (EX, a
        # fraction field such as ZZ(a)): the coefficients as expressions
        pass
    except PolynomialError:
        return None
    try:
        decomposed = _decompose_with_gens(p, (x,))
    except PolynomialError:
        return None
    terms = {m[0]: expand(add(*cs)) for m, cs in decomposed.items()}
    terms = {k: c for k, c in terms.items() if c != 0}
    if not terms:
        return [_ZERO]
    return [terms.get(k, _ZERO) for k in range(max(terms), -1, -1)]


def _degree(coefficients: list[Expr]) -> int:
    """The degree of the coefficients of :func:`_coefficients` (``-1`` for
    the zero polynomial, whose degree is ``-oo`` in SymPy)."""
    if len(coefficients) == 1 and coefficients[0] == 0:
        return -1
    return len(coefficients) - 1


def _linear_coefficients(atom: Relational, x: Symbol) -> Optional[tuple[Expr, Expr]]:
    """``(a, b)`` with ``atom.lhs - atom.rhs == a*x + b``, or ``None``."""
    p = as_expr(atom.lhs - atom.rhs)
    if not p.has(x):
        return _ZERO, p
    coefficients = _coefficients(atom, x)
    if coefficients is None or _degree(coefficients) != 1:
        return None
    a, b = coefficients
    return a, b


def is_linear_in(formula: Boolean, x: Symbol) -> bool:
    """Whether every relation of the formula is linear in ``x``.

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.polys.virtual_substitution import is_linear_in
    >>> x, y = symbols('x y')
    >>> is_linear_in((2*x + y > 0) & (x*y < 1), x)
    True
    >>> is_linear_in(x**2 + y > 0, x)
    False
    """
    return all(_linear_coefficients(a, x) is not None for a in _atoms(formula))


def _kind(atom: Relational) -> Kind:
    return type(atom)


def _at_root(atom: Relational, a: Expr, b: Expr, c: Expr, d: Expr) -> Boolean:
    """The atom ``c x + d rel 0`` at ``x = -b/a`` (``a != 0``): the sign of
    ``(a d - b c)/a``, decided by the sign of ``a (a d - b c)``."""
    numerator = a*d - b*c
    kind = _kind(atom)
    if kind in (Eq, Ne):
        return _relation(kind, numerator)
    return _relation(kind, a*numerator)


def _at_root_plus_epsilon(atom: Relational, a: Expr, b: Expr, c: Expr, d: Expr) -> Boolean:
    """The atom ``c x + d rel 0`` at ``x = -b/a + epsilon``: the value is
    ``(a d - b c)/a + c epsilon``, so its sign is that of the first term
    unless it vanishes, and then that of ``c``."""
    numerator = a*d - b*c
    value_sign = a*numerator  # the sign of (a d - b c)/a
    kind = _kind(atom)
    if kind is Eq:
        return and_(eq(numerator, 0), eq(c, 0))
    if kind is Ne:
        return or_(ne(numerator, 0), ne(c, 0))
    if kind is Lt:
        return or_(value_sign < 0, and_(eq(numerator, 0), c < 0))
    if kind is Le:
        return or_(value_sign < 0, and_(eq(numerator, 0), c <= 0))
    if kind is Gt:
        return or_(value_sign > 0, and_(eq(numerator, 0), c > 0))
    return or_(value_sign > 0, and_(eq(numerator, 0), c >= 0))


def _at_minus_infinity(atom: Relational, c: Expr, d: Expr) -> Boolean:
    """The atom ``c x + d rel 0`` at ``x = -oo``."""
    kind = _kind(atom)
    if kind is Eq:
        return and_(eq(c, 0), eq(d, 0))
    if kind is Ne:
        return or_(ne(c, 0), ne(d, 0))
    if kind is Lt:
        return or_(c > 0, and_(eq(c, 0), d < 0))
    if kind is Le:
        return or_(c > 0, and_(eq(c, 0), d <= 0))
    if kind is Gt:
        return or_(c < 0, and_(eq(c, 0), d > 0))
    return or_(c < 0, and_(eq(c, 0), d >= 0))


def _substitute(formula: Boolean, replacements: dict[Relational, Boolean]) -> Boolean:
    """The relations of the formula replaced, the connectives evaluated."""
    if isinstance(formula, Relational):
        return replacements.get(formula, formula)
    if isinstance(formula, (And, Or, Not, Implies, Equivalent, Xor, ITE)):
        args = tuple(_substitute(to_boolean(a), replacements) for a in formula.args)
        return to_boolean(formula._evaluated(args))
    return formula


# PORTED FROM SYMPY: sympy/logic/boolalg.py:to_nnf
def to_nnf(formula: Boolean) -> Boolean:
    """The negation normal form (SymPy's ``to_nnf(simplify=False)``): the
    negations pushed down to the atoms (the negation of a relation is the
    opposite relation).

    Examples
    ========

    >>> from sympy_extras.core import symbols, Implies, Not
    >>> from sympy_extras.polys.virtual_substitution import to_nnf
    >>> x, y = symbols('x y')
    >>> to_nnf(Implies(x > 0, y < 1)), to_nnf(Not((x > 0) & (y < 1)))
    ((x <= 0) | (y < 1), (y >= 1) | (x <= 0))
    """
    if isinstance(formula, And):
        return and_(*[to_nnf(to_boolean(a)) for a in formula.args])
    if isinstance(formula, Or):
        return or_(*[to_nnf(to_boolean(a)) for a in formula.args])
    if isinstance(formula, Implies):
        a, b = [to_boolean(c) for c in formula.args]
        return or_(to_nnf(not_(a)), to_nnf(b))
    if isinstance(formula, Equivalent):
        items = [to_boolean(c) for c in formula.args]
        clauses = [or_(not_(a), b) for a, b in zip(items, items[1:])]
        clauses.append(or_(not_(items[-1]), items[0]))
        return and_(*[to_nnf(c) for c in clauses])
    if isinstance(formula, Xor):
        items = [to_boolean(c) for c in formula.args]
        clauses = [or_(*[not_(s) if s in negated else s for s in items])
                   for i in range(0, len(items) + 1, 2) for negated in combinations(items, i)]
        return and_(*[to_nnf(c) for c in clauses])
    if isinstance(formula, ITE):
        a, b, c = [to_boolean(d) for d in formula.args]
        return and_(to_nnf(or_(not_(a), b)), to_nnf(or_(a, c)))
    if isinstance(formula, Not):
        return _negation_to_nnf(to_boolean(formula.args[0]))
    return formula


# PORTED FROM SYMPY: sympy/logic/boolalg.py:Not.to_nnf
def _negation_to_nnf(inner: Boolean) -> Boolean:
    """The negation normal form of ``Not(inner)``."""
    if isinstance(inner, And):
        return or_(*[to_nnf(not_(to_boolean(a))) for a in inner.args])
    if isinstance(inner, Or):
        return and_(*[to_nnf(not_(to_boolean(a))) for a in inner.args])
    if isinstance(inner, Implies):
        a, b = [to_boolean(c) for c in inner.args]
        return and_(to_nnf(a), to_nnf(not_(b)))
    if isinstance(inner, Equivalent):
        items = [to_boolean(c) for c in inner.args]
        return and_(to_nnf(or_(*items)), to_nnf(or_(*[not_(a) for a in items])))
    if isinstance(inner, Xor):
        items = [to_boolean(c) for c in inner.args]
        clauses = [or_(*[not_(s) if s in negated else s for s in items])
                   for i in range(1, len(items) + 1, 2) for negated in combinations(items, i)]
        return and_(*[to_nnf(c) for c in clauses])
    if isinstance(inner, ITE):
        a, b, c = [to_boolean(d) for d in inner.args]
        return and_(to_nnf(or_(a, not_(c))), to_nnf(or_(not_(a), not_(b))))
    if isinstance(inner, Not):
        return to_nnf(to_boolean(inner.args[0]))
    return not_(inner)


def eliminate_linear(formula: Boolean, x: Symbol) -> Boolean:
    """`\\exists x\\, \\varphi` for a formula linear in ``x``, as a
    quantifier-free formula in the other variables.

    Examples
    ========

    >>> from sympy_extras.core import symbols, Eq
    >>> from sympy_extras.polys.virtual_substitution import eliminate_linear
    >>> x, y, a = symbols('x y a')
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
    formula = to_nnf(to_boolean(formula))
    if not formula.has(x):
        return formula
    # a top-level equation a x + b = 0 with a nonzero number a pins x: the
    # one test point is -b/a, and no disjunction is needed. Without this a
    # system of five linear equations took a minute to say True, all of it
    # spent on test points that were thrown away (sympy-extras#47).
    pinned = _pinned_value(formula, x)
    if pinned is not None:
        return _clean(to_boolean(formula._xreplace_evaluated({x: pinned})))
    coefficients: dict[Relational, tuple[Expr, Expr]] = {}
    for atom in _atoms(formula):
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
        disjuncts.append(and_(ne(a, 0), _substitute(formula, replacements)))
    return _clean(or_(*disjuncts))


def _pinned_value(formula: Boolean, x: Symbol) -> Optional[Expr]:
    """``-b/a`` when the formula, or a top-level conjunct of it, is the
    equation ``a x + b = 0`` with a nonzero numeric ``a``."""
    conjuncts = [to_boolean(a) for a in formula.args] if isinstance(formula, And) else [formula]
    for atom in conjuncts:
        if not isinstance(atom, Eq):
            continue
        pair = _linear_coefficients(atom, x)
        if pair is None:
            continue
        a, b = pair
        if a.is_number and a != 0:
            return -b/a
    return None


def _sides(atom: Relational) -> Boolean:
    """``p - q rel 0`` written as ``p rel q`` with the negative terms moved
    to the right-hand side."""
    if atom.rhs != 0:
        return atom
    positive: list[Expr] = []
    negative: list[Expr] = []
    for term in Add.make_args(atom.lhs):
        if term.could_extract_minus_sign():
            negative.append(-term)
        else:
            positive.append(term)
    constants = [t for t in positive if t.is_Number]
    if not negative and len(positive) > 1 and constants:
        # ``x + 1 < 0`` as ``x < -1``
        positive.remove(constants[0])
        negative.append(-constants[0])
    if not negative or not positive:
        return atom
    return _RELATIONS[_kind(atom)](add(*positive), add(*negative))


def _clean(formula: Boolean) -> Boolean:
    """Light simplification of the result."""
    result = normalize(formula)
    if isinstance(result, BooleanAtom):
        return result
    if not isinstance(result, BooleanFunction):
        # SymPy's simplify_logic returns a relation as it is
        return with_sides(result)
    from sympy_extras.logic.boolalg import simplify_logic
    try:
        simplified = simplify_logic(result)
    except (ValueError, TypeError):
        simplified = result
    return with_sides(simplified)


def with_sides(formula: Boolean) -> Boolean:
    """Every relation ``p - q rel 0`` of the formula written as ``p rel q``.

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.polys.virtual_substitution import with_sides
    >>> x, y = symbols('x y')
    >>> with_sides((x - y > 0) & (x + 1 < 0))
    (x > y) & (x < -1)
    """
    formula = to_boolean(formula)
    return _substitute(formula, {a: _sides(a) for a in _atoms(formula)})


def linear_quantifier_elimination(formula: Boolean, prefix: Prefix) -> tuple[Boolean, Prefix]:
    """Eliminate, from the innermost outwards, the quantified variables in
    which the formula is linear; the remaining prefix (outermost
    quantifiers whose variables occur nonlinearly) is returned with the
    partially eliminated formula.

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.polys.virtual_substitution import linear_quantifier_elimination
    >>> x, y, z = symbols('x y z')
    >>> linear_quantifier_elimination((x > y) & (y > z), [('exists', x)])
    (y > z, [])
    >>> linear_quantifier_elimination((x > y) & (y > z), [('forall', z), ('exists', x)])
    (False, [])
    >>> linear_quantifier_elimination((x**2 > y) & (y > z), [('exists', x), ('exists', z)])
    (x**2 > y, [('exists', x)])
    """
    remaining = list(prefix)
    current = to_boolean(formula)
    while remaining:
        kind, v = remaining[-1]
        if not is_linear_in(current, v):
            break
        remaining.pop()
        if kind == 'exists':
            current = eliminate_linear(current, v)
        else:
            current = _clean(not_(eliminate_linear(not_(current), v)))
    return current, remaining


# ---------------------------------------------------------------------------
# the quadratic case

def _quadratic_coefficients(atom: Relational, x: Symbol) -> Optional[tuple[Expr, Expr, Expr]]:
    """``(a, b, c)`` with ``atom.lhs - atom.rhs == a*x**2 + b*x + c``."""
    p = as_expr(atom.lhs - atom.rhs)
    if not p.has(x):
        return _ZERO, _ZERO, p
    coefficients = _coefficients(atom, x)
    if coefficients is None or _degree(coefficients) > 2:
        return None
    while len(coefficients) < 3:
        coefficients.insert(0, _ZERO)
    a, b, c = coefficients
    return a, b, c


def is_quadratic_in(formula: Boolean, x: Symbol) -> bool:
    """Whether every relation of the formula has degree at most two in
    ``x``.

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.polys.virtual_substitution import is_quadratic_in
    >>> x, y = symbols('x y')
    >>> is_quadratic_in((x**2 + y*x > 0) & (x < 1), x)
    True
    >>> is_quadratic_in(x**3 > y, x)
    False
    """
    return all(_quadratic_coefficients(a, x) is not None for a in _atoms(formula))


class _Root:
    """A test point ``(p + q*sqrt(r))/s`` with the guard under which it
    is a real root."""

    __slots__ = ('p', 'q', 'r', 's', 'guard')

    def __init__(self, p: Expr, q: Expr, r: Expr, s: Expr, guard: Boolean) -> None:
        self.p: Final[Expr] = p
        self.q: Final[Expr] = q
        self.r: Final[Expr] = r
        self.s: Final[Expr] = s
        self.guard: Final[Boolean] = guard


def _parts(a2: Expr, a1: Expr, a0: Expr, root: _Root) -> tuple[Expr, Expr]:
    """``A, B`` with ``s**2 * g((p + q sqrt(r))/s) == A + B sqrt(r)`` for
    ``g = a2 x**2 + a1 x + a0``."""
    p, q, r, s = root.p, root.q, root.r, root.s
    A = a2*(p**2 + q**2*r) + a1*s*p + a0*s**2
    B = 2*a2*p*q + a1*s*q
    return expand(A), expand(B)


def _sign_with_sqrt(kind: Kind, A: Expr, B: Expr, r: Expr) -> Boolean:
    """``A + B sqrt(r) rel 0`` (``r >= 0``) as a formula in ``A, B, r``."""
    if B == 0:
        return _relation(kind, A)
    D = expand(A**2 - B**2*r)
    if kind is Eq:
        return or_(and_(eq(A, 0), eq(B, 0)), and_(A*B <= 0, eq(D, 0)))
    if kind is Ne:
        return not_(_sign_with_sqrt(Eq, A, B, r))
    if kind is Lt:
        return or_(and_(B <= 0, or_(A < 0, D < 0)), and_(B > 0, A < 0, D > 0))
    if kind is Le:
        return or_(and_(B <= 0, or_(A <= 0, D <= 0)), and_(B > 0, A <= 0, D >= 0))
    if kind is Gt:
        return _sign_with_sqrt(Lt, -A, -B, r)
    return _sign_with_sqrt(Le, -A, -B, r)


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
    derivatives: list[tuple[Expr, Expr, Expr]] = [(a2, a1, a0), (_ZERO, 2*a2, a1), (_ZERO, _ZERO, 2*a2)]
    values = [_parts(d2, d1, d0, root) for d2, d1, d0 in derivatives]
    zero = [_sign_with_sqrt(Eq, A, B, root.r) for A, B in values]
    if kind is Eq:
        return and_(*zero)
    if kind is Ne:
        return not_(and_(*zero))
    strict: Kind = Lt if kind in (Lt, Le) else Gt
    cases: list[Boolean] = []
    for i, (A, B) in enumerate(values):
        cases.append(and_(*zero[:i], _sign_with_sqrt(strict, A, B, root.r)))
    if kind in (Le, Ge):
        cases.append(and_(*zero))
    return or_(*cases)


def _at_minus_infinity_quadratic(atom: Relational, coefficients: tuple[Expr, Expr, Expr]) -> Boolean:
    """The atom ``a x**2 + b x + c rel 0`` at ``x = -oo``: the sign of the
    leading coefficient, ``-b`` when ``a = 0``."""
    a, b, c = coefficients
    kind = _kind(atom)
    if kind is Eq:
        return and_(eq(a, 0), eq(b, 0), eq(c, 0))
    if kind is Ne:
        return or_(ne(a, 0), ne(b, 0), ne(c, 0))
    strict: Kind = Lt if kind in (Lt, Le) else Gt
    reverse: Kind = Gt if strict is Lt else Lt
    return or_(_relation(strict, a), and_(eq(a, 0), _relation(reverse, b)),
               and_(eq(a, 0), eq(b, 0), _relation(kind, c)))


def eliminate_quadratic(formula: Boolean, x: Symbol) -> Boolean:
    """`\\exists x\\, \\varphi` for a formula of degree at most two in ``x``.

    Examples
    ========

    >>> from sympy_extras.core import symbols, Eq
    >>> from sympy_extras.polys.virtual_substitution import eliminate_quadratic
    >>> x, a, b, c = symbols('x a b c')
    >>> eliminate_quadratic(Eq(x**2 + b*x + c, 0), x)
    b**2 >= 4*c
    >>> eliminate_quadratic((x**2 < a) & (x > 1), x)
    a > 1
    """
    formula = to_nnf(to_boolean(formula))
    if not formula.has(x):
        return formula
    # a top-level equation a x + b = 0 with a nonzero number a pins x: the
    # one test point is -b/a, and no disjunction is needed. Without this a
    # system of five linear equations took a minute to say True, all of it
    # spent on test points that were thrown away (sympy-extras#47).
    pinned = _pinned_value(formula, x)
    if pinned is not None:
        return _clean(to_boolean(formula._xreplace_evaluated({x: pinned})))
    coefficients: dict[Relational, tuple[Expr, Expr, Expr]] = {}
    for atom in _atoms(formula):
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
            roots.append(_Root(-c, _ZERO, _ZERO, b, and_(eq(a, 0), ne(b, 0))))
        if a != 0:
            r = expand(b**2 - 4*a*c)
            for sign in (Integer(1), Integer(-1)):
                roots.append(_Root(-b, sign, r, 2*a, and_(ne(a, 0), r >= 0)))
        for root in roots:
            replacements: dict[Relational, Boolean] = {}
            for other, triple in coefficients.items():
                if shifted:
                    replacements[other] = _at_algebraic_plus_epsilon(other, triple, root)
                else:
                    replacements[other] = _at_algebraic(other, triple, root)
            disjuncts.append(and_(root.guard, _substitute(formula, replacements)))
    return _clean(or_(*disjuncts))


def virtual_substitution_elimination(formula: Boolean, prefix: Prefix, max_atoms: int = 12,
                                     max_free: int = 2) -> tuple[Boolean, Prefix]:
    """Eliminate, innermost first, the quantified variables in which the
    formula is linear or quadratic; the remaining prefix is returned with
    the partially eliminated formula. The quadratic case is used with at
    most ``max_atoms`` relations and ``max_free`` free variables (it
    squares the degrees and its results are only simplified afterwards by
    the decomposition, which handles two free variables).

    >>> from sympy_extras.core import symbols, Eq
    >>> from sympy_extras.polys.virtual_substitution import virtual_substitution_elimination
    >>> x, y, b, c = symbols('x y b c')
    >>> virtual_substitution_elimination(Eq(x**2 + b*x + c, 0), [('exists', x)])
    (b**2 >= 4*c, [])
    >>> virtual_substitution_elimination(x**2 + b*x + c > 0, [('forall', x)])
    (b**2 < 4*c, [])
    """
    remaining = list(prefix)
    current = to_boolean(formula)
    bound = {v for _, v in prefix}
    free = len([s for s in current.free_symbols if s not in bound])
    while remaining:
        kind, v = remaining[-1]
        eliminate: Callable[[Boolean, Symbol], Boolean]
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
            current = _clean(not_(eliminate(not_(current), v)))
    return current, remaining
