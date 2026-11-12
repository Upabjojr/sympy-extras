"""The consistency of predicates with the known facts about them, by the
SAT solver of :mod:`sympy_extras.logic.dpll2`.

:func:`predicates_consistent` decides whether a Boolean combination of
predicates of the core's ``Q`` can hold, as SymPy's ``satask`` does: the
formula is put in conjunctive normal form, with the composite predicates
replaced by disjunctions of primitive ones (``Q.real(x)`` is
``Q.negative(x) | Q.zero(x) | Q.positive(x)``) and the relations by the
binary predicates (``x > 1`` is ``Q.gt(x, 1)``); the facts of the
classes of the arguments are added, recursively on their arguments (a
sum of positive terms is positive, a product is zero if and only if a
factor is), and so are the known facts between the unary number
predicates for every argument (a rational number is algebraic, an even
number is not odd, ...). The formula is consistent when the clauses
together have a model.

The module is a typed port of :mod:`sympy.assumptions.satask`
(``get_all_relevant_facts``, ``extract_predargs``),
:mod:`sympy.assumptions.sathandlers` (the class facts),
:mod:`sympy.assumptions.facts` (the composite predicates) and the table of
:mod:`sympy.assumptions.ask_generated` (the known number facts in
conjunctive normal form), on the core's predicates and expressions.

Examples
========

>>> from sympy_extras.core import Q, symbols
>>> from sympy_extras.assumptions.satask import predicates_consistent
>>> x, y = symbols('x y')
>>> predicates_consistent(Q.positive(x) & Q.integer(x)), predicates_consistent(Q.positive(x) & Q.negative(x))
(True, False)
>>> predicates_consistent(Q.positive(x) & Q.positive(y) & Q.negative(x + y))
False
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/assumptions/satask.py, sympy/assumptions/sathandlers.py,
#   sympy/assumptions/facts.py, sympy/assumptions/ask_generated.py (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: strict typing, the class facts as typed functions instead of a
# registry of decorated handlers, the number facts as a table of names,
# no matrix facts, the SAT solver of sympy_extras.logic.dpll2.
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Callable, Final, Optional

from sympy_extras.core.add import Add
from sympy_extras.core.ask import Q
from sympy_extras.core.assume import AppliedPredicate, Predicate
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr
from sympy_extras.core.kind import NumberKind, UndefinedKind
from sympy_extras.core.logic import Boolean, and_, equivalent, implies, not_, or_, to_boolean
from sympy_extras.core.mul import Mul
from sympy_extras.core.numbers import ImaginaryUnit, Number, NumberSymbol
from sympy_extras.core.power import Pow
from sympy_extras.core.relational import Eq, Ge, Gt, Le, Lt, Ne, Relational
from sympy_extras.functions.elementary.complexes import Abs
from sympy_extras.logic.cnf import Clauses, conjunction, to_clauses
from sympy_extras.logic.dpll2 import is_satisfiable

__all__ = ['predicates_consistent']


# PORTED FROM SYMPY: sympy/assumptions/facts.py:get_composite_predicates
#: the composite predicates as disjunctions of primitive ones
_COMPOSITES: Final[dict[str, tuple[str, ...]]] = {
    'real': ('negative', 'zero', 'positive'),
    'integer': ('even', 'odd'),
    'nonpositive': ('negative', 'zero'),
    'nonzero': ('negative', 'positive'),
    'nonnegative': ('zero', 'positive'),
    'extended_real': ('negative_infinite', 'negative', 'zero', 'positive', 'positive_infinite'),
    'extended_positive': ('positive', 'positive_infinite'),
    'extended_negative': ('negative', 'negative_infinite'),
    'extended_nonzero': ('negative_infinite', 'negative', 'positive', 'positive_infinite'),
    'extended_nonpositive': ('negative_infinite', 'negative', 'zero'),
    'extended_nonnegative': ('zero', 'positive', 'positive_infinite'),
    'complex': ('algebraic', 'transcendental'),
}

#: the binary predicates of the relations
_RELATIONS: Final[tuple[tuple[type[Relational], str], ...]] = (
    (Eq, 'eq'), (Ne, 'ne'), (Gt, 'gt'), (Lt, 'lt'), (Ge, 'ge'), (Le, 'le'))

# PORTED FROM SYMPY: sympy/assumptions/ask_generated.py:get_all_known_number_facts
#: the known facts between the unary number predicates, in conjunctive
#: normal form: each clause is a tuple of (predicate, negated)
_NUMBER_FACTS: Final[tuple[tuple[tuple[str, bool], ...], ...]] = (
    (('algebraic', False), ('imaginary', True), ('transcendental', False)),
    (('algebraic', False), ('negative', True), ('transcendental', False)),
    (('algebraic', False), ('positive', True), ('transcendental', False)),
    (('algebraic', False), ('rational', True)),
    (('algebraic', False), ('transcendental', False), ('zero', True)),
    (('algebraic', True), ('finite', False)),
    (('algebraic', True), ('transcendental', True)),
    (('antihermitian', False), ('hermitian', False), ('zero', True)),
    (('antihermitian', False), ('imaginary', True)),
    (('commutative', False), ('finite', True)),
    (('commutative', False), ('infinite', True)),
    (('composite', False), ('even', True), ('positive', True), ('prime', False)),
    (('composite', True), ('even', False), ('odd', False)),
    (('composite', True), ('positive', False)),
    (('composite', True), ('prime', True)),
    (('even', False), ('odd', False), ('prime', True)),
    (('even', False), ('zero', True)),
    (('even', True), ('odd', True)),
    (('even', True), ('rational', False)),
    (('finite', False), ('transcendental', True)),
    (('finite', True), ('infinite', True)),
    (('hermitian', False), ('negative', True)),
    (('hermitian', False), ('positive', True)),
    (('hermitian', False), ('zero', True)),
    (('imaginary', True), ('negative', True)),
    (('imaginary', True), ('positive', True)),
    (('imaginary', True), ('zero', True)),
    (('infinite', False), ('negative_infinite', True)),
    (('infinite', False), ('positive_infinite', True)),
    (('irrational', False), ('negative', True), ('rational', False)),
    (('irrational', False), ('positive', True), ('rational', False)),
    (('irrational', False), ('rational', False), ('zero', True)),
    (('irrational', True), ('negative', False), ('positive', False), ('zero', False)),
    (('irrational', True), ('rational', True)),
    (('negative', False), ('positive', False), ('rational', True), ('zero', False)),
    (('negative', True), ('negative_infinite', True)),
    (('negative', True), ('positive', True)),
    (('negative', True), ('positive_infinite', True)),
    (('negative', True), ('zero', True)),
    (('negative_infinite', True), ('positive', True)),
    (('negative_infinite', True), ('positive_infinite', True)),
    (('negative_infinite', True), ('zero', True)),
    (('odd', True), ('rational', False)),
    (('positive', False), ('prime', True)),
    (('positive', True), ('positive_infinite', True)),
    (('positive', True), ('zero', True)),
    (('positive_infinite', True), ('zero', True)),
)

#: the predicates of the known number facts, in the order of first use
_NUMBER_PREDICATES: Final[tuple[str, ...]] = tuple(dict.fromkeys(name for c in _NUMBER_FACTS for name, _ in c))


def _known(name: str) -> Predicate:
    """The predicate of ``Q`` called ``name``."""
    return Predicate._known(name, 2 if name in ('eq', 'ne', 'gt', 'lt', 'ge', 'le') else 1)


def _expand(atom: Boolean) -> Optional[Boolean]:
    """The relations as binary predicates, the composite predicates as
    disjunctions of primitive ones (SymPy's ``to_NNF``)."""
    for cls, name in _RELATIONS:
        if isinstance(atom, cls):
            return AppliedPredicate(_known(name), atom.lhs, atom.rhs)
    if isinstance(atom, AppliedPredicate):
        parts = _COMPOSITES.get(atom.function.name)
        if parts is not None and atom.function == _known(atom.function.name):
            return or_(*[AppliedPredicate(_known(p), *atom.arguments) for p in parts])
    return None


def _clauses(formula: Boolean) -> Clauses:
    return to_clauses(formula, _expand)


# ---------------------------------------------------------------------------
# the facts of the classes

Fact = Callable[[Expr], Boolean]


def _all_args(fact: Fact, args: tuple[Expr, ...]) -> Boolean:
    return and_(*[fact(a) for a in args])


def _any_arg(fact: Fact, args: tuple[Expr, ...]) -> Boolean:
    return or_(*[fact(a) for a in args])


def _exactly_one_arg(fact: Fact, args: tuple[Expr, ...]) -> Boolean:
    facts = [fact(a) for a in args]
    return or_(*[and_(facts[i], *[not_(f) for f in facts[:i] + facts[i + 1:]]) for i in range(len(facts))])


def _real_or_imaginary(e: Expr) -> Boolean:
    return or_(Q.imaginary(e), Q.real(e))


def _not_integer(e: Expr) -> Boolean:
    return not_(Q.integer(e))


def _not_rational(e: Expr) -> Boolean:
    return not_(Q.rational(e))


# PORTED FROM SYMPY: sympy/assumptions/sathandlers.py:_ (Abs)
def _abs_facts(expr: Abs) -> list[Boolean]:
    arg = expr.args[0]
    return [Q.nonnegative(expr),
            equivalent(not_(Q.zero(arg)), not_(Q.zero(expr))),
            implies(Q.even(arg), Q.even(expr)),
            implies(Q.odd(arg), Q.odd(expr)),
            implies(Q.integer(arg), Q.integer(expr))]


# PORTED FROM SYMPY: sympy/assumptions/sathandlers.py:_ (Add)
def _add_facts(expr: Add) -> list[Boolean]:
    args = expr.args
    return [implies(_all_args(Q.positive, args), Q.positive(expr)),
            implies(_all_args(Q.negative, args), Q.negative(expr)),
            implies(_all_args(Q.real, args), Q.real(expr)),
            implies(_all_args(Q.rational, args), Q.rational(expr)),
            implies(_all_args(Q.integer, args), Q.integer(expr)),
            implies(_exactly_one_arg(_not_integer, args), not_(Q.integer(expr))),
            implies(_all_args(Q.real, args), implies(_exactly_one_arg(Q.irrational, args), Q.irrational(expr)))]


# PORTED FROM SYMPY: sympy/assumptions/sathandlers.py:_ (Mul)
def _mul_facts(expr: Mul) -> list[Boolean]:
    args = expr.args
    return [equivalent(Q.zero(expr), _any_arg(Q.zero, args)),
            implies(_all_args(Q.positive, args), Q.positive(expr)),
            implies(_all_args(Q.real, args), Q.real(expr)),
            implies(_all_args(Q.rational, args), Q.rational(expr)),
            implies(_all_args(Q.integer, args), Q.integer(expr)),
            implies(_exactly_one_arg(_not_rational, args), not_(Q.integer(expr))),
            implies(_all_args(Q.commutative, args), Q.commutative(expr)),
            implies(_all_args(Q.prime, args), not_(Q.prime(expr))),
            implies(_all_args(_real_or_imaginary, args),
                    implies(_exactly_one_arg(Q.imaginary, args), Q.imaginary(expr))),
            implies(_all_args(Q.real, args), implies(_exactly_one_arg(Q.irrational, args), Q.irrational(expr))),
            implies(_all_args(Q.integer, args), equivalent(_any_arg(Q.even, args), Q.even(expr)))]


# PORTED FROM SYMPY: sympy/assumptions/sathandlers.py:_ (Pow)
def _pow_facts(expr: Pow) -> list[Boolean]:
    base, exp = expr.base, expr.exp
    return [implies(and_(Q.real(base), Q.even(exp), Q.nonnegative(exp)), Q.nonnegative(expr)),
            implies(and_(Q.nonnegative(base), Q.odd(exp), Q.nonnegative(exp)), Q.nonnegative(expr)),
            implies(and_(Q.nonpositive(base), Q.odd(exp), Q.nonnegative(exp)), Q.nonpositive(expr)),
            equivalent(Q.zero(expr), and_(Q.zero(base), Q.positive(exp)))]


# PORTED FROM SYMPY: sympy/assumptions/sathandlers.py:_ (Number, NumberSymbol, ImaginaryUnit)
def _number_facts(expr: Expr) -> list[Boolean]:
    known: list[tuple[Predicate, Optional[bool]]] = [
        (Q.positive, expr.is_positive), (Q.zero, expr.is_zero), (Q.negative, expr.is_negative),
        (Q.rational, expr.is_rational), (Q.irrational, expr.is_irrational), (Q.even, expr.is_even),
        (Q.odd, expr.is_odd), (Q.imaginary, expr.is_imaginary), (Q.prime, expr.is_prime),
        (Q.composite, expr.is_composite)]
    return [p(expr) if value else not_(p(expr)) for p, value in known if value is not None]


def _class_facts(expr: Basic) -> list[Boolean]:
    """The facts of the class of ``expr`` (SymPy's ``class_fact_registry``)."""
    if isinstance(expr, Abs):
        return _abs_facts(expr)
    if isinstance(expr, Add):
        return _add_facts(expr)
    if isinstance(expr, Mul):
        return _mul_facts(expr)
    if isinstance(expr, Pow):
        return _pow_facts(expr)
    if isinstance(expr, (Number, NumberSymbol, ImaginaryUnit)):
        return _number_facts(expr)
    return []


def _arguments(cnf: Clauses) -> list[Basic]:
    """The arguments of the predicates among the atoms of ``cnf``, and the
    atoms which are not predicates."""
    found: dict[Basic, None] = {}
    for atom in cnf.atoms:
        if isinstance(atom, AppliedPredicate):
            for a in atom.arguments:
                found.setdefault(a, None)
        else:
            found.setdefault(atom, None)
    return list(found)


def _number_facts_of(expr: Basic) -> Clauses:
    """The known number facts instantiated on ``expr``."""
    atoms = [AppliedPredicate(_known(name), expr) for name in _NUMBER_PREDICATES]
    index = {name: i + 1 for i, name in enumerate(_NUMBER_PREDICATES)}
    return Clauses(atoms, [sorted((-index[name] if negated else index[name] for name, negated in c),
                                  key=lambda lit: (abs(lit), lit)) for c in _NUMBER_FACTS])


def _is_number_kind(e: Basic) -> bool:
    return e.kind == NumberKind or e.kind == UndefinedKind


# PORTED FROM SYMPY: sympy/assumptions/satask.py:get_all_relevant_facts
def predicates_consistent(predicates: Boolean) -> bool:
    """Whether a Boolean combination of predicates of the core's ``Q`` is
    consistent with the known facts about the predicates
    (``Q.positive(x) & Q.negative(x)`` is not), the counterpart of
    :func:`sympy_extras.assumptions.facts_sympy.predicates_consistent`.

    Examples
    ========

    >>> from sympy_extras.core import Q, Symbol
    >>> from sympy_extras.assumptions.satask import predicates_consistent
    >>> x = Symbol('x')
    >>> predicates_consistent(Q.positive(x) & Q.integer(x)), predicates_consistent(Q.positive(x) & Q.negative(x))
    (True, False)
    >>> predicates_consistent(Q.even(x) & Q.prime(x) & Q.composite(x)), predicates_consistent(Q.zero(x**2) & Q.positive(x))
    (False, False)
    """
    cnf = _clauses(to_boolean(predicates))
    if cnf.is_false:
        return False
    parts: list[Clauses] = [cnf]
    seen: dict[Basic, None] = {}
    exprs = _arguments(cnf)
    while exprs:
        new: dict[Basic, None] = {}
        for e in exprs:
            seen.setdefault(e, None)
        for e in exprs:
            for fact in _class_facts(e):
                fact_cnf = _clauses(fact)
                parts.append(fact_cnf)
                for atom in fact_cnf.atoms:
                    if isinstance(atom, AppliedPredicate):
                        for a in atom.arguments:
                            if a not in seen:
                                new.setdefault(a, None)
        exprs = list(new)
    if any(_is_number_kind(e) for e in seen):
        parts.extend(_number_facts_of(e) for e in seen)
    return is_satisfiable(conjunction(parts))
