"""Truth values of statements under assumptions.

The statements and the assumptions are Booleans of the core
(:mod:`sympy_extras.core.logic`): relations, memberships in the sets of
:mod:`sympy_extras.sets`, predicates of the core's ``Q``, their Boolean
combinations and the quantified formulas of
:mod:`sympy_extras.assumptions.quantifiers`.

The module works on the objects of the core and imports no SymPy:
:mod:`sympy_extras.assumptions.ask_sympy` is its face on SymPy objects.
The steps which are not ported go through SymPy at run time (the face
holds them, the functions ending in ``_of_core``): the cylindrical
algebraic decomposition (:mod:`sympy_extras.polys.cad`) and virtual
substitution (:mod:`sympy_extras.polys.virtual_substitution`), both
written on SymPy's polynomials, the zero test of the structure theorem
(:mod:`sympy_extras.simplify.structure`), and the sample points and the
reliable numerical values with which a difference is shown not to be
identically zero (:mod:`sympy_extras.integrals.conditions`,
:mod:`sympy_extras._numeric`). The questions about predicates go to the
core's :func:`~sympy_extras.core.ask.ask` (which asks SymPy's ``ask``,
through the bridge, what it cannot answer).
"""
from __future__ import annotations

from fractions import Fraction
from typing import Iterable, Optional, Union

from sympy_extras.core.add import Add
from sympy_extras.core.ask import Q, ask as _core_ask
from sympy_extras.core.assume import AppliedPredicate
from sympy_extras.core.basic import Basic, preorder_traversal
from sympy_extras.core.expr import Expr, as_expr
from sympy_extras.core.function import Application
from sympy_extras.core.logic import (ITE, And, Boolean, BooleanAtom, BooleanFalse, BooleanLike, BooleanTrue,
    Equivalent, Implies, Not, Or, Xor, and_, not_, or_, to_boolean)
from sympy_extras.core.mul import Mul
from sympy_extras.core.numbers import E, Rational, number_from_fraction, pi
from sympy_extras.core.power import Pow
from sympy_extras.core.relational import Eq, Ge, Gt, Le, Lt, Ne, Relational
from sympy_extras.core.symbol import Symbol
from sympy_extras.functions.elementary.complexes import Abs
from sympy_extras.functions.elementary.exponential import Exp, Log
from sympy_extras.functions.elementary.hyperbolic import Acosh, Asinh, Atanh, Cosh, Sinh, Tanh
from sympy_extras.functions.elementary.integers import Ceiling, Floor
from sympy_extras.functions.elementary.miscellaneous import MinMaxBase
from sympy_extras.functions.elementary.piecewise import Piecewise
from sympy_extras.functions.elementary.trigonometric import Acos, Asin, Atan, Cos, Sin, Tan
from sympy_extras.sets.sets import Contains, Reals, Set

from .context import global_assumptions
from .facts import Facts, _predicate_of_atom, element, normalize, to_polynomial
from .quantifiers import Prefix, Quantifier

__all__ = ['ask', 'Assumptions', 'Truth']

#: a truth value of three states
Truth = Optional[bool]

#: assumptions as the functions take them: one statement or several
Assumptions = Union[None, BooleanLike, Iterable[BooleanLike]]


def _symbols(e: Basic) -> set[Symbol]:
    """The free symbols which are plain symbols."""
    return {s for s in e.free_symbols if isinstance(s, Symbol)}


def _sorted_symbols(symbols: Iterable[Symbol]) -> list[Symbol]:
    return sorted(symbols, key=lambda s: s.name)


def _is_function(e: Basic) -> bool:
    """Whether ``e`` is an applied function as SymPy's ``Function`` sees
    it: ``Max`` and ``Min`` are not, ``Piecewise`` and ``Mod`` are."""
    return (isinstance(e, Application) and not isinstance(e, MinMaxBase)) or isinstance(e, Piecewise)


def _has_function(e: Basic) -> bool:
    return any(_is_function(node) for node in preorder_traversal(e))


def _has_fractional_power(e: Basic) -> bool:
    """Whether ``e`` has a power whose exponent is not known to be an
    integer."""
    return any(isinstance(p, Pow) and not p.exp.is_integer for p in preorder_traversal(e))


def _facts(assumptions: Assumptions, domain: Optional[Set], symbols: Iterable[Basic]) -> Facts:
    """The facts from the global assumptions and the explicit ones."""
    items: list[BooleanLike] = list(global_assumptions)
    if isinstance(assumptions, (Boolean, bool)):
        items.append(assumptions)
    elif assumptions is not None:
        items.extend(assumptions)
    return Facts(items, domain, [s for s in symbols if isinstance(s, Symbol)])


def _cad_ask(formula: Boolean, facts: Facts) -> Truth:
    """Truth value of a Boolean combination of polynomial relations under
    the polynomial part of the facts, by cylindrical algebraic
    decomposition: ``True`` if it holds on every real point satisfying the
    assumptions, ``False`` if it fails on every such point, ``None``
    otherwise (or if the formula is not polynomial in real variables)."""
    poly = to_polynomial(formula, facts.real)
    if poly is None:
        return None
    if isinstance(poly, BooleanAtom):
        return isinstance(poly, BooleanTrue)
    premise = facts.polynomial
    gens = _sorted_symbols(_symbols(poly) | _symbols(premise))
    if not gens:
        return isinstance(poly, BooleanTrue)
    linear = _linear_ask(poly, premise, gens)
    if linear is not None:
        return linear
    from .ask_sympy import truth_tables_of_core
    holds, values = truth_tables_of_core([premise, poly], gens)
    relevant = [v for h, v in zip(holds, values) if h]
    if not relevant:
        # the assumptions have no real solution: nothing can be said
        return None
    if all(relevant):
        return True
    if not any(relevant):
        return False
    return None


def _linear_ask(poly: Boolean, premise: Boolean, gens: list[Symbol]) -> Truth:
    """A formula linear in every variable decided by virtual substitution
    (Loos–Weispfenning), which needs no decomposition: the formula holds
    under the premise when ``premise & ~poly`` has no real solution, fails
    when ``premise & poly`` has none; ``None`` otherwise, or when the
    formula is not linear (the CAD decides then)."""
    from .ask_sympy import is_linear_in_of_core, linear_quantifier_elimination_of_core
    combined = and_(premise, poly)
    if not all(is_linear_in_of_core(combined, g) for g in gens):
        return None
    prefix: Prefix = [('exists', g) for g in gens]
    counterexamples, rest = linear_quantifier_elimination_of_core(and_(premise, not_(poly)), prefix)
    examples, rest_ = linear_quantifier_elimination_of_core(combined, prefix)
    if rest or rest_ or not isinstance(counterexamples, BooleanAtom) or not isinstance(examples, BooleanAtom):
        return None
    if isinstance(examples, BooleanFalse) and isinstance(counterexamples, BooleanFalse):
        # the premise has no real solution: nothing can be said
        return None
    if isinstance(counterexamples, BooleanFalse):
        return True
    if isinstance(examples, BooleanFalse):
        return False
    return None


def _predicate_ask(predicate: Boolean, facts: Facts) -> Truth:
    """The core's ``ask`` of a predicate under the predicates of the
    facts, ``None`` when it gives up."""
    try:
        return _core_ask(predicate, facts.predicates)
    except (ValueError, TypeError, AttributeError, AssertionError):
        # AssertionError: SymPy 1.14's LRA solver (asked through the
        # bridge) asserts on a term without symbols in the encoded CNF
        # (lra_theory.py line 319)
        return None


def _evaluate_atom(atom: Boolean, facts: Facts) -> Truth:
    if isinstance(atom, BooleanAtom):
        return isinstance(atom, BooleanTrue)
    if isinstance(atom, (Gt, Lt, Ge, Le)):
        # an inequality is false when its sides are not both real
        if _predicate_ask(Q.real(atom.lhs - atom.rhs), facts) is False:
            return False
    membership = _real_membership(atom)
    if membership is not None:
        # SymPy's ``ask(Q.real(sqrt(a - 2)), Q.positive(a))`` is True: the
        # reality of roots and logarithms is decided here instead
        value = _realness(membership, facts)
        if value is not None:
            return value
        if _fragile(membership):
            return None
    predicate = _predicate_of_atom(atom)
    if predicate is not None:
        value = _predicate_ask(predicate, facts)
        if value is not None:
            return value
    if isinstance(atom, Relational) and not atom.free_symbols:
        done = atom.doit()
        if isinstance(done, BooleanAtom):
            return isinstance(done, BooleanTrue)
    value = _cad_ask(atom, facts)
    if value is None and isinstance(atom, Relational):
        # the two sides equal as elementary functions: a proof
        value = _structure_ask(atom, facts)
    if value is None and isinstance(atom, Relational):
        # expressions beyond polynomials: calculus and interval arithmetic
        from .analysis import decide_relational
        value = decide_relational(atom, facts)
    if value is None:
        value = _bounded_cad_ask(atom, facts)
    return value


#: the budget of the structure theorem inside ``ask``, and its depth: the
#: tower asks about the signs of the arguments it meets, and those
#: questions do not start a tower of their own
_STRUCTURE_SECONDS = 6.0
_structure_depth = 0


def _structure_ask(atom: Relational, facts: Facts) -> Truth:
    """The truth of a relation whose two sides are the same elementary
    function on the region of the facts, by the zero test of
    :mod:`sympy_extras.simplify.structure`: ``Eq`` holds, ``Ne``, ``Gt`` and
    ``Lt`` fail, ``Ge`` and ``Le`` hold where the sides are real. A
    difference which is not identically zero decides nothing (it may
    vanish somewhere), except between constants, where it is a number
    other than zero.

    >>> from sympy_extras.core import Eq, Rational, pi, symbols
    >>> from sympy_extras.functions import atan, log
    >>> from sympy_extras.assumptions.ask import ask
    >>> x = symbols('x')
    >>> ask(Eq(4*atan(Rational(1, 5)) - atan(Rational(1, 239)), pi/4))
    True
    >>> ask(Eq(log(x**2), 2*log(x)), x > 0), ask(Eq(log(x**2), 2*log(x)))
    (True, None)
    """
    global _structure_depth
    if _structure_depth > 0 or not isinstance(atom, (Eq, Ne, Gt, Ge, Lt, Le)):
        return None
    difference = atom.lhs - atom.rhs
    if not (_has_function(difference) or _has_fractional_power(difference) or difference.has(pi, E)):
        return None                                         # polynomial: the decomposition's business
    from sympy_extras._timeout import attempt
    from .ask_sympy import structure_zero_of_core
    if difference.free_symbols and _clearly_not_zero(difference, facts):
        return None                                         # the common case, told at one point

    conjuncts = list(facts.conjuncts)
    _structure_depth += 1
    try:
        zero = attempt(lambda: structure_zero_of_core(difference, conjuncts), _STRUCTURE_SECONDS)
    finally:
        _structure_depth -= 1
    if zero is None:
        return None
    if isinstance(atom, Eq):
        return zero
    if isinstance(atom, Ne):
        return not zero
    if not zero:
        return None                                         # the sign of a nonzero constant is not told here
    if isinstance(atom, (Gt, Lt)):
        return False
    return True if _realness(atom.lhs, facts) is True else None


def _clearly_not_zero(difference: Expr, facts: Facts) -> bool:
    """Whether ``difference`` has a value away from zero at one point of
    the region of the facts: then it is not identically zero there, and
    the tower has nothing to prove."""
    from .ask_sympy import reliable_complex_of_core, sample_values_of_core
    symbols = _sorted_symbols(_symbols(difference))
    values: Optional[dict[Symbol, Expr]] = {
        s: number_from_fraction(Fraction(3 + 2 * i, 7 + i)) for i, s in enumerate(symbols)}
    if facts.conjuncts:
        values = sample_values_of_core(symbols, list(facts.conjuncts), difference)
    if values is None:
        return False
    # a value whose branch a rounding error chooses shows nothing (the
    # bug: log(-2*sqrt(2) - 2*sqrt(z)) - log(2*sqrt(2)) - I*pi with z a sum
    # which is exactly zero came out as -2*I*pi, "clearly not zero")
    value = reliable_complex_of_core(difference, 20, values)
    return value is not None and abs(value) > 1e-12


def _real_membership(atom: Boolean) -> Optional[Expr]:
    """The expression of an atom ``Contains(e, Reals)`` or ``Q.real(e)``."""
    if isinstance(atom, Contains):
        member, s = atom.args
        if s == Reals and isinstance(member, Expr):
            return member
    if isinstance(atom, AppliedPredicate) and atom.function == Q.real and isinstance(atom.arguments[0], Expr):
        return atom.arguments[0]
    return None


def _fragile(e: Expr) -> bool:
    """Whether SymPy's ``Q.real`` handlers are known to misjudge ``e``:
    non-integer powers, logarithms and inverse trigonometric functions of
    expressions whose sign they do not check."""
    return _has_fractional_power(e) or bool(e.atoms(Log, Asin, Acos, Acosh, Atanh))


def _realness(e: Expr, facts: Facts) -> Truth:
    """Whether ``e`` is a real number, by the structure of the expression
    and the facts about the signs of its parts."""
    if isinstance(e, Symbol):
        return True if e in facts.real or e.is_extended_real else None
    if e.is_number:
        return True if e.is_extended_real else (False if e.is_extended_real is False else None)
    if isinstance(e, (Add, Mul)):
        parts = [_realness(as_expr(a), facts) for a in e.args]
        return True if all(p is True for p in parts) else None
    if isinstance(e, Pow):
        base, exponent = e.base, e.exp
        base_real = _realness(base, facts)
        if exponent.is_integer:
            return base_real
        if base_real is not True:
            return None
        if isinstance(exponent, Rational) or _realness(exponent, facts) is True:
            if _evaluate(base >= 0, facts) is True:
                return True
            if isinstance(exponent, Rational) and _evaluate(base < 0, facts) is True:
                # the principal root of a negative number is not real
                return False
        return None
    if isinstance(e, Log):
        argument = as_expr(e.args[0])
        if _evaluate(argument > 0, facts) is True:
            return True
        if _evaluate(argument < 0, facts) is True:
            return False
        return None
    if isinstance(e, (Asin, Acos)):
        argument = as_expr(e.args[0])
        if _evaluate(and_(argument >= -1, argument <= 1), facts) is True:
            return True
        if _evaluate(or_(argument > 1, argument < -1), facts) is True:
            return False
        return None
    if isinstance(e, Acosh):
        argument = as_expr(e.args[0])
        if _evaluate(argument >= 1, facts) is True:
            return True
        if _evaluate(argument < 1, facts) is True:
            return False
        return None
    if isinstance(e, Atanh):
        argument = as_expr(e.args[0])
        if _evaluate(and_(argument > -1, argument < 1), facts) is True:
            return True
        if _evaluate(or_(argument > 1, argument < -1), facts) is True:
            return False
        return None
    if isinstance(e, Abs):
        return True if as_expr(e.args[0]).is_finite is not False else None
    if isinstance(e, (Exp, Sin, Cos, Tan, Atan, Sinh, Cosh, Tanh, Asinh, Floor, Ceiling)):
        argument = as_expr(e.args[0])
        return True if _realness(argument, facts) is True else None
    return None


def _bounded_cad_ask(formula: Boolean, facts: Facts) -> Truth:
    """Decide a formula with elementary functions by replacing them with
    variables constrained by polynomial bounds (see
    :mod:`sympy_extras.assumptions.bounds`) and asking the CAD whether the
    formula holds, or fails, for every allowed value of the new
    variables."""
    from .bounds import polynomial_abstraction
    if not _has_function(formula):
        return None
    abstraction = polynomial_abstraction(normalize(formula), facts.real)
    if abstraction is None:
        return None
    # the bounds of log(u) hold for u > 0: the facts must put every
    # point there (the bug: log(x) < 0 was True under -1 < x < 1, the
    # points with x <= 0, where log(x) is not real, dropped by the
    # constraint u > 0)
    for requirement in abstraction.requirements:
        if _cad_ask(normalize(requirement), facts) is not True:
            return None
    try:
        extended = Facts(list(facts.conjuncts) + list(abstraction.constraints)
                         + [element(t, Reals) for t in abstraction.variables])
    except (ValueError, TypeError):
        return None
    if len(_symbols(extended.polynomial) | _symbols(abstraction.formula)) > 3:
        return None
    from sympy_extras._timeout import attempt
    from sympy_extras.settings import settings
    limit = None if settings.timeout is None else min(settings.timeout, 5.0)
    bounded = normalize(abstraction.formula)
    return attempt(lambda: _cad_ask(bounded, extended), limit)


def _evaluate(formula: Boolean, facts: Facts) -> Truth:
    """Three-valued evaluation of a normalized formula: ``True``,
    ``False`` or ``None``."""
    if isinstance(formula, BooleanAtom):
        return isinstance(formula, BooleanTrue)
    if isinstance(formula, Not):
        value = _evaluate(formula.arg, facts)
        return None if value is None else not value
    if isinstance(formula, (And, Or, Implies, Equivalent, Xor, ITE)):
        values = [_evaluate(to_boolean(arg), facts) for arg in formula.args]
        value = _combine(formula, values)
        if value is None:
            value = _cad_ask(formula, facts)
        if value is None:
            value = _bounded_cad_ask(formula, facts)
        return value
    return _evaluate_atom(formula, facts)


def _combine(formula: Boolean, values: list[Truth]) -> Truth:
    if isinstance(formula, And):
        if any(v is False for v in values):
            return False
        if all(v is True for v in values):
            return True
        return None
    if isinstance(formula, Or):
        if any(v is True for v in values):
            return True
        if all(v is False for v in values):
            return False
        return None
    if isinstance(formula, Implies):
        a, b = values
        if a is False or b is True:
            return True
        if a is True and b is False:
            return False
        return None
    if isinstance(formula, Equivalent):
        if any(v is None for v in values):
            return None
        return all(v == values[0] for v in values)
    if isinstance(formula, Xor):
        if any(v is None for v in values):
            return None
        return sum(1 for v in values if v) % 2 == 1
    if isinstance(formula, ITE):
        c, a, b = values
        if c is True:
            return a
        if c is False:
            return b
        if a is not None and a == b:
            return a
        return None
    return None


def ask(query: BooleanLike, assumptions: Assumptions = None, domain: Optional[Set] = None) -> Truth:
    """Truth value of a statement under assumptions.

    Parameters
    ==========

    query : Boolean
        A relation (``x > 0``, ``Eq(x, y)``), a membership
        (``element(x, Integers)``), a predicate (``Q.positive(x)``), a
        Boolean combination of those or a quantified formula
        (``forall(x, ...)``, ``exists(x, ...)``).
    assumptions : Boolean or list of Booleans, optional
        Assumptions in the same form. The assumptions in
        :data:`~sympy_extras.assumptions.context.global_assumptions` are
        always used as well.
    domain : Set, optional
        A named set (``Reals``, ``Integers``, ...) all the variables are
        assumed to belong to.

    Returns
    =======

    ``True`` if the statement follows from the assumptions, ``False`` if
    its negation does and ``None`` if neither can be established.

    The assumptions and the query are first translated to the predicates
    of the core's ``Q`` and answered by its
    :func:`~sympy_extras.core.ask.ask`; what remains undecided and is a
    Boolean combination of polynomial relations between real variables is
    decided exactly by cylindrical algebraic decomposition (see
    :mod:`sympy_extras.polys.cad`). Note that an inequality between
    expressions states that both sides are real.

    Examples
    ========

    >>> from sympy_extras.core import Eq, Q, symbols
    >>> from sympy_extras.sets import Integers, Reals
    >>> from sympy_extras.assumptions.ask import ask
    >>> from sympy_extras.assumptions.facts import element
    >>> from sympy_extras.assumptions.quantifiers import ForAll
    >>> x, y = symbols('x y')
    >>> ask(x**3 > 0, x > 0), ask(x**2 - 2*x + 1 >= 0, x > 0), ask(x > 1, x > 2), ask(x > 3, x > 2)
    (True, True, True, None)
    >>> ask(x*y > 0, (x > 0) & (y < 0)), ask(element(x + 1, Integers), element(x, Integers))
    (False, True)
    >>> ask(x**2 >= 0), ask(x**2 >= 0, domain=Reals)
    (None, True)
    >>> ask(ForAll(x, x**2 + 2*x*y + y**2 >= 0), domain=Reals), ask(Q.positive(x), Eq(x, 3))
    (True, True)
    """
    query_ = to_boolean(query)
    if query_.has(Quantifier):
        from .resolve import resolve
        query_ = resolve(query_, assumptions=assumptions, domain=domain)
    query_ = normalize(query_)
    facts = _facts(assumptions, domain, query_.free_symbols)
    return _evaluate(query_, facts)
