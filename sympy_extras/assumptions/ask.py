"""Truth values of statements under assumptions."""
from __future__ import annotations

from typing import Iterable, Optional, Union

from sympy.assumptions import Q, ask as _sympy_ask
from sympy.assumptions.assume import AppliedPredicate
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.function import Function
from sympy.core.mul import Mul
from sympy.core.numbers import Rational
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.sets.contains import Contains
from sympy.core.relational import Relational, Gt, Lt, Ge, Le
from sympy.core.symbol import Symbol
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.hyperbolic import sinh, cosh, tanh, asinh, acosh, atanh
from sympy.functions.elementary.integers import floor, ceiling
from sympy.functions.elementary.trigonometric import sin, cos, tan, asin, acos, atan
from sympy.logic.boolalg import (Boolean, BooleanTrue, BooleanFalse, And, Or,
    Not, Implies, Equivalent, Xor, ITE, true, false)
from sympy.sets.sets import Set

from sympy_extras._typing import Truth, as_boolean, as_expr, free_symbols, sorted_symbols

from sympy_extras.polys.cad import truth_tables
from sympy_extras.polys.virtual_substitution import is_linear_in, linear_quantifier_elimination
from sympy_extras._typing import QuantifierPrefix

from .context import global_assumptions
from .facts import Facts, normalize, to_polynomial, _predicate_of_atom
from .quantifiers import Quantifier

__all__ = ['ask']


Assumptions = Union[None, Boolean, bool, Iterable[Union[Boolean, bool]]]


def _facts(assumptions: Assumptions, domain: Optional[Set], symbols: Iterable[Basic]) -> Facts:
    """The facts from the global assumptions and the explicit ones."""
    items: list[Union[Boolean, bool]] = list(global_assumptions)
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
    if isinstance(poly, (BooleanTrue, BooleanFalse)):
        return bool(poly)
    premise = facts.polynomial
    gens = sorted_symbols(free_symbols(poly) | free_symbols(premise))
    if not gens:
        return bool(poly)
    linear = _linear_ask(poly, premise, gens)
    if linear is not None:
        return linear
    _, (holds, values) = truth_tables([premise, poly], gens)
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
    combined = And(premise, poly)
    if not all(is_linear_in(combined, g) for g in gens):
        return None
    prefix: QuantifierPrefix = [('exists', g) for g in gens]
    counterexamples, rest = linear_quantifier_elimination(And(premise, Not(poly)), prefix)
    examples, rest_ = linear_quantifier_elimination(combined, prefix)
    if rest or rest_ or not isinstance(counterexamples, (BooleanTrue, BooleanFalse)) \
            or not isinstance(examples, (BooleanTrue, BooleanFalse)):
        return None
    if examples is false and counterexamples is false:
        # the premise has no real solution: nothing can be said
        return None
    if counterexamples is false:
        return True
    if examples is false:
        return False
    return None


def _evaluate_atom(atom: Boolean, facts: Facts) -> Truth:
    if isinstance(atom, (BooleanTrue, BooleanFalse)):
        return bool(atom)
    if isinstance(atom, (Gt, Lt, Ge, Le)):
        # an inequality is false when its sides are not both real
        try:
            if _sympy_ask(Q.real(atom.lhs - atom.rhs), facts.predicates) is False:
                return False
        except ValueError:
            pass
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
        try:
            value = _sympy_ask(predicate, facts.predicates)
        except ValueError:
            value = None
        if value is not None:
            return value
    if isinstance(atom, Relational) and not atom.free_symbols:
        value = atom.doit()
        if value in (true, false):
            return bool(value)
    value = _cad_ask(atom, facts)
    if value is None and isinstance(atom, Relational):
        # expressions beyond polynomials: calculus and interval arithmetic
        from .analysis import decide_relational
        value = decide_relational(atom, facts)
    if value is None:
        value = _bounded_cad_ask(atom, facts)
    return value


def _real_membership(atom: Boolean) -> Optional[Expr]:
    """The expression of an atom ``Contains(e, S.Reals)`` or ``Q.real(e)``."""
    if isinstance(atom, Contains) and atom.args[1] == S.Reals and isinstance(atom.args[0], Expr):
        return atom.args[0]
    if isinstance(atom, AppliedPredicate) and atom.function == Q.real and isinstance(atom.arguments[0], Expr):
        return atom.arguments[0]
    return None


def _fragile(e: Expr) -> bool:
    """Whether SymPy's ``Q.real`` handlers are known to misjudge ``e``:
    non-integer powers, logarithms and inverse trigonometric functions of
    expressions whose sign they do not check."""
    return any(isinstance(a, Pow) and not as_expr(a.exp).is_integer for a in e.atoms(Pow)) or \
        bool(e.atoms(log, asin, acos, acosh, atanh))


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
        base, exponent = as_expr(e.base), as_expr(e.exp)
        base_real = _realness(base, facts)
        if exponent.is_integer:
            return base_real
        if base_real is not True:
            return None
        if isinstance(exponent, Rational) or _realness(exponent, facts) is True:
            if _evaluate(as_boolean(base >= 0), facts) is True:
                return True
            if isinstance(exponent, Rational) and _evaluate(as_boolean(base < 0), facts) is True:
                # the principal root of a negative number is not real
                return False
        return None
    if isinstance(e, log):
        argument = as_expr(e.args[0])
        if _evaluate(as_boolean(argument > 0), facts) is True:
            return True
        if _evaluate(as_boolean(argument < 0), facts) is True:
            return False
        return None
    if isinstance(e, (asin, acos)):
        argument = as_expr(e.args[0])
        if _evaluate(as_boolean(And(argument >= -1, argument <= 1)), facts) is True:
            return True
        if _evaluate(as_boolean(Or(argument > 1, argument < -1)), facts) is True:
            return False
        return None
    if isinstance(e, acosh):
        argument = as_expr(e.args[0])
        if _evaluate(as_boolean(argument >= 1), facts) is True:
            return True
        if _evaluate(as_boolean(argument < 1), facts) is True:
            return False
        return None
    if isinstance(e, atanh):
        argument = as_expr(e.args[0])
        if _evaluate(as_boolean(And(argument > -1, argument < 1)), facts) is True:
            return True
        if _evaluate(as_boolean(Or(argument > 1, argument < -1)), facts) is True:
            return False
        return None
    if isinstance(e, Abs):
        return True if e.args[0].is_finite is not False else None
    if isinstance(e, (exp, sin, cos, tan, atan, sinh, cosh, tanh, asinh, floor, ceiling)):
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
    if not formula.atoms(Function):
        return None
    abstraction = polynomial_abstraction(normalize(formula), facts.real)
    if abstraction is None:
        return None
    try:
        extended = Facts(facts.conjuncts + abstraction.constraints
                         + [Contains(t, S.Reals) for t in abstraction.variables])
    except (ValueError, TypeError):
        return None
    if len(free_symbols(extended.polynomial) | free_symbols(abstraction.formula)) > 3:
        return None
    from sympy_extras._timeout import attempt
    from sympy_extras.settings import settings
    limit = None if settings.timeout is None else min(settings.timeout, 5.0)
    return attempt(lambda: _cad_ask(normalize(abstraction.formula), extended), limit)


def _evaluate(formula: Boolean, facts: Facts) -> Truth:
    """Three-valued evaluation of a normalized formula: ``True``,
    ``False`` or ``None``."""
    if isinstance(formula, (BooleanTrue, BooleanFalse)):
        return bool(formula)
    if isinstance(formula, Not):
        value = _evaluate(formula.args[0], facts)
        return None if value is None else not value
    if isinstance(formula, (And, Or, Implies, Equivalent, Xor, ITE)):
        values = [_evaluate(as_boolean(arg), facts) for arg in formula.args]
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


def ask(query: Union[Boolean, bool], assumptions: Assumptions = None, domain: Optional[Set] = None) -> Truth:
    """Truth value of a statement under assumptions.

    Parameters
    ==========

    query : Boolean
        A relation (``x > 0``, ``Eq(x, y)``), a membership
        (``element(x, S.Integers)``), a predicate (``Q.positive(x)``), a
        Boolean combination of those or a quantified formula
        (``ForAll(x, ...)``, ``Exists(x, ...)``).
    assumptions : Boolean or list of Booleans, optional
        Assumptions in the same form. The assumptions in
        :data:`~sympy_extras.assumptions.global_assumptions` are always
        used as well.
    domain : Set, optional
        A named SymPy set (``S.Reals``, ``S.Integers``, ...) all the
        variables are assumed to belong to.

    Returns
    =======

    ``True`` if the statement follows from the assumptions, ``False`` if
    its negation does and ``None`` if neither can be established.

    The assumptions and the query are first translated to the predicates
    of :mod:`sympy.assumptions` and answered by :func:`sympy.ask`; what
    remains undecided and is a Boolean combination of polynomial relations
    between real variables is decided exactly by cylindrical algebraic
    decomposition (see :mod:`sympy_extras.polys.cad`). Note that an
    inequality between expressions states that both sides are real.

    Examples
    ========

    >>> from sympy import S, Eq, Q
    >>> from sympy.abc import x, y
    >>> from sympy_extras.assumptions import ask, element, ForAll
    >>> ask(x**3 > 0, x > 0)
    True
    >>> ask(x**2 - 2*x + 1 >= 0, x > 0)
    True
    >>> ask(x > 1, x > 2)
    True
    >>> ask(x > 3, x > 2) is None
    True
    >>> ask(x*y > 0, (x > 0) & (y < 0))
    False
    >>> ask(element(x + 1, S.Integers), element(x, S.Integers))
    True
    >>> ask(x**2 >= 0) is None
    True
    >>> ask(x**2 >= 0, domain=S.Reals)
    True
    >>> ask(ForAll(x, x**2 + 2*x*y + y**2 >= 0), domain=S.Reals)
    True
    >>> ask(Q.positive(x), Eq(x, 3))
    True
    """
    query_ = as_boolean(query)
    if query_.has(Quantifier):
        from .resolve import resolve
        query_ = resolve(query_, assumptions=assumptions, domain=domain)
    query_ = normalize(query_)
    facts = _facts(assumptions, domain, query_.free_symbols)
    return _evaluate(query_, facts)
