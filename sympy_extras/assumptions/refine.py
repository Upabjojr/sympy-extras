"""Refining and simplifying expressions under assumptions.

The assumptions are statements as described in
:mod:`sympy_extras.assumptions` (``x > 1``, ``element(n, S.Integers)``, ...).
Two mechanisms are combined:

1. *Abstraction.* Every symbol, and every subexpression whose sign or
   nature matters (the base of a fractional power, the argument of
   ``Abs``, ``log``, ``floor``, ...), is decided against the assumptions
   with :func:`~sympy_extras.assumptions.ask` (predicates of SymPy plus
   cylindrical algebraic decomposition) and replaced by a temporary
   :class:`~sympy.core.symbol.Dummy` carrying the equivalent assumptions of
   SymPy's core (``positive``, ``integer``, ``real``, ...). The
   algorithms of SymPy (its automatic evaluation, :func:`sympy.refine`,
   :func:`sympy.simplify`) then apply unchanged, and the dummies are
   substituted back.
2. *Handlers.* What SymPy cannot do is rewritten here: relations between
   several subexpressions (``Max``, ``Min``, ``atan2``, the conditions of
   ``Piecewise``), powers and logarithms of polynomials whose factors
   have a known sign, and Boolean formulas, which are decided atom by atom
   or minimised by quantifier elimination.
"""
from __future__ import annotations

from typing import Callable, Optional, Union

from sympy.assumptions import Q, refine as _sympy_refine
from sympy.assumptions.assume import AppliedPredicate
from sympy.core.facts import InconsistentAssumptions
from sympy.core.basic import Basic
from sympy.core.traversal import preorder_traversal
from sympy.core.expr import Expr
from sympy.core.function import Function, count_ops
from sympy.core.mul import Mul
from sympy.core.numbers import pi
from sympy.core.power import Pow
from sympy.core.relational import Relational, Eq, Ne
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.core.sympify import sympify
from sympy.functions.elementary.complexes import Abs, sign, conjugate, re, im, arg
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.integers import floor, ceiling, frac
from sympy.core.mod import Mod
from sympy.core.numbers import Integer
from itertools import product
from sympy.polys.polytools import Poly
from sympy.functions.elementary.miscellaneous import Max, Min
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import atan, atan2, asin, acos
from sympy.functions.special.gamma_functions import gamma, loggamma
from sympy.functions.combinatorial.factorials import factorial
from sympy.logic.boolalg import Boolean, BooleanTrue, BooleanFalse, Not, true, false
from sympy.polys.polyerrors import PolynomialError
from sympy.polys.polytools import factor_list
from sympy.sets.contains import Contains
from sympy.sets.sets import Set
from sympy.simplify.simplify import simplify as _sympy_simplify

from sympy_extras._typing import Truth, as_boolean, as_expr, free_symbols, sorted_symbols

from .ask import Assumptions, _facts, _evaluate, _evaluate_atom
from .facts import Facts, normalize
from .quantifiers import Quantifier

__all__ = ['refine', 'simplify']

#: assumption keys of SymPy's core derived for an expression
Properties = dict[str, bool]
#: a rewriting of an expression under the facts
Handler = Callable[['_Refiner', Basic], Basic]

#: the functions whose arguments are abstracted
_ARGUMENTS: tuple[type[Function], ...] = (Abs, sign, log, arg, re, im,
    conjugate, floor, ceiling, frac, atan2, gamma, loggamma, factorial,
    asin, acos)

#: what SymPy's core already knows about an expression, by assumption key
_KNOWN: dict[str, Callable[[Expr], Optional[bool]]] = {
    'real': lambda e: e.is_real,
    'positive': lambda e: e.is_positive,
    'negative': lambda e: e.is_negative,
    'nonnegative': lambda e: e.is_nonnegative,
    'nonpositive': lambda e: e.is_nonpositive,
    'zero': lambda e: e.is_zero,
    'nonzero': lambda e: e.is_nonzero,
    'integer': lambda e: e.is_integer,
    'even': lambda e: e.is_even,
    'odd': lambda e: e.is_odd,
    'rational': lambda e: e.is_rational,
    'prime': lambda e: e.is_prime,
}

_MAX_PASSES = 4


def _size(expr: Basic) -> int:
    """The number of operations of an expression, the measure of
    simplicity."""
    return int(count_ops(expr))


class _Refiner:
    """The refinement of expressions under a fixed set of facts, with a
    cache of the statements decided so far."""

    def __init__(self, facts: Facts) -> None:
        self.facts = facts
        self._cache: dict[Boolean, Truth] = {}
        self._dummies: dict[Basic, Symbol] = {}

    # ------------------------------------------------------------------
    # deciding statements

    def holds(self, statement: Union[Boolean, bool]) -> Truth:
        """Three-valued truth of a statement under the facts."""
        formula = normalize(statement)
        if formula not in self._cache:
            self._cache[formula] = _evaluate(formula, self.facts)
        return self._cache[formula]

    def true(self, statement: Union[Boolean, bool]) -> bool:
        return self.holds(statement) is True

    def is_real(self, f: Expr) -> bool:
        if isinstance(f, Symbol):
            return f in self.facts.real
        return self.true(Contains(f, S.Reals))

    def properties(self, f: Expr) -> Properties:
        """The assumption keys of SymPy's core which follow from the
        facts for the expression ``f``."""
        props: Properties = {}
        if self.is_real(f):
            props['real'] = True
            if self.true(f > 0):
                props['positive'] = True
            elif self.true(f < 0):
                props['negative'] = True
            else:
                nonnegative = self.true(f >= 0)
                nonpositive = self.true(f <= 0)
                if nonnegative and nonpositive:
                    props['zero'] = True
                elif nonnegative:
                    props['nonnegative'] = True
                elif nonpositive:
                    props['nonpositive'] = True
                if 'zero' not in props and self.true(Ne(f, 0)):
                    props['nonzero'] = True
            integer = f in self.facts.integer if isinstance(f, Symbol) else False
            if integer or self.true(Contains(f, S.Integers)):
                props['integer'] = True
                parity = self.residue(f, 2)
                if parity is None:
                    parity = 0 if self.true(Q.even(f)) else (1 if self.true(Q.odd(f)) else None)
                if parity == 0:
                    props['even'] = True
                elif parity == 1:
                    props['odd'] = True
                if isinstance(f, Symbol) and self.true(Q.prime(f)):
                    props['prime'] = True
            elif self.true(Contains(f, S.Rationals)):
                props['rational'] = True
        return props

    def residue(self, f: Expr, m: int) -> Optional[int]:
        """The residue of ``f`` modulo ``m`` when ``f`` is a polynomial
        with integer coefficients in integer variables whose value modulo
        ``m`` is the same at every point (a polynomial is even for every
        integer argument exactly when it is even on the residues 0 and 1,
        so the residues are checked on ``{0, ..., m - 1}**k``)."""
        symbols = sorted_symbols(free_symbols(f))
        if not symbols or not all(s in self.facts.integer for s in symbols):
            return None
        try:
            poly = Poly(f, *symbols)
        except PolynomialError:
            return None
        if not all(c.is_Integer for c in poly.coeffs()):
            return None
        if len(symbols)*m > 64:
            return None
        found: Optional[int] = None
        for point in product(range(m), repeat=len(symbols)):
            value = int(poly.eval(dict(zip(symbols, point)))) % m
            if found is None:
                found = value
            elif value != found:
                return None
        return found

    # ------------------------------------------------------------------
    # abstraction: dummies carrying the assumptions

    def _dummy(self, f: Expr, props: Properties) -> Optional[Symbol]:
        """A dummy standing for ``f`` with the properties ``props`` (and,
        for a symbol, its own assumptions), or ``None`` when the
        assumptions are inconsistent."""
        if f in self._dummies:
            return self._dummies[f]
        merged: Properties = {}
        name = 'e'
        if isinstance(f, Symbol):
            name = f.name
            for key, value in f.assumptions0.items():
                if isinstance(value, bool):
                    merged[key] = value
        merged.update(props)
        try:
            dummy = Dummy(name, **merged)
        except InconsistentAssumptions:
            return None
        self._dummies[f] = dummy
        return dummy

    def dummy_for(self, f: Expr) -> Optional[Symbol]:
        """A dummy carrying the properties of ``f`` which follow from the
        facts, or ``None`` when there are none."""
        props = self.properties(f)
        if not props:
            return None
        return self._dummy(f, props)

    def _candidates(self, expr: Basic) -> list[Expr]:
        """The non-atomic subexpressions whose properties matter."""
        found: list[Expr] = []
        seen: set[Basic] = set()

        def add(e: Basic) -> None:
            if isinstance(e, Expr) and not e.is_Atom and e not in seen:
                seen.add(e)
                found.append(e)
                if isinstance(e, Mul):
                    for factor in e.args:
                        add(factor)

        for node in preorder_traversal(expr):
            if isinstance(node, Pow):
                base, exponent = node.args
                if not exponent.is_Integer:
                    add(base)
                if base.is_Number and not base.is_positive:
                    add(exponent)
            elif isinstance(node, _ARGUMENTS):
                for a in node.args:
                    add(a)
        return found

    def abstract(self, expr: Basic) -> tuple[Basic, dict[Basic, Basic], dict[Basic, Basic]]:
        """``expr`` with its symbols and relevant subexpressions replaced by
        dummies carrying the assumptions, and the two substitutions which
        undo it (subexpressions first, then symbols)."""
        symbols: dict[Basic, Basic] = {}
        for s in sorted(free_symbols(expr), key=lambda s: s.name):
            props = self.properties(s)
            if props:
                dummy = self._dummy(s, props)
                if dummy is not None:
                    symbols[s] = dummy
        sub: dict[Basic, Basic] = {}
        for f in self._candidates(expr):
            props = self.properties(f)
            if not props:
                continue
            g = as_expr(f.xreplace(symbols))
            if all(_KNOWN[key](g) is value for key, value in props.items()):
                continue
            dummy = self._dummy(f, props)
            if dummy is not None:
                sub[f] = dummy
        abstracted = expr.xreplace(symbols)
        back_sub: dict[Basic, Basic] = {}
        if sub:
            abstracted = expr.xreplace(sub).xreplace(symbols)
            back_sub = {d: f.xreplace(symbols) for f, d in sub.items()}
        back_symbols: dict[Basic, Basic] = {d: s for s, d in symbols.items()}
        return abstracted, back_sub, back_symbols

    def through_sympy(self, expr: Basic, operation: Callable[[Basic], Basic]) -> Basic:
        """Apply an operation of SymPy to the abstracted expression and
        substitute the dummies back."""
        abstracted, back_sub, back_symbols = self.abstract(expr)
        result = operation(abstracted)
        if back_sub:
            result = result.xreplace(back_sub)
        return result.xreplace(back_symbols)

    def sympy_refine(self, expr: Basic) -> Basic:
        """:func:`sympy.refine` on the abstracted expression, with the
        predicates of the facts. Boolean subexpressions (relations, the
        conditions of ``Piecewise``) are left to the handlers: SymPy's
        ``ask`` is not reliable on them."""
        def operation(e: Basic) -> Basic:
            predicates = self.facts.predicates.xreplace(
                {s: d for s, d in self._dummies.items() if isinstance(s, Symbol)})
            return _refine_around_booleans(e, predicates)
        return self.through_sympy(expr, operation)

    # ------------------------------------------------------------------
    # handlers

    def polynomial_factors(self, f: Expr) -> Optional[tuple[Expr, list[tuple[Expr, int]]]]:
        """The factorization of ``f`` when it is a polynomial in real
        variables, else ``None``."""
        symbols = free_symbols(f)
        if not symbols or not symbols <= self.facts.real:
            return None
        if not f.is_polynomial(*symbols):
            return None
        try:
            c, factors = factor_list(f, *sorted(symbols, key=lambda s: s.name))
        except (PolynomialError, ValueError):
            return None
        return as_expr(c), [(as_expr(g), int(m)) for g, m in factors]

    def _better(self, new: Expr, old: Expr) -> Expr:
        return new if _size(new) <= _size(old) else old

    def refine_abs(self, e: Expr) -> Expr:
        f = as_expr(e.args[0])
        if self.true(f >= 0):
            return f
        if self.true(f <= 0):
            return -f
        factored = self.polynomial_factors(f)
        if factored is None:
            return e
        c, factors = factored
        parts: list[Expr] = [as_expr(abs(c))]
        for g, m in factors:
            if self.true(g >= 0):
                parts.append(g**m)
            elif self.true(g <= 0):
                parts.append((-g)**m)
            elif m % 2 == 0:
                parts.append(g**m)
            else:
                parts.append(Abs(g)**m)
        return self._better(as_expr(Mul(*parts)), e)

    def refine_sign(self, e: Expr) -> Expr:
        f = as_expr(e.args[0])
        if self.true(f > 0):
            return S.One
        if self.true(f < 0):
            return S.NegativeOne
        if self.true(Eq(f, 0)):
            return S.Zero
        return e

    def refine_minmax(self, e: Expr) -> Expr:
        args = [as_expr(a) for a in e.args]
        dominated: set[int] = set()
        for i, a in enumerate(args):
            for j, b in enumerate(args):
                if i == j or j in dominated:
                    continue
                better = (b >= a) if isinstance(e, Max) else (b <= a)
                if self.true(better):
                    dominated.add(i)
                    break
        if not dominated:
            return e
        remaining = [a for i, a in enumerate(args) if i not in dominated]
        return as_expr(e.func(*remaining))

    def refine_floor(self, e: Expr) -> Expr:
        f = as_expr(e.args[0])
        if self.true(Contains(f, S.Integers)):
            return f if not isinstance(e, frac) else S.Zero
        return e

    def refine_real_part(self, e: Expr) -> Expr:
        """``re``, ``im`` and ``conjugate`` of a real expression."""
        f = as_expr(e.args[0])
        if self.is_real(f):
            return S.Zero if isinstance(e, im) else f
        return e

    def refine_arg(self, e: Expr) -> Expr:
        f = as_expr(e.args[0])
        if self.true(f > 0):
            return S.Zero
        if self.true(f < 0):
            return as_expr(pi)
        return e

    def refine_atan2(self, e: Expr) -> Expr:
        y, x = (as_expr(a) for a in e.args)
        if self.true(x > 0):
            return as_expr(atan(y/x))
        if self.true(x < 0):
            if self.true(y >= 0):
                return as_expr(atan(y/x) + pi)
            if self.true(y < 0):
                return as_expr(atan(y/x) - pi)
        if self.true(Eq(x, 0)):
            if self.true(y > 0):
                return as_expr(pi/2)
            if self.true(y < 0):
                return as_expr(-pi/2)
        return e

    def refine_pow(self, e: Expr) -> Expr:
        """Split a power of a polynomial along the factors of known
        sign: ``(f**2*g)**e`` is ``f**(2*e) * g**e`` for ``f >= 0``."""
        base, exponent = (as_expr(a) for a in e.args)
        if exponent.is_Integer:
            return e
        factored = self.polynomial_factors(base)
        if factored is None:
            return e
        c, factors = factored
        known: list[Expr] = []
        unknown: list[Expr] = []
        for g, m in factors:
            if self.true(g >= 0):
                known.append(g**(m*exponent))
            elif self.true(g <= 0):
                known.append((-g)**(m*exponent))
                if m % 2 == 1:
                    c = -c
            elif m % 2 == 0:
                known.append(Abs(g)**(m*exponent))
            else:
                unknown.append(g**m)
        if not known:
            return e
        if c.is_negative:
            unknown.append(c)
        else:
            known.append(c**exponent)
        if unknown:
            known.append(as_expr(Mul(*unknown))**exponent)
        return self._better(as_expr(Mul(*known)), e)

    def refine_log(self, e: Expr) -> Expr:
        """``log(f**k)`` is ``k*log(f)`` for ``f > 0`` and real ``k``;
        the logarithm of a polynomial with positive factors is expanded
        when that makes it simpler."""
        f = as_expr(e.args[0])
        if isinstance(f, Pow):
            base, exponent = (as_expr(a) for a in f.args)
            if self.true(base > 0) and self.is_real(exponent):
                return as_expr(exponent*log(base))
            if self.true(base < 0) and exponent.is_Integer and exponent.is_even:
                return as_expr(exponent*log(-base))
            return e
        factored = self.polynomial_factors(f)
        if factored is None:
            return e
        c, factors = factored
        if not c.is_positive or not all(self.true(g > 0) for g, _ in factors):
            return e
        terms: list[Expr] = [m*log(g) for g, m in factors]
        if c != 1:
            terms.append(as_expr(log(c)))
        return self._better(as_expr(sum(terms)), e)

    def refine_mod(self, e: Expr) -> Expr:
        """``Mod(f, m)`` for an integer polynomial ``f`` whose residue is
        the same at every integer point (``Mod(n**2 + n, 2)`` is ``0``)."""
        f, m = (as_expr(a) for a in e.args)
        if not m.is_Integer or m <= 0:
            return e
        residue = self.residue(f, int(m))
        if residue is None:
            return e
        return as_expr(Integer(residue))

    def refine_piecewise(self, e: Expr) -> Expr:
        """The conditions are decided with the facts; the expression of a
        branch is refined with the facts and its own condition (and the
        negations of the conditions of the previous branches)."""
        pairs: list[tuple[Expr, Boolean]] = []
        excluded: list[Boolean] = []
        for branch in e.args:
            expr, cond = as_expr(branch.args[0]), as_boolean(branch.args[1])
            value = self.holds(cond)
            if value is False:
                excluded.append(cond)
                continue
            extra: list[Boolean] = [Not(c) for c in excluded] + ([cond] if value is None else [])
            try:
                local = _Refiner(Facts(self.facts.conjuncts + extra))
            except (ValueError, TypeError):
                local = self
            expr = as_expr(local.refine_expr(expr))
            excluded.append(cond)
            pairs.append((expr, true if value is True else cond))
            if value is True:
                break
        if not pairs:
            return S.NaN
        return Piecewise(*pairs)

    def refine_relational(self, e: Boolean) -> Boolean:
        value = self.holds(e)
        if value is None:
            return e
        return true if value else false

    # ------------------------------------------------------------------
    # the passes

    def _handlers_pass(self, expr: Basic) -> Basic:
        for cls, handler in _HANDLERS:
            expr = expr.replace(lambda e: isinstance(e, cls),
                                lambda e: handler(self, e))
        return expr

    def refine_expr(self, expr: Basic) -> Basic:
        """Refine a non-Boolean expression to a fixpoint."""
        for _ in range(_MAX_PASSES):
            new = self._handlers_pass(expr)
            new = self.sympy_refine(new)
            if new == expr:
                break
            expr = new
        return expr

    def refine_boolean(self, formula: Boolean) -> Boolean:
        """Replace the atoms of a Boolean formula which are decided by the
        facts."""
        formula = normalize(formula)
        value = self.holds(formula)
        if value is not None:
            return true if value else false
        decided: dict[Basic, Boolean] = {}
        for atom in formula.atoms(Relational, Contains, AppliedPredicate):
            value = _evaluate_atom(atom, self.facts)
            if value is not None:
                decided[atom] = true if value else false
        return formula.xreplace(decided) if decided else formula

    def refine(self, expr: Basic) -> Basic:
        if isinstance(expr, Boolean):
            return self.refine_boolean(expr)
        return self.refine_expr(expr)


def _refine_around_booleans(expr: Basic, predicates: Boolean) -> Basic:
    """:func:`sympy.refine` applied to the maximal subexpressions of
    ``expr`` which contain no Boolean."""
    if isinstance(expr, Boolean):
        return expr
    if expr.has(Boolean):
        return expr.func(*[_refine_around_booleans(a, predicates) for a in expr.args])
    try:
        return sympify(_sympy_refine(expr, predicates))
    except (ValueError, TypeError, NotImplementedError):
        return expr


def _expr_handler(method: Callable[[_Refiner, Expr], Expr]) -> Handler:
    def wrapped(refiner: _Refiner, e: Basic) -> Basic:
        return method(refiner, as_expr(e))
    return wrapped


def _boolean_handler(refiner: _Refiner, e: Basic) -> Basic:
    return refiner.refine_relational(as_boolean(e))


_HANDLERS: list[tuple[Union[type, tuple[type, ...]], Handler]] = [
    (Relational, _boolean_handler),
    (Pow, _expr_handler(_Refiner.refine_pow)),
    (Abs, _expr_handler(_Refiner.refine_abs)),
    (sign, _expr_handler(_Refiner.refine_sign)),
    ((Max, Min), _expr_handler(_Refiner.refine_minmax)),
    ((floor, ceiling, frac), _expr_handler(_Refiner.refine_floor)),
    ((re, im, conjugate), _expr_handler(_Refiner.refine_real_part)),
    (arg, _expr_handler(_Refiner.refine_arg)),
    (atan2, _expr_handler(_Refiner.refine_atan2)),
    (log, _expr_handler(_Refiner.refine_log)),
    (Piecewise, _expr_handler(_Refiner.refine_piecewise)),
    (Mod, _expr_handler(_Refiner.refine_mod)),
]


def _refine_boolean(formula: Boolean, facts: Facts) -> Boolean:
    """Replace the atoms of a Boolean formula which are decided by the
    facts (used by :func:`~sympy_extras.assumptions.resolve`)."""
    return _Refiner(facts).refine_boolean(formula)


def _resolve_quantified(formula: Boolean, assumptions: Assumptions,
                        domain: Optional[Set]) -> Boolean:
    from .resolve import resolve
    return resolve(formula, assumptions=assumptions, domain=domain)


def refine(expr: Union[Expr, Boolean, bool], assumptions: Assumptions = None,
           domain: Optional[Set] = None) -> Basic:
    """Refine an expression using assumptions, the counterpart of
    Mathematica's ``Refine[expr, assum]``.

    Parameters
    ==========

    expr : Expr or Boolean
        The expression to refine.
    assumptions : Boolean or list of Booleans, optional
        Assumptions written as relations, memberships in sets or predicates,
        see :mod:`sympy_extras.assumptions`. The global assumptions are
        used as well.
    domain : Set, optional
        A named SymPy set all the variables are assumed to belong to.

    The symbols and the subexpressions whose sign or nature matters (the
    base of a fractional power, the argument of ``Abs``, ``log``,
    ``floor``, ...) are decided with :func:`~sympy_extras.assumptions.ask`,
    which uses cylindrical algebraic decomposition for polynomial
    inequalities, and temporarily replaced by symbols carrying the
    corresponding assumptions of SymPy's core, so that SymPy's automatic
    evaluation and :func:`sympy.refine` apply. Relations, maxima and
    minima, ``atan2``, powers and logarithms of polynomials with factors of
    known sign, and the conditions of ``Piecewise`` expressions are
    rewritten here.

    Examples
    ========

    >>> from sympy import Abs, sqrt, sign, Max, floor, Piecewise, S, re, im, log, atan2
    >>> from sympy.abc import x, y, n
    >>> from sympy_extras.assumptions import refine, element
    >>> refine(Abs(x - 1) + sqrt(x**2), x > 2)
    2*x - 1
    >>> refine(sqrt(x**2 - 2*x + 1), x > 1)
    x - 1
    >>> refine(log(x**2) + sqrt(x**2*y**2), (x > 0) & (y > 0))
    x*y + 2*log(x)
    >>> refine(sign(x*y), (x > 0) & (y < 0))
    -1
    >>> refine(Max(x, x**2), (x > 0) & (x < 1))
    x
    >>> refine(floor(n + 1), element(n, S.Integers))
    n + 1
    >>> refine(atan2(y, x - 1), x > 1)
    atan(y/(x - 1))
    >>> refine(Piecewise((1, x**2 + y**2 < 1), (2, x > 0), (3, True)), (x > 1) & (y > 0))
    2
    >>> refine(re(x) + im(y), element(x, S.Reals) & element(y, S.Reals))
    x
    >>> refine((x > 0) & (y > 0), x > 1)
    y > 0
    >>> refine(x*y > 0, (x > 0) & (y > 0))
    True
    """
    if expr is True or expr is False:
        return true if expr else false
    expr_ = sympify(expr)
    facts = _facts(assumptions, domain, expr_.free_symbols)
    refiner = _Refiner(facts)
    if isinstance(expr_, Boolean):
        formula = expr_
        if formula.has(Quantifier):
            formula = _resolve_quantified(formula, assumptions, domain)
            if isinstance(formula, (BooleanTrue, BooleanFalse)):
                return formula
        return refiner.refine_boolean(formula)
    return refiner.refine_expr(expr_)


def _simplify_boolean(formula: Boolean, refiner: _Refiner, assumptions: Assumptions,
                      domain: Optional[Set], **kwargs: object) -> Boolean:
    """A Boolean formula is minimised by quantifier elimination when it is
    polynomial in real variables, else simplified by SymPy."""
    if formula.has(Quantifier):
        formula = _resolve_quantified(formula, assumptions, domain)
    formula = refiner.refine_boolean(formula)
    if isinstance(formula, (BooleanTrue, BooleanFalse)):
        return formula
    facts = refiner.facts
    if facts.is_polynomial(formula) and free_symbols(formula) <= facts.real:
        try:
            return _resolve_quantified(formula, assumptions, domain)
        except (NotImplementedError, ValueError):
            pass
    simplified = refiner.through_sympy(formula, lambda e: sympify(_sympy_simplify(e, **kwargs)))
    return refiner.refine_boolean(as_boolean(simplified))


def simplify(expr: Union[Expr, Boolean, bool], assumptions: Assumptions = None,
             domain: Optional[Set] = None, **kwargs: object) -> Basic:
    """Simplify an expression using assumptions, the counterpart of
    Mathematica's ``Simplify[expr, assum]``.

    The expression is refined with :func:`refine`, then simplified with
    :func:`sympy.simplify` (which receives the extra keyword arguments,
    such as ``ratio`` and ``measure``) while its symbols and the relevant
    subexpressions carry the assumptions, and refined again; the result is
    kept when it is not larger than the refined expression. A Boolean
    formula which is polynomial in real variables is minimised by
    quantifier elimination (:func:`~sympy_extras.assumptions.resolve`).

    Examples
    ========

    >>> from sympy import Abs, sqrt, cos, sin, log, exp, Eq
    >>> from sympy.abc import x, y, n
    >>> from sympy_extras.assumptions import simplify, element
    >>> simplify(sqrt(x**2) + Abs(x)*sin(x)**2 + Abs(x)*cos(x)**2, x < 0)
    -2*x
    >>> simplify((x**2 - 1)/(x - 1), x > 1)
    x + 1
    >>> simplify(log(x) + log(y), (x > 0) & (y > 0))
    log(x*y)
    >>> simplify(log(exp(x)), element(x, S.Reals))
    x
    >>> simplify(sin(n*pi) + cos(n*pi), element(n, S.Integers))
    (-1)**n
    >>> simplify(Eq(x**2, 1), x > 0)
    Eq(x, 1)
    >>> simplify((x > 1) | (x**2 > 1), domain=S.Reals)
    (x > 1) | (x < -1)
    """
    if expr is True or expr is False:
        return true if expr else false
    expr_ = sympify(expr)
    facts = _facts(assumptions, domain, expr_.free_symbols)
    refiner = _Refiner(facts)
    if isinstance(expr_, Boolean):
        return _simplify_boolean(expr_, refiner, assumptions, domain, **kwargs)
    refined = refiner.refine_expr(expr_)
    simplified = refiner.through_sympy(refined, lambda e: sympify(_sympy_simplify(e, **kwargs)))
    simplified = refiner.refine_expr(simplified)
    if _size(simplified) <= _size(refined):
        return simplified
    return refined
