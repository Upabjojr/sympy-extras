"""Refining and simplifying expressions under assumptions.

The assumptions are statements as described in
:mod:`sympy_extras.assumptions.facts` (``x > 1``, ``element(n, Integers)``,
...). Two mechanisms are combined:

1. *Abstraction.* Every symbol, and every subexpression whose sign or
   nature matters (the base of a fractional power, the argument of
   ``Abs``, ``log``, ``floor``, ...), is decided against the assumptions
   with :func:`~sympy_extras.assumptions.ask.ask` (predicates plus
   cylindrical algebraic decomposition) and replaced by a temporary
   :class:`~sympy_extras.core.symbol.Dummy` carrying the equivalent
   assumptions of the core (``positive``, ``integer``, ``real``, ...).
   The algorithms of the core (its automatic evaluation, its
   :func:`~sympy_extras.core.refine.refine`,
   :func:`sympy_extras.simplify.simplify.simplify`) then apply unchanged,
   and the dummies are substituted back.
2. *Handlers.* What those cannot do is rewritten here: relations between
   several subexpressions (``Max``, ``Min``, ``atan2``, the conditions of
   ``Piecewise``), powers and logarithms of polynomials whose factors
   have a known sign, and Boolean formulas, which are decided atom by atom
   or minimised by quantifier elimination.

The module works on the objects of the core and imports no SymPy:
:mod:`sympy_extras.assumptions.refine_sympy` is its face on SymPy
objects. Three steps are not ported and go through SymPy at run time (the
face holds them, the functions ending in ``_of_core``): SymPy's
``simplify`` of a Boolean formula (``simplify_logic`` and the rules of the
relations), the factorisation of a polynomial whose coefficients are not
rational (over SymPy's ``EX``), and the proofs of the structure theorem
(:mod:`sympy_extras.simplify.structure`).
"""
from __future__ import annotations

from itertools import product
from typing import Callable, Final, Optional, Union

from sympy_extras.core.assume import AppliedPredicate
from sympy_extras.core.assumptions import is_consistent
from sympy_extras.core.ask import Q
from sympy_extras.core.basic import Basic, preorder_traversal
from sympy_extras.core.expr import Expr, as_expr
from sympy_extras.core.function import count_ops, expand_log
from sympy_extras.core.logic import Boolean, BooleanAtom, BooleanLike, false, not_, to_boolean, true
from sympy_extras.core.mod import Mod
from sympy_extras.core.mul import Mul, mul
from sympy_extras.core.numbers import Integer, nan, pi
from sympy_extras.core.power import Pow
from sympy_extras.core.refine import refine as _core_refine
from sympy_extras.core.relational import Relational, eq, ne
from sympy_extras.core.symbol import Dummy, Symbol
from sympy_extras.functions.combinatorial.factorials import Factorial
from sympy_extras.functions.elementary.complexes import Abs, Arg, Atan2, Conjugate, Im, Re, Sign, abs_
from sympy_extras.functions.elementary.exponential import Log, log
from sympy_extras.functions.elementary.integers import Ceiling, Floor, Frac
from sympy_extras.functions.elementary.miscellaneous import Max, Min, max_, min_
from sympy_extras.functions.elementary.piecewise import Piecewise, piecewise
from sympy_extras.functions.elementary.trigonometric import Acos, Asin, atan
from sympy_extras.functions.special.gamma_functions import Gamma, LogGamma
from sympy_extras.polys.polyerrors import BasePolynomialError
from sympy_extras.sets.fancysets import Integers, Rationals
from sympy_extras.sets.sets import Contains, Reals, Set

from .ask import Assumptions, Truth, _evaluate, _evaluate_atom, _facts, _has_fractional_power, _has_function
from .facts import Facts, element, normalize
from .quantifiers import Quantifier

__all__ = ['refine', 'simplify']

#: assumption keys of the core derived for an expression
Properties = dict[str, bool]
#: a rewriting of an expression under the facts
Handler = Callable[['_Refiner', Basic], Basic]

#: the functions whose arguments are abstracted
_ARGUMENTS: tuple[type[Expr], ...] = (Abs, Sign, Log, Arg, Re, Im, Conjugate, Floor, Ceiling, Frac, Atan2, Gamma,
    LogGamma, Factorial, Asin, Acos)

#: what the core already knows about an expression, by assumption key
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
    return int(as_expr(count_ops(expr)))


def _symbols(e: Basic) -> set[Symbol]:
    return {s for s in e.free_symbols if isinstance(s, Symbol)}


class _Refiner:
    """The refinement of expressions under a fixed set of facts, with a
    cache of the statements decided so far."""

    def __init__(self, facts: Facts) -> None:
        self.facts: Final[Facts] = facts
        self._cache: Final[dict[Boolean, Truth]] = {}
        self._dummies: Final[dict[Basic, Symbol]] = {}

    # ------------------------------------------------------------------
    # deciding statements

    def holds(self, statement: BooleanLike) -> Truth:
        """Three-valued truth of a statement under the facts."""
        formula = normalize(statement)
        if formula not in self._cache:
            self._cache[formula] = _evaluate(formula, self.facts)
        return self._cache[formula]

    def true(self, statement: BooleanLike) -> bool:
        return self.holds(statement) is True

    def is_real(self, f: Expr) -> bool:
        if isinstance(f, Symbol):
            return f in self.facts.real
        return self.true(element(f, Reals))

    def properties(self, f: Expr) -> Properties:
        """The assumption keys of the core which follow from the facts for
        the expression ``f``."""
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
                if 'zero' not in props and self.true(ne(f, 0)):
                    props['nonzero'] = True
            integer = f in self.facts.integer if isinstance(f, Symbol) else False
            if integer or self.true(element(f, Integers)):
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
            elif self.true(element(f, Rationals)):
                props['rational'] = True
        return props

    def residue(self, f: Expr, m: int) -> Optional[int]:
        """The residue of ``f`` modulo ``m`` when ``f`` is a polynomial
        with integer coefficients in integer variables whose value modulo
        ``m`` is the same at every point (a polynomial is even for every
        integer argument exactly when it is even on the residues 0 and 1,
        so the residues are checked on ``{0, ..., m - 1}**k``)."""
        from sympy_extras.polys.polytools import Poly
        symbols = sorted(_symbols(f), key=lambda s: s.name)
        if not symbols or not all(s in self.facts.integer for s in symbols):
            return None
        try:
            poly = Poly(f, *symbols)
        except (BasePolynomialError, ValueError, NotImplementedError):
            return None
        if not all(c.is_Integer for c in poly.coeffs()):
            return None
        if len(symbols)*m > 64:
            return None
        polynomial = poly.as_expr()
        found: Optional[int] = None
        for point in product(range(m), repeat=len(symbols)):
            at_point: dict[Basic, Basic] = {s: Integer(v) for s, v in zip(symbols, point)}
            value = int(as_expr(polynomial._xreplace_evaluated(at_point))) % m
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
            merged.update(f.assumptions0)
        merged.update(props)
        if not is_consistent(merged):
            return None
        dummy = Dummy(name, **merged)
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
                base, exponent = node.base, node.exp
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
        for s in sorted(_symbols(expr), key=lambda s: s.name):
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
            g = as_expr(f._xreplace_evaluated(symbols))
            if all(_KNOWN[key](g) is value for key, value in props.items()):
                continue
            dummy = self._dummy(f, props)
            if dummy is not None:
                sub[f] = dummy
        abstracted = expr._xreplace_evaluated(symbols)
        back_sub: dict[Basic, Basic] = {}
        if sub:
            abstracted = expr._xreplace_evaluated(sub)._xreplace_evaluated(symbols)
            back_sub = {d: f._xreplace_evaluated(symbols) for f, d in sub.items()}
        back_symbols: dict[Basic, Basic] = {d: s for s, d in symbols.items()}
        return abstracted, back_sub, back_symbols

    def through(self, expr: Basic, operation: Callable[[Basic], Basic]) -> Basic:
        """Apply an operation to the abstracted expression and substitute
        the dummies back."""
        abstracted, back_sub, back_symbols = self.abstract(expr)
        result = operation(abstracted)
        if back_sub:
            result = result._xreplace_evaluated(back_sub)
        return result._xreplace_evaluated(back_symbols)

    def core_refine(self, expr: Basic) -> Basic:
        """The core's :func:`~sympy_extras.core.refine.refine` on the
        abstracted expression, with the predicates of the facts. Boolean
        subexpressions (relations, the conditions of ``Piecewise``) are
        left to the handlers: the predicates are not reliable on them."""
        def operation(e: Basic) -> Basic:
            predicates = self.facts.predicates._xreplace_evaluated(
                {s: d for s, d in self._dummies.items() if isinstance(s, Symbol)})
            return _refine_around_booleans(e, to_boolean(predicates))
        return self.through(expr, operation)

    # ------------------------------------------------------------------
    # handlers

    def polynomial_factors(self, f: Expr) -> Optional[tuple[Expr, list[tuple[Expr, int]]]]:
        """The factorization of ``f`` when it is a polynomial in real
        variables, else ``None``."""
        from sympy_extras.polys.polytools import factor_list
        symbols = _symbols(f)
        if not symbols or not symbols <= self.facts.real:
            return None
        gens = sorted(symbols, key=lambda s: s.name)
        if not f.is_polynomial(*gens):
            return None
        try:
            c, factors = factor_list(f, *gens)
            return c, [(g, int(m)) for g, m in factors]
        except (BasePolynomialError, ValueError):
            return None
        except NotImplementedError:
            # coefficients which are not rational: SymPy factors over EX
            from .refine_sympy import factor_list_of_core
            return factor_list_of_core(f, gens)

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
        parts: list[Expr] = [abs(c)]
        for g, m in factors:
            if self.true(g >= 0):
                parts.append(g**m)
            elif self.true(g <= 0):
                parts.append((-g)**m)
            elif m % 2 == 0:
                parts.append(g**m)
            else:
                parts.append(abs_(g)**m)
        return self._better(mul(*parts), e)

    def refine_sign(self, e: Expr) -> Expr:
        f = as_expr(e.args[0])
        if self.true(f > 0):
            return Integer(1)
        if self.true(f < 0):
            return Integer(-1)
        if self.true(eq(f, 0)):
            return Integer(0)
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
        return max_(*remaining) if isinstance(e, Max) else min_(*remaining)

    def refine_floor(self, e: Expr) -> Expr:
        f = as_expr(e.args[0])
        if self.true(element(f, Integers)):
            return f if not isinstance(e, Frac) else Integer(0)
        return e

    def refine_real_part(self, e: Expr) -> Expr:
        """``re``, ``im`` and ``conjugate`` of a real expression."""
        f = as_expr(e.args[0])
        if self.is_real(f):
            return Integer(0) if isinstance(e, Im) else f
        return e

    def refine_arg(self, e: Expr) -> Expr:
        f = as_expr(e.args[0])
        if self.true(f > 0):
            return Integer(0)
        if self.true(f < 0):
            return pi
        return e

    def refine_atan2(self, e: Expr) -> Expr:
        y, x = (as_expr(a) for a in e.args)
        if self.true(x > 0):
            return atan(y/x)
        if self.true(x < 0):
            if self.true(y >= 0):
                return atan(y/x) + pi
            if self.true(y < 0):
                return atan(y/x) - pi
        if self.true(eq(x, 0)):
            if self.true(y > 0):
                return pi/2
            if self.true(y < 0):
                return -pi/2
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
                known.append(abs_(g)**(m*exponent))
            else:
                unknown.append(g**m)
        if not known:
            return e
        if c.is_negative:
            unknown.append(c)
        else:
            known.append(c**exponent)
        if unknown:
            known.append(mul(*unknown)**exponent)
        return self._better(mul(*known), e)

    def refine_log(self, e: Expr) -> Expr:
        """``log(f**k)`` is ``k*log(f)`` for ``f > 0`` and real ``k``;
        the logarithm of a polynomial with positive factors is expanded
        when that makes it simpler."""
        f = as_expr(e.args[0])
        if isinstance(f, Pow):
            base, exponent = f.base, f.exp
            if self.true(base > 0) and self.is_real(exponent):
                return exponent*log(base)
            if self.true(base < 0) and exponent.is_Integer and exponent.is_even:
                return exponent*log(-base)
            return e
        factored = self.polynomial_factors(f)
        if factored is None:
            return e
        c, factors = factored
        if not c.is_positive or not all(self.true(g > 0) for g, _ in factors):
            return e
        total: Expr = Integer(0)
        for g, m in factors:
            total = total + m*log(g)
        if c != Integer(1):
            total = total + log(c)
        return self._better(total, e)

    def refine_mod(self, e: Expr) -> Expr:
        """``Mod(f, m)`` for an integer polynomial ``f`` whose residue is
        the same at every integer point (``Mod(n**2 + n, 2)`` is ``0``)."""
        f, m = (as_expr(a) for a in e.args)
        if not isinstance(m, Integer) or int(m) <= 0:
            return e
        residue = self.residue(f, int(m))
        if residue is None:
            return e
        return Integer(residue)

    def refine_piecewise(self, e: Expr) -> Expr:
        """The conditions are decided with the facts; the expression of a
        branch is refined with the facts and its own condition (and the
        negations of the conditions of the previous branches)."""
        pairs: list[tuple[Expr, Boolean]] = []
        excluded: list[Boolean] = []
        for branch in e.args:
            expr, cond = as_expr(branch.args[0]), to_boolean(branch.args[1])
            value = self.holds(cond)
            if value is False:
                excluded.append(cond)
                continue
            extra: list[Boolean] = [not_(c) for c in excluded] + ([cond] if value is None else [])
            try:
                local = _Refiner(Facts(list(self.facts.conjuncts) + extra))
            except (ValueError, TypeError):
                local = self
            expr = as_expr(local.refine_expr(expr))
            excluded.append(cond)
            pairs.append((expr, true if value is True else cond))
            if value is True:
                break
        if not pairs:
            return nan
        return piecewise(*pairs)

    def refine_relational(self, e: Boolean) -> Boolean:
        value = self.holds(e)
        if value is None:
            return e
        return true if value else false

    # ------------------------------------------------------------------
    # the passes

    def _handlers_pass(self, expr: Basic) -> Basic:
        for cls, handler in _HANDLERS:
            expr = expr.replace(lambda e: isinstance(e, cls), lambda e: handler(self, e))
        return expr

    def refine_expr(self, expr: Basic) -> Basic:
        """Refine a non-Boolean expression to a fixpoint."""
        for _ in range(_MAX_PASSES):
            new = self._handlers_pass(expr)
            new = self.core_refine(new)
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
        decided: dict[Basic, Basic] = {}
        for atom in formula.atoms(Relational, Contains, AppliedPredicate):
            if not isinstance(atom, Boolean):
                continue
            value = _evaluate_atom(atom, self.facts)
            if value is not None:
                decided[atom] = true if value else false
        return to_boolean(formula._xreplace_evaluated(decided)) if decided else formula

    def refine(self, expr: Basic) -> Basic:
        if isinstance(expr, Boolean):
            return self.refine_boolean(expr)
        return self.refine_expr(expr)


def _has_boolean(expr: Basic) -> bool:
    return any(isinstance(node, Boolean) for node in preorder_traversal(expr))


def _refine_around_booleans(expr: Basic, predicates: Boolean) -> Basic:
    """The core's :func:`~sympy_extras.core.refine.refine` applied to the
    maximal subexpressions of ``expr`` which contain no Boolean."""
    if isinstance(expr, Boolean):
        return expr
    if _has_boolean(expr):
        return expr._evaluated(tuple(_refine_around_booleans(a, predicates) for a in expr.args))
    try:
        return _core_refine(expr, predicates)
    except (ValueError, TypeError, NotImplementedError):
        return expr


def _expr_handler(method: Callable[[_Refiner, Expr], Expr]) -> Handler:
    def wrapped(refiner: _Refiner, e: Basic) -> Basic:
        return method(refiner, as_expr(e))
    return wrapped


def _boolean_handler(refiner: _Refiner, e: Basic) -> Basic:
    return refiner.refine_relational(to_boolean(e))


_HANDLERS: list[tuple[Union[type, tuple[type, ...]], Handler]] = [
    (Relational, _boolean_handler),
    (Pow, _expr_handler(_Refiner.refine_pow)),
    (Abs, _expr_handler(_Refiner.refine_abs)),
    (Sign, _expr_handler(_Refiner.refine_sign)),
    ((Max, Min), _expr_handler(_Refiner.refine_minmax)),
    ((Floor, Ceiling, Frac), _expr_handler(_Refiner.refine_floor)),
    ((Re, Im, Conjugate), _expr_handler(_Refiner.refine_real_part)),
    (Arg, _expr_handler(_Refiner.refine_arg)),
    (Atan2, _expr_handler(_Refiner.refine_atan2)),
    (Log, _expr_handler(_Refiner.refine_log)),
    (Piecewise, _expr_handler(_Refiner.refine_piecewise)),
    (Mod, _expr_handler(_Refiner.refine_mod)),
]


def _refine_boolean(formula: Boolean, facts: Facts) -> Boolean:
    """Replace the atoms of a Boolean formula which are decided by the
    facts (used by :func:`~sympy_extras.assumptions.resolve.resolve`)."""
    return _Refiner(facts).refine_boolean(formula)


def _resolve_quantified(formula: Boolean, assumptions: Assumptions, domain: Optional[Set]) -> Boolean:
    from .resolve import resolve
    return resolve(formula, assumptions=assumptions, domain=domain)


def refine(expr: Union[Basic, bool], assumptions: Assumptions = None, domain: Optional[Set] = None) -> Basic:
    """Refine an expression using assumptions, the counterpart of
    Mathematica's ``Refine[expr, assum]``.

    Parameters
    ==========

    expr : Expr or Boolean
        The expression to refine.
    assumptions : Boolean or list of Booleans, optional
        Assumptions written as relations, memberships in sets or predicates,
        see :mod:`sympy_extras.assumptions.facts`. The global assumptions
        are used as well.
    domain : Set, optional
        A named set all the variables are assumed to belong to.

    The symbols and the subexpressions whose sign or nature matters (the
    base of a fractional power, the argument of ``Abs``, ``log``,
    ``floor``, ...) are decided with :func:`~sympy_extras.assumptions.ask.ask`,
    which uses cylindrical algebraic decomposition for polynomial
    inequalities, and temporarily replaced by symbols carrying the
    corresponding assumptions of the core, so that its automatic
    evaluation and :func:`~sympy_extras.core.refine.refine` apply.
    Relations, maxima and minima, ``atan2``, powers and logarithms of
    polynomials with factors of known sign, and the conditions of
    ``Piecewise`` expressions are rewritten here.

    Examples
    ========

    >>> from sympy_extras.core import symbols, sqrt
    >>> from sympy_extras.functions import Abs, sign, Max, floor, log, atan2
    >>> from sympy_extras.sets import Integers
    >>> from sympy_extras.assumptions.refine import refine
    >>> from sympy_extras.assumptions.facts import element
    >>> x, y, n = symbols('x y n')
    >>> refine(Abs(x - 1) + sqrt(x**2), x > 2), refine(sqrt(x**2 - 2*x + 1), x > 1)
    (2*x - 1, x - 1)
    >>> refine(log(x**2) + sqrt(x**2*y**2), (x > 0) & (y > 0)), refine(sign(x*y), (x > 0) & (y < 0))
    (x*y + 2*log(x), -1)
    >>> refine(Max(x, x**2), (x > 0) & (x < 1)), refine(floor(n + 1), element(n, Integers))
    (x, n + 1)
    >>> refine(atan2(y, x - 1), x > 1), refine((x > 0) & (y > 0), x > 1), refine(x*y > 0, (x > 0) & (y > 0))
    (atan(y/(x - 1)), y > 0, True)
    """
    if expr is True or expr is False:
        return true if expr else false
    if isinstance(expr, bool):
        raise TypeError("expected an expression")
    facts = _facts(assumptions, domain, expr.free_symbols)
    refiner = _Refiner(facts)
    if isinstance(expr, Boolean):
        formula = expr
        if formula.has(Quantifier):
            formula = _resolve_quantified(formula, assumptions, domain)
            if isinstance(formula, BooleanAtom):
                return formula
        return refiner.refine_boolean(formula)
    return refiner.refine_expr(expr)


def _simplify_boolean(formula: Boolean, refiner: _Refiner, assumptions: Assumptions, domain: Optional[Set],
                      ratio: float, rational: Optional[bool], inverse: bool, doit: bool) -> Boolean:
    """A Boolean formula is minimised by quantifier elimination when it is
    polynomial in real variables, else simplified by SymPy's ``simplify``
    (at run time, :func:`~sympy_extras.assumptions.refine_sympy.simplify_boolean_of_core`)."""
    from .refine_sympy import simplify_boolean_of_core
    if formula.has(Quantifier):
        formula = _resolve_quantified(formula, assumptions, domain)
    formula = refiner.refine_boolean(formula)
    if isinstance(formula, BooleanAtom):
        return formula
    facts = refiner.facts
    if facts.is_polynomial(formula) and _symbols(formula) <= facts.real:
        try:
            return _resolve_quantified(formula, assumptions, domain)
        except (NotImplementedError, ValueError):
            pass

    def operation(e: Basic) -> Basic:
        return simplify_boolean_of_core(to_boolean(e), ratio, rational, inverse, doit)
    simplified = refiner.through(formula, operation)
    return refiner.refine_boolean(to_boolean(simplified))


def simplify(expr: Union[Basic, bool], assumptions: Assumptions = None, domain: Optional[Set] = None,
             ratio: float = 1.7, rational: Optional[bool] = False, inverse: bool = False,
             doit: bool = True) -> Basic:
    """Simplify an expression using assumptions, the counterpart of
    Mathematica's ``Simplify[expr, assum]``.

    The expression is refined with :func:`refine`, then simplified with
    :func:`sympy_extras.simplify.simplify.simplify` (which receives
    ``ratio``, ``rational``, ``inverse`` and ``doit``) while its symbols
    and the relevant subexpressions carry the assumptions, and refined
    again; the result is kept when it is not larger than the refined
    expression. A Boolean formula which is polynomial in real variables is
    minimised by quantifier elimination
    (:func:`~sympy_extras.assumptions.resolve.resolve`).

    Examples
    ========

    >>> from sympy_extras.core import symbols, sqrt, Eq, pi
    >>> from sympy_extras.functions import Abs, cos, sin, log, exp
    >>> from sympy_extras.sets import Integers, Reals
    >>> from sympy_extras.assumptions.refine import simplify
    >>> from sympy_extras.assumptions.facts import element
    >>> x, y, n = symbols('x y n')
    >>> simplify(sqrt(x**2) + Abs(x)*sin(x)**2 + Abs(x)*cos(x)**2, x < 0), simplify((x**2 - 1)/(x - 1), x > 1)
    (-2*x, x + 1)
    >>> simplify(log(x) + log(y), (x > 0) & (y > 0)), simplify(log(exp(x)), element(x, Reals))
    (log(x*y), x)
    >>> simplify(sin(n*pi) + cos(n*pi), element(n, Integers))
    (-1)**n
    >>> simplify(Eq(x**2, 1), x > 0), simplify((x > 1) | (x**2 > 1), domain=Reals)
    (Eq(x, 1), (x > 1) | (x < -1))
    """
    from sympy_extras.simplify.simplify import simplify as _simplify
    if expr is True or expr is False:
        return true if expr else false
    if isinstance(expr, bool):
        raise TypeError("expected an expression")
    facts = _facts(assumptions, domain, expr.free_symbols)
    refiner = _Refiner(facts)
    if isinstance(expr, Boolean):
        return _simplify_boolean(expr, refiner, assumptions, domain, ratio, rational, inverse, doit)

    def operation(e: Basic) -> Basic:
        if not isinstance(e, Expr):
            return e
        return _simplify(e, ratio=ratio, rational=rational, inverse=inverse, doit=doit)
    refined = refiner.refine_expr(expr)
    simplified = refiner.through(refined, operation)
    simplified = refiner.refine_expr(simplified)
    best = simplified if _size(simplified) <= _size(refined) else refined
    if isinstance(best, Expr):
        return _proved_simplification(best, facts)
    return best


#: the budget of the proofs of :func:`_proved_simplification`
_PROOF_SECONDS = 5.0


def _forced(expr: Expr) -> list[Expr]:
    """What the transformations of :mod:`sympy_extras.simplify` give when
    they are told to disregard the signs: ``powsimp``, ``powdenest``,
    ``logcombine``, ``expand_log`` with ``force=True``, ``simplify`` with
    every symbol positive."""
    from sympy_extras.simplify.powsimp import powdenest, powsimp
    from sympy_extras.simplify.simplify import logcombine, posify, simplify as _simplify
    positive, back = posify(expr)
    return [powsimp(expr, force=True), powdenest(expr, force=True), logcombine(expr, force=True),
            expand_log(expr, force=True), as_expr(_simplify(positive)._xreplace_evaluated(back))]


def _proved_simplification(expr: Expr, facts: Facts) -> Expr:
    """``expr`` or a smaller expression *proved* equal to it on the region
    of the facts by the structure theorem
    (:mod:`sympy_extras.simplify.structure`, at run time through
    :func:`~sympy_extras.assumptions.refine_sympy.proved_simplification_of_core`):
    zero when it vanishes, its canonical form, or what the transformations
    of :func:`_forced` give, which are right on part of the plane only and
    are kept where the proof says so.

    >>> from sympy_extras.core import Rational, sqrt, symbols
    >>> from sympy_extras.functions import atan, asin, log
    >>> from sympy_extras.assumptions.refine import simplify
    >>> x, y = symbols('x y')
    >>> simplify(4*atan(Rational(1, 5)) - atan(Rational(1, 239))), simplify(log(x*y) - log(y), x > 0)
    (pi/4, log(x))
    >>> simplify(asin(x) - atan(x/sqrt(1 - x**2)), (x > -1) & (x < 1))
    0
    """
    from sympy_extras._timeout import attempt
    from .refine_sympy import proved_simplification_of_core
    if not (_has_function(expr) or _has_fractional_power(expr) or expr.has(pi)):
        return expr
    conjuncts = list(facts.conjuncts)
    found = attempt(lambda: proved_simplification_of_core(expr, conjuncts, lambda: _forced(expr)), _PROOF_SECONDS)
    return expr if found is None else found
