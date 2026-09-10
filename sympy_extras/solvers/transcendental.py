"""Transcendental equations and inequalities in one real unknown, reduced
to polynomial problems.

A formula in which the unknown `x` occurs only inside applications of
elementary functions to one argument `u(x)` — such as
`e^{2x} - 3 e^{x} + 2 = 0` or `\\sin^2 x - \\sin x > 0` — is rewritten as a
polynomial formula in a **kernel** variable which stands for the
transcendental part, together with the *side conditions* the kernel
satisfies (`t = e^{u} > 0`; `w = \\tan(u/2)` with
`\\cos u = (1 - w^2)/(1 + w^2)`, `\\sin u = 2w/(1 + w^2)`; `y = u^{1/q} \\ge 0`
with `u \\ge 0`). The functional relations of the elementary functions
(`e^{2x} = (e^{x})^2`, `\\sin 2x = 2 \\sin x \\cos x`, `\\sinh x =
(e^x - e^{-x})/2`, `2^x = e^{x \\log 2}`) bring the occurrences with
proportional arguments to a single kernel. The polynomial formula is
solved exactly by the cylindrical algebraic decomposition, and the
solutions are pulled back through a database of **inverse images** of the
kernels (`e^{u} = v \\iff u = \\log v` for `v > 0`, `\\tan(u/2) = v \\iff
u \\in 2 \\arctan v + 2\\pi\\mathbb{Z}`, ...), recursively when the argument
`u` is not the unknown itself. Absolute values of polynomials are removed
by case distinctions.

The reduction is for the real numbers; anything else is left to SymPy's
:func:`~sympy.solvers.solveset.solveset`.
"""
from __future__ import annotations

import math
from typing import Callable, Optional

from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.function import Lambda
from sympy.core.numbers import Rational, pi
from sympy.core.power import Pow
from sympy.core.relational import Relational, Eq, Ne, Gt, Ge
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.hyperbolic import sinh, cosh, tanh, coth
from sympy.functions.elementary.trigonometric import sin, cos, tan, cot, sec, csc, atan
from sympy.logic.boolalg import (Boolean, BooleanTrue, BooleanFalse, And, Or, Not, Implies, Equivalent,
    Xor, ITE)
from sympy.polys.polyerrors import PolynomialError
from sympy.polys.polytools import Poly, cancel
from sympy.sets.conditionset import ConditionSet
from sympy.sets.contains import Contains
from sympy.sets.fancysets import ImageSet
from sympy.sets.sets import Set, FiniteSet, Interval, Union as SetUnion, Intersection, EmptySet
from sympy.core.function import expand, expand_trig, expand_log

from sympy.core.evalf import N

from sympy_extras._timeout import attempt
from sympy_extras._typing import Truth, as_boolean, as_expr, as_set
from sympy_extras.settings import settings
from sympy_extras.polys.cad import solution_set
from sympy_extras.polys.roots import in_radicals

__all__ = ['solve_transcendental', 'polynomialize', 'Reduction', 'Solver', 'Decider']

#: solves a formula, polynomial in the given variable, for it (``None``
#: when it cannot)
Solver = Callable[[Boolean, Symbol], Optional[Set]]
#: decides a condition on the parameters (``True``, ``False`` or ``None``)
Decider = Callable[[Boolean], Truth]

_COMPOUND = (And, Or, Not, Implies, Equivalent, Xor, ITE)


class Reduction:
    """A formula in the kernel variable equivalent to a transcendental
    formula in ``x``, with the way back.

    Attributes
    ==========

    formula : Boolean
        The polynomial formula in ``variable``.
    variable : Symbol
        The kernel variable.
    argument : Expr
        The argument ``u`` of the kernel, an expression in ``x``.
    family : str
        ``'exp'`` (``variable = exp(u)``), ``'trig'`` (``variable =
        tan(u/2)``), ``'log'`` (``variable = log(u)``) or ``'root'``
        (``variable = u**(1/q)``).
    order : int
        ``q`` for the root family.
    special : Boolean
        For the trig family, whether the formula holds at the points
        ``u = pi (mod 2 pi)`` missed by the half-angle substitution.
    """

    def __init__(self, formula: Boolean, variable: Symbol, argument: Expr, family: str,
                 order: int = 1, special: Boolean = S.false) -> None:
        self.formula = formula
        self.variable = variable
        self.argument = argument
        self.family = family
        self.order = order
        self.special = special

    def __repr__(self) -> str:
        return "Reduction(%s, %s = %s)" % (self.formula, self.variable, self.kernel())

    def kernel(self) -> Expr:
        """The transcendental expression the variable stands for."""
        u = self.argument
        if self.family == 'exp':
            return exp(u)
        if self.family == 'trig':
            return tan(u/2)
        if self.family == 'log':
            return log(u)
        return as_expr(u**Rational(1, self.order))


# ---------------------------------------------------------------------------
# rewriting with the functional relations

def _rewrite(e: Expr, x: Symbol) -> Expr:
    """Hyperbolic functions and general powers as exponentials, the
    remaining trigonometric functions as sines and cosines, and the
    exponentials of sums expanded."""
    e = as_expr(e.replace(lambda a: isinstance(a, (sinh, cosh, tanh, coth)) and a.has(x),
                          lambda a: a.rewrite(exp)))
    e = as_expr(e.replace(lambda a: isinstance(a, Pow) and a.exp.has(x) and not a.base.has(x),
                          lambda a: exp(a.exp*log(a.base))))
    e = as_expr(e.replace(lambda a: isinstance(a, tan) and a.has(x), lambda a: sin(a.args[0])/cos(a.args[0])))
    e = as_expr(e.replace(lambda a: isinstance(a, cot) and a.has(x), lambda a: cos(a.args[0])/sin(a.args[0])))
    e = as_expr(e.replace(lambda a: isinstance(a, sec) and a.has(x), lambda a: 1/cos(a.args[0])))
    e = as_expr(e.replace(lambda a: isinstance(a, csc) and a.has(x), lambda a: 1/sin(a.args[0])))
    return as_expr(expand(e, power_exp=True, power_base=False, mul=False, multinomial=False, log=False))


def _atoms(formula: Boolean) -> list[Relational]:
    return [a for a in formula.atoms(Relational)]


def _map_atoms(formula: Boolean, f: Callable[[Relational], Boolean]) -> Boolean:
    if isinstance(formula, Relational):
        return f(formula)
    if isinstance(formula, _COMPOUND):
        return as_boolean(formula.func(*[_map_atoms(as_boolean(a), f) for a in formula.args]))
    return formula


def _kernels(e: Expr, x: Symbol) -> list[Expr]:
    """The maximal transcendental subexpressions containing ``x``:
    exponentials, logarithms, sines and cosines, and rational powers."""
    found: list[Expr] = []

    def visit(a: Basic) -> None:
        if not a.has(x):
            return
        if isinstance(a, (exp, log, sin, cos)) or \
                (isinstance(a, Pow) and isinstance(a.exp, Rational) and not a.exp.is_integer):
            if a not in found:
                found.append(as_expr(a))
            return
        for b in a.args:
            visit(b)

    visit(e)
    return found


def _lcm(values: list[int]) -> int:
    result = 1
    for v in values:
        result = math.lcm(result, v)
    return result


def _ratio(a: Expr, b: Expr) -> Optional[Rational]:
    r = cancel(a/b)
    return r if isinstance(r, Rational) else None


def _common_argument(arguments: list[Expr]) -> Optional[tuple[Expr, list[Rational]]]:
    """``u`` and the rationals ``r_i`` with ``arguments[i] == r_i*u``, all
    ``r_i`` integers, if the arguments are proportional."""
    first = arguments[0]
    ratios: list[Rational] = []
    for a in arguments:
        r = _ratio(a, first)
        if r is None:
            return None
        ratios.append(r)
    d = _lcm([int(r.q) for r in ratios])
    u = as_expr(first/d)
    return u, [Rational(r.p*d, r.q) for r in ratios]


def polynomialize(formula: Boolean, x: Symbol) -> Optional[Reduction]:
    """Rewrite a formula in ``x`` as a polynomial formula in a kernel
    variable, if ``x`` occurs only inside elementary functions of one
    family with proportional arguments.

    Examples
    ========

    >>> from sympy import exp, sin, cos, Eq
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.transcendental import polynomialize
    >>> polynomialize(Eq(exp(2*x) - 3*exp(x) + 2, 0), x)
    Reduction(Eq(_t**2 - 3*_t + 2, 0), _t = exp(x))
    >>> polynomialize(Eq(sin(x) + cos(x), 1), x)
    Reduction(Eq(-2*_w**2 + 2*_w, 0), _w = tan(x/2))
    """
    formula = as_boolean(formula)
    rewritten = _map_atoms(formula, lambda a: as_boolean(a.func(_rewrite(as_expr(a.lhs), x),
                                                                _rewrite(as_expr(a.rhs), x))))
    kernels: list[Expr] = []
    for atom in _atoms(rewritten):
        for k in _kernels(as_expr(atom.lhs - atom.rhs), x):
            if k not in kernels:
                kernels.append(k)
    if not kernels:
        return None
    families = set('exp' if isinstance(k, exp) else 'log' if isinstance(k, log)
                   else 'trig' if isinstance(k, (sin, cos)) else 'root' for k in kernels)
    if len(families) != 1:
        return None
    family = families.pop()
    if family == 'root':
        return _polynomialize_roots(rewritten, kernels, x)
    if family == 'log':
        arguments = [as_expr(k.args[0]) for k in kernels]
        if any(a != arguments[0] for a in arguments):
            return None
        u = arguments[0]
        variable = Dummy('l', real=True)
        reduced = rewritten.xreplace({k: variable for k in kernels})
        if reduced.has(x):
            return None
        return Reduction(as_boolean(reduced), variable, u, 'log')
    common = _common_argument([as_expr(k.args[0]) for k in kernels])
    if common is None:
        return None
    u, multiples = common
    if family == 'exp':
        variable = Dummy('t', positive=True)
        replacements: dict[Basic, Basic] = {k: variable**int(m) for k, m in zip(kernels, multiples)}
        reduced = rewritten.xreplace(replacements)
        if reduced.has(x):
            return None
        return _clear(reduced, variable, 'exp', u)
    # sines and cosines of integer multiples of u, in terms of those of u,
    # then the half-angle substitution
    w = Dummy('w', real=True)
    c, s = Dummy('c'), Dummy('s')
    # the multiple angle is expanded around a stand-in for u: expanding
    # k.func(m*u) with u = 2x turns sin(2x) into 2 sin(x) cos(x), which the
    # kernel then never matches (sympy-extras#36)
    v = Dummy('v')
    reduced = rewritten
    for k, m in zip(kernels, multiples):
        image = as_expr(expand_trig(k.func(int(m)*v)))
        image = image.xreplace({cos(v): c, sin(v): s})
        reduced = reduced.xreplace({k: image})
    if reduced.has(x):
        return None
    special = as_boolean(reduced.xreplace({c: S.NegativeOne, s: S.Zero}))
    if not isinstance(special, (BooleanTrue, BooleanFalse)):
        return None
    reduced = reduced.xreplace({c: (1 - w**2)/(1 + w**2), s: 2*w/(1 + w**2)})
    return _clear(reduced, w, 'trig', u, special=special)


def _polynomialize_roots(formula: Boolean, kernels: list[Expr], x: Symbol) -> Optional[Reduction]:
    bases: list[Expr] = []
    orders: list[int] = []
    for k in kernels:
        if not isinstance(k, Pow) or not isinstance(k.exp, Rational):
            return None
        bases.append(as_expr(k.base))
        orders.append(int(k.exp.q))
    if any(b != bases[0] for b in bases):
        return None
    u = bases[0]
    q = _lcm(orders)
    y = Dummy('y', nonnegative=True)
    replacements: dict[Basic, Basic] = {}
    for k in kernels:
        if isinstance(k, Pow) and isinstance(k.exp, Rational):
            replacements[k] = y**int(k.exp.p*q//k.exp.q)
    reduced = formula.xreplace(replacements)
    if reduced.has(x):
        return None
    return _clear(reduced, y, 'root', u, order=q)


def _clear(formula: Boolean, variable: Symbol, family: str, u: Expr, order: int = 1,
           special: Boolean = S.false) -> Optional[Reduction]:
    """Clear the denominators of the atoms: divided out when positive
    (powers of ``1 + w**2`` or polynomials with nonnegative coefficients
    in the positive kernel), otherwise by a case distinction on their
    sign. The atoms must be polynomial with rational coefficients."""
    def polynomial(e: Expr) -> Expr:
        try:
            poly = Poly(e, variable)
        except PolynomialError:
            raise ValueError("not polynomial")
        if any(c.has(variable) for c in poly.coeffs()):
            raise ValueError("not polynomial")
        return as_expr(poly.as_expr())

    def clear(atom: Relational) -> Boolean:
        e = cancel(as_expr(atom.lhs - atom.rhs))
        numerator, denominator = e.as_numer_denom()
        f = polynomial(as_expr(numerator))
        if _positive_denominator(as_expr(denominator), variable, family):
            return as_boolean(atom.func(f, 0))
        g = polynomial(as_expr(denominator))
        kind = atom.func
        if kind is Eq:
            return And(Eq(f, 0), Ne(g, 0))
        if kind is Ne:
            return And(Ne(f, 0), Ne(g, 0))
        if kind in (Gt, Ge):
            return Or(And(kind(f, 0), g > 0), And(kind(-f, 0), g < 0))
        return Or(And(kind(f, 0), g > 0), And(kind(-f, 0), g < 0))

    try:
        cleared = _map_atoms(formula, clear)
    except ValueError:
        return None
    return Reduction(cleared, variable, u, family, order, special)


def _positive_denominator(d: Expr, variable: Symbol, family: str) -> bool:
    """Whether the denominator is a positive constant times a power of
    ``1 + w**2`` (trig) or of the positive kernel variable (exp, root)."""
    if not d.has(variable):
        return bool(d.is_positive)
    try:
        poly = Poly(d, variable)
    except PolynomialError:
        return False
    if family == 'trig':
        divisor = Poly(1 + variable**2, variable)
        while poly.degree() > 0:
            poly, remainder = poly.div(divisor)
            if not remainder.is_zero:
                return False
        return bool(as_expr(poly.as_expr()).is_positive)
    if family in ('exp', 'root'):
        # positive on the positive reals
        coefficients = [as_expr(c) for c in poly.coeffs()]
        return all(c.is_nonnegative for c in coefficients) and any(c.is_positive for c in coefficients)
    return False


# ---------------------------------------------------------------------------
# inverse images

class _Context:
    def __init__(self, solver: Optional[Solver], decide: Optional[Decider], depth: int) -> None:
        self.solver = solver
        self.decide = decide
        self.depth = depth

    def truth(self, condition: Boolean) -> Truth:
        condition = as_boolean(condition)
        if isinstance(condition, BooleanTrue):
            return True
        if isinstance(condition, BooleanFalse):
            return False
        if self.decide is not None:
            return self.decide(condition)
        return None

    def deeper(self) -> _Context:
        return _Context(self.solver, self.decide, self.depth + 1)


#: a set of values together with the condition under which they are
#: solutions
Conditioned = tuple[Set, Boolean]


def _integer(name: str) -> Symbol:
    return Dummy(name, integer=True)


def _image(expr: Expr, k: Symbol) -> Set:
    return ImageSet(Lambda(k, expr), S.Integers)


def _simplify(e: Expr) -> Expr:
    """``log(8)/log(2)`` as ``3`` (when expanding the logarithms makes
    the expression smaller)."""
    from sympy.core.function import count_ops
    candidate = as_expr(cancel(expand_log(e, force=True)))
    return candidate if count_ops(candidate) < count_ops(e) else e


def _kernel_preimage(reduction: Reduction, values: Set, ctx: _Context) -> Optional[list[Conditioned]]:
    """The values of the argument ``u`` at which the kernel takes a value
    of the set, with the conditions on the parameters under which they
    are real."""
    family = reduction.family
    if family == 'trig':
        k = _integer('k')
        regular = _trig_preimage(values)
        if regular is None:
            return None
        result: list[Conditioned] = [(regular, S.true)]
        if reduction.special is S.true:
            result.append((_image(pi + 2*pi*k, k), S.true))
        return result
    if isinstance(values, EmptySet):
        return []
    if isinstance(values, (SetUnion, Intersection)):
        parts: list[Conditioned] = []
        for a in values.args:
            part = _kernel_preimage(reduction, as_set(a), ctx)
            if part is None:
                return None
            parts.extend(part)
        if isinstance(values, Intersection):
            return None if any(c is not S.true for _, c in parts) else \
                [(Intersection(*[s for s, _ in parts]), S.true)]
        return parts
    if family == 'log':
        if isinstance(values, FiniteSet):
            return [(FiniteSet(*[exp(v) for v in values.args]), S.true)]
        if isinstance(values, Interval):
            return [(Interval(exp(values.start), exp(values.end), values.left_open, values.right_open), S.true)]
        return None
    # exp is increasing onto the positive reals, the root onto the
    # nonnegative reals
    if family == 'exp':
        inverse: Callable[[Expr], Expr] = lambda v: as_expr(log(v))
        lower, strict = S.Zero, True
    else:
        q = reduction.order
        inverse = lambda v: as_expr(v**q)
        lower, strict = S.Zero, False
    if isinstance(values, FiniteSet):
        conditioned: list[Conditioned] = []
        for v in values.args:
            v_ = as_expr(v)
            condition = as_boolean(v_ > lower if strict else v_ >= lower)
            truth = ctx.truth(condition)
            if truth is False:
                continue
            conditioned.append((FiniteSet(inverse(v_)), S.true if truth is True else condition))
        return conditioned
    if isinstance(values, Interval):
        start, end = as_expr(values.start), as_expr(values.end)
        left_open, right_open = bool(values.left_open), bool(values.right_open)
        below = ctx.truth(start > lower)
        if below is None and start != lower:
            return None
        if below is not True:
            # the interval reaches below the range: cut at the bound
            open_at_bound = start == lower and left_open
            start, left_open = lower, strict or open_at_bound
        if ctx.truth(end > lower) is False or (strict and ctx.truth(end >= lower) is False):
            return []
        if end != S.Infinity and ctx.truth(end > lower) is None:
            return None
        if end == lower and (strict or right_open):
            return []
        if family == 'exp':
            image_start = S.NegativeInfinity if start == 0 else inverse(start)
            image_end = S.Infinity if end == S.Infinity else inverse(end)
            return [(Interval(image_start, image_end, start == 0 or left_open, right_open), S.true)]
        image_end = S.Infinity if end == S.Infinity else inverse(end)
        return [(Interval(inverse(start), image_end, left_open, right_open), S.true)]
    return None


def _trig_preimage(values: Set) -> Optional[Set]:
    """``tan(u/2) = v  <=>  u = 2 atan(v) + 2 pi k``."""
    if isinstance(values, EmptySet):
        return S.EmptySet
    if isinstance(values, (SetUnion, Intersection)):
        members = [_trig_preimage(as_set(a)) for a in values.args]
        if any(p is None for p in members):
            return None
        return as_set(values.func(*[p for p in members if p is not None]))
    if isinstance(values, FiniteSet):
        k = _integer('k')
        images: list[Set] = [_image(2*atan(v) + 2*pi*k, k) for v in values.args]
        return SetUnion(*images) if images else S.EmptySet
    return None


def _argument_preimage(u: Expr, x: Symbol, values: Set, ctx: _Context) -> Optional[Set]:
    """The set of ``x`` with ``u(x)`` in the set of values."""
    if u == x:
        if isinstance(values, FiniteSet):
            return FiniteSet(*[_simplify(as_expr(v)) for v in values.args])
        return values
    if isinstance(values, EmptySet):
        return S.EmptySet
    if isinstance(values, (SetUnion, Intersection)):
        members = [_argument_preimage(u, x, as_set(a), ctx) for a in values.args]
        if any(p is None for p in members):
            return None
        return as_set(values.func(*[p for p in members if p is not None]))
    if isinstance(values, FiniteSet):
        pieces: list[Set] = []
        for v in values.args:
            piece = _solve_real(Eq(u, v), x, ctx)
            if piece is None:
                return None
            if isinstance(piece, FiniteSet):
                piece = FiniteSet(*[_simplify(as_expr(e)) for e in piece.args])
            pieces.append(piece)
        return SetUnion(*pieces) if pieces else S.EmptySet
    if isinstance(values, Interval):
        conditions: list[Boolean] = []
        if values.start != S.NegativeInfinity:
            conditions.append(u > values.start if values.left_open else u >= values.start)
        if values.end != S.Infinity:
            conditions.append(u < values.end if values.right_open else u <= values.end)
        return _solve_real(And(*conditions), x, ctx)
    if isinstance(values, ImageSet):
        lamda = values.lamda
        if not isinstance(lamda, Lambda) or len(lamda.variables) != 1 or values.base_set != S.Integers:
            return None
        k = lamda.variables[0]
        inner = _solve_real(Eq(u, as_expr(lamda.expr)), x, ctx)
        if inner is None:
            return None
        return _parametric_union(inner, k)
    return None


def _real_condition(e: Expr, k: Symbol) -> Boolean:
    """A condition on ``k`` for ``e`` to be real."""
    base, power = e.as_base_exp()
    if e.could_extract_minus_sign():
        base, power = as_expr(-e).as_base_exp()
    if isinstance(power, Rational) and power.q == 2 and not base.has(S.ImaginaryUnit):
        return as_boolean(base >= 0)
    return Contains(e, S.Reals)


def _parametric_union(solutions: Set, k: Symbol) -> Optional[Set]:
    """The union over the integers ``k`` of a set depending on ``k``."""
    if isinstance(solutions, FiniteSet):
        return SetUnion(*[_image(as_expr(e), k) for e in solutions.args])
    if isinstance(solutions, EmptySet):
        return solutions
    if isinstance(solutions, SetUnion):
        members = [_parametric_union(as_set(a), k) for a in solutions.args]
        if any(p is None for p in members):
            return None
        return SetUnion(*[p for p in members if p is not None])
    if isinstance(solutions, Intersection) and len(solutions.args) == 2 and S.Reals in solutions.args:
        other = as_set([a for a in solutions.args if a != S.Reals][0])
        if isinstance(other, FiniteSet):
            # the members which are real: for the integers satisfying a condition
            images: list[Set] = []
            for e in other.args:
                e_ = as_expr(e)
                condition = _real_condition(e_, k)
                base: Set = S.Integers if condition is S.true else ConditionSet(k, condition, S.Integers)
                images.append(ImageSet(Lambda(k, e_), base))
            return SetUnion(*images)
    return None


def _polynomial(formula: Boolean, x: Symbol) -> bool:
    """Polynomial in ``x`` with rational coefficients."""
    for atom in _atoms(formula):
        e = as_expr(atom.lhs - atom.rhs)
        try:
            poly = Poly(e, x)
        except PolynomialError:
            return False
        if not all(c.is_rational for c in poly.coeffs()):
            return False
    return True


def _solve_real(formula: Boolean, x: Symbol, ctx: _Context) -> Optional[Set]:
    """The real solutions of a formula in ``x``: by the cylindrical
    algebraic decomposition when polynomial with rational coefficients,
    by the solver of the context (or SymPy's) when polynomial with other
    constant coefficients, through the kernels otherwise."""
    from sympy.solvers.solveset import solveset
    formula = as_boolean(formula)
    if isinstance(formula, BooleanTrue):
        return S.Reals
    if isinstance(formula, BooleanFalse):
        return S.EmptySet
    if not formula.has(x):
        return None
    if _polynomial(formula, x):
        return in_radicals(solution_set(formula, x))
    if all(as_expr(a.lhs - a.rhs).is_polynomial(x) for a in _atoms(formula)):
        # polynomial in x with transcendental constants or parameters
        if ctx.solver is not None:
            return ctx.solver(formula, x)
        if isinstance(formula, Relational):
            try:
                return as_set(solveset(formula, x, S.Reals))
            except (NotImplementedError, ValueError, TypeError):
                return None
        if isinstance(formula, (And, Or)):
            parts = [_solve_real(as_boolean(a), x, ctx) for a in formula.args]
            if any(p is None for p in parts):
                return None
            kept = [p for p in parts if p is not None]
            return Intersection(*kept) if isinstance(formula, And) else SetUnion(*kept)
        return None
    if ctx.depth > 3:
        return None
    return _solve_transcendental(formula, x, ctx.deeper())


def _lambert(formula: Boolean, x: Symbol, ctx: _Context) -> Optional[Set]:
    """Equations ``x`` appears in both inside and outside an exponential
    or logarithm (``x*exp(x) = 1``, ``x + log(x) = 2``): SymPy's ``solve``
    has the Lambert W patterns; its candidates are checked on the
    equation and kept only when real."""
    from sympy.functions.elementary.exponential import LambertW
    from sympy.solvers.solvers import solve as _solve
    if not isinstance(formula, Eq):
        return None
    e = as_expr(formula.lhs - formula.rhs)
    if e.free_symbols - {x}:
        return None
    if not (e.has(exp) or e.has(log)):
        return None
    found = attempt(lambda: _solve(e, x), settings.timeout)
    if not found:
        return None
    candidates: list[Expr] = []
    for v in found:
        if not isinstance(v, Expr) or v.has(x) or not v.has(LambertW):
            continue
        # SymPy returns the principal branch only; the branch -1 is real
        # too for arguments in [-1/e, 0) and gives the second real root
        for candidate in _lambert_branches(v):
            if _real_root(e, x, candidate) is not False and candidate not in candidates:
                candidates.append(candidate)
    if not candidates:
        return None
    return FiniteSet(*candidates)


def _lambert_branches(v: Expr) -> list[Expr]:
    from sympy.functions.elementary.exponential import LambertW
    result = [v]
    for w in v.atoms(LambertW):
        if len(w.args) != 1:
            continue
        argument = as_expr(w.args[0])
        if argument.is_number and argument.is_real and \
                bool(argument < 0) and bool(argument >= -exp(-1)):
            result.append(as_expr(v.xreplace({w: LambertW(argument, -1)})))
    return result


def _real_root(e: Expr, x: Symbol, v: Expr) -> Truth:
    """Whether the candidate is a real root of ``e``: ``False`` when it is
    not real or the residual is clearly nonzero (with the numerical
    checks), ``None`` when this cannot be told."""
    real = v.is_real
    if real is None and settings.numerical_checks:
        value = N(v, settings.precision)
        real = bool(value.is_real) if isinstance(value, Expr) and value.is_number else None
    if real is False:
        return False
    if not settings.numerical_checks:
        return None
    residual = N(e.subs(x, v), settings.precision)
    if isinstance(residual, Expr) and residual.is_number and residual.is_finite:
        if abs(residual) > Rational(1, 10)**(settings.precision*2//3):
            return False
        return True
    return None


def _without_abs(formula: Boolean, x: Symbol) -> Boolean:
    """Absolute values of expressions in ``x`` removed by case
    distinctions."""
    while True:
        found = [a for a in formula.atoms(Abs) if a.has(x)]
        if not found:
            return formula
        a = found[0]
        inner = as_expr(a.args[0])
        formula = as_boolean(Or(And(inner >= 0, formula.xreplace({a: inner})),
                                And(inner < 0, formula.xreplace({a: -inner}))))


def solve_transcendental(formula: Boolean, x: Symbol, solver: Optional[Solver] = None,
                         decide: Optional[Decider] = None) -> Optional[Set]:
    """The real solutions of a transcendental formula in ``x``, or
    ``None`` when the formula is not reducible to a polynomial one.

    Parameters
    ==========

    formula : Boolean
        A Boolean combination of relations in ``x`` (and parameters).
    x : Symbol
        The real unknown.
    solver : callable, optional
        Solves a formula polynomial in a variable (with parameters or
        transcendental constants) for it; SymPy's ``solveset`` by default.
        The formulas with rational coefficients and no parameters are
        always solved by the cylindrical algebraic decomposition.
    decide : callable, optional
        Decides the conditions on the parameters met when inverting the
        kernels (``v > 0`` for ``exp(u) = v``); undecided conditions are
        kept in :class:`~sympy.sets.conditionset.ConditionSet`.

    Examples
    ========

    >>> from sympy import exp, sin, log, sqrt, cbrt, Eq
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.transcendental import solve_transcendental
    >>> solve_transcendental(Eq(exp(2*x) - 3*exp(x) + 2, 0), x)
    {0, log(2)}
    >>> solve_transcendental(exp(x) + exp(-x) > 3, x)
    Union(Interval.open(-oo, log(3/2 - sqrt(5)/2)), Interval.open(log(sqrt(5)/2 + 3/2), oo))
    >>> solve_transcendental(Eq(sin(x) + cos(x), 1), x)
    Union(ImageSet(Lambda(_k, 2*_k*pi), Integers), ImageSet(Lambda(_k, 2*_k*pi + pi/2), Integers))
    >>> solve_transcendental(Eq(log(x)**2, 3*log(x) - 2), x)
    {E, exp(2)}
    >>> solve_transcendental(Eq(sqrt(x + 1) + cbrt(x + 1), 2), x)
    {0}
    """
    return _solve_transcendental(as_boolean(formula), x, _Context(solver, decide, 0))


def _solve_transcendental(formula: Boolean, x: Symbol, ctx: _Context) -> Optional[Set]:
    formula = _without_abs(formula, x)
    if _polynomial(formula, x):
        return in_radicals(solution_set(formula, x))
    reduction = polynomialize(formula, x)
    if reduction is None:
        return _lambert(formula, x, ctx)
    side: Boolean = S.true
    if reduction.family == 'exp':
        side = reduction.variable > 0
    elif reduction.family == 'root':
        side = reduction.variable >= 0
    values = _solve_real(as_boolean(And(reduction.formula, side)), reduction.variable, ctx)
    if values is None:
        return None
    conditioned = _kernel_preimage(reduction, values, ctx)
    if conditioned is None:
        return None
    results: list[Set] = []
    for argument_values, condition in conditioned:
        result = _argument_preimage(reduction.argument, x, argument_values, ctx)
        if result is None:
            return None
        results.append(result if condition is S.true else ConditionSet(x, condition, result))
    return SetUnion(*results) if results else S.EmptySet
