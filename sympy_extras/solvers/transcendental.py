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

The reduction is for the real numbers. The formulas, the unknowns and the
solution sets are objects of the core (:mod:`sympy_extras.core`,
:mod:`sympy_extras.sets`) and the module imports no SymPy: the
cylindrical algebraic decomposition (:mod:`sympy_extras.polys.cad.qe`),
the radical forms of its roots (:mod:`sympy_extras.polys.roots`) and the
relations linear in the unknown with real coefficients are native, and
so are the Lambert W candidates (the native port of SymPy's ``solve``,
:mod:`sympy_extras.solvers.solvers` and :mod:`sympy_extras.solvers.bivariate`)
and the reliable numerical values of :mod:`sympy_extras._numeric` which
check them, and the other relations polynomial in the unknown with
transcendental or symbolic coefficients, when no solver is given, are
solved by the native port of SymPy's ``solveset``
(:mod:`sympy_extras.solvers.solveset`).
:mod:`sympy_extras.solvers.transcendental_sympy` is its face on SymPy
objects.

Examples
========

>>> from sympy_extras.core import Symbol, Eq
>>> from sympy_extras.functions import exp, sin, cos
>>> from sympy_extras.solvers.transcendental import solve_transcendental
>>> x = Symbol('x')
>>> solve_transcendental(Eq(exp(2*x) - 3*exp(x) + 2, 0), x)
{0, log(2)}
>>> solve_transcendental(exp(x) + exp(-x) > 3, x)
Union(Interval.open(-oo, log(3/2 - sqrt(5)/2)), Interval.open(log(sqrt(5)/2 + 3/2), oo))
>>> solve_transcendental(Eq(sin(x) + cos(x), 1), x)
Union(ImageSet(Lambda(_k, 2*_k*pi), Integers), ImageSet(Lambda(_k, 2*_k*pi + pi/2), Integers))
"""
from __future__ import annotations

import math
from typing import Callable, Final, Optional

from sympy_extras._timeout import attempt
from sympy_extras.core.add import Add, add
from sympy_extras.core.assumptions import Truth
from sympy_extras.core.basic import Basic
from sympy_extras.core.containers import Tuple
from sympy_extras.core.expr import Expr, as_expr
from sympy_extras.core.function import count_ops, expand, expand_log, expand_trig
from sympy_extras.core.logic import (ITE, And, Boolean, BooleanFalse, BooleanTrue, Equivalent, Implies, Not, Or,
    Xor, and_, false, or_, true)
from sympy_extras.core.mul import mul
from sympy_extras.core.numbers import I, Integer, NegativeInfinity, Rational, oo, pi
from sympy_extras.core.power import Pow, pow
from sympy_extras.core.relational import Relational, eq, ge, gt, le, lt, ne
from sympy_extras.core.symbol import Dummy, Symbol
from sympy_extras.functions.elementary.complexes import Abs, abs_
from sympy_extras.functions.elementary.exponential import Exp, LambertW, Log, exp, lambertw, log
from sympy_extras.functions.elementary.hyperbolic import Cosh, Coth, Sinh, Tanh
from sympy_extras.functions.elementary.trigonometric import Cos, Cot, Csc, Sec, Sin, Tan, atan, cos, sin, tan
from sympy_extras.polys.cad.qe import solution_set
from sympy_extras.polys.polytools import cancel
from sympy_extras.polys.roots import in_radicals
from sympy_extras.sets.conditionset import condition_set
from sympy_extras.sets.fancysets import ImageSet, Integers, image_set
from sympy_extras.sets.sets import (EmptySet, EmptySetClass, FiniteSet, Intersection, Interval, Reals, Set,
    Union as SetUnion, contains, finite_set, intersection, interval, union)
from sympy_extras.settings import settings

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
                 order: int = 1, special: Boolean = false) -> None:
        self.formula: Final[Boolean] = formula
        self.variable: Final[Symbol] = variable
        self.argument: Final[Expr] = argument
        self.family: Final[str] = family
        self.order: Final[int] = order
        self.special: Final[Boolean] = special

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
        return pow(u, Rational(1, self.order))


# ---------------------------------------------------------------------------
# helpers on the core

def _boolean(b: Basic) -> Boolean:
    if not isinstance(b, Boolean):
        raise TypeError("expected a condition, got %s" % (b,))
    return b


def _set(b: Basic) -> Set:
    if not isinstance(b, Set):
        raise TypeError("expected a set, got %s" % (b,))
    return b


def _xreplace(b: Boolean, rule: dict[Basic, Basic]) -> Boolean:
    """SymPy's ``xreplace`` of a formula: the rebuilt nodes evaluated."""
    return _boolean(b._xreplace_evaluated(rule))


def _relation(op: str, lhs: Expr, rhs: Expr) -> Boolean:
    """SymPy's ``atom.func(lhs, rhs)``: the relation, evaluated."""
    return {'<': lt, '<=': le, '>': gt, '>=': ge, '==': eq, '!=': ne}[op](lhs, rhs)


def _coefficients(e: Expr, v: Symbol) -> Optional[dict[int, Expr]]:
    """The nonzero coefficients of ``e`` as a polynomial in ``v`` (by
    degree), ``None`` when it is not one (SymPy's ``Poly(e, v)``, whose
    domain ``EX`` is not ported, for the transcendental coefficients)."""
    ex = expand(e)
    gathered: dict[int, list[Expr]] = {}
    for t in (ex.args if isinstance(ex, Add) else (ex,)):
        term = as_expr(t)
        c, k = term.as_coeff_exponent(v)
        if c.has(v) or not isinstance(k, Integer) or k.p < 0:
            return None
        gathered.setdefault(int(k.p), []).append(c)
    result: dict[int, Expr] = {}
    for degree, cs in gathered.items():
        coefficient = add(*cs)
        if coefficient != 0:
            result[degree] = coefficient
    return result


def _polynomial_expr(coefficients: dict[int, Expr], v: Symbol) -> Expr:
    """SymPy's ``Poly.as_expr``."""
    return add(*[mul(c, pow(v, k)) for k, c in coefficients.items()])


def _operations(e: Expr) -> int:
    count = count_ops(e)
    if not isinstance(count, Integer):
        raise TypeError("expected a count, got %s" % (count,))
    return int(count.p)


# ---------------------------------------------------------------------------
# rewriting with the functional relations

def _rewrite(e: Expr, x: Symbol) -> Expr:
    """Hyperbolic functions and general powers as exponentials, the
    remaining trigonometric functions as sines and cosines, and the
    exponentials of sums expanded."""
    def hyperbolic(a: Basic) -> bool:
        return isinstance(a, (Sinh, Cosh, Tanh, Coth)) and a.has(x)

    def general_power(a: Basic) -> bool:
        # a negative base is not exp(u*log(b)) with a real logarithm:
        # (-1)**x is 1 at every even integer, not only at 0
        return isinstance(a, Pow) and a.exp.has(x) and not a.base.has(x) and a.base.is_positive is True

    def as_exponential(a: Basic) -> Basic:
        if not isinstance(a, Pow):
            raise TypeError("expected a power, got %s" % (a,))
        return exp(a.exp*log(a.base))

    def argument(a: Basic) -> Expr:
        return as_expr(a.args[0])

    e = as_expr(e.replace(hyperbolic, lambda a: a.rewrite(Exp)))
    e = as_expr(e.replace(general_power, as_exponential))
    e = as_expr(e.replace(lambda a: isinstance(a, Tan) and a.has(x),
                          lambda a: sin(argument(a))/cos(argument(a))))
    e = as_expr(e.replace(lambda a: isinstance(a, Cot) and a.has(x),
                          lambda a: cos(argument(a))/sin(argument(a))))
    e = as_expr(e.replace(lambda a: isinstance(a, Sec) and a.has(x), lambda a: 1/cos(argument(a))))
    e = as_expr(e.replace(lambda a: isinstance(a, Csc) and a.has(x), lambda a: 1/sin(argument(a))))
    return expand(e, power_exp=True, power_base=False, mul=False, multinomial=False, log=False)


def _atoms(formula: Boolean) -> list[Relational]:
    return [a for a in formula.atoms(Relational) if isinstance(a, Relational)]


def _map_atoms(formula: Boolean, f: Callable[[Relational], Boolean]) -> Boolean:
    if isinstance(formula, Relational):
        return f(formula)
    if isinstance(formula, _COMPOUND):
        return _boolean(formula._evaluated(tuple(_map_atoms(_boolean(a), f) for a in formula.args)))
    return formula


def _kernels(e: Expr, x: Symbol) -> list[Expr]:
    """The maximal transcendental subexpressions containing ``x``:
    exponentials, logarithms, sines and cosines, and rational powers."""
    found: list[Expr] = []

    def visit(a: Basic) -> None:
        if not a.has(x):
            return
        if isinstance(a, (Exp, Log, Sin, Cos)) or \
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
    u = first/d
    return u, [Rational(r.p*d, r.q) for r in ratios]


def polynomialize(formula: Boolean, x: Symbol) -> Optional[Reduction]:
    """Rewrite a formula in ``x`` as a polynomial formula in a kernel
    variable, if ``x`` occurs only inside elementary functions of one
    family with proportional arguments.

    Examples
    ========

    >>> from sympy_extras.core import Symbol, Eq
    >>> from sympy_extras.functions import exp, sin, cos
    >>> from sympy_extras.solvers.transcendental import polynomialize
    >>> x = Symbol('x')
    >>> polynomialize(Eq(exp(2*x) - 3*exp(x) + 2, 0), x)
    Reduction(Eq(_t**2 - 3*_t + 2, 0), _t = exp(x))
    >>> polynomialize(Eq(sin(x) + cos(x), 1), x)
    Reduction(Eq(-2*_w**2 + 2*_w, 0), _w = tan(x/2))
    """
    rewritten = _map_atoms(formula, lambda a: _relation(a.rel_op, _rewrite(a.lhs, x), _rewrite(a.rhs, x)))
    kernels: list[Expr] = []
    for atom in _atoms(rewritten):
        for k in _kernels(atom.lhs - atom.rhs, x):
            if k not in kernels:
                kernels.append(k)
    if not kernels:
        return None
    families = set('exp' if isinstance(k, Exp) else 'log' if isinstance(k, Log)
                   else 'trig' if isinstance(k, (Sin, Cos)) else 'root' for k in kernels)
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
        reduced = _xreplace(rewritten, {k: variable for k in kernels})
        if reduced.has(x):
            return None
        return Reduction(reduced, variable, u, 'log')
    common = _common_argument([as_expr(k.args[0]) for k in kernels])
    if common is None:
        return None
    u, multiples = common
    if family == 'exp':
        t = Dummy('t', positive=True)
        replacements: dict[Basic, Basic] = {k: pow(t, int(m.p)) for k, m in zip(kernels, multiples)}
        reduced = _xreplace(rewritten, replacements)
        if reduced.has(x):
            return None
        return _clear(reduced, t, 'exp', u)
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
        multiple = int(m.p)*v
        image = expand_trig(sin(multiple) if isinstance(k, Sin) else cos(multiple))
        image = as_expr(image._xreplace_evaluated({cos(v): c, sin(v): s}))
        reduced = _xreplace(reduced, {k: image})
    if reduced.has(x):
        return None
    special = _xreplace(reduced, {c: Integer(-1), s: Integer(0)})
    if not isinstance(special, (BooleanTrue, BooleanFalse)):
        return None
    reduced = _xreplace(reduced, {c: (1 - w**2)/(1 + w**2), s: 2*w/(1 + w**2)})
    return _clear(reduced, w, 'trig', u, special=special)


def _polynomialize_roots(formula: Boolean, kernels: list[Expr], x: Symbol) -> Optional[Reduction]:
    bases: list[Expr] = []
    orders: list[int] = []
    for k in kernels:
        if not isinstance(k, Pow) or not isinstance(k.exp, Rational):
            return None
        bases.append(k.base)
        orders.append(int(k.exp.q))
    if any(b != bases[0] for b in bases):
        return None
    u = bases[0]
    q = _lcm(orders)
    y = Dummy('y', nonnegative=True)
    replacements: dict[Basic, Basic] = {}
    for k in kernels:
        if isinstance(k, Pow) and isinstance(k.exp, Rational):
            replacements[k] = pow(y, int(k.exp.p)*q//int(k.exp.q))
    reduced = _xreplace(formula, replacements)
    if reduced.has(x):
        # x outside the roots of a linear u = a*x + b (x - sqrt(x)) is
        # (y**q - b)/a
        linear = _coefficients(u, x)
        if linear is None or set(linear) != {0, 1} and set(linear) != {1} or \
                not all(isinstance(c, Rational) for c in linear.values()):
            return None
        a, b = linear[1], linear.get(0, Integer(0))
        reduced = _xreplace(reduced, {x: (pow(y, q) - b)/a})
        if reduced.has(x):
            return None
    return _clear(reduced, y, 'root', u, order=q)


class _NotPolynomial(Exception):
    pass


def _clear(formula: Boolean, variable: Symbol, family: str, u: Expr, order: int = 1,
           special: Boolean = false) -> Optional[Reduction]:
    """Clear the denominators of the atoms: divided out when positive
    (powers of ``1 + w**2`` or polynomials with nonnegative coefficients
    in the positive kernel), otherwise by a case distinction on their
    sign. The atoms must be polynomial with rational coefficients."""
    def polynomial(e: Expr) -> Expr:
        coefficients = _coefficients(e, variable)
        if coefficients is None:
            raise _NotPolynomial()
        return _polynomial_expr(coefficients, variable)

    def clear(atom: Relational) -> Boolean:
        e = cancel(atom.lhs - atom.rhs)
        numerator, denominator = e.as_numer_denom()
        f = polynomial(numerator)
        op = atom.rel_op
        if _positive_denominator(denominator, variable, family):
            return _relation(op, f, Integer(0))
        g = polynomial(denominator)
        if op == '==':
            return and_(eq(f, 0), ne(g, 0))
        if op == '!=':
            return and_(ne(f, 0), ne(g, 0))
        return or_(and_(_relation(op, f, Integer(0)), g > 0), and_(_relation(op, -f, Integer(0)), g < 0))

    try:
        cleared = _map_atoms(formula, clear)
    except _NotPolynomial:
        return None
    return Reduction(cleared, variable, u, family, order, special)


def _positive_denominator(d: Expr, variable: Symbol, family: str) -> bool:
    """Whether the denominator is a positive constant times a power of
    ``1 + w**2`` (trig) or of the positive kernel variable (exp, root)."""
    if not d.has(variable):
        return d.is_positive is True
    coefficients = _coefficients(d, variable)
    if coefficients is None:
        return False
    if family == 'trig':
        # the remainders of the divisions by 1 + w**2 must vanish
        degree = max(coefficients)
        dense = [coefficients.get(k, Integer(0)) for k in range(degree + 1)]
        while degree > 0:
            if degree == 1:
                return False
            quotient: list[Expr] = [Integer(0)]*(degree - 1)
            for k in range(degree, 1, -1):
                quotient[k - 2] = dense[k]
                dense[k - 2] = expand(dense[k - 2] - dense[k])
            if dense[0] != 0 or dense[1] != 0:
                return False
            dense = quotient
            degree = len(dense) - 1
        return dense[0].is_positive is True
    if family in ('exp', 'root'):
        # positive on the positive reals
        values = list(coefficients.values())
        return all(c.is_nonnegative is True for c in values) and any(c.is_positive is True for c in values)
    return False


# ---------------------------------------------------------------------------
# inverse images

class _Context:
    def __init__(self, solver: Optional[Solver], decide: Optional[Decider], depth: int, lambert: bool = True) -> None:
        self.solver: Final[Optional[Solver]] = solver
        self.decide: Final[Optional[Decider]] = decide
        self.depth: Final[int] = depth
        self.lambert: Final[bool] = lambert

    def truth(self, condition: Boolean) -> Truth:
        if isinstance(condition, BooleanTrue):
            return True
        if isinstance(condition, BooleanFalse):
            return False
        if self.decide is not None:
            return self.decide(condition)
        return None

    def deeper(self) -> _Context:
        return _Context(self.solver, self.decide, self.depth + 1, self.lambert)


#: a set of values together with the condition under which they are
#: solutions
Conditioned = tuple[Set, Boolean]


def _integer(name: str) -> Symbol:
    return Dummy(name, integer=True)


def _image(expr: Expr, k: Symbol) -> Set:
    return image_set(Tuple(k), expr, Integers)


def _simplify(e: Expr) -> Expr:
    """``log(8)/log(2)`` as ``3`` (when expanding the logarithms makes
    the expression smaller)."""
    candidate = cancel(expand_log(e, force=True))
    return candidate if _operations(candidate) < _operations(e) else e


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
        result: list[Conditioned] = [(regular, true)]
        if reduction.special is true:
            result.append((_image(pi + 2*pi*k, k), true))
        return result
    if isinstance(values, EmptySetClass):
        return []
    if isinstance(values, (SetUnion, Intersection)):
        parts: list[Conditioned] = []
        for a in values.args:
            part = _kernel_preimage(reduction, _set(a), ctx)
            if part is None:
                return None
            parts.extend(part)
        if isinstance(values, Intersection):
            return None if any(c is not true for _, c in parts) else \
                [(intersection(*[s for s, _ in parts]), true)]
        return parts
    if family == 'log':
        if isinstance(values, FiniteSet):
            return [(finite_set(*[exp(as_expr(v)) for v in values.args]), true)]
        if isinstance(values, Interval):
            return [(interval(exp(values.start), exp(values.end), values.left_open, values.right_open), true)]
        return None
    # exp is increasing onto the positive reals, the root onto the
    # nonnegative reals
    lower: Expr = Integer(0)
    if family == 'exp':
        inverse: Callable[[Expr], Expr] = log
        strict = True
    else:
        q = reduction.order
        inverse = lambda v: pow(v, q)
        strict = False
    if isinstance(values, FiniteSet):
        conditioned: list[Conditioned] = []
        for v in values.args:
            v_ = as_expr(v)
            condition = v_ > lower if strict else v_ >= lower
            truth = ctx.truth(condition)
            if truth is False:
                continue
            conditioned.append((finite_set(inverse(v_)), true if truth is True else condition))
        return conditioned
    if isinstance(values, Interval):
        start, end = values.start, values.end
        left_open, right_open = values.left_open, values.right_open
        below = ctx.truth(start > lower)
        if below is None and start != lower:
            return None
        if below is not True:
            # the interval reaches below the range: cut at the bound
            open_at_bound = start == lower and left_open
            start, left_open = lower, strict or open_at_bound
        if ctx.truth(end > lower) is False or (strict and ctx.truth(end >= lower) is False):
            return []
        if end != oo and ctx.truth(end > lower) is None:
            return None
        if end == lower and (strict or right_open):
            return []
        image_end = oo if end == oo else inverse(end)
        if family == 'exp':
            image_start: Expr = NegativeInfinity() if start == 0 else inverse(start)
            return [(interval(image_start, image_end, start == 0 or left_open, right_open), true)]
        return [(interval(inverse(start), image_end, left_open, right_open), true)]
    return None


def _trig_preimage(values: Set) -> Optional[Set]:
    """``tan(u/2) = v  <=>  u = 2 atan(v) + 2 pi k``."""
    if isinstance(values, EmptySetClass):
        return EmptySet
    if isinstance(values, (SetUnion, Intersection)):
        members: list[Set] = []
        for a in values.args:
            member = _trig_preimage(_set(a))
            if member is None:
                return None
            members.append(member)
        return union(*members) if isinstance(values, SetUnion) else intersection(*members)
    if isinstance(values, FiniteSet):
        k = _integer('k')
        images: list[Set] = [_image(2*atan(as_expr(v)) + 2*pi*k, k) for v in values.args]
        return union(*images) if images else EmptySet
    return None


def _argument_preimage(u: Expr, x: Symbol, values: Set, ctx: _Context) -> Optional[Set]:
    """The set of ``x`` with ``u(x)`` in the set of values."""
    if u == x:
        if isinstance(values, FiniteSet):
            return finite_set(*[_simplify(as_expr(v)) for v in values.args])
        return values
    if isinstance(values, EmptySetClass):
        return EmptySet
    if isinstance(values, (SetUnion, Intersection)):
        members: list[Set] = []
        for a in values.args:
            member = _argument_preimage(u, x, _set(a), ctx)
            if member is None:
                return None
            members.append(member)
        return union(*members) if isinstance(values, SetUnion) else intersection(*members)
    if isinstance(values, FiniteSet):
        pieces: list[Set] = []
        for v in values.args:
            piece = _solve_real(eq(u, as_expr(v)), x, ctx)
            if piece is None:
                return None
            if isinstance(piece, FiniteSet):
                piece = finite_set(*[_simplify(as_expr(e)) for e in piece.args])
            pieces.append(piece)
        return union(*pieces) if pieces else EmptySet
    if isinstance(values, Interval):
        conditions: list[Boolean] = []
        if values.start != NegativeInfinity():
            conditions.append(u > values.start if values.left_open else u >= values.start)
        if values.end != oo:
            conditions.append(u < values.end if values.right_open else u <= values.end)
        return _solve_real(and_(*conditions), x, ctx)
    if isinstance(values, ImageSet):
        variables, image = values.variables, values.image
        if len(variables) != 1 or values.base_sets != (Integers,) or not isinstance(image, Expr):
            return None
        k = variables[0]
        inner = _solve_real(eq(u, image), x, ctx)
        if inner is None:
            return None
        return _parametric_union(inner, k)
    return None


def _real_condition(e: Expr, k: Symbol) -> Boolean:
    """A condition on ``k`` for ``e`` to be real."""
    base, power = e.as_base_exp()
    if e.could_extract_minus_sign():
        base, power = (-e).as_base_exp()
    if isinstance(power, Rational) and power.q == 2 and not base.has(I):
        return base >= 0
    return contains(e, Reals)


def _parametric_union(solutions: Set, k: Symbol) -> Optional[Set]:
    """The union over the integers ``k`` of a set depending on ``k``."""
    if isinstance(solutions, FiniteSet):
        return union(*[_image(as_expr(e), k) for e in solutions.args])
    if isinstance(solutions, EmptySetClass):
        return solutions
    if isinstance(solutions, SetUnion):
        members: list[Set] = []
        for a in solutions.args:
            member = _parametric_union(_set(a), k)
            if member is None:
                return None
            members.append(member)
        return union(*members)
    if isinstance(solutions, Intersection) and len(solutions.args) == 2 and Reals in solutions.args:
        other = _set([a for a in solutions.args if a != Reals][0])
        if isinstance(other, FiniteSet):
            # the members which are real: for the integers satisfying a condition
            images: list[Set] = []
            for e in other.args:
                e_ = as_expr(e)
                condition = _real_condition(e_, k)
                base: Set = Integers if condition is true else condition_set(k, condition, Integers)
                images.append(image_set(Tuple(k), e_, base))
            return union(*images)
    return None


def _polynomial(formula: Boolean, x: Symbol) -> bool:
    """Polynomial in ``x`` with rational coefficients."""
    for atom in _atoms(formula):
        coefficients = _coefficients(atom.lhs - atom.rhs, x)
        if coefficients is None or not all(c.is_rational is True for c in coefficients.values()):
            return False
    return True


def _linear_solution(relation: Relational, x: Symbol) -> Optional[Set]:
    """The real solutions of a relation linear in ``x``, the leading
    coefficient a real number of known sign, the other one real (SymPy's
    ``solveset`` of one, natively), ``None`` for the others."""
    coefficients = _coefficients(relation.lhs - relation.rhs, x)
    if coefficients is None or set(coefficients) - {0, 1} or 1 not in coefficients:
        return None
    # the constant term unexpanded: expanding writes log(4) as 2*log(2)
    a, b = coefficients[1], as_expr((relation.lhs - relation.rhs).subs(x, Integer(0)))
    if not (a.is_number and a.is_real is True and b.is_real is True):
        return None
    if a.is_positive is True:
        op = relation.rel_op
    elif a.is_negative is True:
        op = {'<': '>', '<=': '>=', '>': '<', '>=': '<=', '==': '==', '!=': '!='}[relation.rel_op]
    else:
        return None
    root = -b/a
    if op == '==':
        return finite_set(root)
    if op == '!=':
        return union(interval(NegativeInfinity(), root, True, True), interval(root, oo, True, True))
    if op in ('>', '>='):
        return interval(root, oo, op == '>', True)
    return interval(NegativeInfinity(), root, True, op == '<')


def _solve_real(formula: Boolean, x: Symbol, ctx: _Context) -> Optional[Set]:
    """The real solutions of a formula in ``x``: by the cylindrical
    algebraic decomposition when polynomial with rational coefficients,
    by the solver of the context (or the native port of SymPy's
    ``solveset``) when polynomial with other constant coefficients, through
    the kernels otherwise."""
    if isinstance(formula, BooleanTrue):
        return Reals
    if isinstance(formula, BooleanFalse):
        return EmptySet
    if not formula.has(x):
        return None
    if _polynomial(formula, x):
        return in_radicals(solution_set(formula, x))
    if all((a.lhs - a.rhs).is_polynomial(x) for a in _atoms(formula)):
        # polynomial in x with transcendental constants or parameters
        if ctx.solver is not None:
            return ctx.solver(formula, x)
        if isinstance(formula, Relational):
            linear = _linear_solution(formula, x)
            if linear is not None:
                return linear
            from .solveset import solveset
            return solveset(formula, x, Reals)
        if isinstance(formula, (And, Or)):
            parts: list[Set] = []
            for a in formula.args:
                part = _solve_real(_boolean(a), x, ctx)
                if part is None:
                    return None
                parts.append(part)
            return intersection(*parts) if isinstance(formula, And) else union(*parts)
        return None
    if ctx.depth > 3:
        return None
    return _solve_transcendental(formula, x, ctx.deeper())


def _lambert(formula: Boolean, x: Symbol, ctx: _Context) -> Optional[Set]:
    """Equations ``x`` appears in both inside and outside an exponential
    or logarithm (``x*exp(x) = 1``, ``x + log(x) = 2``): the Lambert W
    patterns of the native port of SymPy's ``solve``; its candidates are
    checked on the equation and kept only when real."""
    if not ctx.lambert or not isinstance(formula, Relational) or formula.rel_op != '==':
        return None
    e = formula.lhs - formula.rhs
    if e.free_symbols - {x}:
        return None
    if not (e.has(Exp) or e.has(Log)):
        return None
    from .solvers import solve_equation
    found = attempt(lambda: solve_equation(e, x), settings.timeout, native=True)
    if not found:
        return None
    candidates: list[Expr] = []
    for v in found:
        if v.has(x) or not v.has(LambertW):
            continue
        # SymPy returns the principal branch only; the branch -1 is real
        # too for arguments in [-1/e, 0) and gives the second real root
        for candidate in _lambert_branches(v):
            if _real_root(e, x, candidate) is not False and candidate not in candidates:
                candidates.append(candidate)
    if not candidates:
        return None
    return finite_set(*candidates)


def _lambert_branches(v: Expr) -> list[Expr]:
    result = [v]
    for w in v.atoms(LambertW):
        if len(w.args) != 1:
            continue
        argument = as_expr(w.args[0])
        if argument.is_number and argument.is_real and \
                (argument < 0) is true and (argument >= -exp(Integer(-1))) is true:
            result.append(as_expr(v._xreplace_evaluated({w: lambertw(argument, -1)})))
    return result


def _real_root(e: Expr, x: Symbol, v: Expr) -> Truth:
    """Whether the candidate is a real root of ``e``: ``False`` when it is
    not real or the residual is clearly nonzero (with the numerical
    checks), ``None`` when this cannot be told."""
    # the sums which are exactly zero are written 0 first: SymPy's is_real
    # of log(-2*sqrt(2) - 2*sqrt(z)) - I*pi, with z such a sum, is True in
    # some runs and False in others (sympy-extras#25), and a value whose
    # branch a rounding error chooses tells nothing (the bug: this root of
    # x - log(2*sqrt(2)) was refused in some runs)
    from sympy_extras._numeric import reliable_form, reliable_value
    if settings.numerical_checks:
        form = reliable_form(v, settings.precision)
        if form is None:
            return None
        v = form
    real = v.is_real
    if real is None and settings.numerical_checks:
        value = reliable_value(v, settings.precision)
        real = value.is_real is True if value is not None else None
    if real is False:
        return False
    if not settings.numerical_checks:
        return None
    residual = reliable_value(as_expr(e.subs(x, v)), settings.precision)
    if residual is not None:
        if (abs_(residual) > Rational(1, 10)**(settings.precision*2//3)) is true:
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
        formula = or_(and_(inner >= 0, _xreplace(formula, {a: inner})),
                      and_(inner < 0, _xreplace(formula, {a: -inner})))


def solve_transcendental(formula: Boolean, x: Symbol, solver: Optional[Solver] = None,
                         decide: Optional[Decider] = None, lambert: bool = True) -> Optional[Set]:
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
        transcendental constants) for it; the native port of SymPy's
        ``solveset`` (:func:`~sympy_extras.solvers.solveset.solveset`) by
        default.
        The formulas with rational coefficients and no parameters are
        always solved by the cylindrical algebraic decomposition.
    decide : callable, optional
        Decides the conditions on the parameters met when inverting the
        kernels (``v > 0`` for ``exp(u) = v``); undecided conditions are
        kept in :class:`~sympy_extras.sets.conditionset.ConditionSet`.
    lambert : bool, optional
        Whether the equations with the unknown inside and outside an
        exponential or a logarithm are tried with the Lambert W candidates
        of the native port of SymPy's ``solve``; ``False`` gives ``None``
        for them.

    Examples
    ========

    >>> from sympy_extras.core import Symbol, Eq, Rational, sqrt, cbrt
    >>> from sympy_extras.functions import exp, log
    >>> from sympy_extras.solvers.transcendental import solve_transcendental
    >>> x = Symbol('x')
    >>> solve_transcendental(Eq(log(x)**2, 3*log(x) - 2), x)
    {E, exp(2)}
    >>> solve_transcendental(Eq(sqrt(x + 1) + cbrt(x + 1), 2), x)
    {0}
    >>> solve_transcendental(exp(x) < 2, x)
    Interval.open(-oo, log(2))
    """
    return _solve_transcendental(formula, x, _Context(solver, decide, 0, lambert))


def _cleared(formula: Boolean, x: Symbol) -> Optional[Boolean]:
    """A formula rational in ``x`` with rational coefficients as a
    polynomial one: each relation ``p/q op 0`` as ``p*q op 0`` with
    ``q != 0``; ``None`` when an atom is not rational so."""
    from sympy_extras.polys.rationaltools import together

    def rational(e: Expr) -> bool:
        coefficients = _coefficients(e, x)
        return coefficients is not None and all(c.is_rational is True for c in coefficients.values())

    cleared: dict[Basic, Basic] = {}
    for atom in _atoms(formula):
        numerator, denominator = together(atom.lhs - atom.rhs).as_numer_denom()
        if not rational(numerator) or not rational(denominator):
            return None
        relation = _relation(atom.rel_op, expand(numerator*denominator), Integer(0))
        cleared[atom] = and_(relation, ne(denominator, 0)) if denominator.has(x) else relation
    return _xreplace(formula, cleared)


def _solve_transcendental(formula: Boolean, x: Symbol, ctx: _Context) -> Optional[Set]:
    formula = _without_abs(formula, x)
    # the relations evaluated: Eq(exp(x) + 2, 0) is False for a real x
    formula = _map_atoms(formula, lambda a: _relation(a.rel_op, a.lhs, a.rhs))
    if isinstance(formula, BooleanFalse):
        return EmptySet
    if isinstance(formula, BooleanTrue):
        return Reals
    if _polynomial(formula, x):
        return in_radicals(solution_set(formula, x))
    rational = _cleared(formula, x)
    if rational is not None and _polynomial(rational, x):
        # a rational formula (after the absolute values): 1/c > 1
        return in_radicals(solution_set(rational, x))
    reduction = polynomialize(formula, x)
    if reduction is None:
        return _lambert(formula, x, ctx)
    side: Boolean = true
    if reduction.family == 'exp':
        side = reduction.variable > 0
    elif reduction.family == 'root':
        side = reduction.variable >= 0
    values = _solve_real(and_(reduction.formula, side), x=reduction.variable, ctx=ctx)
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
        results.append(result if condition is true else condition_set(x, condition, result))
    return union(*results) if results else EmptySet
