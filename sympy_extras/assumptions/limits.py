"""Limits and series expansions under statement assumptions.

SymPy's :func:`~sympy.limit` and :func:`~sympy.series` know the
properties of a parameter only through the assumptions of its
:class:`~sympy.Symbol` (``Symbol('a', positive=True)``), and
``limit(exp(a*x), x, oo)`` with a plain ``a`` raises ``NotImplementedError:
Result depends on the sign of sign(a)``. Here the parameters carry the
statement assumptions (``a > 0``, ``a < 0``, ``element(a, S.Integers)``,
...) into SymPy through dummies with the matching core assumptions, and
when the result still depends on the sign of some expression the limit is
computed in each case (positive, zero, negative) compatible with the
assumptions and returned as a ``Piecewise``, the counterpart of
Mathematica's ``GenerateConditions``.
"""
from __future__ import annotations

import itertools

import re
from typing import Optional, Union

from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.core.power import Pow
from sympy.core.sympify import sympify
from sympy.parsing.sympy_parser import parse_expr
from sympy.functions.elementary.complexes import sign
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.piecewise import Piecewise
from sympy.core.mul import Mul
from sympy.calculus.accumulationbounds import AccumBounds
from sympy.simplify.simplify import simplify
from sympy.logic.boolalg import Boolean, And, Or, Not, true
from sympy.series.limits import Limit, limit as _limit
from sympy.series.limitseq import limit_seq as _limit_seq
from typing import Callable
from sympy.sets.sets import Set

from sympy_extras._typing import as_expr, free_symbols
from sympy_extras._timeout import attempt
from sympy_extras.settings import settings

#: computes a limit of an expression in a variable (raising
#: ``NotImplementedError`` or returning an unevaluated ``Limit`` when it
#: cannot)
Engine = Callable[[Expr, Symbol], Expr]

from .ask import Assumptions, _facts, _evaluate
from .facts import Facts, normalize
from .refine import _Refiner
from .sat import satisfiable

__all__ = ['limit', 'limit_seq', 'series']


def _abstract(expr: Expr, facts: Facts, x: Symbol) -> tuple[Expr, dict[Basic, Basic]]:
    """The parameters replaced by dummies carrying their assumptions."""
    refiner = _Refiner(facts)
    forward: dict[Basic, Basic] = {}
    back: dict[Basic, Basic] = {}
    for parameter in sorted((s for s in free_symbols(expr) if s != x), key=lambda s: s.name):
        if _evaluate(normalize(as_boolean_(Eq(parameter, 0))), facts) is True:
            # a parameter known to vanish is replaced by zero
            forward[parameter] = S.Zero
            continue
        value = _known_value(parameter, facts)
        if value is not None:
            # a parameter known to equal a constant is replaced by it
            forward[parameter] = value
            continue
        if _is_base(parameter, expr, x):
            # a parameter compared with 1 in the base of a power: p = exp(q)
            # with q of known sign carries the comparison into SymPy
            if _evaluate(normalize(as_boolean_(parameter > 1)), facts) is True:
                q = Dummy('q', positive=True)
                forward[parameter] = exp(q)
                back[q] = log(parameter)
                continue
            if _evaluate(normalize(as_boolean_(And(parameter > 0, parameter < 1))), facts) is True:
                q = Dummy('q', positive=True)
                forward[parameter] = exp(-q)
                back[q] = -log(parameter)
                continue
        dummy = refiner.dummy_for(parameter)
        if dummy is not None:
            forward[parameter] = dummy
            back[dummy] = parameter
    return as_expr(expr.xreplace(forward)), back


def _is_base(parameter: Symbol, expr: Expr, x: Symbol) -> bool:
    """Whether the parameter is the base of a power whose exponent
    contains the variable."""
    return any(isinstance(power, Pow) and power.base == parameter and as_expr(power.exp).has(x)
               for power in expr.atoms(Pow))


def _known_value(parameter: Symbol, facts: Facts) -> Optional[Expr]:
    """The constant a parameter is assumed to equal, if any."""
    for conjunct in facts.conjuncts:
        if isinstance(conjunct, Eq):
            lhs, rhs = as_expr(conjunct.lhs), as_expr(conjunct.rhs)
            if lhs == parameter and not free_symbols(rhs):
                return rhs
            if rhs == parameter and not free_symbols(lhs):
                return lhs
    return None


def _sign_dependency(message: str, back: dict[Basic, Basic]) -> Optional[Expr]:
    """The expression whose sign SymPy needs, from its error message."""
    found = re.search(r"sign of (.*)$", message)
    if not found:
        return None
    text = found.group(1).strip()
    try:
        e = parse_expr(text, local_dict={str(d): d for d in back})
    except (ValueError, TypeError, SyntaxError, NameError):
        return None
    if isinstance(e, sign):
        e = e.args[0]
    if not isinstance(e, Expr):
        return None
    return as_expr(e.xreplace(back))


def _cases(e: Expr, facts: Facts, assumptions: Assumptions) -> list[Boolean]:
    """The sign cases of ``e`` compatible with the assumptions (the sign
    of ``log(u)`` is the position of ``u`` with respect to 1)."""
    result: list[Boolean] = []
    if e.is_extended_real is False or e.has(AccumBounds):
        # a sign SymPy asked for which is not the sign of a real number:
        # the logarithm of a negative base, or the bounds of an
        # oscillating factor. There is no case to distinguish, and the
        # unevaluated limit is returned instead of a formula whose
        # conditions compare an AccumBounds with zero
        return []
    candidates: tuple[Boolean, ...] = (as_boolean_(e > 0), as_boolean_(Eq(e, 0)), as_boolean_(e < 0))
    if isinstance(e, log):
        # the sign of log(u) is the position of u with respect to 1, and
        # only for u > 0: a power of a non-positive base is not covered by
        # the formula SymPy asks the sign for, and needs its own cases
        u = as_expr(e.args[0])
        if _evaluate(normalize(as_boolean_(u > 0)), facts) is True:
            candidates = (as_boolean_(u > 1), as_boolean_(Eq(u, 1)), as_boolean_(u < 1))
        else:
            candidates = (as_boolean_(u > 1), as_boolean_(Eq(u, 1)), as_boolean_(And(u > 0, u < 1)),
                          as_boolean_(Eq(u, 0)), as_boolean_(And(u > -1, u < 0)),
                          as_boolean_(Eq(u, -1)), as_boolean_(u < -1))
    for case in candidates:
        case_ = normalize(as_boolean_(case))
        value = _evaluate(case_, facts)
        if value is False:
            continue
        if value is True:
            return [true]
        if satisfiable(case_, assumptions) is False:
            continue
        result.append(case_)
    return result


def as_boolean_(b: object) -> Boolean:
    from sympy_extras._typing import as_boolean
    return as_boolean(b)


def limit(expr: Union[Expr, int], x: Symbol, x0: Union[Expr, int], direction: str = '+',
          assumptions: Assumptions = None, domain: Optional[Set] = None) -> Expr:
    """The limit of ``expr`` as ``x`` tends to ``x0`` under assumptions on
    the parameters (``limit(expr, x, x0, dir)`` of SymPy otherwise).

    Examples
    ========

    >>> from sympy import exp, oo, log
    >>> from sympy.abc import a, x
    >>> from sympy_extras.assumptions import limit
    >>> limit(exp(a*x), x, oo, assumptions=a < 0)
    0
    >>> limit(exp(a*x), x, oo)
    Piecewise((oo, a > 0), (1, Eq(a, 0)), (0, a < 0))
    >>> limit(x**a, x, oo, assumptions=a > 0)
    oo
    >>> limit((1 + a/x)**x, x, oo)
    exp(a)
    >>> limit(x**a*log(x), x, 0, assumptions=a > 0)
    0
    """
    expr_ = as_expr(sympify(expr))
    x0_ = as_expr(sympify(x0))
    facts = _facts(assumptions, S.Reals if domain is None else domain, free_symbols(expr_) | {x})
    return _limit_under(expr_, x, x0_, direction, facts, assumptions, depth=0)


def _continuous(x0: Expr, direction: str) -> Engine:
    def engine(e: Expr, x: Symbol) -> Expr:
        return as_expr(_limit(e, x, x0, direction))
    return engine


def _sequence(e: Expr, n: Symbol) -> Expr:
    """SymPy's ``limit_seq`` as an engine (an unevaluated ``Limit`` when
    it gives up)."""
    value = _limit_seq(e, n)
    if value is None:
        # the limit of the function of a real variable, when it exists,
        # is the limit of the sequence (an oscillation in the real
        # variable says nothing about the integers)
        continuous = _limit(e, n, S.Infinity, '-')
        if isinstance(continuous, (Limit, AccumBounds)):
            return as_expr(Limit(e, n, S.Infinity, '-'))
        return as_expr(continuous)
    return as_expr(value)


def _nonpositive_base(expr: Expr, x: Symbol, x0: Expr, facts: Facts) -> Optional[Expr]:
    """The limit of ``c*b**(k*x)`` at infinity when the base is known to be
    non-positive, where SymPy asks for the sign of ``log(b)`` (which is not
    real there).

    The modulus of ``b**(k*x)`` is ``|b|**(k*x)`` and its argument turns by
    ``pi*k`` at every step, so the limit is ``0`` when the modulus tends to
    zero, ``zoo`` when it tends to infinity, and ``nan`` for ``b = -1``,
    where the value keeps turning on the unit circle.
    """
    if x0 not in (S.Infinity, S.NegativeInfinity):
        return None
    coefficient, power = expr.as_independent(x, as_Add=False)
    if not isinstance(power, Pow):
        return None
    base, exponent = as_expr(power.base), as_expr(power.exp)
    slope = as_expr(exponent.diff(x))
    if x in free_symbols(base) or x in free_symbols(slope) or as_expr(exponent - slope*x) != 0:
        return None
    upwards = _decide_sign(slope, facts)
    if upwards is None or upwards.is_zero:
        return None
    # whether the exponent tends to plus infinity
    growing = (upwards == 1) == (x0 is S.Infinity)
    if _evaluate(normalize(as_boolean_(Eq(base, 0))), facts) is True:
        return S.Zero if growing else S.ComplexInfinity
    if _evaluate(normalize(as_boolean_(base < 0)), facts) is not True:
        return None
    if _evaluate(normalize(as_boolean_(Eq(base, -1))), facts) is True:
        return S.NaN
    small = _evaluate(normalize(as_boolean_(And(base > -1, base < 0))), facts)
    large = _evaluate(normalize(as_boolean_(base < -1)), facts)
    if small is not True and large is not True:
        return None
    if (small is True) == growing:
        return S.Zero
    return as_expr(S.ComplexInfinity*coefficient) if coefficient != 1 else S.ComplexInfinity


def _limit_under(expr: Expr, x: Symbol, x0: Expr, direction: str, facts: Facts,
                 assumptions: Assumptions, depth: int, engine: Optional[Engine] = None) -> Expr:
    if engine is None:
        # the limit of a function of a real variable; a sequence with a
        # negative base oscillates and is left to ``limit_seq``
        special = _nonpositive_base(expr, x, x0, facts)
        if special is not None:
            return special
    engine_ = engine if engine is not None else _continuous(x0, direction)
    abstracted, back = _abstract(expr, facts, x)
    try:
        value = engine_(abstracted, x)
    except NotImplementedError as error:
        dependency = _sign_dependency(str(error), back)
        if dependency is None or depth > 3:
            return Limit(expr, x, x0, direction)
        return _split(expr, x, x0, direction, facts, assumptions, dependency, depth, engine)
    except (ValueError, TypeError):
        return Limit(expr, x, x0, direction)
    result = as_expr(value.xreplace(back))
    # signs SymPy could not decide, and powers of infinity with a
    # parameter in the exponent, are decided from the assumptions or
    # split into cases
    undecided = _undecided(result)
    if undecided is not None and depth <= 3:
        decided = _decide_sign(undecided, facts)
        if decided is not None:
            result = as_expr(result.xreplace({sign(undecided): decided}))
            return _finish(result, undecided, decided, expr, x, x0, direction, facts, assumptions, depth, engine)
        return _split(expr, x, x0, direction, facts, assumptions, undecided, depth, engine)
    if undecided is not None:
        # a sign which could not be decided within the depth: the formula
        # SymPy returned says nothing, the unevaluated limit says so
        return as_expr(Limit(expr, x, x0, direction))
    if isinstance(result, Limit) and depth <= 3:
        # an unevaluated limit: try the sign cases of the parameters' atoms
        for parameter in sorted((s for s in free_symbols(expr) if s != x), key=lambda s: s.name):
            cases = _cases(parameter, facts, assumptions)
            if len(cases) > 1:
                return _split(expr, x, x0, direction, facts, assumptions, parameter, depth, engine)
    if depth <= 3 and not isinstance(result, Limit):
        # the generic answer need not hold at the degenerate values of a
        # parameter: (a x + 1)/(b x + 2) -> a/b says nothing at b = 0, where
        # the limit is +-oo (sympy-extras#30). A parameter at which the
        # engine's answer for the instance disagrees with the generic one
        # gets its own cases.
        degenerate = _degenerate_parameter(result, expr, x, x0, direction, facts, assumptions, engine_)
        if degenerate is not None:
            return _split(expr, x, x0, direction, facts, assumptions, degenerate, depth, engine)
    return _undirected(result, expr, x, x0, direction)


def _degenerate_parameter(result: Expr, expr: Expr, x: Symbol, x0: Expr, direction: str, facts: Facts,
                          assumptions: Assumptions, engine: Engine) -> Optional[Symbol]:
    """A parameter whose value 0 is not covered by the generic answer."""
    for parameter in sorted((s for s in free_symbols(expr) if s != x), key=lambda s: s.name):
        cases = _cases(parameter, facts, assumptions)
        if len(cases) <= 1:
            continue                            # its sign is already known
        generic = as_expr(result.xreplace({parameter: S.Zero}))
        instance = as_expr(expr.xreplace({parameter: S.Zero}))
        if generic.has(S.ComplexInfinity, S.NaN):
            return parameter
        try:
            value = attempt(lambda: as_expr(engine(instance, x)), settings.timeout)
        except (NotImplementedError, ValueError, TypeError):
            continue
        if value is None or isinstance(value, Limit) or value.has(AccumBounds):
            continue
        if _same_limit(value, generic) is False:
            return parameter
    return None


def _same_limit(first: Expr, second: Expr) -> Optional[bool]:
    """Whether two limit values are the same; ``None`` when undecided."""
    if first == second:
        return True
    if first.is_infinite or second.is_infinite:
        return first == second
    difference = attempt(lambda: as_expr(simplify(first - second)), settings.timeout)
    if difference is None:
        return None
    if difference == 0:
        return True
    if difference.is_number and difference.is_zero is False:
        return False
    return None


def _undirected(result: Expr, expr: Expr, x: Symbol, x0: Expr, direction: str) -> Expr:
    """A directed infinity (``oo``, ``-oo``, ``oo*I``, ...) is only right
    for a function which is real -- or on one ray -- along the approach.
    ``exp(x**2 + I*x)`` has a diverging modulus and an argument which never
    settles: the limit is ``zoo``, not ``oo`` (sympy-extras#52). A function
    which takes complex values on the approach cannot run off along a real
    ray, so its infinite limit is written ``zoo``."""
    if not (result.is_infinite and result not in (S.ComplexInfinity, S.NaN)):
        return result
    if free_symbols(expr) != {x}:
        return result                           # parameters: not sampled
    for sample in _approach_samples(x0, direction):
        try:
            value = complex(expr.xreplace({x: sample}).evalf(30))
        except (TypeError, ValueError, ArithmeticError, OverflowError):
            return result
        if abs(value.imag) > 1e-12*max(1.0, abs(value.real)):
            return S.ComplexInfinity
    return result


def _approach_samples(x0: Expr, direction: str) -> list[Expr]:
    """A few points on the approach to ``x0``."""
    from sympy.core.numbers import Rational
    if x0 is S.Infinity:
        return [Rational(7), Rational(31, 2), Rational(101)]
    if x0 is S.NegativeInfinity:
        return [Rational(-7), Rational(-31, 2), Rational(-101)]
    if not x0.is_number:
        return []
    steps = [Rational(1, 10), Rational(1, 100), Rational(1, 1000)]
    return [as_expr(x0 + (h if direction == '+' else -h)) for h in steps]


def _finish(result: Expr, undecided: Expr, decided: Expr, expr: Expr, x: Symbol, x0: Expr, direction: str,
            facts: Facts, assumptions: Assumptions, depth: int, engine: Optional[Engine] = None) -> Expr:
    """The result with a decided sign substituted; when the substitution
    leaves other undecided parts, the corresponding case is recomputed
    with the sign as an extra assumption."""
    if _undecided(result) is None:
        return result
    case = undecided > 0 if decided == 1 else (undecided < 0 if decided == -1 else Eq(undecided, 0))
    extra: list[Union[Boolean, bool]] = [as_boolean_(case)]
    if isinstance(assumptions, (Boolean, bool)):
        extra.append(assumptions)
    elif assumptions is not None:
        extra.extend(assumptions)
    return _limit_under(expr, x, x0, direction, Facts(extra, S.Reals, list(facts.real)), And(*extra), depth + 1,
                        engine)


def _undecided(result: Expr) -> Optional[Expr]:
    """An expression whose sign the result depends on: the argument of a
    ``sign`` or the exponent of a power of infinity."""
    for atom in result.atoms(sign):
        return as_expr(atom.args[0])
    from sympy.core.power import Pow
    for atom in result.atoms(Pow):
        if atom.base in (S.Infinity, S.NegativeInfinity) and free_symbols(as_expr(atom.exp)):
            return as_expr(atom.exp)
    for atom in result.atoms(Mul):
        if any(a in (S.Infinity, S.NegativeInfinity) for a in atom.args) and free_symbols(atom):
            others = [as_expr(a) for a in atom.args if a not in (S.Infinity, S.NegativeInfinity)]
            if others:
                return as_expr(Mul(*others))
    return None


def _decide_sign(e: Expr, facts: Facts) -> Optional[Expr]:
    if _evaluate(normalize(as_boolean_(e > 0)), facts) is True:
        return S.One
    if _evaluate(normalize(as_boolean_(e < 0)), facts) is True:
        return S.NegativeOne
    if _evaluate(normalize(as_boolean_(Eq(e, 0))), facts) is True:
        return S.Zero
    return None


def _split(expr: Expr, x: Symbol, x0: Expr, direction: str, facts: Facts, assumptions: Assumptions,
           dependency: Expr, depth: int, engine: Optional[Engine] = None) -> Expr:
    pieces: list[tuple[Expr, Boolean]] = []
    for case in _cases(dependency, facts, assumptions):
        extra: list[Union[Boolean, bool]] = [case]
        if isinstance(assumptions, (Boolean, bool)):
            extra.append(assumptions)
        elif assumptions is not None:
            extra.extend(assumptions)
        case_facts = Facts(extra, S.Reals, list(facts.real))
        if case is true:
            return _limit_under(expr, x, x0, direction, case_facts, And(*extra), depth + 1, engine)
        value = _limit_under(expr, x, x0, direction, case_facts, And(*extra), depth + 1, engine)
        value = _verified_piece(value, case, expr, x, x0, direction, case_facts, And(*extra), depth, engine)
        pieces.append((value, case))
    if not pieces:
        return Limit(expr, x, x0, direction)
    return _piecewise(pieces)


def _verified_piece(value: Expr, case: Boolean, expr: Expr, x: Symbol, x0: Expr, direction: str,
                    facts: Facts, assumptions: Assumptions, depth: int, engine: Optional[Engine]) -> Expr:
    """The piece, checked at a point of its case against the engine on the
    instance. A wrong piece -- ``log(x**a + x**b)/log(x)`` split on the sign
    of ``b`` alone gives ``b`` where the answer is ``max(a, b)``
    (sympy-extras#28) -- is refined on the differences of the parameters,
    and left as the unevaluated limit when that does not settle it."""
    if isinstance(value, (Limit, Piecewise)) or value.has(AccumBounds) or depth > 3:
        return value
    parameters = sorted((s for s in free_symbols(expr) if s != x), key=lambda s: s.name)
    engine_ = engine if engine is not None else _continuous(x0, direction)
    # several points of the case: one can agree with a wrong piece by
    # accident (a = b = 2 satisfies both b and max(a, b))
    wrong = False
    for point in _points_of(case, parameters, 4):
        try:
            instance = attempt(lambda: as_expr(engine_(as_expr(expr.xreplace(point)), x)), settings.timeout)
        except (NotImplementedError, ValueError, TypeError):
            continue
        if instance is None or isinstance(instance, Limit) or instance.has(AccumBounds):
            continue
        if _same_limit(instance, as_expr(value.xreplace(point))) is False:
            wrong = True
            break
    if not wrong:
        return value
    for first, second in itertools.combinations(parameters, 2):
        difference = as_expr(first - second)
        if len(_cases(difference, facts, assumptions)) > 1:
            return _split(expr, x, x0, direction, facts, assumptions, difference, depth + 1, engine)
    return as_expr(Limit(expr, x, x0, direction))


def _points_of(case: Boolean, parameters: list[Symbol], count: int) -> list[dict[Symbol, Expr]]:
    """Up to ``count`` rational points of the parameters at which the case
    holds, spread over the candidate values."""
    from sympy.core.numbers import Rational
    candidates = [Rational(2), Rational(1, 2), Rational(-1), Rational(3), Rational(-5, 2), Rational(0),
                  Rational(1), Rational(5), Rational(-3)]
    found: list[dict[Symbol, Expr]] = []
    for choice in itertools.product(candidates, repeat=len(parameters)):
        point = dict(zip(parameters, choice))
        if case.xreplace(point) is true:
            found.append(point)
            if len(found) >= count:
                break
    return found


def _piecewise(pieces: list[tuple[Expr, Boolean]]) -> Expr:
    """A ``Piecewise`` over exhaustive cases, flattened, with the cases of
    equal value merged (a single value is returned as it is)."""
    flat: list[tuple[Expr, Boolean]] = []
    for value, condition in pieces:
        if isinstance(value, Piecewise):
            # the pieces of a Piecewise are ordered: a condition holds
            # when the previous ones fail
            previous: list[Boolean] = []
            for pair in value.args:
                inner_value, inner_condition = as_expr(pair.args[0]), as_boolean_(pair.args[1])
                exclusive = And(condition, inner_condition, *[Not(p) for p in previous])
                flat.append((inner_value, as_boolean_(exclusive)))
                previous.append(inner_condition)
        else:
            flat.append((value, condition))
    merged: list[tuple[Expr, list[Boolean]]] = []
    for value, condition in flat:
        for existing in merged:
            if existing[0] == value:
                existing[1].append(condition)
                break
        else:
            merged.append((value, [condition]))
    if len(merged) == 1:
        return merged[0][0]
    return as_expr(Piecewise(*[(value, _condition(Or(*conditions))) for value, conditions in merged]))


def _condition(formula: Boolean) -> Boolean:
    """A condition on the parameters simplified over the reals when it is
    polynomial (``(a > 0) & (a > 1)`` is ``a > 1``)."""
    from .resolve import resolve
    simplified = attempt(lambda: resolve(formula), settings.timeout)
    return simplified if isinstance(simplified, Boolean) else formula


def limit_seq(expr: Union[Expr, int], n: Symbol, assumptions: Assumptions = None,
              domain: Optional[Set] = None) -> Expr:
    """The limit of the sequence ``expr`` as the integer ``n`` tends to
    infinity under assumptions on the parameters (SymPy's ``limit_seq``
    otherwise), with case distinctions on the signs the result depends
    on, as :func:`limit`.

    Examples
    ========

    >>> from sympy import factorial, binomial
    >>> from sympy.abc import a, n, x
    >>> from sympy_extras.assumptions import limit_seq
    >>> limit_seq(x**n/factorial(n), n)
    0
    >>> limit_seq(a**n, n, assumptions=(a > 0) & (a < 1))
    0
    >>> limit_seq(a**n, n, assumptions=a > 0)
    Piecewise((oo, a > 1), (1, Eq(a, 1)), (0, a < 1))
    >>> limit_seq(a**n, n, assumptions=(a > -1) & (a < 0))
    0
    >>> limit_seq((1 + x/n)**n, n)
    exp(x)
    >>> limit_seq(binomial(2*n, n)/4**n, n)
    0
    """
    expr_ = as_expr(sympify(expr))
    facts = _facts(assumptions, S.Reals if domain is None else domain, free_symbols(expr_) | {n})
    return _limit_seq_under(expr_, n, facts, assumptions, 0)


def _limit_seq_under(expr: Expr, n: Symbol, facts: Facts, assumptions: Assumptions, depth: int) -> Expr:
    """The limit of a sequence: powers with a parameter in the base and
    ``n`` in the exponent oscillate for a negative base, so the sign of
    the base is decided or split into cases first; a negative base gives
    ``(-1)**e * (-b)**e``, whose limit is zero when ``(-b)**e`` tends to
    zero and does not exist otherwise."""
    negative: list[Pow] = []
    for power in sorted(expr.atoms(Pow), key=str):
        if not isinstance(power, Pow):
            continue
        base, exponent = as_expr(power.base), as_expr(power.exp)
        if not (exponent.has(n) and not base.has(n) and free_symbols(base)):
            continue
        s = _decide_sign(base, facts)
        if s is None and depth <= 3:
            pieces: list[tuple[Expr, Boolean]] = []
            for case in _cases(base, facts, assumptions):
                extra: list[Union[Boolean, bool]] = [case]
                if isinstance(assumptions, (Boolean, bool)):
                    extra.append(assumptions)
                elif assumptions is not None:
                    extra.extend(assumptions)
                case_facts = Facts(extra, S.Reals, list(facts.real))
                value = _limit_seq_under(expr, n, case_facts, And(*extra), depth + 1)
                if case is true:
                    return value
                pieces.append((value, case))
            return _piecewise(pieces) if pieces else as_expr(Limit(expr, n, S.Infinity, '-'))
        if s == S.NegativeOne:
            negative.append(power)
    if negative:
        # b**e = (-1)**e * (-b)**e: the sequence tends to zero when the
        # rest does, and has no limit when the rest has a nonzero one
        magnitude = expr
        for power in negative:
            base, exponent = as_expr(power.base), as_expr(power.exp)
            magnitude = as_expr(magnitude.xreplace({power: (-base)**exponent}))
        size = _limit_under(magnitude, n, S.Infinity, '-', facts, assumptions, depth, _sequence)
        pieces_: list[tuple[Expr, Boolean]] = [(size, as_boolean_(true))]
        if isinstance(size, Piecewise):
            pieces_ = [(as_expr(pair.args[0]), as_boolean_(pair.args[1])) for pair in size.args]
        return _piecewise([(S.Zero if v == 0 else as_expr(Limit(expr, n, S.Infinity, '-')), c)
                           for v, c in pieces_])
    return _limit_under(expr, n, S.Infinity, '-', facts, assumptions, depth, _sequence)


def series(expr: Union[Expr, int], x: Symbol, x0: Union[Expr, int] = 0, n: int = 6,
           assumptions: Assumptions = None, domain: Optional[Set] = None) -> Expr:
    """The series expansion of ``expr`` about ``x0`` with the parameters
    carrying the assumptions.

    Examples
    ========

    >>> from sympy import sqrt, log
    >>> from sympy.abc import a, x
    >>> from sympy_extras.assumptions import series
    >>> series(sqrt(a**2 + x), x, 0, 2, assumptions=a > 0)
    x/(2*a) + a + O(x**2)
    >>> series(sqrt(a**2 + x), x, 0, 2, assumptions=a < 0)
    -x/(2*a) - a + O(x**2)
    >>> series(log(a*x), x, 1, 2, assumptions=a > 0)
    -1 + log(a) + x + O((x - 1)**2, (x, 1))
    """
    expr_ = as_expr(sympify(expr))
    x0_ = as_expr(sympify(x0))
    facts = _facts(assumptions, S.Reals if domain is None else domain, free_symbols(expr_) | {x})
    abstracted, back = _abstract(expr_, facts, x)
    expansion = abstracted.series(x, x0_, n)
    result = as_expr(expansion.xreplace(back))
    return result
