"""Limits and series expansions under statement assumptions.

The core's :func:`~sympy_extras.core.limits.limit` and
:meth:`~sympy_extras.core.expr.Expr.series` know the properties of a
parameter only through the assumptions of its
:class:`~sympy_extras.core.symbol.Symbol` (``Symbol('a', positive=True)``),
and ``limit(exp(a*x), x, oo)`` with a plain ``a`` raises
``NotImplementedError: Result depends on the sign of sign(a)``. Here the
parameters carry the statement assumptions (``a > 0``, ``a < 0``,
``element(a, Integers)``, ...) into the core through dummies with the
matching assumptions, and when the result still depends on the sign of
some expression the limit is computed in each case (positive, zero,
negative) compatible with the assumptions and returned as a
``Piecewise``, the counterpart of Mathematica's ``GenerateConditions``.

The module works on the objects of the core and imports no SymPy:
:mod:`sympy_extras.assumptions.limits_sympy` is its face on SymPy objects.
One step is not ported and goes through SymPy at run time (the face
holds it, :func:`~sympy_extras.assumptions.limits_sympy.limit_seq_of_core`):
SymPy's ``limit_seq`` (:func:`sympy.series.limitseq.limit_seq`, the
limits of sequences by the Stolz--Cesaro theorem and the dominant terms).
The reliable numerical values of
:func:`sympy_extras._numeric.reliable_value`, with which an infinite limit
of a function taking complex values on the approach is recognised as
``zoo``, are native.
"""
from __future__ import annotations

import itertools
import re
from typing import Callable, Optional

from sympy_extras.core.accumbounds import AccumBounds
from sympy_extras.core.basic import Basic
from sympy_extras.core.coreerrors import PoleError
from sympy_extras.core.expr import Expr, ExprLike, as_expr, to_expr
from sympy_extras.core.limits import Limit, limit as _limit
from sympy_extras.core.logic import Boolean, BooleanLike, BooleanTrue, and_, not_, or_, true
from sympy_extras.core.mul import Mul, mul
from sympy_extras.core.numbers import Integer, NegativeInfinity, Infinity, Rational, nan, oo, zoo
from sympy_extras.core.parsing import ParseUnsupported, parse
from sympy_extras.core.power import Pow
from sympy_extras.core.relational import Eq, eq
from sympy_extras.core.symbol import Dummy, Symbol
from sympy_extras.functions.elementary.complexes import Sign
from sympy_extras.functions.elementary.exponential import Log, exp, log
from sympy_extras.functions.elementary.piecewise import Piecewise, piecewise
from sympy_extras.sets.sets import Reals, Set
from sympy_extras._timeout import attempt, prepare_sympy
from sympy_extras.settings import settings

from .ask import Assumptions, _evaluate, _facts
from .facts import Facts, normalize
from .refine import _Refiner
from .sat import satisfiable

__all__ = ['limit', 'limit_seq', 'series', 'Engine']

#: computes a limit of an expression in a variable (raising
#: ``NotImplementedError`` or returning an unevaluated ``Limit`` when it
#: cannot)
Engine = Callable[[Expr, Symbol], Expr]

#: the depth of the case distinctions
_MAX_DEPTH = 3


def _parameters(expr: Expr, x: Symbol) -> list[Symbol]:
    """The free symbols other than the variable, sorted by name."""
    return sorted((s for s in expr.free_symbols if s != x), key=lambda s: s.name)


def _replaced(e: Expr, rule: dict[Basic, Basic]) -> Expr:
    """``e`` with the keys of ``rule`` replaced, evaluated (SymPy's
    ``xreplace``)."""
    return as_expr(e._xreplace_evaluated(rule))


def _holds(statement: BooleanLike, facts: Facts) -> bool:
    """Whether a statement follows from the facts."""
    return _evaluate(normalize(statement), facts) is True


def _abstract(expr: Expr, facts: Facts, x: Symbol) -> tuple[Expr, dict[Basic, Basic]]:
    """The parameters replaced by dummies carrying their assumptions."""
    refiner = _Refiner(facts)
    forward: dict[Basic, Basic] = {}
    back: dict[Basic, Basic] = {}
    for parameter in _parameters(expr, x):
        if _holds(eq(parameter, 0), facts):
            # a parameter known to vanish is replaced by zero
            forward[parameter] = Integer(0)
            continue
        value = _known_value(parameter, facts)
        if value is not None:
            # a parameter known to equal a constant is replaced by it
            forward[parameter] = value
            continue
        if _is_base(parameter, expr, x):
            # a parameter compared with 1 in the base of a power: p = exp(q)
            # with q of known sign carries the comparison into the core
            if _holds(parameter > 1, facts):
                q = Dummy('q', positive=True)
                forward[parameter] = exp(q)
                back[q] = log(parameter)
                continue
            if _holds(and_(parameter > 0, parameter < 1), facts):
                q = Dummy('q', positive=True)
                forward[parameter] = exp(-q)
                back[q] = -log(parameter)
                continue
        dummy = refiner.dummy_for(parameter)
        if dummy is not None:
            forward[parameter] = dummy
            back[dummy] = parameter
    return _replaced(expr, forward), back


def _is_base(parameter: Symbol, expr: Expr, x: Symbol) -> bool:
    """Whether the parameter is the base of a power whose exponent
    contains the variable."""
    return any(isinstance(power, Pow) and power.base == parameter and power.exp.has(x)
               for power in expr.atoms(Pow))


def _known_value(parameter: Symbol, facts: Facts) -> Optional[Expr]:
    """The constant a parameter is assumed to equal, if any."""
    for conjunct in facts.conjuncts:
        if isinstance(conjunct, Eq):
            lhs, rhs = conjunct.lhs, conjunct.rhs
            if lhs == parameter and not rhs.free_symbols:
                return rhs
            if rhs == parameter and not lhs.free_symbols:
                return lhs
    return None


def _sign_dependency(message: str, back: dict[Basic, Basic]) -> Optional[Expr]:
    """The expression whose sign the engine needs, from its error message
    (the dummies are found by their names, as SymPy's ``parse_expr`` with
    a ``local_dict`` did)."""
    found = re.search(r"sign of (.*)$", message)
    if not found:
        return None
    text = found.group(1).strip()
    try:
        e = parse(text)
    except (ParseUnsupported, ValueError, TypeError, SyntaxError, NameError):
        return None
    names: dict[Basic, Basic] = {Symbol(str(d)): d for d in back if isinstance(d, Symbol)}
    e = _replaced(e, names)
    if isinstance(e, Sign):
        e = e.args[0]
    return _replaced(e, back)


def _cases(e: Expr, facts: Facts, assumptions: Assumptions) -> list[Boolean]:
    """The sign cases of ``e`` compatible with the assumptions (the sign
    of ``log(u)`` is the position of ``u`` with respect to 1)."""
    result: list[Boolean] = []
    if e.is_extended_real is False or e.has(AccumBounds):
        # a sign the engine asked for which is not the sign of a real
        # number: the logarithm of a negative base, or the bounds of an
        # oscillating factor. There is no case to distinguish, and the
        # unevaluated limit is returned instead of a formula whose
        # conditions compare an AccumBounds with zero
        return []
    candidates: tuple[Boolean, ...] = (e > 0, eq(e, 0), e < 0)
    if isinstance(e, Log):
        # the sign of log(u) is the position of u with respect to 1, and
        # only for u > 0: a power of a non-positive base is not covered by
        # the formula the engine asks the sign for, and needs its own cases
        u = e.args[0]
        if _holds(u > 0, facts):
            candidates = (u > 1, eq(u, 1), u < 1)
        else:
            candidates = (u > 1, eq(u, 1), and_(u > 0, u < 1), eq(u, 0), and_(u > -1, u < 0), eq(u, -1),
                          u < -1)
    for case in candidates:
        case_ = normalize(case)
        value = _evaluate(case_, facts)
        if value is False:
            continue
        if value is True:
            return [true]
        if satisfiable(case_, assumptions) is False:
            continue
        result.append(case_)
    return result


def _with(case: Boolean, assumptions: Assumptions) -> list[BooleanLike]:
    """The assumptions with one more statement."""
    extra: list[BooleanLike] = [case]
    if isinstance(assumptions, (Basic, bool)):
        extra.append(assumptions)
    elif assumptions is not None:
        extra.extend(assumptions)
    return extra


def limit(expr: ExprLike, x: Symbol, x0: ExprLike, direction: str = '+', assumptions: Assumptions = None,
          domain: Optional[Set] = None) -> Expr:
    """The limit of ``expr`` as ``x`` tends to ``x0`` under assumptions on
    the parameters (the core's ``limit(expr, x, x0, dir)`` otherwise).

    Examples
    ========

    >>> from sympy_extras.core import oo, symbols
    >>> from sympy_extras.functions import exp, log
    >>> from sympy_extras.assumptions.limits import limit
    >>> a, x = symbols('a x')
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
    expr_ = to_expr(expr)
    x0_ = to_expr(x0)
    facts = _facts(assumptions, Reals if domain is None else domain, expr_.free_symbols | {x})
    return _limit_under(expr_, x, x0_, direction, facts, assumptions, depth=0)


def _continuous(x0: Expr, direction: str) -> Engine:
    def engine(e: Expr, x: Symbol) -> Expr:
        return _limit(e, x, x0, direction)
    return engine


def _sequence(e: Expr, n: Symbol) -> Expr:
    """SymPy's ``limit_seq`` as an engine (an unevaluated ``Limit`` when
    it gives up); called under native time limits, it prepares SymPy's
    tables first."""
    from .limits_sympy import limit_seq_of_core
    prepare_sympy()
    value = limit_seq_of_core(e, n)
    if value is None:
        # the limit of the function of a real variable, when it exists,
        # is the limit of the sequence (an oscillation in the real
        # variable says nothing about the integers)
        continuous = _limit(e, n, oo, '-')
        if isinstance(continuous, (Limit, AccumBounds)):
            return Limit(e, n, oo, '-')
        return continuous
    return value


def _nonpositive_base(expr: Expr, x: Symbol, x0: Expr, facts: Facts) -> Optional[Expr]:
    """The limit of ``c*b**(k*x)`` at infinity when the base is known to be
    non-positive, where the engine asks for the sign of ``log(b)`` (which
    is not real there).

    The modulus of ``b**(k*x)`` is ``|b|**(k*x)`` and its argument turns by
    ``pi*k`` at every step, so the limit is ``0`` when the modulus tends to
    zero, ``zoo`` when it tends to infinity, and ``nan`` for ``b = -1``,
    where the value keeps turning on the unit circle.
    """
    if not isinstance(x0, (Infinity, NegativeInfinity)):
        return None
    coefficient, power = expr.as_independent(x, as_Add=False)
    if not isinstance(power, Pow):
        return None
    base, exponent = power.base, power.exp
    slope = exponent.diff(x)
    if x in base.free_symbols or x in slope.free_symbols or (exponent - slope*x) != 0:
        return None
    upwards = _decide_sign(slope, facts)
    if upwards is None or upwards.is_zero:
        return None
    # whether the exponent tends to plus infinity
    growing = (upwards == 1) == isinstance(x0, Infinity)
    if _holds(eq(base, 0), facts):
        return Integer(0) if growing else zoo
    if not _holds(base < 0, facts):
        return None
    if _holds(eq(base, -1), facts):
        return nan
    small = _evaluate(normalize(and_(base > -1, base < 0)), facts)
    large = _evaluate(normalize(base < -1), facts)
    if small is not True and large is not True:
        return None
    if (small is True) == growing:
        return Integer(0)
    return mul(zoo, coefficient) if coefficient != 1 else zoo


def _limit_under(expr: Expr, x: Symbol, x0: Expr, direction: str, facts: Facts, assumptions: Assumptions,
                 depth: int, engine: Optional[Engine] = None) -> Expr:
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
        if dependency is None or depth > _MAX_DEPTH:
            return Limit(expr, x, x0, direction)
        return _split(expr, x, x0, direction, facts, assumptions, dependency, depth, engine)
    except (ValueError, TypeError, PoleError):
        return Limit(expr, x, x0, direction)
    result = _replaced(value, back)
    # signs the engine could not decide, and powers of infinity with a
    # parameter in the exponent, are decided from the assumptions or
    # split into cases
    undecided = _undecided(result)
    if undecided is not None and depth <= _MAX_DEPTH:
        decided = _decide_sign(undecided, facts)
        if decided is not None:
            result = _replaced(result, {Sign(undecided): decided})
            return _finish(result, undecided, decided, expr, x, x0, direction, facts, assumptions, depth, engine)
        return _split(expr, x, x0, direction, facts, assumptions, undecided, depth, engine)
    if undecided is not None:
        # a sign which could not be decided within the depth: the formula
        # the engine returned says nothing, the unevaluated limit says so
        return Limit(expr, x, x0, direction)
    if isinstance(result, Limit) and depth <= _MAX_DEPTH:
        # an unevaluated limit: try the sign cases of the parameters' atoms
        for parameter in _parameters(expr, x):
            cases = _cases(parameter, facts, assumptions)
            if len(cases) > 1:
                return _split(expr, x, x0, direction, facts, assumptions, parameter, depth, engine)
    if depth <= _MAX_DEPTH and not isinstance(result, Limit):
        # the generic answer need not hold at the degenerate values of a
        # parameter: (a x + 1)/(b x + 2) -> a/b says nothing at b = 0, where
        # the limit is +-oo (sympy-extras#30). A parameter at which the
        # engine's answer for the instance disagrees with the generic one
        # gets its own cases.
        degenerate = _degenerate_parameter(result, expr, x, facts, assumptions, engine_)
        if degenerate is not None:
            return _split(expr, x, x0, direction, facts, assumptions, degenerate, depth, engine)
    return _undirected(result, expr, x, x0, direction)


def _engine_value(engine: Engine, instance: Expr, x: Symbol) -> Optional[Expr]:
    """The engine's limit of an instance within the time limit; ``None``
    when it fails, gives up or returns bounds."""
    try:
        value = attempt(lambda: engine(instance, x), settings.timeout, native=True)
    except (NotImplementedError, ValueError, TypeError, PoleError):
        return None
    if value is None or isinstance(value, Limit) or value.has(AccumBounds):
        return None
    return value


def _degenerate_parameter(result: Expr, expr: Expr, x: Symbol, facts: Facts, assumptions: Assumptions,
                          engine: Engine) -> Optional[Symbol]:
    """A parameter whose value 0 is not covered by the generic answer."""
    for parameter in _parameters(expr, x):
        cases = _cases(parameter, facts, assumptions)
        if len(cases) <= 1:
            continue                            # its sign is already known
        generic = _replaced(result, {parameter: Integer(0)})
        instance = _replaced(expr, {parameter: Integer(0)})
        if generic.has(zoo, nan):
            return parameter
        value = _engine_value(engine, instance, x)
        if value is not None and _same_limit(value, generic) is False:
            return parameter
    return None


def _same_limit(first: Expr, second: Expr) -> Optional[bool]:
    """Whether two limit values are the same; ``None`` when undecided."""
    from sympy_extras.simplify.simplify import simplify
    if first == second:
        return True
    if first.is_infinite or second.is_infinite:
        return first == second
    difference = attempt(lambda: simplify(first - second), settings.timeout, native=True)
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
    if not (result.is_infinite and result != zoo and result != nan):
        return result
    if expr.free_symbols != {x}:
        return result                           # parameters: not sampled
    from sympy_extras._numeric import reliable_value
    for sample in _approach_samples(x0, direction):
        number = reliable_value(expr, 30, {x: sample})
        if number is None:
            return result
        try:
            value = complex(number)
        except (TypeError, ValueError, OverflowError):
            return result
        if abs(value.imag) > 1e-12*max(1.0, abs(value.real)):
            return zoo
    return result


def _approach_samples(x0: Expr, direction: str) -> list[Expr]:
    """A few points on the approach to ``x0``."""
    if isinstance(x0, Infinity):
        return [Integer(7), Rational(31, 2), Integer(101)]
    if isinstance(x0, NegativeInfinity):
        return [Integer(-7), Rational(-31, 2), Integer(-101)]
    if not x0.is_number:
        return []
    steps = [Rational(1, 10), Rational(1, 100), Rational(1, 1000)]
    return [x0 + (h if direction == '+' else -h) for h in steps]


def _sign_case(undecided: Expr, decided: Expr) -> Boolean:
    if decided == 1:
        return undecided > 0
    if decided == -1:
        return undecided < 0
    return eq(undecided, 0)


def _finish(result: Expr, undecided: Expr, decided: Expr, expr: Expr, x: Symbol, x0: Expr, direction: str,
            facts: Facts, assumptions: Assumptions, depth: int, engine: Optional[Engine] = None) -> Expr:
    """The result with a decided sign substituted; when the substitution
    leaves other undecided parts, the corresponding case is recomputed
    with the sign as an extra assumption."""
    if _undecided(result) is None:
        return result
    extra = _with(_sign_case(undecided, decided), assumptions)
    return _limit_under(expr, x, x0, direction, Facts(extra, Reals, list(facts.real)), and_(*extra), depth + 1,
                        engine)


def _is_infinity(e: Basic) -> bool:
    return isinstance(e, (Infinity, NegativeInfinity))


def _undecided(result: Expr) -> Optional[Expr]:
    """An expression whose sign the result depends on: the argument of a
    ``sign`` or the exponent of a power of infinity."""
    for atom in result.atoms(Sign):
        if isinstance(atom, Sign):
            return atom.args[0]
    for atom in result.atoms(Pow):
        if isinstance(atom, Pow) and _is_infinity(atom.base) and atom.exp.free_symbols:
            return atom.exp
    for atom in result.atoms(Mul):
        if isinstance(atom, Mul) and any(_is_infinity(a) for a in atom.args) and atom.free_symbols:
            others = [as_expr(a) for a in atom.args if not _is_infinity(a)]
            if others:
                return mul(*others)
    return None


def _decide_sign(e: Expr, facts: Facts) -> Optional[Expr]:
    if _holds(e > 0, facts):
        return Integer(1)
    if _holds(e < 0, facts):
        return Integer(-1)
    if _holds(eq(e, 0), facts):
        return Integer(0)
    return None


def _split(expr: Expr, x: Symbol, x0: Expr, direction: str, facts: Facts, assumptions: Assumptions,
           dependency: Expr, depth: int, engine: Optional[Engine] = None) -> Expr:
    pieces: list[tuple[Expr, Boolean]] = []
    for case in _cases(dependency, facts, assumptions):
        extra = _with(case, assumptions)
        case_facts = Facts(extra, Reals, list(facts.real))
        if isinstance(case, BooleanTrue):
            return _limit_under(expr, x, x0, direction, case_facts, and_(*extra), depth + 1, engine)
        value = _limit_under(expr, x, x0, direction, case_facts, and_(*extra), depth + 1, engine)
        value = _verified_piece(value, case, expr, x, x0, direction, case_facts, and_(*extra), depth, engine)
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
    if isinstance(value, (Limit, Piecewise)) or value.has(AccumBounds) or depth > _MAX_DEPTH:
        return value
    parameters = _parameters(expr, x)
    engine_ = engine if engine is not None else _continuous(x0, direction)
    # several points of the case: one can agree with a wrong piece by
    # accident (a = b = 2 satisfies both b and max(a, b))
    wrong = False
    for point in _points_of(case, parameters, 4):
        instance = _engine_value(engine_, _replaced(expr, point), x)
        if instance is None:
            continue
        if _same_limit(instance, _replaced(value, point)) is False:
            wrong = True
            break
    if not wrong:
        return value
    for first, second in itertools.combinations(parameters, 2):
        difference = first - second
        if len(_cases(difference, facts, assumptions)) > 1:
            return _split(expr, x, x0, direction, facts, assumptions, difference, depth + 1, engine)
    return Limit(expr, x, x0, direction)


def _points_of(case: Boolean, parameters: list[Symbol], count: int) -> list[dict[Basic, Basic]]:
    """Up to ``count`` rational points of the parameters at which the case
    holds, spread over the candidate values."""
    candidates: list[Expr] = [Integer(2), Rational(1, 2), Integer(-1), Integer(3), Rational(-5, 2), Integer(0),
                              Integer(1), Integer(5), Integer(-3)]
    found: list[dict[Basic, Basic]] = []
    for choice in itertools.product(candidates, repeat=len(parameters)):
        point: dict[Basic, Basic] = dict(zip(parameters, choice))
        if isinstance(case._xreplace_evaluated(point), BooleanTrue):
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
                exclusive = and_(condition, pair.cond, *[not_(p) for p in previous])
                flat.append((pair.expr, exclusive))
                previous.append(pair.cond)
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
    return piecewise(*[(value, _condition(or_(*conditions))) for value, conditions in merged])


def _condition(formula: Boolean) -> Boolean:
    """A condition on the parameters simplified over the reals when it is
    polynomial (``(a > 0) & (a > 1)`` is ``a > 1``)."""
    from .resolve import resolve
    simplified = attempt(lambda: resolve(formula), settings.timeout, native=True)
    return simplified if isinstance(simplified, Boolean) else formula


def limit_seq(expr: ExprLike, n: Symbol, assumptions: Assumptions = None, domain: Optional[Set] = None) -> Expr:
    """The limit of the sequence ``expr`` as the integer ``n`` tends to
    infinity under assumptions on the parameters (SymPy's ``limit_seq``
    otherwise), with case distinctions on the signs the result depends
    on, as :func:`limit`.

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.functions import binomial, factorial
    >>> from sympy_extras.assumptions.limits import limit_seq
    >>> a, n, x = symbols('a n x')
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
    expr_ = to_expr(expr)
    facts = _facts(assumptions, Reals if domain is None else domain, expr_.free_symbols | {n})
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
        base, exponent = power.base, power.exp
        if not (exponent.has(n) and not base.has(n) and base.free_symbols):
            continue
        s = _decide_sign(base, facts)
        if s is None and depth <= _MAX_DEPTH:
            pieces: list[tuple[Expr, Boolean]] = []
            for case in _cases(base, facts, assumptions):
                extra = _with(case, assumptions)
                case_facts = Facts(extra, Reals, list(facts.real))
                value = _limit_seq_under(expr, n, case_facts, and_(*extra), depth + 1)
                if isinstance(case, BooleanTrue):
                    return value
                pieces.append((value, case))
            return _piecewise(pieces) if pieces else Limit(expr, n, oo, '-')
        if s == -1:
            negative.append(power)
    if negative:
        # b**e = (-1)**e * (-b)**e: the sequence tends to zero when the
        # rest does, and has no limit when the rest has a nonzero one
        magnitude = expr
        for power in negative:
            magnitude = _replaced(magnitude, {power: (-power.base)**power.exp})
        size = _limit_under(magnitude, n, oo, '-', facts, assumptions, depth, _sequence)
        pieces_: list[tuple[Expr, Boolean]] = [(size, true)]
        if isinstance(size, Piecewise):
            pieces_ = [(pair.expr, pair.cond) for pair in size.args]
        return _piecewise([(Integer(0) if v == 0 else Limit(expr, n, oo, '-'), c) for v, c in pieces_])
    return _limit_under(expr, n, oo, '-', facts, assumptions, depth, _sequence)


def series(expr: ExprLike, x: Symbol, x0: ExprLike = 0, n: int = 6, assumptions: Assumptions = None,
           domain: Optional[Set] = None) -> Expr:
    """The series expansion of ``expr`` about ``x0`` with the parameters
    carrying the assumptions.

    Examples
    ========

    >>> from sympy_extras.core import symbols, sqrt
    >>> from sympy_extras.functions import log
    >>> from sympy_extras.assumptions.limits import series
    >>> a, x = symbols('a x')
    >>> series(sqrt(a**2 + x), x, 0, 2, assumptions=a > 0)
    x/(2*a) + a + O(x**2)
    >>> series(sqrt(a**2 + x), x, 0, 2, assumptions=a < 0)
    -x/(2*a) - a + O(x**2)
    >>> series(log(a*x), x, 1, 2, assumptions=a > 0)
    -1 + log(a) + x + O((x - 1)**2, (x, 1))
    """
    expr_ = to_expr(expr)
    x0_ = to_expr(x0)
    facts = _facts(assumptions, Reals if domain is None else domain, expr_.free_symbols | {x})
    abstracted, back = _abstract(expr_, facts, x)
    expansion = abstracted.series(x, x0_, n)
    return _replaced(expansion, back)
