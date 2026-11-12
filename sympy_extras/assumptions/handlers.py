"""The handlers of the predicates by class of the argument, and the
last steps of the core's ``ask``.

:func:`handler_ask` answers ``Q.real(sqrt(x)*y)``, ``Q.positive(log(x))``,
``Q.even(2*n)``, ... by the rules of the class of the argument, asking the
same question (or another one) about its arguments under the same
assumptions: a product of real factors is real, ``log(x)`` is positive
when ``x - 1`` is, ``Abs(x)`` is real. The core's ``ask``
(:func:`sympy_extras.core.ask.ask`) calls it, through :func:`final_ask`,
when the facts of the assumptions about the subjects and the old
assumptions do not decide, before the SAT search of
:func:`sympy_extras.assumptions.satask.predicates_consistent` (SymPy's
``satask``): the proposition is ``True`` when its negation is not
consistent with the assumptions and the known facts, ``False`` when the
proposition itself is not.

The module is a typed port of the handlers of SymPy's new assumptions
(:mod:`sympy.assumptions.handlers.sets`, ``order``, ``ntheory``,
``common``) and of the last steps of SymPy's ``ask``
(:mod:`sympy.assumptions.ask`). The handler of ``Expr`` (the old
assumptions of the node) is the core's own first step, so it is not
repeated; the numbers are evaluated by
:func:`sympy_extras._numeric.reliable_complex` rather than by a plain
``evalf`` at two digits; and the rule ``Real**Real -> Real`` when the
base may be zero and the exponent is positive is not ported: SymPy
answers ``ask(Q.real(sqrt(x)), Q.real(x))`` and
``ask(Q.real(sqrt(-1 - x**2)), Q.real(x))`` with ``True`` (both are
imaginary at ``x = -1``), a SymPy bug (issue #25). The relational step
of SymPy's ``ask`` (``lra_satask``, linear real arithmetic over the
relations of the assumptions) is not ported, except for its most frequent case: a sign
predicate of ``e`` decided by an assumed one of ``f`` when ``e/f`` is a
nonzero rational number (``Q.positive(1 - a)`` from ``Q.negative(a -
1)``).

Examples
========

>>> from sympy_extras.core import Q, Symbol, sqrt, ask
>>> from sympy_extras.functions import log, Abs, re
>>> x, k = Symbol('x'), Symbol('k')
>>> ask(Q.real(re(k) + 1)), ask(Q.positive(log(x)), Q.positive(x) & Q.negative(x - 1))
(True, False)
>>> ask(Q.real(Abs(x) - 1)), ask(Q.real(sqrt(x)), Q.real(x))
(True, None)

References
==========

.. [1] SymPy Development Team, *SymPy documentation: ask*,
       https://docs.sympy.org/latest/modules/assumptions/ask.html
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/assumptions/handlers/sets.py, sympy/assumptions/handlers/order.py,
#   sympy/assumptions/handlers/ntheory.py, sympy/assumptions/handlers/common.py,
#   sympy/assumptions/ask.py (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: strict typing, a table of typed functions instead of the
# multiple dispatch, no Expr handler (the core's first step), numbers by
# reliable_complex, the zero-base Pow rule of Q.real not ported (bug).
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Callable, Final, Optional

from sympy_extras.core.add import Add
from sympy_extras.core.assume import AppliedPredicate
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr
from sympy_extras.core.fuzzy import fuzzy_and, fuzzy_group, fuzzy_not, fuzzy_or
from sympy_extras.core.logic import Boolean, and_, not_, or_, true
from sympy_extras.core.mul import Mul
from sympy_extras.core.numbers import (Exp1, Float, GoldenRatioType, I, ImaginaryUnit, Infinity, Integer, NaN,
    NegativeInfinity, Number, NumberSymbol, Pi, Rational, TribonacciConstantType, pi)
from sympy_extras.core.power import Pow
from sympy_extras.functions.combinatorial.factorials import Factorial
from sympy_extras.functions.elementary.complexes import Abs, Im, Re
from sympy_extras.functions.elementary.exponential import Exp, Log, log
from sympy_extras.functions.elementary.trigonometric import Acos, Acot, Asin, Atan, Cos, Cot, Sin, Tan

__all__ = ['final_ask', 'handler_ask']

Truth = Optional[bool]
#: a handler: the argument of the predicate and the assumptions
Handler = Callable[[Expr, Boolean], Truth]


def _ask(proposition: Boolean, assumptions: Boolean) -> Truth:
    """The core's ``ask`` (the assumptions include the global ones)."""
    from sympy_extras.core.ask import ask
    return ask(proposition, assumptions, context=())


def _ask_global(proposition: Boolean) -> Truth:
    """The core's ``ask`` without local assumptions (SymPy's handlers ask
    some questions so)."""
    from sympy_extras.core.ask import ask
    return ask(proposition)


def _q(name: str, *args: Expr) -> Boolean:
    from sympy_extras.core.ask import PREDICATES_OF_Q
    return AppliedPredicate(PREDICATES_OF_Q[name], *args)


def _exp_argument(e: Expr) -> Optional[Expr]:
    """``x`` for ``exp(x)`` and ``E**x``."""
    if isinstance(e, Exp):
        return _arg(e)
    if isinstance(e, Pow) and isinstance(e.base, Exp1):
        return e.exp
    return None


def _arg(e: Expr, i: int = 0) -> Expr:
    a = e.args[i]
    assert isinstance(a, Expr)
    return a


def _value(e: Expr) -> Optional[complex]:
    from sympy_extras._numeric import reliable_complex
    return reliable_complex(e, 15)


# ---------------------------------------------------------------------
# the numbers

def _is_zero_part(part: float, z: complex) -> Truth:
    """Whether a part of the value ``z`` is zero: ``True`` when it is
    exactly (the hidden zeros are written 0), ``False`` when it is clearly
    not, ``None`` when it is below the precision (a zero not proved)."""
    if part == 0:
        return True
    if abs(part) > 1e-12 * max(1.0, abs(z)):
        return False
    return None


# PORTED FROM SYMPY: sympy/assumptions/handlers/order.py:_NegativePredicate_number, _PositivePredicate_number
def _sign_of_number(e: Expr, positive: bool) -> Truth:
    z = _value(e)
    if z is None:
        return None
    if abs(z) < 1e-10:
        # a hidden zero evaluates to noise (sqrt(2)*4/(2*sqrt(2) + 3) -
        # 12*sqrt(2) + 16 to 1e-87): zero only when proved, else undecided
        return False if _proved_number_zero(e) else None
    real = _is_zero_part(z.imag, z)
    if real is not True:
        return None if real is None else False
    return z.real > 0 if positive else z.real < 0


# PORTED FROM SYMPY: sympy/assumptions/handlers/sets.py:_RealPredicate_number
def _real_number(e: Expr) -> Truth:
    z = _value(e)
    return None if z is None else _is_zero_part(z.imag, z)


# PORTED FROM SYMPY: sympy/assumptions/handlers/sets.py:_Imaginary_number
def _imaginary_number(e: Expr) -> Truth:
    z = _value(e)
    return None if z is None else _is_zero_part(z.real, z)


def _proved_number_zero(e: Expr) -> bool:
    """Whether the number is proved zero (by its minimal polynomial)."""
    from sympy_extras._numeric import _proved_zero
    return _proved_zero(e)


def _not_integer_number(e: Expr) -> Truth:
    """``True`` when the number is certainly not an integer."""
    z = _value(e)
    if z is None:
        return None
    if _is_zero_part(z.imag, z) is False:
        return True
    return abs(z.real - round(z.real)) > 1e-10 * max(1.0, abs(z))


def _proved_integer(e: Expr) -> Optional[int]:
    """The integer the number equals, proved (by its minimal polynomial:
    SymPy's ``_IntegerPredicate_number`` proves it with ``equals``);
    ``None`` when it is not proved."""
    from sympy_extras._numeric import _proved_zero
    z = _value(e)
    if z is None or _is_zero_part(z.imag, z) is False:
        return None
    i = round(z.real)
    return i if _proved_zero(e - Integer(i)) else None


# PORTED FROM SYMPY: sympy/assumptions/handlers/sets.py:_IntegerPredicate_number
def _integer_of_number(e: Expr) -> Truth:
    if _not_integer_number(e):
        return False
    return True if _proved_integer(e) is not None else None


# ---------------------------------------------------------------------
# common

# PORTED FROM SYMPY: sympy/assumptions/handlers/common.py:test_closed_group
def _closed_group(e: Expr, assumptions: Boolean, name: str) -> Truth:
    return fuzzy_group((_ask(_q(name, _e(a)), assumptions) for a in e.args), quick_exit=True)


def _e(b: Basic) -> Expr:
    assert isinstance(b, Expr)
    return b


# ---------------------------------------------------------------------
# sets.py: integer, rational, irrational, real, extended_real, complex, imaginary

# PORTED FROM SYMPY: sympy/assumptions/handlers/sets.py (IntegerPredicate)
def _integer(e: Expr, assumptions: Boolean) -> Truth:
    if isinstance(e, Integer):
        return True
    if isinstance(e, (Exp1, GoldenRatioType, ImaginaryUnit, Infinity, NegativeInfinity, Pi, Rational,
                      TribonacciConstantType)):
        return False
    if isinstance(e, Add):
        if e.is_number:
            return _integer_of_number(e)
        return _closed_group(e, assumptions, 'integer')
    if isinstance(e, Pow):
        if e.is_number:
            return _integer_of_number(e)
        base, exp = e.base, e.exp
        if fuzzy_and([_ask(not_(_q('zero', base)), assumptions), _ask(_q('finite', base), assumptions),
                      _ask(_q('zero', exp), assumptions)]):
            return True
        if fuzzy_and([_ask(_q('integer', base), assumptions), _ask(_q('integer', exp), assumptions)]):
            if fuzzy_or([_ask(_q('positive', exp), assumptions),
                         _ask(and_(_q('nonnegative', exp), not_(_q('zero', base))), assumptions),
                         _ask(_q('zero', base - 1), assumptions), _ask(_q('zero', base + 1), assumptions)]):
                return True
        return None
    if isinstance(e, Mul):
        if e.is_number:
            return _integer_of_number(e)
        output = True
        for a in e.args:
            arg = _e(a)
            if not _ask(_q('integer', arg), assumptions):
                if isinstance(arg, Rational):
                    if arg.q == 2:
                        return _ask(_q('even', 2 * e), assumptions)
                    # SymPy tests ``~(arg.q & 1)``, which is always true
                    return None
                elif _ask(_q('irrational', arg), assumptions):
                    if output:
                        output = False
                    else:
                        return None
                else:
                    return None
        return output
    if isinstance(e, Abs):
        if _ask(_q('integer', _arg(e)), assumptions):
            return True
        return None
    return None


# PORTED FROM SYMPY: sympy/assumptions/handlers/sets.py (RationalPredicate)
def _rational(e: Expr, assumptions: Boolean) -> Truth:
    if isinstance(e, Rational):
        return True
    if isinstance(e, Float):
        return None
    if isinstance(e, (Exp1, GoldenRatioType, ImaginaryUnit, Infinity, NegativeInfinity, Pi,
                      TribonacciConstantType)):
        return False
    if isinstance(e, (Add, Mul)):
        if e.is_number:
            z = _value(e)
            if z is not None and _is_zero_part(z.imag, z) is False:
                return False
        return _closed_group(e, assumptions, 'rational')
    x = _exp_argument(e)
    if x is not None:
        if _ask(_q('rational', x), assumptions):
            return _ask(not_(_q('nonzero', x)), assumptions)
        return None
    if isinstance(e, Pow):
        base, exp = e.base, e.exp
        is_exp_integer = _ask(_q('integer', exp), assumptions)
        if is_exp_integer:
            if _ask(_q('rational', base), assumptions):
                is_base_zero = _ask(_q('zero', base), assumptions)
                if is_base_zero is False:
                    return True
                if is_base_zero and _ask_global(_q('positive', exp)):
                    return True
            if _ask(_q('algebraic', base), assumptions) is False:
                return _ask(_q('zero', exp), assumptions)
            if _ask(_q('irrational', base), assumptions) and _ask_global(_q('eq', exp, Integer(-1))):
                return False
            return None
        if _ask(_q('rational', exp), assumptions):
            if _ask(_q('prime', base), assumptions) and is_exp_integer is False:
                return False
            if _ask_global(_q('zero', base)) and _ask_global(_q('positive', exp)):
                return True
            if _ask_global(_q('eq', base, Integer(1))):
                return True
        return None
    if isinstance(e, (Asin, Atan, Cos, Sin, Tan)):
        x = _arg(e)
        if _ask(_q('rational', x), assumptions):
            return _ask(not_(_q('nonzero', x)), assumptions)
        return None
    if isinstance(e, (Acot, Cot)):
        if _ask(_q('rational', _arg(e)), assumptions):
            return False
        return None
    if isinstance(e, (Acos, Log)) and len(e.args) == 1:
        x = _arg(e)
        if _ask(_q('rational', x), assumptions):
            return _ask(not_(_q('nonzero', x - 1)), assumptions)
        return None
    return None


# PORTED FROM SYMPY: sympy/assumptions/handlers/sets.py (IrrationalPredicate, Basic)
def _irrational(e: Expr, assumptions: Boolean) -> Truth:
    real = _ask(_q('real', e), assumptions)
    if real:
        rational = _ask(_q('rational', e), assumptions)
        if rational is None:
            return None
        return not rational
    return real


# PORTED FROM SYMPY: sympy/assumptions/handlers/sets.py (RealPredicate)
def _real(e: Expr, assumptions: Boolean) -> Truth:
    if isinstance(e, (Abs, Exp1, Float, GoldenRatioType, Im, Pi, Rational, Re, TribonacciConstantType)):
        return True
    if isinstance(e, (ImaginaryUnit, Infinity, NegativeInfinity)):
        return False
    if isinstance(e, Add):
        if e.is_number:
            return _real_number(e)
        return _closed_group(e, assumptions, 'real')
    if isinstance(e, Mul):
        if e.is_number:
            return _real_number(e)
        result = True
        for a in e.args:
            arg = _e(a)
            if _ask(_q('real', arg), assumptions):
                pass
            elif _ask(_q('imaginary', arg), assumptions):
                result = not result
            else:
                return None
        return result
    x = _exp_argument(e)
    if x is not None:
        return _ask(or_(_q('integer', x / I / pi), _q('real', x)), assumptions)
    if isinstance(e, Pow):
        return _real_pow(e, assumptions)
    if isinstance(e, (Cos, Sin)):
        if _ask(_q('real', _arg(e)), assumptions):
            return True
        return None
    if isinstance(e, Log) and len(e.args) == 1:
        return _ask(_q('positive', _arg(e)), assumptions)
    return None


# PORTED FROM SYMPY: sympy/assumptions/handlers/sets.py:_ (RealPredicate, Pow)
def _real_pow(e: Pow, assumptions: Boolean) -> Truth:
    if e.is_number:
        return _real_number(e)
    base, exp = e.base, e.exp
    inner = _exp_argument(base)
    if inner is not None:
        if _ask(_q('imaginary', inner), assumptions):
            if _ask(_q('imaginary', exp), assumptions):
                return True
        # exp(i*I*pi)**exp is ((-1)**i)**exp when 2*i is an integer
        i = inner / I / pi
        if _ask(_q('integer', 2 * i), assumptions):
            return _ask(_q('real', (Integer(-1) ** i) ** exp), assumptions)
        return None
    if _ask(_q('imaginary', base), assumptions):
        if _ask(_q('integer', exp), assumptions):
            odd = _ask(_q('odd', exp), assumptions)
            if odd is not None:
                return not odd
            return None
    if _ask(_q('imaginary', exp), assumptions):
        imlog = _ask(_q('imaginary', log(base)), assumptions)
        if imlog is not None:
            return imlog
    if _ask(_q('real', base), assumptions):
        if _ask(_q('real', exp), assumptions):
            zero = _ask(_q('zero', base), assumptions)
            if zero is True:
                if _ask(_q('positive', exp), assumptions):
                    return True
                return None
            if zero is None:
                # SymPy answers True here for a positive exponent: wrong at
                # a negative base (sqrt(x) with x real), see the docstring
                return None
            if isinstance(exp, Rational) and exp.q % 2 == 0:
                return _ask(_q('positive', base), assumptions)
            if _ask(_q('integer', exp), assumptions):
                return True
            if _ask(_q('positive', base), assumptions):
                return True
    return None


# PORTED FROM SYMPY: sympy/assumptions/handlers/sets.py (ExtendedRealPredicate)
def _extended_real(e: Expr, assumptions: Boolean) -> Truth:
    if isinstance(e, (Infinity, NegativeInfinity)):
        return True
    if isinstance(e, (Add, Mul, Pow)):
        return _closed_group(e, assumptions, 'extended_real')
    return None


# PORTED FROM SYMPY: sympy/assumptions/handlers/sets.py (ComplexPredicate)
def _complex(e: Expr, assumptions: Boolean) -> Truth:
    if isinstance(e, (Abs, Cos, Exp, Im, ImaginaryUnit, Log, Number, NumberSymbol, Re, Sin)):
        if isinstance(e, (Infinity, NegativeInfinity, NaN)):
            return False
        return True
    if isinstance(e, (Add, Mul)):
        return _closed_group(e, assumptions, 'complex')
    if isinstance(e, Pow):
        if e.base == Exp1():
            return True
        return _closed_group(e, assumptions, 'complex')
    return None


# PORTED FROM SYMPY: sympy/assumptions/handlers/sets.py (ImaginaryPredicate)
def _imaginary(e: Expr, assumptions: Boolean) -> Truth:
    if isinstance(e, ImaginaryUnit):
        return True
    if isinstance(e, NaN):
        return None
    if isinstance(e, (Number, NumberSymbol)):
        z = _value(e)
        return None if z is None else fuzzy_not(_is_zero_part(z.imag, z))
    if isinstance(e, Add):
        if e.is_number:
            return _imaginary_number(e)
        reals = 0
        for a in e.args:
            arg = _e(a)
            if _ask(_q('imaginary', arg), assumptions):
                pass
            elif _ask(_q('real', arg), assumptions):
                reals += 1
            else:
                return None
        if reals == 0:
            return True
        if reals in (1, len(e.args)):
            # two reals could sum 0 thus giving an imaginary
            return False
        return None
    if isinstance(e, Mul):
        if e.is_number:
            return _imaginary_number(e)
        result = False
        for a in e.args:
            arg = _e(a)
            if _ask(_q('imaginary', arg), assumptions):
                result = not result
            elif not _ask(_q('real', arg), assumptions):
                return None
        return result
    x = _exp_argument(e)
    if x is not None:
        a = x / I / pi
        return _ask(and_(_q('integer', 2 * a), not_(_q('integer', a))), assumptions)
    if isinstance(e, Pow):
        return _imaginary_pow(e, assumptions)
    if isinstance(e, Log) and len(e.args) == 1:
        x = _arg(e)
        if _ask(_q('real', x), assumptions):
            if _ask(_q('positive', x), assumptions):
                return False
            return None
        inner = _exp_argument(x)
        if inner is not None and (inner == I or inner == -I):
            return True
        if _ask(_q('imaginary', x), assumptions) is False:
            return False
        return None
    return None


# PORTED FROM SYMPY: sympy/assumptions/handlers/sets.py:_ (ImaginaryPredicate, Pow)
def _imaginary_pow(e: Pow, assumptions: Boolean) -> Truth:
    if e.is_number:
        return _imaginary_number(e)
    base, exp = e.base, e.exp
    inner = _exp_argument(base)
    if inner is not None:
        if _ask(_q('imaginary', inner), assumptions):
            if _ask(_q('imaginary', exp), assumptions):
                return False
            i = inner / I / pi
            if _ask(_q('integer', 2 * i), assumptions):
                return _ask(_q('imaginary', (Integer(-1) ** i) ** exp), assumptions)
    if _ask(_q('imaginary', base), assumptions):
        if _ask(_q('integer', exp), assumptions):
            return _ask(_q('odd', exp), assumptions)
    if _ask(_q('imaginary', exp), assumptions):
        if _ask(_q('imaginary', log(base)), assumptions) is not None:
            return False
    if _ask(and_(_q('real', base), _q('real', exp)), assumptions):
        if _ask(_q('positive', base), assumptions):
            return False
        rat = _ask(_q('rational', exp), assumptions)
        if not rat:
            return rat
        if _ask(_q('integer', exp), assumptions):
            return False
        half = _ask(_q('integer', 2 * exp), assumptions)
        if half:
            return _ask(_q('negative', base), assumptions)
        return half
    return None


# ---------------------------------------------------------------------
# order.py: negative, nonnegative, nonzero, zero, nonpositive, positive

# PORTED FROM SYMPY: sympy/assumptions/handlers/order.py (NegativePredicate)
def _negative(e: Expr, assumptions: Boolean) -> Truth:
    if isinstance(e, (Abs, ImaginaryUnit)):
        return False
    x = _exp_argument(e)
    if x is not None:
        if _ask(_q('real', x), assumptions):
            return False
        return None
    if e.is_number:
        return _sign_of_number(e, False)
    if isinstance(e, Add):
        r = _ask(_q('real', e), assumptions)
        if r is not True:
            return r
        nonpos = 0
        for a in e.args:
            arg = _e(a)
            if _ask(_q('negative', arg), assumptions) is not True:
                if _ask(_q('positive', arg), assumptions) is False:
                    nonpos += 1
                else:
                    return None
        return True if nonpos < len(e.args) else None
    if isinstance(e, Mul):
        result: Truth = None
        for a in e.args:
            arg = _e(a)
            if result is None:
                result = False
            if _ask(_q('negative', arg), assumptions):
                result = not result
            elif _ask(_q('positive', arg), assumptions):
                pass
            else:
                return None
        return result
    if isinstance(e, Pow):
        base, exp = e.base, e.exp
        if _ask(_q('real', base), assumptions):
            if _ask(_q('positive', base), assumptions):
                if _ask(_q('real', exp), assumptions):
                    return False
            if _ask(_q('even', exp), assumptions):
                return False
            if _ask(_q('odd', exp), assumptions):
                return _ask(_q('negative', base), assumptions)
        return None
    return None


# PORTED FROM SYMPY: sympy/assumptions/handlers/order.py (NonNegativePredicate, Basic)
def _nonnegative(e: Expr, assumptions: Boolean) -> Truth:
    if e.is_number:
        notnegative = fuzzy_not(_sign_of_number(e, False))
        if notnegative:
            return _ask(_q('real', e), assumptions)
        return notnegative
    return None


# PORTED FROM SYMPY: sympy/assumptions/handlers/order.py (NonPositivePredicate, Basic)
def _nonpositive(e: Expr, assumptions: Boolean) -> Truth:
    if e.is_number:
        notpositive = fuzzy_not(_sign_of_number(e, True))
        if notpositive:
            return _ask(_q('real', e), assumptions)
        return notpositive
    return None


# PORTED FROM SYMPY: sympy/assumptions/handlers/order.py (NonZeroPredicate)
def _nonzero(e: Expr, assumptions: Boolean) -> Truth:
    if isinstance(e, NaN):
        return None
    if e.is_number and _proved_number_zero(e):
        return False
    if isinstance(e, Add):
        if all(_ask(_q('positive', _e(a)), assumptions) for a in e.args) or all(
                _ask(_q('negative', _e(a)), assumptions) for a in e.args):
            return True
        return None
    if isinstance(e, Mul):
        for a in e.args:
            result = _ask(_q('nonzero', _e(a)), assumptions)
            if result:
                continue
            return result
        return True
    if isinstance(e, Pow):
        return _ask(_q('nonzero', e.base), assumptions)
    if isinstance(e, Abs):
        return _ask(_q('nonzero', _arg(e)), assumptions)
    if _ask_global(_q('real', e)) is False:
        return False
    if e.is_number:
        z = _value(e)
        if z is None:
            return None
        return fuzzy_not(fuzzy_and([_is_zero_part(z.real, z), _is_zero_part(z.imag, z)]))
    return None


# PORTED FROM SYMPY: sympy/assumptions/handlers/order.py (ZeroPredicate)
def _zero(e: Expr, assumptions: Boolean) -> Truth:
    if e.is_number and _proved_number_zero(e):
        # SymPy proves it with equals (here the minimal polynomial)
        return True
    if isinstance(e, Mul):
        return fuzzy_or(_ask(_q('zero', _e(a)), assumptions) for a in e.args)
    return fuzzy_and([fuzzy_not(_ask(_q('nonzero', e), assumptions)), _ask(_q('real', e), assumptions)])


# PORTED FROM SYMPY: sympy/assumptions/handlers/order.py (PositivePredicate)
def _positive(e: Expr, assumptions: Boolean) -> Truth:
    if isinstance(e, (ImaginaryUnit, NaN)):
        return None if isinstance(e, NaN) else False
    x = _exp_argument(e)
    if x is not None:
        if _ask(_q('real', x), assumptions):
            return True
        if _ask(_q('imaginary', x), assumptions):
            return _ask(_q('even', x / (I * pi)), assumptions)
        return None
    if e.is_number:
        return _sign_of_number(e, True)
    if isinstance(e, Mul):
        result = True
        for a in e.args:
            arg = _e(a)
            if _ask(_q('positive', arg), assumptions):
                continue
            if _ask(_q('negative', arg), assumptions):
                result = not result
            else:
                return None
        return result
    if isinstance(e, Add):
        r = _ask(_q('real', e), assumptions)
        if r is not True:
            return r
        nonneg = 0
        for a in e.args:
            arg = _e(a)
            if _ask(_q('positive', arg), assumptions) is not True:
                if _ask(_q('negative', arg), assumptions) is False:
                    nonneg += 1
                else:
                    return None
        return True if nonneg < len(e.args) else None
    if isinstance(e, Pow):
        base, exp = e.base, e.exp
        if _ask(_q('positive', base), assumptions):
            if _ask(_q('real', exp), assumptions):
                return True
        if _ask(_q('negative', base), assumptions):
            if _ask(_q('even', exp), assumptions):
                return True
            if _ask(_q('odd', exp), assumptions):
                return False
        return None
    if isinstance(e, Log) and len(e.args) == 1:
        arg = _arg(e)
        r = _ask(_q('real', arg), assumptions)
        if r is not True:
            return r
        if _ask(_q('positive', arg - 1), assumptions):
            return True
        if _ask(_q('negative', arg - 1), assumptions):
            return False
        return None
    if isinstance(e, Factorial):
        arg = _arg(e)
        if _ask(and_(_q('integer', arg), _q('positive', arg)), assumptions):
            return True
        return None
    if isinstance(e, Abs):
        return _ask(_q('nonzero', e), assumptions)
    if isinstance(e, Atan):
        return _ask(_q('positive', _arg(e)), assumptions)
    if isinstance(e, Asin):
        arg = _arg(e)
        if _ask(and_(_q('positive', arg), _q('nonpositive', arg - 1)), assumptions):
            return True
        if _ask(and_(_q('negative', arg), _q('nonnegative', arg + 1)), assumptions):
            return False
        return None
    if isinstance(e, Acos):
        arg = _arg(e)
        if _ask(and_(_q('nonpositive', arg - 1), _q('nonnegative', arg + 1)), assumptions):
            return True
        return None
    if isinstance(e, Acot):
        return _ask(_q('real', _arg(e)), assumptions)
    return None


# ---------------------------------------------------------------------
# ntheory.py: prime, composite, even, odd

# PORTED FROM SYMPY: sympy/assumptions/handlers/ntheory.py (PrimePredicate)
def _prime(e: Expr, assumptions: Boolean) -> Truth:
    from sympy_extras.ntheory.primetest import isprime
    if isinstance(e, Integer):
        return isprime(e.p)
    if isinstance(e, (Rational, Infinity, NegativeInfinity, ImaginaryUnit)):
        return False
    if isinstance(e, NaN):
        return None
    if e.is_number:
        # PORTED FROM SYMPY: sympy/assumptions/handlers/ntheory.py:_PrimePredicate_number
        if _not_integer_number(e):
            return False
        i = _proved_integer(e)
        return None if i is None or e.has(Float) else isprime(i)
    if isinstance(e, Mul):
        for a in e.args:
            if not _ask(_q('integer', _e(a)), assumptions):
                return None
        for a in e.args:
            if _e(a).is_number and _e(a).is_composite:
                return False
        return None
    if isinstance(e, Pow):
        base, exp = e.base, e.exp
        if _ask(_q('integer', exp), assumptions) and _ask(_q('integer', base), assumptions):
            prime_base = _ask(_q('prime', base), assumptions)
            if prime_base is False:
                return False
            is_exp_one = _ask(_q('eq', exp, Integer(1)), assumptions)
            if is_exp_one is False:
                return False
            if prime_base is True and is_exp_one is True:
                return True
        return None
    return None


# PORTED FROM SYMPY: sympy/assumptions/handlers/ntheory.py (CompositePredicate, Basic)
def _composite(e: Expr, assumptions: Boolean) -> Truth:
    positive = _ask(_q('positive', e), assumptions)
    if positive:
        integer = _ask(_q('integer', e), assumptions)
        if integer:
            prime = _ask(_q('prime', e), assumptions)
            if prime is None:
                return None
            is_one = _ask(_q('eq', e, Integer(1)), assumptions)
            if is_one:
                return False
            if is_one is None:
                return None
            return not prime
        return integer
    return positive


# PORTED FROM SYMPY: sympy/assumptions/handlers/ntheory.py (EvenPredicate)
def _even(e: Expr, assumptions: Boolean) -> Truth:
    if isinstance(e, Integer):
        return e.p % 2 == 0
    if isinstance(e, (Rational, Infinity, NegativeInfinity, ImaginaryUnit)):
        return False
    if isinstance(e, NaN):
        return None
    if e.is_number:
        # PORTED FROM SYMPY: sympy/assumptions/handlers/ntheory.py:_EvenPredicate_number
        if _not_integer_number(e):
            return False
        i = _proved_integer(e)
        return None if i is None else i % 2 == 0
    if isinstance(e, Mul):
        even, odd, irrational = False, 0, False
        acc: Expr = Integer(1)
        for a in e.args:
            arg = _e(a)
            if _ask(_q('integer', arg), assumptions):
                if _ask(_q('even', arg), assumptions):
                    even = True
                elif _ask(_q('odd', arg), assumptions):
                    odd += 1
                elif not even and acc != Integer(1):
                    if _ask(_q('odd', acc + arg), assumptions):
                        even = True
            elif _ask(_q('irrational', arg), assumptions):
                # one irrational makes the result False, two make it undefined
                if irrational:
                    return None
                irrational = True
            else:
                return None
            acc = arg
        if irrational:
            return False
        if even:
            return True
        if odd == len(e.args):
            return False
        return None
    if isinstance(e, Add):
        result = True
        for a in e.args:
            arg = _e(a)
            if _ask(_q('even', arg), assumptions):
                pass
            elif _ask(_q('odd', arg), assumptions):
                result = not result
            else:
                return None
        return result
    if isinstance(e, Pow):
        base, exp = e.base, e.exp
        if _ask(_q('integer', exp), assumptions):
            if _ask(_q('positive', exp), assumptions):
                return _ask(_q('even', base), assumptions)
            if _ask(and_(not_(_q('negative', exp)), _q('odd', base)), assumptions):
                return False
            if base == Integer(-1):
                return False
        return None
    if isinstance(e, (Abs, Re)):
        if _ask(_q('real', _arg(e)), assumptions):
            return _ask(_q('even', _arg(e)), assumptions)
        return None
    if isinstance(e, Im):
        if _ask(_q('real', _arg(e)), assumptions):
            return True
        return None
    return None


# PORTED FROM SYMPY: sympy/assumptions/handlers/ntheory.py (OddPredicate, Basic)
def _odd(e: Expr, assumptions: Boolean) -> Truth:
    integer = _ask(_q('integer', e), assumptions)
    if integer:
        even = _ask(_q('even', e), assumptions)
        if even is None:
            return None
        return not even
    return integer


#: the handlers by name of the predicate
_HANDLERS: Final[dict[str, Handler]] = {
    'integer': _integer, 'rational': _rational, 'irrational': _irrational, 'real': _real,
    'extended_real': _extended_real, 'complex': _complex, 'imaginary': _imaginary, 'negative': _negative,
    'nonnegative': _nonnegative, 'nonpositive': _nonpositive, 'nonzero': _nonzero, 'zero': _zero,
    'positive': _positive, 'prime': _prime, 'composite': _composite, 'even': _even, 'odd': _odd,
}

#: the sign predicates, and their counterparts under a negative factor
_SIGN_FLIPS: Final[dict[str, str]] = {
    'positive': 'negative', 'negative': 'positive', 'nonnegative': 'nonpositive', 'nonpositive': 'nonnegative',
    'zero': 'zero', 'nonzero': 'nonzero'}

#: what a sign fact implies about the other sign predicates (True: holds,
#: False: fails)
_SIGN_IMPLIES: Final[dict[str, dict[str, bool]]] = {
    'positive': {'positive': True, 'nonnegative': True, 'nonzero': True, 'negative': False, 'nonpositive': False,
                 'zero': False},
    'negative': {'negative': True, 'nonpositive': True, 'nonzero': True, 'positive': False, 'nonnegative': False,
                 'zero': False},
    'zero': {'zero': True, 'nonnegative': True, 'nonpositive': True, 'positive': False, 'negative': False,
             'nonzero': False},
    'nonnegative': {'nonnegative': True, 'negative': False},
    'nonpositive': {'nonpositive': True, 'positive': False},
    'nonzero': {'nonzero': True, 'zero': False},
}


def _scaled_literal(proposition: Boolean, assumptions: Boolean) -> Truth:
    """A sign predicate of ``e`` decided by an assumed sign predicate of
    ``f`` with ``e/f`` a nonzero rational number (``Q.positive(1 - a)``
    from ``Q.negative(a - 1)``): the case of SymPy's ``lra_satask``
    (linear real arithmetic over the assumptions, not ported) met in
    practice."""
    from sympy_extras.core.logic import And
    from sympy_extras.polys.polytools import cancel
    if not (isinstance(proposition, AppliedPredicate) and proposition.function._defined):
        return None
    name = proposition.function.name
    if name not in _SIGN_FLIPS or len(proposition.arguments) != 1:
        return None
    e = proposition.arguments[0]
    if not isinstance(e, Expr) or e.is_number:
        return None
    for literal in And.make_args(assumptions):
        if not (isinstance(literal, AppliedPredicate) and literal.function._defined):
            continue
        known = literal.function.name
        if known not in _SIGN_FLIPS or len(literal.arguments) != 1:
            continue
        f = literal.arguments[0]
        if not isinstance(f, Expr) or f.is_number or f.free_symbols != e.free_symbols:
            continue
        ratio = cancel(e / f)
        if not isinstance(ratio, Rational) or ratio.p == 0:
            continue
        sign = known if ratio.p > 0 else _SIGN_FLIPS[known]
        answer = _SIGN_IMPLIES[sign].get(name)
        if answer is not None:
            return answer
    return None


#: the depth of the nested questions of the handlers
_depth: list[int] = [0]
#: the deepest nesting: past it the handlers give up
_MAX_DEPTH: Final[int] = 40


def handler_ask(proposition: Boolean, assumptions: Boolean) -> Truth:
    """The answer of the handler of the predicate for the class of its
    argument (``None`` without one, or when it does not decide).

    Examples
    ========

    >>> from sympy_extras.core import Q, Symbol, I
    >>> from sympy_extras.functions import Abs
    >>> from sympy_extras.assumptions.handlers import handler_ask
    >>> x = Symbol('x')
    >>> handler_ask(Q.real(Abs(x)), Q.real(x)), handler_ask(Q.real(I*x), Q.real(x) & Q.nonzero(x))
    (True, False)
    """
    if not isinstance(proposition, AppliedPredicate) or not proposition.function._defined:
        return None
    handler = _HANDLERS.get(proposition.function.name)
    args = proposition.arguments
    if handler is None or len(args) != 1:
        return None
    arg = args[0]
    if not isinstance(arg, Expr) or _depth[0] >= _MAX_DEPTH:
        return None
    _depth[0] += 1
    try:
        return handler(arg, assumptions)
    finally:
        _depth[0] -= 1


# PORTED FROM SYMPY: sympy/assumptions/ask.py:_ask (the handler and satask steps)
def final_ask(proposition: Boolean, assumptions: Boolean) -> Truth:
    """The last steps of the core's ``ask``: the handlers of the class of
    the argument, then the consistency of the proposition and of its
    negation with the assumptions and the known facts (SymPy's
    ``satask``).

    Examples
    ========

    >>> from sympy_extras.core import Q, Symbol
    >>> from sympy_extras.functions import Abs
    >>> from sympy_extras.assumptions.handlers import final_ask
    >>> x = Symbol('x')
    >>> final_ask(Q.zero(Abs(x)), Q.zero(x)), final_ask(Q.positive(x), Q.real(x))
    (True, None)
    """
    from .satask import predicates_consistent
    answer = handler_ask(proposition, assumptions)
    if answer is not None:
        return answer
    answer = _scaled_literal(proposition, assumptions)
    if answer is not None:
        return answer
    # PORTED FROM SYMPY: sympy/assumptions/satask.py:satask, check_satisfiability
    can_be_true = predicates_consistent(and_(assumptions, proposition))
    can_be_false = predicates_consistent(and_(assumptions, not_(proposition)))
    if can_be_true and can_be_false:
        return None
    if can_be_true:
        return True
    if can_be_false:
        return False
    if assumptions == true:
        return None
    raise ValueError("Inconsistent assumptions")
