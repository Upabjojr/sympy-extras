"""Numerical values which do not depend on a rounding error.

The numerical checks of the package evaluate constants numerically: the
value of an integral against a quadrature, a difference at a sample
point, the residual of an equation at a candidate. Two things make such
a value a matter of luck:

* a sum which is exactly zero (*a hidden zero*) evaluates to a rounding
  error of either sign (or to an exact ``0.0``), and what is built on it
  carries the precision asked for: for ``z = -6 + (-2 + sqrt(2))**2 +
  4*sqrt(2)``, which is 0, ``sqrt(z)`` evaluates to a number of the size
  of the rounding, real at some precisions and imaginary at others;
* the argument of a logarithm, of a fractional power or of an inverse
  function whose real or imaginary part is exactly zero and evaluates to
  a rounding error lies on an axis, where the cuts are, on the side the
  rounding chooses: ``log(-2*sqrt(2) - 2*sqrt(z))`` is ``log(2*sqrt(2))``
  plus or minus ``I*pi``.

A wrong value passed a check this way (sympy-extras#65), and a right
identity was refuted (the zero test of ``atan(x)**(1/3)`` against its
form in logarithms). :func:`reliable_form` writes the hidden zeros ``0``
where they are proved and refuses (``None``) the constants whose value
still hangs on a rounding; :func:`reliable_value` evaluates the others.

The module works on the expressions of the typed core
(:mod:`sympy_extras.core`) with mpmath and imports no SymPy:
:mod:`sympy_extras._numeric_sympy` is its face on SymPy objects.

The rules
=========

The values are computed by :func:`~sympy_extras.core.evalf.numeric_value`,
mpmath at a fixed working precision, so a cancellation loses digits where
SymPy's ``evalf`` raises the precision. Hence:

* a part (real or imaginary) of a value is a *suspect* when it is not
  zero and negligible next to the scale of the terms, or when it is
  exactly zero while some term of the sum has that part (a hidden zero
  can cancel exactly in binary);
* a suspect is a *rounding error* unless two evaluations at successive
  higher precisions (15, 30 and 60 more digits) agree on it: a small
  value which is there (``exp(-200) + I``, or ``(1 + exp(-100)) - 1``,
  which needs more precision) is found again, a hidden zero changes with
  the precision;
* a sum whose part is a rounding error is written 0 only when that part
  is *proved* zero: by the core's ``is_zero``, or, for an algebraic
  number, by its minimal polynomial being ``x`` (the step by which SymPy's
  ``is_zero`` decides such numbers); otherwise the value is undecided;
* the value returned by :func:`reliable_value` is evaluated at doubling
  precisions until two evaluations agree to the digits asked for.
"""
from __future__ import annotations

from typing import Callable, Mapping, Optional

import mpmath

from sympy_extras._timeout import attempt
from sympy_extras.core.add import Add, add
from sympy_extras.core.basic import Basic, preorder_traversal
from sympy_extras.core.evalf import mpf_to_fraction, numeric_value
from sympy_extras.core.expr import Expr
from sympy_extras.core.function import Application
from sympy_extras.core.mul import mul
from sympy_extras.core.numbers import Float, ImaginaryUnit, Integer, Rational, dps_to_prec, number_from_fraction
from sympy_extras.core.power import Pow
from sympy_extras.core.symbol import Symbol
from sympy_extras.functions.elementary.complexes import Arg
from sympy_extras.functions.elementary.exponential import Log
from sympy_extras.functions.elementary.hyperbolic import Acosh, Acoth, Asinh, Atanh
from sympy_extras.functions.elementary.piecewise import Piecewise
from sympy_extras.functions.elementary.trigonometric import Acos, Acot, Asin, Atan

__all__ = ['reliable_form', 'reliable_value', 'reliable_complex']

#: the seconds given to prove one hidden zero
_PROOF_SECONDS = 2.0
#: the additional digits of the evaluations of a part which looks like a
#: rounding error
_MORE_DIGITS = (15, 30, 60)
#: the guard bits of every evaluation
_GUARD_BITS = 20
#: the doublings of the precision allowed for the final value
_DOUBLINGS = 2

_ANGULAR = (Log, Arg)
_INVERSE = (Asin, Acos, Atan, Acot, Asinh, Acosh, Atanh, Acoth)

#: the real and the imaginary part of a value
_Parts = tuple[mpmath.mpf, mpmath.mpf]


class _Undecided(Exception):
    """A sum without a significant digit which is not proved zero."""


def _number(e: Expr, prec: int) -> Optional[mpmath.mpc]:
    """The value of ``e`` at the working precision ``prec`` (bits), ``None``
    when it is not a finite number."""
    z = numeric_value(e, prec)
    if z is None or not mpmath.isfinite(z):
        return None
    return z


def _parts(e: Expr, digits: int) -> Optional[_Parts]:
    """The real and imaginary parts of the value with ``digits`` digits."""
    z = _number(e, dps_to_prec(digits) + _GUARD_BITS)
    if z is None:
        return None
    return mpmath.mpf(z.real), mpmath.mpf(z.imag)


def _size(x: mpmath.mpf) -> float:
    return float(abs(x))


def _is_function(node: Basic) -> bool:
    """A function application (SymPy's ``Function``, ``Piecewise`` among
    them)."""
    return isinstance(node, (Application, Piecewise))


def _noise(e: Expr, digits: int, scale: float, parts: _Parts,
           structural: tuple[bool, bool]) -> Optional[tuple[bool, bool]]:
    """Whether the real and the imaginary part of the value of ``e`` are
    rounding errors. A part is a suspect when it is not zero and
    negligible next to ``scale``, or zero while ``structural`` says that
    the terms have such a part; a suspect is a rounding error unless two
    evaluations at successive higher precisions agree on it (the
    imaginary part of ``I*(log(1 + 3*I) - log(1 - 3*I))``, which is zero,
    changes with the precision; a small part which is there,
    ``exp(-200) + I``, does not). ``None`` when a value at a higher
    precision cannot be computed."""
    threshold = 10.0**(-(digits * 3 // 5)) * scale
    suspects = [(part != 0 and _size(part) < threshold) or (part == 0 and has_part)
                for part, has_part in zip(parts, structural)]
    if not any(suspects):
        return False, False
    genuine = [False, False]
    previous = parts
    for more in _MORE_DIGITS:
        again = _parts(e, digits + more)
        if again is None:
            return None
        for k in (0, 1):
            if suspects[k] and not genuine[k]:
                first, second = previous[k], again[k]
                if first != 0 and second != 0 and _size(first - second) <= 1e-5 * _size(second):
                    genuine[k] = True
        if all(genuine[k] or not suspects[k] for k in (0, 1)):
            break
        previous = again
    return suspects[0] and not genuine[0], suspects[1] and not genuine[1]


def _has_foreign(e: Basic) -> bool:
    """Whether ``e`` contains a wrapped object of a foreign system (SymPy's
    ``CRootOf``, ...)."""
    return any(node._is_foreign_node() for node in preorder_traversal(e))


def _is_constant(e: Expr) -> bool:
    """Whether ``e`` is a number: the core's ``is_number``, or no symbol in
    an expression with wrapped foreign objects (whose ``is_number`` the
    core does not know)."""
    return e.is_number or (not e.free_symbols and _has_foreign(e))


def _foreign_minimal_polynomial_is_x(part: Expr) -> bool:
    """Whether the foreign system's minimal polynomial of ``part`` is ``x``
    (SymPy's ``minimal_polynomial`` knows ``CRootOf``, the native one
    does not); ``False`` when it cannot tell."""
    from sympy_extras.core.bridge import bridge
    x = Symbol('x')

    def foreign() -> bool:
        try:
            return bridge().minimal_polynomial(part, x) == x
        except Exception:
            # any error of the foreign system (not algebraic, not
            # implemented, not available) proves nothing
            return False

    return attempt(foreign, _PROOF_SECONDS) is True


def _proved_zero(part: Expr) -> bool:
    """Whether ``part`` is proved zero: by the core's ``is_zero`` (its
    ``is_positive`` and ``is_negative`` are not asked: with a hidden zero
    inside they would be guesses, sympy-extras#25), or, for an algebraic
    number without functions, by its minimal polynomial ``x`` (the step by
    which SymPy's ``is_zero`` decides such numbers; through the foreign
    system for a number with wrapped foreign objects)."""
    if part.is_zero is True:
        return True
    if part.is_zero is False or part.free_symbols or part.has(Application, Piecewise):
        return False
    from sympy_extras.polys.numberfields.minpoly import minimal_polynomial_coeffs
    from sympy_extras.polys.polyerrors import NotAlgebraic
    try:
        return minimal_polynomial_coeffs(part) == [1, 0]
    except (NotAlgebraic, NotImplementedError):
        return _has_foreign(part) and _foreign_minimal_polynomial_is_x(part)


def _structural(terms: tuple[Expr, ...], digits: int) -> Optional[tuple[tuple[bool, bool], float]]:
    """Whether some term has a real part, and an imaginary part, which is
    not exactly zero, and the size of the largest term; ``None`` when a
    term has no value."""
    has_real, has_imaginary, scale = False, False, 0.0
    for term in terms:
        parts = _parts(term, digits)
        if parts is None:
            return None
        has_real = has_real or parts[0] != 0
        has_imaginary = has_imaginary or parts[1] != 0
        scale = max(scale, abs(complex(parts[0], parts[1])))
    return (has_real, has_imaginary), scale


def _resolved_sum(node: Add, digits: int) -> Expr:
    """The sum with its parts which are hidden zeros written 0."""
    parts = _parts(node, digits)
    if parts is None:
        return node
    found = _structural(node.args, digits)
    if found is None:
        raise _Undecided
    structural, scale = found
    noise = _noise(node, digits, scale, parts, structural)
    if noise is None:
        raise _Undecided
    if not any(noise):
        return node
    if parts[1] == 0 and node.is_extended_real:
        exact: tuple[Expr, Expr] = (node, Integer(0))
    else:
        split = attempt(lambda: node.as_real_imag(), _PROOF_SECONDS, native=True)
        if split is None:
            raise _Undecided
        exact = split
    kept: list[Expr] = []
    for noisy, part in zip(noise, exact):
        if noisy:
            proof: Callable[[], bool] = lambda: _proved_zero(part)
            if attempt(proof, _PROOF_SECONDS, native=True) is not True:
                raise _Undecided
            kept.append(Integer(0))
        else:
            kept.append(part)
    return add(kept[0], mul(ImaginaryUnit(), kept[1]))


def _on_a_cut(e: Expr, digits: int) -> bool:
    """Whether a constant argument of a logarithm, of a fractional power or
    of an inverse function in ``e`` has a part which is a rounding
    error and decides the branch: the imaginary part of the argument of a
    logarithm or of the base of a power whose real part is not clearly
    positive (or the real part when the imaginary one is not clearly
    there); any part for the inverse functions, whose cuts lie on both
    axes."""
    stack: list[Basic] = [e]
    while stack:
        node = stack.pop()
        stack.extend(node.args)
        argument: Expr
        if isinstance(node, Pow):
            if node.exp.is_integer:
                continue
            argument, angular = node.base, True
        elif isinstance(node, _ANGULAR) and isinstance(node.args[0], Expr):
            argument, angular = node.args[0], True
        elif isinstance(node, _INVERSE) and isinstance(node.args[0], Expr):
            argument, angular = node.args[0], False
        else:
            continue
        if not _is_constant(argument) or isinstance(argument, Rational):
            continue
        parts = _parts(argument, digits)
        if parts is None:
            return True
        structural = (False, False)
        if isinstance(argument, Add):
            found = _structural(argument.args, digits)
            if found is None:
                return True
            structural = found[0]
        noise = _noise(argument, digits, _size(parts[0]) + _size(parts[1]), parts, structural)
        if noise is None:
            return True
        if not any(noise):
            continue
        if not angular or all(noise):
            return True
        if noise[1] and not parts[0] > 0:
            return True
        if noise[0] and parts[1] == 0:
            return True
    return False


def _walk(node: Expr, exposed: bool, digits: int) -> Expr:
    """``node`` with the constant sums under a function or a fractional
    power resolved (:func:`_resolved_sum`), rebuilt with the evaluating
    functions where an argument changed."""
    if not node.args:
        return node
    inside = exposed or _is_function(node) or (isinstance(node, Pow) and not node.exp.is_integer)
    changed = False
    args: list[Basic] = []
    for argument in node.args:
        new = _walk(argument, inside, digits) if isinstance(argument, Expr) else argument
        changed = changed or new is not argument
        args.append(new)
    rebuilt = node
    if changed:
        evaluated = node._evaluated(tuple(args))
        if not isinstance(evaluated, Expr):
            raise TypeError("expected an expression, got %s" % (evaluated,))
        rebuilt = evaluated
    if exposed and isinstance(rebuilt, Add) and _is_constant(rebuilt):
        return _resolved_sum(rebuilt, digits)
    return rebuilt


def reliable_form(e: Expr, digits: int = 20) -> Optional[Expr]:
    """The expression ``e`` with its constant sums which are exactly zero
    written 0, or ``None`` when the numerical value of a constant in it
    depends on a rounding error: a sum without a significant digit which
    is not proved zero, or an argument on a cut on a side which the
    rounding chooses.

    Only the sums under a function or a fractional power are looked at:
    a value which is itself zero (an area at the sample where the region
    is empty, ``asin(-1) + pi/2``) is a sum without a significant digit
    too, and harmless. The sums of an expression with floating point
    numbers are left alone (there is nothing exact to decide).

    Examples
    ========

    >>> from sympy_extras.core import sqrt
    >>> from sympy_extras.functions import Log, sin, cos
    >>> from sympy_extras._numeric import reliable_form
    >>> z = -6 + (-2 + sqrt(2))**2 + 4*sqrt(2)
    >>> reliable_form(Log(-2*sqrt(2) - 2*sqrt(z)))
    log(2*sqrt(2)) + I*pi
    >>> reliable_form(sqrt(sin(1)**2 + cos(1)**2 - 1) + 1) is None
    True
    >>> reliable_form(sin(1)**2 + cos(1)**2 - 1) is None
    False
    """
    try:
        resolved = e if e.has(Float) else _walk(e, False, digits)
        if _on_a_cut(resolved, digits):
            return None
    except (_Undecided, RecursionError, TypeError, ValueError, ArithmeticError):
        # a value which cannot be computed (a hypergeometric series at a
        # pole) is not reliable either
        return None
    return resolved


def _float(x: mpmath.mpf, prec: int) -> Float:
    return Float(number_from_fraction(mpf_to_fraction(x)), precision=prec)


def _as_expr(z: mpmath.mpc, prec: int) -> Expr:
    """``z`` as ``re + im*I`` with floats of precision ``prec`` (a real
    float alone when the imaginary part is zero)."""
    with mpmath.workprec(prec):
        real, imaginary = mpmath.mpf(z.real), mpmath.mpf(z.imag)
    if imaginary == 0:
        return _float(real, prec)
    term = mul(_float(imaginary, prec), ImaginaryUnit())
    if real == 0:
        return term
    return add(_float(real, prec), term)


def _accurate(e: Expr, digits: int) -> Optional[mpmath.mpc]:
    """The value of ``e`` to ``digits`` digits: evaluated at doubling
    precisions until two evaluations agree (the last one when they never
    do, a value without significant digits, as SymPy's ``evalf`` returns
    it); ``None`` when it is not a finite number."""
    target = dps_to_prec(digits)
    prec = target + _GUARD_BITS
    value = _number(e, prec)
    if value is None:
        return None
    for _ in range(_DOUBLINGS):
        prec *= 2
        again = _number(e, prec)
        if again is None:
            return None
        if abs(again - value) <= mpmath.ldexp(abs(again), 4 - target):
            return again
        value = again
    return value


def reliable_value(e: Expr, digits: int, values: Optional[Mapping[Symbol, Expr]] = None) -> Optional[Expr]:
    """The numerical value with ``digits`` digits of ``e`` at the
    ``values`` of its symbols, or ``None`` when it depends on a rounding
    error (see :func:`reliable_form`) or is not a finite number. The
    hidden zeros are written 0 before the substitution too, which
    evaluates the functions again (the logarithm of a sum with a hidden
    zero inside can evaluate to the wrong branch).

    Examples
    ========

    >>> from sympy_extras.core import Integer, sqrt, symbols
    >>> from sympy_extras.functions import Log, sin, cos, log
    >>> from sympy_extras._numeric import reliable_value
    >>> x = symbols('x')
    >>> z = -6 + (-2 + sqrt(2))**2 + 4*sqrt(2)
    >>> reliable_value(Log(-2*sqrt(2) - 2*sqrt(z)), 15)
    1.03972077083992 + 3.14159265358979*I
    >>> reliable_value(Log(-x - sqrt(z)), 15, {x: Integer(2)})
    0.693147180559945 + 3.14159265358979*I
    >>> reliable_value(log(-1 - sqrt(sin(1)**2 + cos(1)**2 - 1)), 15) is None
    True
    """
    form = reliable_form(e, digits)
    if form is not None and values:
        rule: dict[Basic, Basic] = {s: v for s, v in values.items()}
        try:
            substituted = form._xreplace_evaluated(rule)
        except (RecursionError, TypeError, ValueError, ArithmeticError):
            return None
        if not isinstance(substituted, Expr):
            return None
        form = reliable_form(substituted, digits)
    if form is None or form.free_symbols:
        return None
    try:
        z = _accurate(form, digits)
    except (RecursionError, TypeError, ValueError, ArithmeticError):
        return None
    if z is None:
        return None
    return _as_expr(z, dps_to_prec(digits))


def reliable_complex(e: Expr, digits: int, values: Optional[Mapping[Symbol, Expr]] = None) -> Optional[complex]:
    """:func:`reliable_value` as a Python complex number, ``None`` when it
    is not reliable or does not fit in a float.

    >>> from sympy_extras.core import Integer, sqrt, symbols
    >>> from sympy_extras._numeric import reliable_complex
    >>> x = symbols('x')
    >>> reliable_complex(sqrt(x), 15, {x: Integer(-4)})
    2j
    """
    value = reliable_value(e, digits, values)
    if value is None:
        return None
    z = numeric_value(value, dps_to_prec(digits) + _GUARD_BITS)
    if z is None:
        return None
    try:
        return complex(z)
    except (TypeError, ValueError, OverflowError):
        return None
