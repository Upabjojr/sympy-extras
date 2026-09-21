"""Numerical values which do not depend on a rounding error.

The numerical checks of the package evaluate constants with ``evalf``:
the value of an integral against a quadrature, a difference at a sample
point, the residual of an equation at a candidate. Two things make such
a value a matter of luck:

* a sum which is exactly zero (*a hidden zero*) evaluates to a rounding
  error of either sign, and what is built on it carries the precision
  ``evalf`` asked for: for ``z = -6 + (-2 + sqrt(2))**2 + 4*sqrt(2)``,
  which is 0, ``sqrt(z).evalf(20)`` is ``7.07...e-74`` with twenty digits,
  real at some precisions and imaginary at others;
* the argument of a logarithm, of a fractional power or of an inverse
  function whose real or imaginary part is exactly zero and evaluates to
  a rounding error lies on an axis, where the cuts are, on the side the
  rounding chooses: ``log(-2*sqrt(2) - 2*sqrt(z))`` is ``log(2*sqrt(2))``
  plus or minus ``I*pi``.

A wrong value passed a check this way (sympy-extras#65), and a right
identity was refuted (the zero test of ``atan(x)**(1/3)`` against its
form in logarithms). :func:`reliable_form` writes the hidden zeros ``0``
where SymPy proves them and refuses (``None``) the constants whose value
still hangs on a rounding; :func:`reliable_value` evaluates the others.
"""
from __future__ import annotations

from typing import Mapping, Optional

from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.function import Function
from sympy.core.numbers import Float, I
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.complexes import arg
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.hyperbolic import acosh, acoth, asinh, atanh
from sympy.functions.elementary.trigonometric import acos, acot, asin, atan

from sympy_extras._timeout import attempt
from sympy_extras._typing import as_expr

__all__ = ['reliable_form', 'reliable_value']

#: the seconds given to SymPy to prove one hidden zero
_PROOF_SECONDS = 2.0
#: the additional digits of the second evaluation of a part which looks
#: like a rounding error
_MORE_DIGITS = 15

_ANGULAR = (log, arg)
_INVERSE = (asin, acos, atan, acot, asinh, acosh, atanh, acoth)


def _parts(e: Expr, digits: int) -> Optional[tuple[Expr, Expr]]:
    """The real and imaginary parts of the numerical value."""
    value = as_expr(e.evalf(digits))
    real_part, imaginary_part = value.as_real_imag()
    if not (real_part.is_number and imaginary_part.is_number
            and real_part.is_finite and imaginary_part.is_finite):
        return None
    return as_expr(real_part), as_expr(imaginary_part)


def _size(part: Expr) -> float:
    return abs(complex(part))


def _noise(e: Expr, digits: int, scale: float, parts: tuple[Expr, Expr]) -> Optional[tuple[bool, bool]]:
    """Whether the real and the imaginary part of the value of ``e`` are
    rounding errors: not zero, negligible next to ``scale``, and either
    without a significant digit or different at a higher precision
    (``evalf`` gives the imaginary part of ``I*(log(1 + 3*I) - log(1 -
    3*I))``, which is zero, as ``5.7e-29`` with twenty digits and as
    ``-1.3e-48`` with forty; a small part which is there, ``exp(-200) +
    I``, has the same digits at both)."""
    threshold = 10.0**(-(digits * 3 // 5)) * scale
    suspects = [part != 0 and _size(part) < threshold for part in parts]
    if not any(suspects):
        return False, False
    again = _parts(e, digits + _MORE_DIGITS)
    if again is None:
        return None
    found: list[bool] = []
    for suspect, first, second in zip(suspects, parts, again):
        if not suspect:
            found.append(False)
        elif isinstance(first, Float) and first._prec <= 1:
            found.append(True)
        else:
            found.append(_size(as_expr(first - second)) > 1e-5 * _size(first))
    return found[0], found[1]


class _Undecided(Exception):
    """A sum without a significant digit which is not proved zero."""


def _resolved_sum(node: Expr, digits: int) -> Expr:
    parts = _parts(node, digits)
    if parts is None:
        return node
    scale = max(_size(as_expr(as_expr(term).evalf(digits))) for term in node.args)
    noise = _noise(node, digits, scale, parts)
    if noise is None:
        raise _Undecided
    if not any(noise):
        return node
    if parts[1] == 0 and node.is_extended_real:
        exact: tuple[Expr, Expr] = (node, S.Zero)
    else:
        split = attempt(lambda: node.as_real_imag(), _PROOF_SECONDS)
        if split is None:
            raise _Undecided
        exact = (as_expr(split[0]), as_expr(split[1]))
    kept: list[Expr] = []
    for noisy, part in zip(noise, exact):
        if noisy:
            # SymPy's is_zero alone: is_positive and is_negative of a sum
            # with a hidden zero inside are random (sympy-extras#25); a
            # part without a digit which is "not zero" is not believed
            if attempt(lambda: part.is_zero, _PROOF_SECONDS) is not True:
                raise _Undecided
            kept.append(S.Zero)
        else:
            kept.append(part)
    return as_expr(kept[0] + I * kept[1])


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
        if isinstance(node, Pow):
            if node.exp.is_integer:
                continue
            argument, angular = as_expr(node.base), True
        elif isinstance(node, _ANGULAR):
            argument, angular = as_expr(node.args[0]), True
        elif isinstance(node, _INVERSE):
            argument, angular = as_expr(node.args[0]), False
        else:
            continue
        if not argument.is_number or argument.is_Rational:
            continue
        parts = _parts(argument, digits)
        if parts is None:
            return True
        noise = _noise(argument, digits, _size(parts[0]) + _size(parts[1]), parts)
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


def reliable_form(e: Expr, digits: int = 20) -> Optional[Expr]:
    """The expression ``e`` with its constant sums which are exactly zero
    written 0, or ``None`` when the numerical value of a constant in it
    depends on a rounding error: a sum without a significant digit which
    SymPy does not prove zero, or an argument on a cut on a side which the
    rounding chooses.

    Only the sums under a function or a fractional power are looked at:
    a value which is itself zero (an area at the sample where the region
    is empty, ``asin(-1) + pi/2``) is a sum without a significant digit
    too, and harmless. The sums of an expression with floating point
    numbers are left alone (there is nothing exact to decide).

    Examples
    ========

    >>> from sympy import log, sqrt, sin, cos
    >>> from sympy_extras._numeric import reliable_form
    >>> z = -6 + (-2 + sqrt(2))**2 + 4*sqrt(2)
    >>> reliable_form(log(-2*sqrt(2) - 2*sqrt(z), evaluate=False))
    log(2*sqrt(2)) + I*pi
    >>> reliable_form(sqrt(sin(1)**2 + cos(1)**2 - 1) + 1) is None
    True
    >>> reliable_form(sin(1)**2 + cos(1)**2 - 1) is None
    False
    """
    def walk(node: Basic, exposed: bool) -> Basic:
        if not node.args:
            return node
        inside = exposed or isinstance(node, Function) or (isinstance(node, Pow) and not node.exp.is_integer)
        args = [walk(argument, inside) for argument in node.args]
        rebuilt = node if all(new is old for new, old in zip(args, node.args)) else node.func(*args)
        if exposed and isinstance(rebuilt, Add) and rebuilt.is_number:
            return _resolved_sum(rebuilt, digits)
        return rebuilt

    try:
        resolved = e if e.has(Float) else as_expr(walk(e, False))
        if _on_a_cut(resolved, digits):
            return None
    except (_Undecided, RecursionError, TypeError, ValueError, ArithmeticError):
        # SymPy's log of a sum with a hidden zero inside recurses for ever
        # in some runs (sympy-extras#25); a value which cannot be computed
        # (a hypergeometric series at a pole) is not reliable either
        return None
    return resolved


def reliable_value(e: Expr, digits: int, values: Optional[Mapping[Symbol, Expr]] = None) -> Optional[Expr]:
    """The numerical value with ``digits`` digits of ``e`` at the
    ``values`` of its symbols, or ``None`` when it depends on a rounding
    error (see :func:`reliable_form`) or is not a finite number. The
    hidden zeros are written 0 before the substitution too, which
    evaluates the functions again (the bug: ``numerically_equal`` raised
    ``RecursionError`` in some runs, from SymPy's ``log`` of a sum with a
    hidden zero inside).

    Examples
    ========

    >>> from sympy import log, sqrt, sin, cos
    >>> from sympy.abc import x
    >>> from sympy_extras._numeric import reliable_value
    >>> z = -6 + (-2 + sqrt(2))**2 + 4*sqrt(2)
    >>> reliable_value(log(-2*sqrt(2) - 2*sqrt(z), evaluate=False), 15)
    1.03972077083992 + 3.14159265358979*I
    >>> reliable_value(log(-x - sqrt(z), evaluate=False), 15, {x: 2})
    0.693147180559945 + 3.14159265358979*I
    >>> reliable_value(log(-1 - sqrt(sin(1)**2 + cos(1)**2 - 1)), 15) is None
    True
    """
    form = reliable_form(e, digits)
    if form is not None and values:
        try:
            form = reliable_form(as_expr(form.xreplace(dict(values))), digits)
        except RecursionError:
            return None
    if form is None or form.free_symbols:
        return None
    try:
        value = as_expr(form.evalf(digits))
    except (TypeError, ValueError, ArithmeticError):
        return None
    if not value.is_number or value.is_finite is not True:
        return None
    return value
