"""``init_printing``, the ``_repr_latex_`` of the native objects and the
hook of the pretty printer."""
from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout

from sympy.testing.pytest import raises

from sympy_extras.core import Integer, Symbol
from sympy_extras.matrices.dense import ExprMatrix
from sympy_extras.polys.polytools import Poly
from sympy_extras.printing import init_printing
from sympy_extras.printing.latex import LatexInput
from sympy_extras.printing.interactive import (pretty_printer, printing_state, repr_latex,
    reset_printing, set_pretty_printer, text_form)

x, y = Symbol('x'), Symbol('y')


def _pretty(e: LatexInput, use_unicode: bool) -> str:
    return ('unicode ' if use_unicode else 'ascii ') + str(e)


def test_repr_latex_of_the_native_objects() -> None:
    reset_printing()
    assert (x**2/2)._repr_latex_() == r'$\displaystyle \frac{x^{2}}{2}$'
    assert (x < y)._repr_latex_() == r'$\displaystyle x < y$'
    assert ExprMatrix.from_rows([[x, Integer(1)]])._repr_latex_() == \
        r'$\displaystyle \left[\begin{matrix}x & 1\end{matrix}\right]$'
    assert Poly(x + 1, x)._repr_latex_() == \
        r'$\displaystyle \operatorname{Poly}{\left( x + 1, x, domain=\mathbb{Z} \right)}$'


def test_init_printing_sets_the_latex() -> None:
    try:
        init_printing(mul_symbol='dot', fold_frac_powers=True, display_hook=False)
        assert (2*x)._repr_latex_() == r'$\displaystyle 2 \cdot x$'
        init_printing(latex_mode='equation*', display_hook=False)
        assert (2*x)._repr_latex_() == r'\begin{equation*}2 x\end{equation*}'
        init_printing(use_latex=False, display_hook=False)
        assert (2*x)._repr_latex_() is None and repr_latex(x) is None
        raises(ValueError, lambda: init_printing(latex_mode='wrong', display_hook=False))
        assert printing_state().use_latex is False  # a refused call changes nothing
    finally:
        reset_printing()
    assert (2*x)._repr_latex_() == r'$\displaystyle 2 x$'


def test_pretty_printer_hook() -> None:
    previous = sys.displayhook
    try:
        assert pretty_printer() is None and text_form(x + 1) == 'x + 1'
        set_pretty_printer(_pretty)
        init_printing(display_hook=True, use_unicode=False)
        assert sys.displayhook is not previous
        assert text_form(x + 1) == 'ascii x + 1'
        init_printing(display_hook=True, pretty_print=False)
        assert text_form(x + 1) == 'x + 1'
    finally:
        set_pretty_printer(None)
        reset_printing()
    assert sys.displayhook is previous and pretty_printer() is None


def test_display_hook_prints_the_native_objects() -> None:
    previous = sys.displayhook
    out = io.StringIO()
    try:
        set_pretty_printer(_pretty)
        init_printing(display_hook=True)
        with redirect_stdout(out):
            sys.displayhook(x + 1)
            sys.displayhook(3)
            sys.displayhook(None)
    finally:
        set_pretty_printer(None)
        reset_printing()
    assert out.getvalue() == 'unicode x + 1\n3\n'
    assert sys.displayhook is previous
