r"""Printing in interactive sessions: ``init_printing`` and the
representations the notebooks ask for.

The native objects have a ``_repr_latex_`` method, which Jupyter calls to
show a result: it returns what :func:`repr_latex` gives, the LaTeX of the
object in display style with the settings chosen by :func:`init_printing`
(``None``, so that the notebook falls back on the text, once LaTeX is
switched off).

In a plain Python session :func:`init_printing` installs a display hook
which prints the results with the text printer: the string form of
:func:`~sympy_extras.core.printing.sstr`, or a two-dimensional printer
once one is plugged in with :func:`set_pretty_printer` (the hook the
pretty printer of this package uses).

Unlike SymPy's ``init_printing`` nothing of IPython is imported or
configured: the notebook finds the ``_repr_latex_`` methods by itself.

References
==========

.. [1] SymPy Development Team, *SymPy documentation: Printing*,
       https://docs.sympy.org/latest/modules/interactive.html
.. [2] IPython Development Team, *Integrating your objects with IPython*,
       https://ipython.readthedocs.io/en/stable/config/integrating.html
"""
from __future__ import annotations

import builtins
import sys
from dataclasses import dataclass
from types import TracebackType
from typing import Callable, Mapping, Optional

from sympy_extras.core.basic import Basic
from sympy_extras.core.symbol import Symbol
from sympy_extras.matrices.dense import ExprMatrix
from sympy_extras.polys.polytools import Poly

from .latex import LatexInput, LatexPrinter, LatexSettings

__all__ = ['init_printing', 'reset_printing', 'set_pretty_printer', 'pretty_printer', 'text_form', 'repr_latex',
           'printing_state', 'printing', 'PrintingState', 'PrettyPrinter']

#: a two-dimensional text printer: the object, whether Unicode may be used
PrettyPrinter = Callable[[LatexInput, bool], str]


@dataclass(frozen=True)
class PrintingState:
    """How the results are shown: what :func:`init_printing` chose."""

    #: ``_repr_latex_`` returns LaTeX
    use_latex: bool = True
    #: the results are shown with the pretty printer, when there is one
    pretty_print: bool = True
    #: the pretty printer may use Unicode
    use_unicode: bool = True
    #: the settings of the LaTeX of ``_repr_latex_``
    latex_settings: LatexSettings = LatexSettings()
    #: the LaTeX wrapped for the notebook (``%s`` is the formula)
    latex_wrapper: str = r'$\displaystyle %s$'


class _Session:
    """The state of the session (a single instance, ``_SESSION``)."""

    def __init__(self) -> None:
        self.state: PrintingState = PrintingState()
        self.pretty: Optional[PrettyPrinter] = None
        self.previous_hook: Optional[Callable[[object], None]] = None
        self.hook: Optional[Callable[[object], None]] = None


_SESSION = _Session()


def printing_state() -> PrintingState:
    """The current state of the interactive printing.

    Examples
    ========

    >>> from sympy_extras.printing.interactive import printing_state
    >>> printing_state().use_latex, printing_state().latex_settings.mode
    (True, 'plain')
    """
    return _SESSION.state


def set_pretty_printer(printer: Optional[PrettyPrinter]) -> None:
    """Plug in the two-dimensional text printer which
    :func:`init_printing` shows the results with: a function of the
    object and of whether Unicode may be used. ``None`` removes it.

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.printing.interactive import set_pretty_printer, text_form
    >>> x = symbols('x')
    >>> text_form(x**2)
    x**2
    >>> set_pretty_printer(lambda e, use_unicode: 'pretty(%s)' % (e,))
    >>> text_form(x**2)
    pretty(x**2)
    >>> set_pretty_printer(None)
    """
    _SESSION.pretty = printer


def pretty_printer() -> Optional[PrettyPrinter]:
    """The pretty printer plugged in, ``None`` when there is none."""
    return _SESSION.pretty


def text_form(obj: LatexInput) -> str:
    """The text a result is shown with: the form of the pretty printer
    when one is plugged in and ``pretty_print`` is on, ``str`` otherwise."""
    state = _SESSION.state
    if state.pretty_print and _SESSION.pretty is not None:
        return _SESSION.pretty(obj, state.use_unicode)
    return str(obj)


def repr_latex(obj: LatexInput) -> Optional[str]:
    r"""What the ``_repr_latex_`` methods of the native objects return:
    the LaTeX of ``obj`` wrapped for the notebook, ``None`` when LaTeX is
    switched off.

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.printing.interactive import repr_latex
    >>> x = symbols('x')
    >>> print(repr_latex(x**2/2)), print((x**2/2)._repr_latex_())
    $\displaystyle \frac{x^{2}}{2}$
    $\displaystyle \frac{x^{2}}{2}$
    (None, None)
    """
    state = _SESSION.state
    if not state.use_latex:
        return None
    return state.latex_wrapper % LatexPrinter(state.latex_settings).doprint(obj)


def _display(value: object) -> None:
    """The display hook of a plain Python session: the native objects
    with the text printer, the others as Python shows them."""
    if value is None:
        return
    setattr(builtins, '_', None)
    if isinstance(value, (Basic, ExprMatrix, Poly)):
        print(text_form(value))
    else:
        print(repr(value))
    setattr(builtins, '_', value)


def init_printing(*, pretty_print: bool = True, use_latex: bool = True, use_unicode: bool = True,
                  display_hook: Optional[bool] = None, latex_mode: str = 'plain',
                  latex_wrapper: Optional[str] = None, fold_frac_powers: bool = False,
                  fold_func_brackets: bool = False, fold_short_frac: Optional[bool] = None,
                  inv_trig_style: str = 'abbreviated', ln_notation: bool = False,
                  long_frac_ratio: Optional[float] = None, mat_delim: Optional[str] = '[',
                  mat_str: Optional[str] = None, mul_symbol: Optional[str] = None, order: Optional[str] = None,
                  symbol_names: Optional[Mapping[Symbol, str]] = None, root_notation: bool = True,
                  imaginary_unit: str = 'i', gothic_re_im: bool = False, decimal_separator: str = 'period',
                  parenthesize_super: bool = True, min: Optional[int] = None, max: Optional[int] = None,
                  full_prec: bool = False, diff_operator: str = 'd') -> None:
    r"""Choose how the native objects are shown in an interactive session.

    Parameters
    ==========

    pretty_print : show the results with the pretty printer plugged in
        by :func:`set_pretty_printer` (the string form when there is none)
    use_latex : whether ``_repr_latex_`` returns LaTeX, which a notebook
        renders; ``False`` leaves the notebook with the text
    use_unicode : whether the pretty printer may use Unicode
    display_hook : install the display hook of a plain Python session;
        by default only in an interactive session which is not IPython
    latex_mode : the mode of the LaTeX ('plain', 'inline', 'equation',
        'equation*'); the plain LaTeX is wrapped in ``latex_wrapper``
    latex_wrapper : what the plain LaTeX is wrapped in for the notebook,
        with ``%s`` for the formula (default ``$\displaystyle %s$``)
    fold_frac_powers, ..., diff_operator : the settings of
        :func:`~sympy_extras.printing.latex.latex`

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.printing import init_printing
    >>> from sympy_extras.printing.interactive import reset_printing
    >>> x = symbols('x')
    >>> init_printing(mul_symbol='dot', display_hook=False)
    >>> print((2*x)._repr_latex_())
    $\displaystyle 2 \cdot x$
    >>> init_printing(use_latex=False, display_hook=False)
    >>> print((2*x)._repr_latex_())
    None
    >>> reset_printing()
    """
    settings = LatexSettings(mode=latex_mode, fold_frac_powers=fold_frac_powers,
                             fold_func_brackets=fold_func_brackets, fold_short_frac=fold_short_frac,
                             inv_trig_style=inv_trig_style, ln_notation=ln_notation, long_frac_ratio=long_frac_ratio,
                             mat_delim=mat_delim, mat_str=mat_str, mul_symbol=mul_symbol, order=order,
                             symbol_names={} if symbol_names is None else symbol_names,
                             root_notation=root_notation, imaginary_unit=imaginary_unit, gothic_re_im=gothic_re_im,
                             decimal_separator=decimal_separator, parenthesize_super=parenthesize_super, min=min,
                             max=max, full_prec=full_prec, diff_operator=diff_operator)
    LatexPrinter(settings)  # the settings are checked now, not at the first result
    wrapper = latex_wrapper
    if wrapper is None:
        wrapper = r'$\displaystyle %s$' if latex_mode == 'plain' else '%s'
    _SESSION.state = PrintingState(use_latex=use_latex, pretty_print=pretty_print, use_unicode=use_unicode,
                                   latex_settings=settings, latex_wrapper=wrapper)
    install = display_hook
    if install is None:
        install = _interactive() and 'IPython' not in sys.modules
    if install:
        _install_hook()


def _interactive() -> bool:
    """Whether Python runs an interactive session (``sys.ps1`` is set)."""
    return 'ps1' in vars(sys)


def _install_hook() -> None:
    if _SESSION.hook is not None and sys.displayhook is _SESSION.hook:
        return
    _SESSION.previous_hook = sys.displayhook
    _SESSION.hook = _display
    sys.displayhook = _display


def reset_printing() -> None:
    """Undo :func:`init_printing`: the default state, the display hook
    which was there before."""
    _SESSION.state = PrintingState()
    if _SESSION.hook is not None and sys.displayhook is _SESSION.hook and _SESSION.previous_hook is not None:
        sys.displayhook = _SESSION.previous_hook
    _SESSION.hook = None
    _SESSION.previous_hook = None


class printing:
    r"""A context manager: the settings of :func:`init_printing` inside
    the block, the previous state after it.

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.printing.interactive import PrintingState, printing
    >>> x = symbols('x')
    >>> with printing(PrintingState(use_latex=False)):
    ...     print((2*x)._repr_latex_())
    None
    >>> print((2*x)._repr_latex_())
    $\displaystyle 2 x$
    """

    def __init__(self, state: PrintingState) -> None:
        self.state: PrintingState = state
        self.previous: PrintingState = _SESSION.state

    def __enter__(self) -> PrintingState:
        self.previous = _SESSION.state
        _SESSION.state = self.state
        return self.state

    def __exit__(self, kind: Optional[type[BaseException]], value: Optional[BaseException],
                 traceback: Optional[TracebackType]) -> None:
        _SESSION.state = self.previous
