"""The operations the core delegates to a foreign computer algebra system
until they are ported: an interface with the core's own types only.

The core never imports SymPy. :mod:`sympy_extras.interop` implements this
interface with SymPy and registers it when imported (``import
sympy_extras.interop``). Without it the queries of knowledge answer
"unknown" (``None``, no facts, no rule) and the operations which must
produce a result raise :class:`ForeignOperationUnavailable`, naming the
operation. Every method is a debt: the goal is to port each one and
remove it from here.

Examples
========

>>> from sympy_extras.core.bridge import Bridge, bridge, register_bridge
>>> isinstance(bridge(), Bridge)
True
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

import mpmath

# the core modules import this module inside their functions (as they
# imported the SymPy bridge before): the core types are loaded already
from .basic import Basic
from .expr import Expr
from .function import FunctionClass
from .logic import Boolean
from .symbol import Symbol

__all__ = ['Bridge', 'ForeignOperationUnavailable', 'bridge', 'register_bridge']

#: a truth value of three states
Truth = Optional[bool]
#: the target of a rewrite: a node, a function class or a class
RewriteTarget = Union[Basic, FunctionClass, type]


class ForeignOperationUnavailable(NotImplementedError):
    """The operation is not ported and no foreign system is registered."""


class Bridge:
    """The interface; every method raises until a system is registered."""

    def _missing(self, name: str) -> ForeignOperationUnavailable:
        return ForeignOperationUnavailable(
            '%s is not ported to the core: import sympy_extras.interop to use SymPy for it' % name)

    # -- conversion of foreign objects and strings

    def is_foreign(self, a: object) -> bool:
        """Whether ``a`` is an object of the foreign system (converted by
        :meth:`from_foreign`)."""
        return False

    def from_foreign(self, a: object) -> Basic:
        raise self._missing('the conversion of %r' % (type(a).__name__,))

    def parse(self, text: str, evaluate: bool) -> Basic:
        raise self._missing('parsing strings')

    # -- numbers and assumptions

    def numeric(self, e: Expr, dps: int) -> Optional[mpmath.mpc]:
        """The value of a number node at ``dps`` digits, ``None`` unknown."""
        return None

    def ask(self, proposition: Boolean, assumptions: Boolean) -> Truth:
        return None

    def equals_zero(self, e: Expr) -> Truth:
        return None

    def is_constant(self, e: Expr, wrt: Sequence[Symbol]) -> Truth:
        return None

    def is_hypergeometric(self, e: Expr, k: Symbol) -> Truth:
        return None

    def minimal_polynomial(self, e: Expr, x: Symbol) -> Expr:
        raise self._missing('minimal_polynomial')

    def degree(self, e: Expr, x: Expr) -> int:
        raise self._missing('degree')

    # -- operations on nodes whose rules are not ported

    def nseries(self, node: Expr, x: Symbol, n: int, logx: Optional[Expr], cdir: Expr) -> Expr:
        raise self._missing('the series of %s' % (type(node).__name__,))

    def meromorphic(self, node: Expr, x: Symbol, a: Expr, own_only: bool) -> tuple[bool, Truth]:
        """``(found, answer)``: whether the foreign class has a rule (with
        ``own_only``) and its answer."""
        return (False, None)

    def rewrite(self, e: Basic, targets: Sequence[RewriteTarget], piecewise: bool) -> Basic:
        raise self._missing('rewrite')

    def rewrite_node(self, e: Basic, args: tuple[Basic, ...], target: type, piecewise: bool) -> Optional[Basic]:
        """The node rewritten by the foreign class's rule, ``None`` without one."""
        return None

    def rewrite_named_node(self, e: Basic, args: tuple[Basic, ...], name: str, limitvar: Optional[Symbol]
                           ) -> Optional[Basic]:
        return None

    def expand_func_node(self, e: Basic) -> Optional[Basic]:
        return None

    def solve_first(self, equation: Expr, x: Symbol) -> Optional[Expr]:
        raise self._missing('solving %s' % (equation,))

    # -- whole operations not ported

    def simplify(self, e: Basic, ratio: float = 1.7, rational: Optional[bool] = False, inverse: bool = False,
                 doit: bool = True) -> Basic:
        raise self._missing('simplify of %s' % (type(e).__name__,))

    def factor_foreign_integrals(self, e: Expr) -> Expr:
        """``e`` with the content of its foreign integrals taken out."""
        raise self._missing('factor_terms of foreign integrals')

    def integrate(self, e: Expr, specs: Sequence[Union[Symbol, tuple[Symbol, Expr, Expr]]]) -> Expr:
        raise self._missing('integrate')

    def fps(self, e: Expr, x: Optional[Symbol], x0: Expr, dir: int) -> Expr:
        raise self._missing('fps')

    def fourier_series(self, e: Expr, limits: Optional[tuple[Symbol, Expr, Expr]]) -> Expr:
        raise self._missing('fourier_series')

    def invert(self, e: Expr, g: Expr, gens: Sequence[Symbol]) -> Expr:
        raise self._missing('invert in several variables')


_registered: list[Bridge] = [Bridge()]


def bridge() -> Bridge:
    """The registered bridge (one raising on every operation when none is)."""
    return _registered[0]


def register_bridge(b: Bridge) -> None:
    """Make ``b`` the bridge of the core."""
    _registered[0] = b
