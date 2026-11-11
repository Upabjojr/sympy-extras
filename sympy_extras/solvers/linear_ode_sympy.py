"""The SymPy face of the linear ODE solvers
(:mod:`sympy_extras.solvers.linear_ode`), and the step which is not
ported.

:class:`LinearOperator` holds SymPy coefficients and its arithmetic, and
the functions :func:`polynomial_solutions`, :func:`rational_solutions`,
:func:`hyperexponential_solutions`, :func:`hyperexponential_search`,
:func:`reduce_order_linear`, :func:`dsolve_linear`, :func:`gcrd`,
:func:`lclm` and :func:`log_derivative` take and return SymPy objects,
converted by :mod:`sympy_extras.interop` around the native module. The
modules still written on SymPy (:mod:`~sympy_extras.solvers.linear_systems`,
:mod:`~sympy_extras.solvers.first_order`) use them, and the private
helpers they share (``_solve_operator``, ``_nullspace``, ``_indicial_at``,
``_indicial_at_infinity``).

The step of the native module which stays written on SymPy is called from
it at run time (it takes and returns objects of the core):

- :func:`dsolve_of_core`: SymPy's ``dsolve``, the last resort of
  :func:`~sympy_extras.solvers.linear_ode.dsolve_linear` on the reduced
  equations; the independent parts of its general solution.

The antiderivatives go through
:func:`~sympy_extras.solvers.kovacic_sympy.integrate_of_core` and
:func:`~sympy_extras.solvers.kovacic_sympy.integral_of_core`.

Examples
========

>>> from sympy import Function
>>> from sympy.abc import x
>>> from sympy_extras.solvers.linear_ode_sympy import dsolve_linear
>>> y = Function('y')(x)
>>> dsolve_linear(x*y.diff(x, 2) - (x + 2)*y.diff(x) + 2*y, y)
[x**2 + 2*x + 2, exp(x)]
"""
from __future__ import annotations

from typing import Optional, Sequence

from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.function import AppliedUndef
from sympy.core.symbol import Symbol
from sympy.polys.polytools import Poly

from sympy_extras._timeout import attempt
from sympy_extras._typing import as_expr, free_symbols
from sympy_extras.assumptions.ask_sympy import _core_symbol, _sympy_symbol
from sympy_extras.assumptions.intervals_sympy import core_expr, sympy_expr
from sympy_extras.core.basic import Basic as CoreBasic
from sympy_extras.core.expr import Expr as CoreExpr
from sympy_extras.core.function import AppliedUndef as CoreAppliedUndef
from sympy_extras.interop import from_sympy, to_sympy
from sympy_extras.settings import settings

from . import linear_ode as native

__all__ = ['LinearOperator', 'polynomial_solutions', 'rational_solutions',
           'hyperexponential_solutions', 'reduce_order_linear', 'dsolve_linear',
           'gcrd', 'lclm', 'log_derivative', 'hyperexponential_search', 'dsolve_of_core']


# ---------------------------------------------------------------------------
# conversions

def _core_basic(e: Basic) -> CoreBasic:
    converted = from_sympy(e)
    if not isinstance(converted, CoreBasic):
        raise TypeError("%s cannot be converted" % (e,))
    return converted


def _core_function(f: AppliedUndef) -> CoreAppliedUndef:
    converted = from_sympy(f)
    if not isinstance(converted, CoreAppliedUndef):
        raise TypeError("%s is not an applied undefined function" % (f,))
    return converted


def _sympy_function(f: CoreAppliedUndef) -> AppliedUndef:
    converted = to_sympy(f)
    if not isinstance(converted, AppliedUndef):
        raise TypeError("%s is not an applied undefined function" % (f,))
    return converted


def _sympy_list(es: Sequence[CoreExpr]) -> list[Expr]:
    return [sympy_expr(e) for e in es]


def _core_list(es: Sequence[Expr]) -> list[CoreExpr]:
    return [core_expr(as_expr(e)) for e in es]


def core_upoly(p: Poly) -> native.UPoly:
    """A SymPy ``Poly`` in one generator as a polynomial of the native
    module."""
    return native.UPoly.of(core_expr(as_expr(p.as_expr())), _core_symbol(p.gen))


def sympy_poly(p: native.UPoly) -> Poly:
    """A polynomial of the native module as a SymPy ``Poly``."""
    return Poly(sympy_expr(p.as_expr()), _sympy_symbol(p.x))


# ---------------------------------------------------------------------------
# the step of the native module which stays on SymPy

def dsolve_of_core(equation: CoreExpr, f: CoreAppliedUndef) -> Optional[list[CoreExpr]]:
    """The independent parts of the general solution of ``equation = 0``
    which SymPy's ``dsolve`` finds under the time limit of the settings
    (the coefficients of the constants ``C1, C2, ...`` of an explicit
    solution), ``None`` when it finds none."""
    from sympy.core.function import expand
    from sympy.core.relational import Eq
    from sympy.integrals.integrals import Integral
    from sympy.series.order import Order
    from sympy.solvers.ode import dsolve
    e, y = sympy_expr(equation), _sympy_function(f)
    try:
        result = attempt(lambda: dsolve(e, y), settings.timeout)
    except IndexError:
        # SymPy 1.14: the reduced equation's power series truncated to
        # Eq(g(x), O(1)), then dsolve(Eq(f(x).diff(x), O(1)), f(x))
        # raises IndexError in the matcher of the hint factorable
        # (sympy-extras#25): nothing found by dsolve
        result = None
    # a truncated power series with an O term is not a solution either
    # (sympy-extras#37)
    if not isinstance(result, Eq) or result.rhs.has(Integral, Order):
        return None
    rhs = as_expr(expand(as_expr(result.rhs)))
    constants = sorted((s for s in free_symbols(rhs) if s.name.startswith('C')), key=lambda s: s.name)
    parts: list[CoreExpr] = []
    for c in constants:
        part = as_expr(rhs.coeff(c))
        if part != 0 and not part.has(*constants):
            parts.append(core_expr(part))
    return parts


# ---------------------------------------------------------------------------
# the face

class LinearOperator:
    """``a_n D**n + ... + a_1 D + a_0`` with polynomial coefficients
    ``[a_0, ..., a_n]`` in ``x`` (SymPy expressions; the arithmetic is
    that of :class:`sympy_extras.solvers.linear_ode.LinearOperator`).

    Examples
    ========

    >>> from sympy import Function
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.linear_ode_sympy import LinearOperator
    >>> y = Function('y')(x)
    >>> L = LinearOperator.from_equation(x*y.diff(x, 2) - (x + 2)*y.diff(x) + 2*y, y)
    >>> L.coefficients, L.order
    ([2, -x - 2, x], 2)
    >>> L(x**2 + 2*x + 2)
    0
    """

    def __init__(self, coefficients: Sequence[Expr], x: Symbol) -> None:
        coefficients_ = [as_expr(c) for c in coefficients]
        while len(coefficients_) > 1 and coefficients_[-1] == 0:
            coefficients_.pop()
        self.coefficients = coefficients_
        self.x = x

    @staticmethod
    def of_core(L: native.LinearOperator) -> LinearOperator:
        """The operator of the native module with SymPy coefficients."""
        return LinearOperator(_sympy_list(L.coefficients), _sympy_symbol(L.x))

    def core(self) -> native.LinearOperator:
        """The operator of the native module."""
        return native.LinearOperator(_core_list(self.coefficients), _core_symbol(self.x))

    @property
    def order(self) -> int:
        return len(self.coefficients) - 1

    @classmethod
    def from_equation(cls, equation: Basic, f: AppliedUndef) -> LinearOperator:
        """The operator of a homogeneous linear equation in ``f``, the
        coefficients made polynomial by clearing denominators."""
        return cls.of_core(native.LinearOperator.from_equation(_core_basic(equation), _core_function(f)))

    def __call__(self, y: Expr) -> Expr:
        return sympy_expr(self.core()(core_expr(as_expr(y))))

    def __repr__(self) -> str:
        return "LinearOperator(%s, %s)" % (self.coefficients, self.x)

    def transformed(self, factor: Expr) -> LinearOperator:
        """The operator ``M`` with ``L(factor * u) = factor * M(u)``,
        denominators cleared."""
        return LinearOperator.of_core(self.core().transformed(core_expr(as_expr(factor))))

    @property
    def leading_coefficient(self) -> Expr:
        return self.coefficients[-1]

    @property
    def is_zero(self) -> bool:
        return len(self.coefficients) == 1 and self.coefficients[0] == 0

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LinearOperator):
            return NotImplemented
        return self.core() == other.core()

    def __hash__(self) -> int:
        return hash((self.x, self.order))

    def __add__(self, other: LinearOperator) -> LinearOperator:
        return LinearOperator.of_core(self.core() + other.core())

    def __neg__(self) -> LinearOperator:
        return LinearOperator([as_expr(-c) for c in self.coefficients], self.x)

    def __sub__(self, other: LinearOperator) -> LinearOperator:
        return LinearOperator.of_core(self.core() - other.core())

    def __mul__(self, other: LinearOperator) -> LinearOperator:
        """The composition ``self(other(y))``.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy_extras.solvers.linear_ode_sympy import LinearOperator
        >>> D = LinearOperator([0, 1], x)
        >>> D*LinearOperator([x], x)
        LinearOperator([1, x], x)
        >>> (D - LinearOperator([1/x], x))*(D + LinearOperator([1/x], x))
        LinearOperator([-2/x**2, 0, 1], x)
        """
        return LinearOperator.of_core(self.core()*other.core())

    def scaled(self, c: Expr) -> LinearOperator:
        """The operator ``c L``."""
        return LinearOperator.of_core(self.core().scaled(core_expr(as_expr(c))))

    def monic(self) -> LinearOperator:
        """The operator divided by its leading coefficient.

        >>> from sympy.abc import x
        >>> from sympy_extras.solvers.linear_ode_sympy import LinearOperator
        >>> LinearOperator([1, x, x**2], x).monic()
        LinearOperator([x**(-2), 1/x, 1], x)
        """
        return LinearOperator.of_core(self.core().monic())

    def primitive(self) -> LinearOperator:
        """The operator times a rational function so that the
        coefficients are polynomials in ``x`` without common factor.

        >>> from sympy.abc import x
        >>> from sympy_extras.solvers.linear_ode_sympy import LinearOperator
        >>> LinearOperator([2/x**2, 2/x, 2], x).primitive()
        LinearOperator([1, x, x**2], x)
        """
        return LinearOperator.of_core(self.core().primitive())

    def cleared(self) -> LinearOperator:
        """The operator times the least common multiple of the
        denominators of its coefficients."""
        return LinearOperator.of_core(self.core().cleared())

    def right_divmod(self, other: LinearOperator) -> tuple[LinearOperator, LinearOperator]:
        """``(Q, R)`` with ``self = Q*other + R`` and ``R`` of order less
        than ``other``.

        >>> from sympy.abc import x
        >>> from sympy_extras.solvers.linear_ode_sympy import LinearOperator
        >>> L = LinearOperator([0, 0, 0, 1], x)
        >>> L.right_divmod(LinearOperator([-1, x], x))
        (LinearOperator([0, -1/x**2, 1/x], x), LinearOperator([0], x))
        """
        Q, R = self.core().right_divmod(other.core())
        return LinearOperator.of_core(Q), LinearOperator.of_core(R)

    def right_quotient(self, other: LinearOperator) -> LinearOperator:
        return self.right_divmod(other)[0]

    def right_remainder(self, other: LinearOperator) -> LinearOperator:
        return self.right_divmod(other)[1]

    def adjoint(self) -> LinearOperator:
        """The adjoint ``L* = sum (-1)**i D**i a_i``.

        >>> from sympy.abc import x
        >>> from sympy_extras.solvers.linear_ode_sympy import LinearOperator
        >>> LinearOperator([0, x, 1], x).adjoint()
        LinearOperator([-1, -x, 1], x)
        """
        return LinearOperator.of_core(self.core().adjoint())

    def exterior_power(self, k: int) -> LinearOperator:
        """The ``k``-th associated operator.

        >>> from sympy.abc import x
        >>> from sympy_extras.solvers.linear_ode_sympy import LinearOperator
        >>> LinearOperator([-1, 0, 0, 1], x).exterior_power(2)
        LinearOperator([1, 0, 0, 1], x)
        """
        return LinearOperator.of_core(self.core().exterior_power(k))

    def symmetric_power(self, m: int) -> LinearOperator:
        """The ``m``-th symmetric power.

        >>> from sympy.abc import x
        >>> from sympy_extras.solvers.linear_ode_sympy import LinearOperator
        >>> LinearOperator([1, 0, 1], x).symmetric_power(2)
        LinearOperator([0, 4, 0, 1], x)
        """
        return LinearOperator.of_core(self.core().symmetric_power(m))


def log_derivative(h: Expr, x: Symbol) -> Expr:
    """``h'/h``, rational for a hyperexponential ``h``
    (:func:`sympy_extras.solvers.linear_ode.log_derivative` on SymPy
    objects).

    >>> from sympy import exp, sqrt, Integral
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.linear_ode_sympy import log_derivative
    >>> log_derivative(sqrt(x - 1)*exp(x**2)/x, x)
    (4*x**3 - 4*x**2 - x + 2)/(2*x**2 - 2*x)
    >>> log_derivative(exp(Integral(1/(x**3 + 1), x)), x)
    1/(x**3 + 1)
    """
    return sympy_expr(native.log_derivative(core_expr(as_expr(h)), _core_symbol(x)))


def gcrd(A: LinearOperator, B: LinearOperator) -> LinearOperator:
    """The monic greatest common right divisor of ``A`` and ``B``.

    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.linear_ode_sympy import LinearOperator, gcrd
    >>> D = LinearOperator([0, 1], x)
    >>> R = D - LinearOperator([1/x], x)
    >>> gcrd((D*D + LinearOperator([1], x))*R, (D - LinearOperator([x], x))*R)
    LinearOperator([-1/x, 1], x)
    """
    return LinearOperator.of_core(native.gcrd(A.core(), B.core()))


def lclm(A: LinearOperator, B: LinearOperator) -> LinearOperator:
    """The monic least common left multiple of ``A`` and ``B``.

    >>> from sympy.abc import x
    >>> from sympy import exp, simplify
    >>> from sympy_extras.solvers.linear_ode_sympy import LinearOperator, lclm
    >>> M = lclm(LinearOperator([-1, 1], x), LinearOperator([-1, x], x))
    >>> M.order, simplify(M(exp(x))), M(x)
    (2, 0, 0)
    """
    return LinearOperator.of_core(native.lclm(A.core(), B.core()))


def polynomial_solutions(L: LinearOperator) -> list[Expr]:
    """A basis of the polynomial solutions of ``L(y) = 0``.

    >>> from sympy import Function
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.linear_ode_sympy import LinearOperator, polynomial_solutions
    >>> y = Function('y')(x)
    >>> L = LinearOperator.from_equation(x*y.diff(x, 2) - (x + 2)*y.diff(x) + 2*y, y)
    >>> polynomial_solutions(L)
    [x**2 + 2*x + 2]
    >>> polynomial_solutions(LinearOperator([-2, 0, 1], x))
    []
    """
    return _sympy_list(native.polynomial_solutions(L.core()))


def rational_solutions(L: LinearOperator) -> list[Expr]:
    """A basis of the rational solutions of ``L(y) = 0``.

    >>> from sympy import Function
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.linear_ode_sympy import LinearOperator, rational_solutions
    >>> y = Function('y')(x)
    >>> L = LinearOperator.from_equation(x**2*y.diff(x, 2) + 4*x*y.diff(x) + 2*y, y)
    >>> rational_solutions(L)
    [x**(-2), 1/x]
    """
    return _sympy_list(native.rational_solutions(L.core()))


def hyperexponential_solutions(L: LinearOperator, max_combinations: int = 200) -> list[Expr]:
    """Hyperexponential solutions of ``L(y) = 0``
    (:func:`sympy_extras.solvers.linear_ode.hyperexponential_solutions`).

    >>> from sympy import Function
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.linear_ode_sympy import LinearOperator, hyperexponential_solutions
    >>> y = Function('y')(x)
    >>> L = LinearOperator.from_equation(4*x**2*y.diff(x, 2) + 4*x*y.diff(x) - y, y)
    >>> hyperexponential_solutions(L)
    [1/sqrt(x), sqrt(x)]
    >>> hyperexponential_solutions(LinearOperator.from_equation(y.diff(x, 2) - 2*x*y.diff(x) + 4*y, y))
    [x**2 - 1/2]
    >>> hyperexponential_solutions(LinearOperator.from_equation(x*y.diff(x, 2) - (x + 2)*y.diff(x) + 2*y, y))
    [x**2 + 2*x + 2, exp(x)]
    >>> hyperexponential_solutions(LinearOperator.from_equation(x**4*y.diff(x, 2) + 2*x**3*y.diff(x) - y, y))
    [exp(1/x), exp(-1/x)]
    """
    return _sympy_list(native.hyperexponential_solutions(L.core(), max_combinations))


def hyperexponential_search(L: LinearOperator, max_combinations: int = 200) -> tuple[list[Expr], bool]:
    """The hyperexponential solutions and whether the search was
    exhaustive (:func:`sympy_extras.solvers.linear_ode.hyperexponential_search`).

    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.linear_ode_sympy import LinearOperator, hyperexponential_search
    >>> hyperexponential_search(LinearOperator([-x, 0, 1], x))
    ([], True)
    >>> hyperexponential_search(LinearOperator([-1, 0, x**2], x))
    ([], False)
    """
    found, complete = native.hyperexponential_search(L.core(), max_combinations)
    return _sympy_list(found), complete


def reduce_order_linear(L: LinearOperator, y1: Expr) -> LinearOperator:
    """The operator ``M`` of order ``n - 1`` with ``L(y1 * w) = y1 * M(w')``.

    >>> from sympy import Function, exp
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.linear_ode_sympy import LinearOperator, reduce_order_linear
    >>> y = Function('y')(x)
    >>> L = LinearOperator.from_equation(y.diff(x, 2) - 3*y.diff(x) + 2*y, y)
    >>> reduce_order_linear(L, exp(x))
    LinearOperator([-1, 1], x)
    """
    return LinearOperator.of_core(native.reduce_order_linear(L.core(), core_expr(as_expr(y1))))


def dsolve_linear(equation: Basic, f: AppliedUndef, use_kovacic: bool = True,
                  use_dsolve: bool = True, special_values: bool = True) -> list[Expr]:
    """Independent solutions of a homogeneous linear equation with
    rational coefficients (:func:`sympy_extras.solvers.linear_ode.dsolve_linear`
    on SymPy objects).

    Examples
    ========

    >>> from sympy import Function, symbols
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.linear_ode_sympy import dsolve_linear
    >>> y = Function('y')(x)
    >>> dsolve_linear(x*y.diff(x, 2) - (x + 2)*y.diff(x) + 2*y, y)
    [x**2 + 2*x + 2, exp(x)]
    >>> dsolve_linear(x**2*y.diff(x, 2) + 4*x*y.diff(x) + 2*y, y)
    [x**(-2), 1/x]
    >>> a, b = symbols('a b')
    >>> dsolve_linear(y.diff(x, 2) - (a + b)*y.diff(x) + a*b*y, y)
    [exp(b*x), Piecewise((x*exp(b*x), Eq(a, b)), (exp(a*x), True))]
    """
    found = native.dsolve_linear(_core_basic(equation), _core_function(f), use_kovacic, use_dsolve,
                                 special_values)
    return _sympy_list(found)


# ---------------------------------------------------------------------------
# the helpers shared with the modules still on SymPy (linear_systems) and
# the tests

def _solve_operator(L: LinearOperator, f: AppliedUndef, use_kovacic: bool, use_dsolve: bool,
                    factorize: bool = True) -> list[Expr]:
    return _sympy_list(native._solve_operator(L.core(), _core_function(f), use_kovacic, use_dsolve, factorize))


def _nullspace(rows: list[list[Expr]], columns: int) -> list[list[Expr]]:
    """A basis of the vectors annihilated by the rows (SymPy's
    ``Matrix.nullspace``)."""
    found = native._nullspace([_core_list(row) for row in rows], columns)
    return [_sympy_list(vector) for vector in found]


def _indicial_at(L: LinearOperator, q: Poly) -> Poly:
    return sympy_poly(native._indicial_at(L.core(), core_upoly(q)))


def _indicial_at_infinity(L: LinearOperator) -> Poly:
    return sympy_poly(native._indicial_at_infinity(L.core()))


def _exponents(L: LinearOperator, q: Poly) -> list[Expr]:
    return _sympy_list(native._exponents(L.core(), core_upoly(q)))


def _exponential_parts(L: LinearOperator) -> tuple[list[Expr], bool]:
    found, complete = native._exponential_parts(L.core())
    return _sympy_list(found), complete


def _independent(solutions: list[Expr], x: Symbol) -> list[Expr]:
    return _sympy_list(native._independent(_core_list(solutions), _core_symbol(x)))


def _proportional(s: Expr, k: Expr, x: Symbol) -> bool:
    return native._proportional(core_expr(as_expr(s)), core_expr(as_expr(k)), _core_symbol(x))

