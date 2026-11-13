"""The SymPy face of the Lie point symmetries
(:mod:`sympy_extras.solvers.lie`).

:class:`JetSpace`, :class:`Symmetry` and the functions take and return
SymPy objects, converted by :mod:`sympy_extras.interop` around the native
module: a :class:`JetSpace` and a :class:`Symmetry` keep their native
counterpart (``native``) and their coordinates as SymPy objects. The
faces :mod:`sympy_extras.solvers.ode_sympy`,
:mod:`sympy_extras.solvers.second_order_sympy` and
:mod:`sympy_extras.solvers.pde_sympy` use it.

Examples
========

>>> from sympy import Function, symbols
>>> from sympy_extras.solvers.lie_sympy import symmetries
>>> x, t = symbols('x t')
>>> u = Function('u')(x, t)
>>> for X in symmetries(u.diff(t) + u*u.diff(x) - u.diff(x, 2), u):
...     print(X.generator())
d/dx
t*d/dx + d/du
x*d/dx + 2*t*d/dt - u*d/du
t*x*d/dx + t**2*d/dt + (-t*u + x)*d/du
d/dt
"""
from __future__ import annotations

from typing import Final, Optional, Sequence, Union

from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.function import AppliedUndef, Derivative
from sympy.core.relational import Eq
from sympy.core.symbol import Symbol

from sympy_extras._typing_sympy import Monomial, as_expr, as_symbol
from sympy_extras.core.basic import Basic as CoreBasic
from sympy_extras.core.expr import Expr as CoreExpr
from sympy_extras.core.function import AppliedUndef as CoreAppliedUndef
from sympy_extras.core.relational import Eq as CoreEq
from sympy_extras.core.symbol import Symbol as CoreSymbol
from sympy_extras.interop import from_sympy, to_sympy
from sympy_extras.polys.differential_sympy import JanetBasis

from . import lie as native

__all__ = ['JetSpace', 'Symmetry', 'symmetries', 'check_symmetry',
    'infinitesimal_condition', 'determining_equations', 'determining_system',
    'symmetry_janet_basis', 'Equation']

#: an expression of a differential equation: ``Eq`` or expression ``= 0``
Equation = Union[Expr, Eq]
Jet = native.Jet


# ---------------------------------------------------------------------------
# conversions

def _core_expr(e: Basic) -> CoreExpr:
    converted = from_sympy(e)
    if not isinstance(converted, CoreExpr):
        raise TypeError("expected an expression, got %s" % (e,))
    return converted


def _core_symbol(s: Basic) -> CoreSymbol:
    converted = from_sympy(as_symbol(s))
    if not isinstance(converted, CoreSymbol):
        raise TypeError("%s is not a symbol" % (s,))
    return converted


def _core_function(f: Basic) -> CoreAppliedUndef:
    converted = from_sympy(f)
    if not isinstance(converted, CoreAppliedUndef):
        raise TypeError("an applied undefined function is expected, got %s" % (f,))
    return converted


def _core_equation(e: Basic) -> native.Equation:
    converted = from_sympy(e)
    if not isinstance(converted, (CoreExpr, CoreEq)):
        raise TypeError("an equation or an expression is expected, got %s" % (e,))
    return converted


def _sympy_expr(e: CoreBasic) -> Expr:
    """An expression of the core as a SymPy expression, its derivatives of
    undefined functions in SymPy's canonical order of the variables
    (``u.diff(x, t)`` is ``Derivative(u, t, x)`` in SymPy; the conversion
    builds the unevaluated ``Derivative(u, x, t)``, which is not equal)."""
    converted = as_expr(to_sympy(e))
    canonical: dict[Basic, Basic] = {}
    for d in converted.atoms(Derivative):
        if isinstance(d, Derivative) and isinstance(d.expr, AppliedUndef):
            ordered = Derivative(d.expr, *d.variable_count, evaluate=True)
            if ordered != d:
                canonical[d] = ordered
    return as_expr(converted.xreplace(canonical)) if canonical else converted


def _sympy_symbol(s: CoreSymbol) -> Symbol:
    return as_symbol(to_sympy(s))


def _sympy_function(f: CoreAppliedUndef) -> AppliedUndef:
    converted = to_sympy(f)
    if not isinstance(converted, AppliedUndef):
        raise TypeError("expected an applied function, got %s" % (converted,))
    return converted


def _core_equations(equations: Union[Equation, Sequence[Equation]]) -> list[native.Equation]:
    if isinstance(equations, (Expr, Eq)):
        return [_core_equation(equations)]
    return [_core_equation(e) for e in equations]


def _core_functions(functions: Union[AppliedUndef, Sequence[AppliedUndef]]) -> list[CoreAppliedUndef]:
    if isinstance(functions, AppliedUndef):
        return [_core_function(functions)]
    return [_core_function(f) for f in functions]


def _as_zero(eq: Equation) -> Expr:
    """The expression which is zero."""
    if isinstance(eq, Eq):
        return as_expr(eq.lhs - eq.rhs)
    return as_expr(eq)


# ---------------------------------------------------------------------------
# the jet space and the vector fields

class JetSpace:
    """Jet coordinates of dependent variables
    (:class:`sympy_extras.solvers.lie.JetSpace` on SymPy objects).

    Examples
    ========

    >>> from sympy import Function, symbols
    >>> from sympy_extras.solvers.lie_sympy import JetSpace
    >>> x, t = symbols('x t')
    >>> u = Function('u')(x, t)
    >>> jets = JetSpace([u])
    >>> jets.to_jet(u.diff(t) - u.diff(x, 2))
    u_t - u_xx
    >>> jets.total_derivative(jets.u[0]**2, 0)
    2*u*u_x
    """

    def __init__(self, functions: Sequence[AppliedUndef]) -> None:
        if not functions:
            raise ValueError("at least one dependent variable is needed")
        args = functions[0].args
        for f in functions:
            if f.args != args:
                raise ValueError("the dependent variables must have the same arguments")
        if not all(isinstance(a, Symbol) for a in args):
            raise TypeError("the arguments of the dependent variables must be symbols")
        self.native: Final[native.JetSpace] = native.JetSpace([_core_function(f) for f in functions])
        self.functions: Final[tuple[AppliedUndef, ...]] = tuple(functions)
        self.x: Final[tuple[Symbol, ...]] = tuple(_sympy_symbol(s) for s in self.native.x)
        self.u: Final[tuple[Symbol, ...]] = tuple(_sympy_symbol(s) for s in self.native.u)

    def __repr__(self) -> str:
        return "JetSpace(%s)" % (list(self.functions),)

    @property
    def p(self) -> int:
        """Number of independent variables."""
        return self.native.p

    @property
    def q(self) -> int:
        """Number of dependent variables."""
        return self.native.q

    def jet(self, a: int, J: Monomial) -> Symbol:
        """The jet coordinate `u^a_J`."""
        return _sympy_symbol(self.native.jet(a, J))

    def index(self, s: Symbol) -> Optional[Jet]:
        """The ``(a, J)`` of a jet coordinate, or ``None``."""
        return self.native.index(_core_symbol(s))

    def jets_in(self, expr: Basic) -> list[Jet]:
        """The jet coordinates occurring in ``expr``."""
        return self.native.jets_in(_core_expr(expr))

    def order(self, expr: Basic) -> int:
        """The highest order of a jet coordinate occurring in ``expr``."""
        return self.native.order(_core_expr(expr))

    def to_jet(self, expr: Basic) -> Expr:
        """Rewrite the derivatives of the dependent variables as jet
        coordinates."""
        return _sympy_expr(self.native.to_jet(_core_expr(expr)))

    def from_jet(self, expr: Basic) -> Expr:
        """Rewrite the jet coordinates as derivatives of the functions."""
        return _sympy_expr(self.native.from_jet(_core_expr(expr)))

    def total_derivative(self, expr: Expr, i: int) -> Expr:
        """The total derivative `D_i` of an expression in the jet
        coordinates."""
        return _sympy_expr(self.native.total_derivative(_core_expr(expr), i))

    def monomials(self, degree: int) -> list[Expr]:
        """The monomials in the independent and dependent variables of
        total degree at most ``degree``."""
        return [_sympy_expr(m) for m in self.native.monomials(degree)]


class Symmetry:
    """A vector field on the space of the independent and dependent
    variables (:class:`sympy_extras.solvers.lie.Symmetry` on SymPy
    objects): ``xi`` and ``eta`` are SymPy expressions in the symbols of
    the :class:`JetSpace`.

    Examples
    ========

    >>> from sympy import Function, symbols
    >>> from sympy_extras.solvers.lie_sympy import JetSpace, Symmetry
    >>> x, t = symbols('x t')
    >>> u = Function('u')(x, t)
    >>> jets = JetSpace([u])
    >>> X = Symmetry(jets, [2*t, 0], [-x*jets.u[0]])
    >>> X
    Symmetry(xi=(2*t, 0), eta=(-u*x,))
    >>> print(X.generator())
    2*t*d/dx - u*x*d/du
    >>> X.characteristic()
    (-2*t*u_x - u*x,)
    """

    def __init__(self, jets: JetSpace, xi: Sequence[Union[Expr, int]],
                 eta: Sequence[Union[Expr, int]]) -> None:
        if len(xi) != jets.p or len(eta) != jets.q:
            raise ValueError("expected %d xi and %d eta" % (jets.p, jets.q))
        self.jets: Final[JetSpace] = jets
        self.xi: Final[tuple[Expr, ...]] = tuple(as_expr(e) for e in xi)
        self.eta: Final[tuple[Expr, ...]] = tuple(as_expr(e) for e in eta)
        self.native: Final[native.Symmetry] = native.Symmetry(
            jets.native, [_core_expr(e) for e in self.xi], [_core_expr(e) for e in self.eta])

    @classmethod
    def _of_native(cls, jets: JetSpace, symmetry: native.Symmetry) -> Symmetry:
        return cls(jets, [_sympy_expr(e) for e in symmetry.xi], [_sympy_expr(e) for e in symmetry.eta])

    def __repr__(self) -> str:
        return "Symmetry(xi=%s, eta=%s)" % (self.xi, self.eta)

    def generator(self) -> str:
        """The vector field written with ``d/dx`` and ``d/du``."""
        return self.native.generator()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Symmetry):
            return NotImplemented
        return self.xi == other.xi and self.eta == other.eta

    def __hash__(self) -> int:
        return hash((self.xi, self.eta))

    def is_zero(self) -> bool:
        """Whether every coefficient is zero."""
        return all(c == 0 for c in self.xi + self.eta)

    def __add__(self, other: Symmetry) -> Symmetry:
        return Symmetry(self.jets, [a + b for a, b in zip(self.xi, other.xi)],
                        [a + b for a, b in zip(self.eta, other.eta)])

    def __sub__(self, other: Symmetry) -> Symmetry:
        return self + (-other)

    def __neg__(self) -> Symmetry:
        return Symmetry(self.jets, [-a for a in self.xi], [-a for a in self.eta])

    def __mul__(self, c: Union[Expr, int]) -> Symmetry:
        c_ = as_expr(c)
        return Symmetry(self.jets, [c_*a for a in self.xi], [c_*a for a in self.eta])

    def __rmul__(self, c: Union[Expr, int]) -> Symmetry:
        return self*c

    def apply_point(self, f: Expr) -> Expr:
        """`X f` for a function of the independent and dependent
        variables."""
        return _sympy_expr(self.native.apply_point(_core_expr(f)))

    def commutator(self, other: Symmetry) -> Symmetry:
        """The Lie bracket `[X, Y]`."""
        return Symmetry._of_native(self.jets, self.native.commutator(other.native))

    def characteristic(self) -> tuple[Expr, ...]:
        """The characteristic `Q^a = \\eta^a - \\sum_i \\xi^i u^a_i`."""
        return tuple(_sympy_expr(e) for e in self.native.characteristic())

    def prolongation(self, order: int) -> dict[Jet, Expr]:
        """The coefficients `\\eta^a_J` of the prolonged vector field for
        all multi-indices with `|J| \\le` ``order``."""
        return {key: _sympy_expr(e) for key, e in self.native.prolongation(order).items()}

    def apply(self, expr: Expr) -> Expr:
        """`\\mathrm{pr} X` applied to an expression in the jet
        coordinates."""
        return _sympy_expr(self.native.apply(_core_expr(expr)))


# ---------------------------------------------------------------------------
# the functions

def _leading_derivatives(jets: JetSpace, eqs: list[Expr]) -> dict[Symbol, Expr]:
    """Solve every equation (in the jet coordinates) for its highest
    derivative, preferring one it is linear in."""
    solved = native.leading_derivatives(jets.native, [_core_expr(e) for e in eqs])
    return {_sympy_symbol(s): _sympy_expr(e) for s, e in solved.items()}


def _jets_of(functions: Union[AppliedUndef, Sequence[AppliedUndef]]) -> JetSpace:
    return JetSpace([functions] if isinstance(functions, AppliedUndef) else list(functions))


def infinitesimal_condition(equations: Union[Equation, Sequence[Equation]],
                            functions: Union[AppliedUndef, Sequence[AppliedUndef]],
                            symmetry: Symmetry) -> list[Expr]:
    """`\\mathrm{pr} X(\\Delta)` restricted to the solutions of the
    equations (:func:`sympy_extras.solvers.lie.infinitesimal_condition` on
    SymPy objects).

    Examples
    ========

    >>> from sympy import Function, symbols
    >>> from sympy_extras.solvers.lie_sympy import JetSpace, Symmetry, infinitesimal_condition
    >>> x, t = symbols('x t')
    >>> u = Function('u')(x, t)
    >>> jets = JetSpace([u])
    >>> infinitesimal_condition(u.diff(t) - u.diff(x, 2), u, Symmetry(jets, [t, 0], [0]))
    [-u_x]
    """
    conditions = native.infinitesimal_condition(_core_equations(equations), _core_functions(functions),
                                                symmetry.native)
    return [_sympy_expr(e) for e in conditions]


def check_symmetry(equations: Union[Equation, Sequence[Equation]],
                   functions: Union[AppliedUndef, Sequence[AppliedUndef]],
                   symmetry: Symmetry) -> bool:
    """Whether the vector field is a point symmetry of the equations
    (:func:`sympy_extras.solvers.lie.check_symmetry` on SymPy objects).

    Examples
    ========

    >>> from sympy import Function, symbols
    >>> from sympy_extras.solvers.lie_sympy import JetSpace, Symmetry, check_symmetry
    >>> x, t = symbols('x t')
    >>> u = Function('u')(x, t)
    >>> heat = u.diff(t) - u.diff(x, 2)
    >>> jets = JetSpace([u])
    >>> check_symmetry(heat, u, Symmetry(jets, [2*t, 0], [-x*jets.u[0]]))
    True
    >>> check_symmetry(heat, u, Symmetry(jets, [t, 0], [0]))
    False
    """
    return native.check_symmetry(_core_equations(equations), _core_functions(functions), symmetry.native)


def determining_equations(equations: Union[Equation, Sequence[Equation]],
                          functions: Union[AppliedUndef, Sequence[AppliedUndef]],
                          degree: int = 2, basis: Sequence[Expr] = ()
                          ) -> tuple[JetSpace, list[Symbol], list[Expr], list[Expr], list[Expr]]:
    """The determining equations for the ansatz
    (:func:`sympy_extras.solvers.lie.determining_equations` on SymPy
    objects): the jet space, the unknown coefficients (dummies), the
    ansatz for `\\xi^i` and `\\eta^a` and the linear equations.

    Examples
    ========

    >>> from sympy import Function, symbols
    >>> from sympy_extras.solvers.lie_sympy import determining_equations
    >>> x = symbols('x')
    >>> y = Function('y')(x)
    >>> jets, unknowns, xi, eta, linear = determining_equations(y.diff(x, 2), y)
    >>> len(unknowns), len(linear)
    (12, 4)
    """
    jets = _jets_of(functions)
    _, unknowns, xi, eta, linear = native.determining_equations(
        _core_equations(equations), list(jets.native.functions), degree, [_core_expr(b) for b in basis])
    return (jets, [_sympy_symbol(c) for c in unknowns], [_sympy_expr(e) for e in xi],
            [_sympy_expr(e) for e in eta], [_sympy_expr(e) for e in linear])


def determining_system(equations: Union[Equation, Sequence[Equation]],
                       functions: Union[AppliedUndef, Sequence[AppliedUndef]]
                       ) -> tuple[list[Expr], list[AppliedUndef], list[Symbol]]:
    """The determining equations of the point symmetries as a linear
    system of partial differential equations
    (:func:`sympy_extras.solvers.lie.determining_system` on SymPy
    objects).

    Examples
    ========

    >>> from sympy import Function, symbols
    >>> from sympy_extras.solvers.lie_sympy import determining_system
    >>> x = symbols('x')
    >>> y = Function('y')(x)
    >>> system, unknowns, variables = determining_system(y.diff(x, 2), y)
    >>> unknowns
    [xi0(x, y), eta0(x, y)]
    >>> len(system)
    4
    """
    system, unknowns, base = native.determining_system(_core_equations(equations), _core_functions(functions))
    return ([_sympy_expr(e) for e in system], [_sympy_function(f) for f in unknowns],
            [_sympy_symbol(s) for s in base])


def symmetry_janet_basis(equations: Union[Equation, Sequence[Equation]],
                         functions: Union[AppliedUndef, Sequence[AppliedUndef]]) -> JanetBasis:
    """The Janet basis of the determining equations of the point
    symmetries (:func:`sympy_extras.solvers.lie.symmetry_janet_basis` on
    SymPy objects; the basis of :mod:`sympy_extras.polys.differential_sympy`).

    Examples
    ========

    >>> from sympy import Function, symbols
    >>> from sympy_extras.solvers.lie_sympy import symmetry_janet_basis
    >>> x, t = symbols('x t')
    >>> y = Function('y')(x)
    >>> symmetry_janet_basis(y.diff(x, 2), y).dimension
    8
    >>> symmetry_janet_basis(y.diff(x, 3) + y*y.diff(x, 2), y).dimension
    2
    >>> u = Function('u')(x, t)
    >>> symmetry_janet_basis(u.diff(t) + u*u.diff(x) + u.diff(x, 3), u).dimension
    4
    """
    return JanetBasis(native.symmetry_janet_basis(_core_equations(equations), _core_functions(functions)))


def symmetries(equations: Union[Equation, Sequence[Equation]],
               functions: Union[AppliedUndef, Sequence[AppliedUndef]],
               degree: int = 2, basis: Sequence[Expr] = (),
               superposition: bool = True) -> list[Symmetry]:
    """The point symmetries of the equations with polynomial infinitesimals
    of total degree at most ``degree`` (times the functions of ``basis``)
    (:func:`sympy_extras.solvers.lie.symmetries` on SymPy objects).

    Examples
    ========

    >>> from sympy import Function, symbols
    >>> from sympy_extras.solvers.lie_sympy import symmetries
    >>> x, t = symbols('x t')
    >>> u = Function('u')(x, t)
    >>> for X in symmetries(u.diff(t) - u.diff(x, 2), u, degree=3, superposition=False):
    ...     print(X.generator())
    d/dx
    t*d/dx - u*x/2*d/du
    x*d/dx + 2*t*d/dt
    t*x*d/dx + t**2*d/dt + (-t*u/2 - u*x**2/4)*d/du
    d/dt
    u*d/du
    """
    jets = _jets_of(functions)
    found = native.symmetries(_core_equations(equations), list(jets.native.functions), degree,
                              [_core_expr(b) for b in basis], superposition)
    return [Symmetry._of_native(jets, s) for s in found]
