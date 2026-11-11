"""The SymPy face of satisfiability (:mod:`sympy_extras.assumptions.sat`),
and the steps of the SAT and theory check which are not ported.

:func:`satisfiable`, :func:`tautology` and :func:`find_instance` take and
return SymPy objects (relations, memberships in SymPy's sets, predicates
of SymPy's ``Q``, the quantifiers of
:mod:`sympy_extras.assumptions.quantifiers_sympy`), converted by
:mod:`sympy_extras.interop` around the native functions.

The steps of the native functions which stay written on SymPy are called
from them at run time (the functions ending in ``_of_core``, which take
and return objects of the core): SymPy's SAT solver
(:func:`sympy.logic.inference.satisfiable`) on the propositional
abstraction, the consistency of predicates with SymPy's known facts
(:func:`sympy_extras.assumptions.facts_sympy.predicates_consistent`), the
sample points and the solution sets of the cylindrical algebraic
decomposition (:func:`sympy_extras.polys.cad.sample_points`,
:func:`sympy_extras.polys.cad.solution_set`), and the floor and the
comparison of real algebraic numbers
(:mod:`sympy_extras.polys.cad.samplepoints_sympy`, on SymPy's ``CRootOf``).
"""
from __future__ import annotations

from typing import Iterator, Literal, Optional, Sequence, Union, overload

from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.logic.boolalg import Boolean
from sympy.sets.sets import Set

from sympy_extras._typing_sympy import Truth, as_symbol
from sympy_extras.core.basic import Basic as CoreBasic
from sympy_extras.core.expr import Expr as CoreExpr
from sympy_extras.core.logic import Boolean as CoreBoolean
from sympy_extras.interop import from_sympy, to_sympy

from . import sat as native
from .ask_sympy import Assumptions, _core_symbol, _sympy_symbol, core_assumptions, core_boolean, core_domain, \
    sympy_boolean
from .intervals_sympy import sympy_expr

__all__ = ['satisfiable', 'tautology', 'find_instance', 'Model', 'Witness', 'propositional_models_of_core',
    'predicates_consistent_of_core', 'floor_of_core', 'compare_real_of_core']


#: a model: Boolean variables to truth values, real variables to values
Model = dict[Basic, Union[bool, Expr]]
#: a witness point
Witness = dict[Symbol, Expr]


# ---------------------------------------------------------------------------
# the steps of the native functions which stay on SymPy

def propositional_models_of_core(formula: CoreBoolean) -> Iterator[dict[CoreBasic, bool]]:
    """The models of a propositional formula, one at a time, by SymPy's
    SAT solver (:func:`sympy.logic.inference.satisfiable` with
    ``all_models=True``); none when it is unsatisfiable."""
    from sympy.logic.inference import satisfiable as sympy_satisfiable
    models = sympy_satisfiable(sympy_boolean(formula), all_models=True)
    if models is False:
        return
    for model in models:
        if not isinstance(model, dict):
            return
        yield {from_sympy(k): bool(v) for k, v in model.items()}


def predicates_consistent_of_core(predicates: CoreBoolean) -> bool:
    """:func:`sympy_extras.assumptions.facts_sympy.predicates_consistent`."""
    from .facts_sympy import predicates_consistent
    return predicates_consistent(sympy_boolean(predicates))


def floor_of_core(a: CoreExpr) -> int:
    """The floor of a real algebraic number (a rational number, a real
    ``CRootOf`` or a product of the two), exactly."""
    from sympy_extras.polys.cad.samplepoints_sympy import _floor_scaled
    return _floor_scaled(sympy_expr(a), 0)


def compare_real_of_core(a: CoreExpr, b: CoreExpr) -> int:
    """:func:`sympy_extras.polys.cad.samplepoints_sympy.compare_real`: the sign
    of ``a - b`` for real algebraic numbers."""
    from sympy_extras.polys.cad.samplepoints_sympy import compare_real
    return compare_real(sympy_expr(a), sympy_expr(b))


# ---------------------------------------------------------------------------
# the face

def _sympy_model(model: native.Model) -> Model:
    return {to_sympy(k): (v if isinstance(v, bool) else sympy_expr(v)) for k, v in model.items()}


def _sympy_witness(witness: native.Witness) -> Witness:
    return {_sympy_symbol(k): sympy_expr(v) for k, v in witness.items()}


def _without_real_solution(formula: Boolean, gens: Sequence[Symbol]) -> bool:
    """:func:`sympy_extras.assumptions.sat._without_real_solution` on SymPy
    objects."""
    return native._without_real_solution(core_boolean(formula), [_core_symbol(g) for g in gens])


@overload
def satisfiable(formula: Union[Boolean, bool], assumptions: Assumptions = None,
                domain: Optional[Set] = None, all_models: Literal[False] = False
                ) -> Union[Model, bool, None]: ...


@overload
def satisfiable(formula: Union[Boolean, bool], assumptions: Assumptions = None,
                domain: Optional[Set] = None, *, all_models: Literal[True]
                ) -> Optional[list[Model]]: ...


def satisfiable(formula: Union[Boolean, bool], assumptions: Assumptions = None,
                domain: Optional[Set] = None, all_models: bool = False
                ) -> Union[Model, bool, None, list[Model]]:
    """Whether a formula can be true
    (:func:`sympy_extras.assumptions.sat.satisfiable` on SymPy objects),
    the counterpart of Mathematica's ``SatisfiableQ`` (together with
    :func:`find_instance`).

    Parameters
    ==========

    formula : Boolean
        A Boolean combination of Boolean variables, relations between
        expressions, memberships in sets and predicates.
    assumptions : Boolean or list of Booleans, optional
        Assumptions which must hold as well. The global assumptions are
        used too.
    domain : Set, optional
        A named SymPy set all the variables are assumed to belong to.
    all_models : bool
        Return the list of all models of the propositional abstraction
        which are consistent with the theory instead of the first one.

    Returns
    =======

    A model, a dict mapping the Boolean variables to ``True``/``False`` and
    the real variables of the polynomial relations to a witness point, if
    the formula is satisfiable; ``False`` if it is not; ``None`` if it
    could not be decided (for instance if a relation is not polynomial).
    With ``all_models=True`` a list of models is returned (empty if the
    formula is unsatisfiable, ``None`` if undecided).

    Examples
    ========

    >>> from sympy import symbols, S, Eq
    >>> from sympy.abc import x, y
    >>> from sympy_extras.assumptions.sat_sympy import satisfiable
    >>> from sympy_extras.assumptions.facts_sympy import element
    >>> p, q = symbols('p q')
    >>> satisfiable(p & ~q)
    {p: True, q: False}
    >>> satisfiable(p & ~p)
    False
    >>> satisfiable((x**2 + y**2 < 1) & (x + y > 1))
    {x: 1/2, y: 2/3}
    >>> satisfiable((x**2 + y**2 < 1) & (x + y > 2))
    False
    >>> satisfiable((p | (x > 1)) & (~p | (x < 0)) & (x > 0))
    {p: False, x: 2}
    >>> satisfiable(Eq(x**2, 2), domain=S.Integers)
    False
    >>> satisfiable((x > 1) & (x < 3), domain=S.Integers)
    {x: 2}
    >>> satisfiable(Eq(x**2, -1), domain=S.Reals)
    False
    >>> satisfiable(Eq(x**2, -1)) is None
    True
    """
    formula_ = core_boolean(formula)
    assumptions_ = core_assumptions(assumptions)
    domain_ = core_domain(domain)
    if all_models:
        models = native.satisfiable(formula_, assumptions_, domain_, all_models=True)
        return None if models is None else [_sympy_model(m) for m in models]
    result = native.satisfiable(formula_, assumptions_, domain_)
    if isinstance(result, dict):
        return _sympy_model(result)
    return result


def tautology(formula: Union[Boolean, bool], assumptions: Assumptions = None,
              domain: Optional[Set] = None) -> Truth:
    """Whether a formula is always true
    (:func:`sympy_extras.assumptions.sat.tautology` on SymPy objects), the
    counterpart of Mathematica's ``TautologyQ``: ``True``, ``False`` or
    ``None`` if undecided.

    Examples
    ========

    >>> from sympy import symbols, S
    >>> from sympy.abc import x, y
    >>> from sympy_extras.assumptions.sat_sympy import tautology
    >>> p, q = symbols('p q')
    >>> tautology(p | ~p)
    True
    >>> tautology(p | q)
    False
    >>> tautology(x**2 + y**2 >= 2*x*y, domain=S.Reals)
    True
    >>> tautology((x > 0) | (x <= 0), domain=S.Reals)
    True
    >>> tautology(x**2 > 0, domain=S.Reals)
    False
    """
    return native.tautology(core_boolean(formula), core_assumptions(assumptions), core_domain(domain))


def find_instance(formula: Union[Boolean, bool], variables: Union[Symbol, Sequence[Symbol]],
                  domain: Set = S.Reals, assumptions: Assumptions = None,
                  count: int = 1) -> Optional[list[Witness]]:
    """Values of the variables satisfying a formula
    (:func:`sympy_extras.assumptions.sat.find_instance` on SymPy objects),
    the counterpart of Mathematica's ``FindInstance[expr, vars, dom, n]``.

    Parameters
    ==========

    formula : Boolean
        A Boolean combination of relations, memberships and predicates.
    variables : Symbol or list of Symbols
        The variables to find values for.
    domain : Set
        ``S.Reals`` (default) or ``S.Integers``: the set the variables
        belong to.
    assumptions : Boolean or list of Booleans, optional
        Assumptions which must hold as well.
    count : int
        The number of instances wanted.

    Returns
    =======

    A list of dicts mapping the variables to exact values, empty if there
    is no instance, or ``None`` if it could not be decided. Over the reals
    the instances are the sample points of the cells of a cylindrical
    algebraic decomposition satisfying the formula, so ``count`` large
    enough returns one point per cell.

    Examples
    ========

    >>> from sympy import S, Eq
    >>> from sympy.abc import x, y
    >>> from sympy_extras.assumptions.sat_sympy import find_instance
    >>> find_instance((x**2 + y**2 < 1) & (x > y), [x, y])
    [{x: 0, y: -1/2}]
    >>> found = find_instance((x**2 + y**2 < 1) & (x > y), [x, y], count=3)
    >>> found[0], found[2], found[1][x].evalf(10), found[1][y]    # SymPy prints a CRootOf in the variable it met first
    ({x: 0, y: -1/2}, {x: 3/4, y: 0}, 0.7071067812, 0)
    >>> [s[x].evalf(10) for s in find_instance(Eq(x**2, 2), [x])]    # a CRootOf, printed in a cached variable
    [-1.414213562]
    >>> find_instance(Eq(x**2, 2), [x], S.Integers)
    []
    >>> find_instance((x**2 > 5) & (x < 0), [x], S.Integers)
    [{x: -3}]
    >>> find_instance(x**2 < 0, [x])
    []
    """
    symbols = [variables] if isinstance(variables, Symbol) else [as_symbol(v) for v in variables]
    found = native.find_instance(core_boolean(formula), [_core_symbol(v) for v in symbols], core_domain(domain),
                                 core_assumptions(assumptions), count)
    return None if found is None else [_sympy_witness(w) for w in found]
