"""An experimental core: unevaluating classes, evaluating functions.

The classes of the expression tree (``Add``, ``Mul``, ``Pow``, ``Eq``,
``Derivative``, ...) build exactly the node they are asked for, so that
their types are honest: ``Add(x, x)`` is an ``Add`` with the arguments
``(x, x)``. The evaluation of SymPy's constructors is done by the
functions of the same name in lower case: ``add(x, x)`` is ``2*x``,
``mul``, ``pow``, ``eq``, ``diff``, ...; the operators of ``Expr`` call
them, and ``doit`` rebuilds a tree with them. Everything else keeps
SymPy's names and meanings (``symbols``, ``S``, ``sympify``, ``args``,
``func``, ``subs``, ``as_coeff_Mul``, the ``is_*`` assumptions, the
printing), and ``to_sympy``/``from_sympy`` convert in both directions,
so that the rest of SymPy stays usable.

See ``docs/core.md`` for the design and the differences with
``sympy.core``.
"""
from __future__ import annotations

from .add import Add, add, sub
from .assumptions import Truth
from .basic import Atom, Basic, preorder_traversal
from .containers import Tuple
from .expr import AtomicExpr, Expr, ExprLike, as_expr
from .function import (AppliedSymPyFunction, AppliedUndef, Application, Derivative, Function,
    Lambda, SymPyFunction, UndefinedFunction, apply_sympy, diff, expand)
from .interop import from_sympy, to_sympy
from .logic import Boolean, BooleanAtom, BooleanFalse, BooleanTrue, false, true
from .mul import Mul, div, mul, neg
from .numbers import (ComplexInfinity, Float, ImaginaryUnit, Infinity, Integer, NaN,
    NegativeInfinity, Number, NumberSymbol, Rational, E, I, nan, oo, pi, zoo)
from .power import Pow, cbrt, pow, root, sqrt
from .printing import srepr, sstr
from .relational import (Eq, Equality, Ge, GreaterThan, Gt, Le, LessThan, Lt, Ne, Relational,
    StrictGreaterThan, StrictLessThan, Unequality, eq, ge, gt, le, lt, ne)
from .singleton import S
from .sorting import default_sort_key, ordered
from .symbol import Dummy, Symbol, symbols
from .sympify import SympifyError, sympify

__all__ = [
    'Basic', 'Atom', 'preorder_traversal', 'Expr', 'AtomicExpr', 'ExprLike', 'as_expr', 'Truth',
    'Symbol', 'Dummy', 'symbols',
    'Number', 'Integer', 'Rational', 'Float', 'NumberSymbol', 'Infinity', 'NegativeInfinity',
    'ComplexInfinity', 'NaN', 'ImaginaryUnit', 'pi', 'E', 'I', 'oo', 'zoo', 'nan',
    'Add', 'add', 'sub', 'Mul', 'mul', 'div', 'neg', 'Pow', 'pow', 'sqrt', 'cbrt', 'root',
    'Function', 'UndefinedFunction', 'AppliedUndef', 'Application', 'SymPyFunction',
    'AppliedSymPyFunction', 'apply_sympy', 'Lambda', 'Derivative', 'diff', 'expand',
    'Tuple', 'Boolean', 'BooleanAtom', 'BooleanTrue', 'BooleanFalse', 'true', 'false',
    'Relational', 'Eq', 'Ne', 'Lt', 'Le', 'Gt', 'Ge', 'Equality', 'Unequality',
    'StrictLessThan', 'LessThan', 'StrictGreaterThan', 'GreaterThan',
    'eq', 'ne', 'lt', 'le', 'gt', 'ge',
    'S', 'sympify', 'SympifyError', 'to_sympy', 'from_sympy', 'sstr', 'srepr',
    'default_sort_key', 'ordered',
]
