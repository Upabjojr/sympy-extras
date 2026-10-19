"""An experimental core: unevaluating classes, evaluating functions.

The classes of the expression tree (``Add``, ``Mul``, ``Pow``, ``Mod``,
``Eq``, ``Derivative``, ``Subs``, ``Order``, ...) build exactly the node
they are asked for, so that their types are honest: ``Add(x, x)`` is an
``Add`` with the arguments ``(x, x)``. The evaluation of SymPy's
constructors is done by the functions of the same name in lower case:
``add(x, x)`` is ``2*x``, ``mul``, ``pow``, ``mod``, ``eq``, ``diff``,
``order``, ...; the operators of ``Expr`` call them, and ``doit`` rebuilds
a tree with them. Everything else keeps SymPy's names and meanings
(``symbols``, ``S``, ``sympify``, ``args``, ``func``, ``subs``, ``match``,
``replace``, ``as_coeff_Mul``, the ``is_*`` assumptions, ``series``,
``simplify``, the printing), and ``to_sympy``/``from_sympy`` convert in
both directions, so that the rest of SymPy stays usable.

See ``docs/core.md`` for the design and the differences with
``sympy.core``.
"""
from __future__ import annotations

from .add import Add, add, sub
from .ask import AssumptionKeys, Q, ask
from .assume import AppliedPredicate, AssumptionsContext, Predicate, assuming, global_assumptions
from .alphabets import greeks
from .assumptions import (Truth, assumptions, check_assumptions, common_assumptions, failing_assumptions)
from .basic import Atom, Basic, preorder_traversal
from .cache import cacheit, clear_cache
from .containers import Dict, Tuple
from .coreerrors import BaseCoreError, NonCommutativeExpression, PoleError, PrecisionExhausted
from .evalf import N, evalf
from .expr import AtomicExpr, Expr, ExprLike, UnevaluatedExpr, as_expr
from .exprtools import factor_nc, factor_terms, gcd_terms
from .function import (AppliedSymPyFunction, AppliedUndef, Application, Derivative, Function, FunctionClass,
    Lambda, Subs, SymPyFunction, UndefinedFunction, WildFunction, apply_sympy, arity, count_ops, diff, expand,
    expand_complex, expand_func, expand_log, expand_mul, expand_multinomial, expand_power_base,
    expand_power_exp, expand_trig, nfloat)
from .interop import SymPyExprNode, SymPyNode, from_sympy, to_sympy
from .intfunc import igcd, igcdex, ilcm, integer_log, integer_nthroot, isqrt, mod_inverse, num_digits, trailing
from .kind import BooleanKind, Kind, NumberKind, TupleKind, UndefinedKind
from .logic import (ITE, And, Boolean, BooleanAtom, BooleanFalse, BooleanFunction, BooleanTrue, Equivalent,
    Implies, Nand, Nor, Not, Or, Xnor, Xor, and_, equivalent, false, implies, ite, not_, or_, true, xor)
from .mod import Mod, mod
from .mul import Mul, div, mul, neg, prod
from .multidimensional import vectorize
from .numbers import (AlgebraicNumber, Catalan, ComplexInfinity, EulerGamma, Float, GoldenRatio, ImaginaryUnit,
    Infinity, Integer, NaN, NegativeInfinity, Number, NumberSymbol, Rational, RealNumber, TribonacciConstant,
    E, I, comp, nan, oo, pi, seterr, zoo)
from .operations import AssocOp, LatticeOp
from .order import O, Order, order
from .parameters import distribute, evaluate, global_parameters
from .power import Pow, cbrt, pow, root, sqrt
from .printing import srepr, sstr
from .refine import refine
from .relational import (Eq, Equality, Ge, GreaterThan, Gt, Le, LessThan, Lt, Ne, Rel, Relational,
    StrictGreaterThan, StrictLessThan, Unequality, eq, ge, gt, le, lt, ne)
from .rules import Transform
from .singleton import S
from .sorting import default_sort_key, ordered
from .symbol import Dummy, Symbol, Wild, symbols, var
from .sympify import SympifyError, sympify
from .traversal import bottom_up, iterargs, iterfreeargs, postorder_traversal, use

__all__ = [
    'Basic', 'Atom', 'preorder_traversal', 'postorder_traversal', 'bottom_up', 'use', 'iterargs',
    'iterfreeargs', 'Expr', 'AtomicExpr', 'ExprLike', 'UnevaluatedExpr', 'as_expr', 'Truth',
    'Symbol', 'Dummy', 'Wild', 'symbols', 'var',
    'Number', 'Integer', 'Rational', 'Float', 'RealNumber', 'NumberSymbol', 'Infinity', 'NegativeInfinity',
    'ComplexInfinity', 'NaN', 'ImaginaryUnit', 'AlgebraicNumber', 'pi', 'E', 'EulerGamma', 'Catalan',
    'GoldenRatio', 'TribonacciConstant', 'I', 'oo', 'zoo', 'nan', 'comp', 'seterr',
    'Add', 'add', 'sub', 'Mul', 'mul', 'div', 'neg', 'prod', 'Pow', 'pow', 'sqrt', 'cbrt', 'root',
    'Mod', 'mod', 'AssocOp',
    'FunctionClass', 'Function', 'UndefinedFunction', 'AppliedUndef', 'Application', 'SymPyFunction',
    'AppliedSymPyFunction', 'WildFunction', 'apply_sympy', 'arity', 'Lambda', 'Derivative', 'Subs', 'diff',
    'expand', 'expand_mul', 'expand_multinomial', 'expand_power_base', 'expand_power_exp', 'expand_log',
    'expand_func', 'expand_trig', 'expand_complex', 'count_ops', 'nfloat', 'Order', 'order', 'O',
    'Tuple', 'Dict', 'Boolean', 'BooleanAtom', 'BooleanTrue', 'BooleanFalse', 'true', 'false', 'BooleanFunction',
    'And', 'Or', 'Not', 'Xor', 'Implies', 'Equivalent', 'ITE', 'and_', 'or_', 'not_', 'xor', 'implies',
    'equivalent', 'ite', 'Nand', 'Nor', 'Xnor', 'LatticeOp',
    'Relational', 'Eq', 'Ne', 'Lt', 'Le', 'Gt', 'Ge', 'Equality', 'Unequality',
    'StrictLessThan', 'LessThan', 'StrictGreaterThan', 'GreaterThan', 'Rel',
    'eq', 'ne', 'lt', 'le', 'gt', 'ge',
    'S', 'sympify', 'SympifyError', 'to_sympy', 'from_sympy', 'SymPyNode', 'SymPyExprNode', 'sstr', 'srepr',
    'default_sort_key', 'ordered', 'evalf', 'N',
    'assumptions', 'check_assumptions', 'failing_assumptions', 'common_assumptions',
    'Q', 'ask', 'AssumptionKeys', 'Predicate', 'AppliedPredicate', 'AssumptionsContext', 'global_assumptions',
    'assuming', 'refine',
    'global_parameters', 'evaluate', 'distribute', 'cacheit', 'clear_cache', 'Transform',
    'Kind', 'NumberKind', 'BooleanKind', 'UndefinedKind', 'TupleKind',
    'BaseCoreError', 'NonCommutativeExpression', 'PoleError', 'PrecisionExhausted',
    'igcd', 'ilcm', 'igcdex', 'mod_inverse', 'integer_nthroot', 'integer_log', 'num_digits', 'trailing', 'isqrt',
    'factor_terms', 'gcd_terms', 'factor_nc', 'vectorize', 'greeks',
]
