"""Every native node class has a LaTeX rule: the subclasses of ``Basic``
are enumerated by importing the native modules, and each one must find a
rule in the table (its own or the one of a base), except the abstract
bases, which have no instances. Every function class prints alone, as a
factor and under an exponent."""
from __future__ import annotations

import importlib
import pkgutil

import sympy_extras
from sympy_extras.core import Integer, Mul, Pow, Symbol
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr
from sympy_extras.core.function import DefinedFunction
from sympy_extras.printing import latex
from sympy_extras.printing.latex import RULES

#: the classes without instances of their own, and the wrappers of foreign
#: objects (printed by the fallback, as text)
ABSTRACT = {
    'Atom', 'AtomicExpr', 'Expr', 'Boolean', 'BooleanFunction', '_Connective', 'AssocOp', 'LatticeOp', 'Number',
    'NumberSymbol', 'Set', '_NamedSet', 'SeqBase', '_SeqExpr', '_SeqExprOp', 'SeriesBase', '_FourierBase',
    'ExprWithLimits', 'Quantifier',
    'SymPyNode', 'SymPyExprNode', 'SymPyBooleanNode', 'SymPySetNode',
}


def _native_classes() -> list[type[Basic]]:
    for module in pkgutil.walk_packages(sympy_extras.__path__, 'sympy_extras.'):
        name = module.name
        if '.tests' in name or '_sympy' in name or name.startswith('sympy_extras.interop'):
            continue
        importlib.import_module(name)
    found: list[type[Basic]] = []
    pending: list[type[Basic]] = [Basic]
    while pending:
        cls = pending.pop()
        for sub in cls.__subclasses__():
            if sub not in found and '.tests.' not in sub.__module__:
                found.append(sub)
                pending.append(sub)
    return found


def test_every_node_class_has_a_rule() -> None:
    classes = _native_classes()
    assert len(classes) > 280
    missing = sorted(cls.__name__ for cls in classes if RULES.lookup(cls) is None)
    assert set(missing) <= ABSTRACT, sorted(set(missing) - ABSTRACT)


def _balanced(tex: str) -> bool:
    depth = 0
    for i, c in enumerate(tex):
        if c == '{' and (i == 0 or tex[i - 1] != '\\'):
            depth += 1
        elif c == '}' and (i == 0 or tex[i - 1] != '\\'):
            depth -= 1
            if depth < 0:
                return False
    return depth == 0 and tex.count(r'\left') == tex.count(r'\right')


def test_every_function_prints_alone_as_a_factor_and_under_an_exponent() -> None:
    x, y, z, t = Symbol('x'), Symbol('y'), Symbol('z'), Symbol('t')
    arguments: list[Expr] = [x, y, z, t, x, y]
    printed = 0
    for cls in _native_classes():
        if not issubclass(cls, DefinedFunction) or cls.function_name == 'f':
            continue  # 'f': the abstract bases of families of functions
        count = 2 if cls.variadic or not cls.nargs else min(cls.nargs)
        try:
            node = cls(*arguments[:count])
        except (TypeError, ValueError):
            continue
        alone = latex(node)
        squared = latex(Pow(node, Integer(2)))
        factor = latex(Mul(node, x))
        assert _balanced(alone) and _balanced(squared) and _balanced(factor), (cls.__name__, alone, squared)
        # an exponent never follows a superscript directly
        assert '}^{2}^' not in squared and '^{2}^{' not in squared and r'^\prime^' not in squared, squared
        assert squared != alone and '2' in squared, (cls.__name__, squared)
        printed += 1
    assert printed > 150, printed
