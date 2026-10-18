"""The base class of the expression tree.

:class:`Basic` is the root of the class hierarchy of this experimental core.
Its instances are immutable trees: a node is identified by its class and
its ``args``, two nodes are equal exactly when their classes and their
arguments are equal, and the hash is derived from the same data. Nothing
is evaluated when a node is built: ``Add(x, x)`` is a node with the two
arguments ``x`` and ``x``, and stays so. Evaluation is the job of the
functions ``add``, ``mul``, ``pow``, ``diff``, ... and of :meth:`Basic.doit`,
which rebuilds a tree with those functions. The rule is the one followed
by ``sympy.matrices.expressions`` (``MatAdd(A, A)`` is not ``2*A``) and by
the classes of this package (see AGENTS.md, rule 7): a constructor is a
constructor, so its return type is the class, and the tree is typed.

The methods are those of ``sympy.core.basic.Basic`` with the same names
and meanings (``args``, ``func``, ``atoms``, ``free_symbols``, ``has``,
``xreplace``, ``subs``, ``doit``, ``count``, ``sort_key``), with one
difference: ``xreplace`` is purely structural (the constructors do not
evaluate, so the result is not simplified) while ``subs`` evaluates the
nodes it rebuilds, as SymPy's does.

References
==========

.. [1] SymPy Development Team, *SymPy documentation: Basic*,
       https://docs.sympy.org/latest/modules/core.html#basic
.. [2] O. Čertík et al., *SymPy: symbolic computing in Python*, PeerJ
       Computer Science 3:e103, 2017, section 2 (the expression tree).
"""
from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING, Callable, Iterable, Iterator, Mapping, Optional, Union

from .sorting import ArgsKey, ClassKey, SortKey

if TYPE_CHECKING:
    from .expr import Expr
    from .printing import StrPrinter
    from .symbol import Symbol

__all__ = ['Basic', 'Atom', 'preorder_traversal', 'SubsValue', 'SubsMapping', 'SubsRules', 'SubsArg']

#: a type pattern for :meth:`Basic.atoms` and :meth:`Basic.has`
TypePattern = type['Basic']
#: what a substitution may replace a subtree with
SubsValue = Union['Basic', int, float, Fraction]
#: a mapping of a substitution (``Mapping`` is invariant in its key type,
#: so the usual key types are listed)
SubsMapping = Union[Mapping['Basic', SubsValue], Mapping['Expr', SubsValue], Mapping['Symbol', SubsValue]]
#: the rules of a substitution: a mapping or pairs
SubsRules = Union[SubsMapping, Iterable[tuple['Basic', SubsValue]]]
#: one positional argument of :meth:`Basic.subs`
SubsArg = Union[SubsValue, SubsRules]

_ONE_KEY = Fraction(1)


class Basic:
    """A node of the expression tree: a class and a tuple of arguments.

    Subclasses store their arguments in slots, expose them through
    :attr:`args`, and never evaluate in their constructor.

    Examples
    ========

    >>> from sympy_extras.core import Symbol, Add
    >>> x = Symbol('x')
    >>> e = Add(x, x)
    >>> e.args
    (x, x)
    >>> e.func(*e.args) == e
    True
    >>> e.doit()
    2*x
    """

    __slots__ = ('_hash',)

    # Class flags of SymPy, usable without importing the classes.
    is_Atom: bool = False
    is_Symbol: bool = False
    is_Dummy: bool = False
    is_Number: bool = False
    is_Rational: bool = False
    is_Float: bool = False
    is_NumberSymbol: bool = False
    is_Add: bool = False
    is_Mul: bool = False
    is_Pow: bool = False
    is_Function: bool = False
    is_Derivative: bool = False
    is_Relational: bool = False
    is_Equality: bool = False
    is_Boolean: bool = False
    is_Lambda: bool = False
    is_Tuple: bool = False

    def __init__(self) -> None:
        self._hash: int = hash((type(self).__name__, self._hashable_content()))

    @property
    def is_Integer(self) -> bool:
        """Whether the node is an integer (``Rational(2, 1)`` is one)."""
        return False

    # ------------------------------------------------------------------
    # structure

    @property
    def args(self) -> tuple[Basic, ...]:
        """The arguments of the node."""
        return ()

    @property
    def func(self) -> Callable[..., Basic]:
        """The constructor of the node: ``e.func(*e.args) == e``."""
        return type(self)

    def _hashable_content(self) -> tuple[Basic, ...]:
        return self.args

    def _rebuild(self, args: tuple[Basic, ...]) -> Basic:
        """The node with ``args`` in place of its arguments, unevaluated."""
        if not args and not self.args:
            return self
        return self.func(*args)

    def _evaluated(self, args: tuple[Basic, ...]) -> Basic:
        """The evaluated node with ``args`` in place of its arguments.

        Subclasses with an evaluating function (``Add`` and ``add``, ...)
        override this; the default rebuilds the node.
        """
        return self._rebuild(args)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Basic):
            return NotImplemented
        if type(self) is not type(other):
            return False
        return self._hashable_content() == other._hashable_content()

    def __ne__(self, other: object) -> bool:
        result = self.__eq__(other)
        if result is NotImplemented:
            return NotImplemented
        return not result

    def __hash__(self) -> int:
        return self._hash

    # ------------------------------------------------------------------
    # printing

    def __str__(self) -> str:
        from .printing import sstr
        return sstr(self)

    def __repr__(self) -> str:
        return str(self)

    def _sympystr(self, printer: StrPrinter) -> str:
        """The printed form; subclasses with a special form override it."""
        return printer.generic(self)

    # ------------------------------------------------------------------
    # ordering

    @classmethod
    def class_key(cls) -> ClassKey:
        """The class part of :meth:`sort_key`."""
        return (5, 0, cls.__name__)

    def sort_key(self) -> SortKey:
        """A key which orders expressions canonically (``default_sort_key``)."""
        args: ArgsKey = (len(self.args), tuple(a.sort_key() for a in self.args))
        return (self.class_key(), args, '', (), _ONE_KEY)

    def compare(self, other: Basic) -> int:
        """``-1``, ``0`` or ``1`` as ``self`` sorts before, with, or after
        ``other``."""
        a, b = self.sort_key(), other.sort_key()
        if a < b:
            return -1
        if a > b:
            return 1
        return 0

    # ------------------------------------------------------------------
    # traversal

    def atoms(self, *types: TypePattern) -> set[Basic]:
        """The atoms of the tree, or its nodes of the given classes.

        Examples
        ========

        >>> from sympy_extras.core import symbols, Integer, Symbol
        >>> x, y = symbols('x y')
        >>> sorted((x + 2*y).atoms(), key=str)
        [2, x, y]
        >>> (x + 2*y).atoms(Symbol) == {x, y}
        True
        """
        found: set[Basic] = set()
        for node in preorder_traversal(self):
            if types:
                if isinstance(node, types):
                    found.add(node)
            elif node.is_Atom:
                found.add(node)
        return found

    @property
    def free_symbols(self) -> set[Symbol]:
        """The symbols of the tree which are not bound (see
        :class:`~sympy_extras.core.function.Lambda`)."""
        result: set[Symbol] = set()
        for a in self.args:
            result |= a.free_symbols
        return result

    def has(self, *patterns: Union[Basic, TypePattern]) -> bool:
        """Whether a subtree equals one of the ``patterns``, or is an
        instance of one of the classes among them.

        Examples
        ========

        >>> from sympy_extras.core import symbols, Integer
        >>> x, y = symbols('x y')
        >>> (x + 1).has(x), (x + 1).has(y), (x + 1).has(Integer)
        (True, False, True)
        """
        classes = tuple(p for p in patterns if isinstance(p, type))
        nodes = [p for p in patterns if isinstance(p, Basic)]
        for node in preorder_traversal(self):
            if classes and isinstance(node, classes):
                return True
            if any(node == p for p in nodes):
                return True
        return False

    def count(self, pattern: Union[Basic, TypePattern]) -> int:
        """The number of subtrees equal to ``pattern`` (or instances of it)."""
        if isinstance(pattern, type):
            return sum(1 for node in preorder_traversal(self) if isinstance(node, pattern))
        return sum(1 for node in preorder_traversal(self) if node == pattern)

    # ------------------------------------------------------------------
    # replacement

    def xreplace(self, rule: Mapping[Basic, Basic]) -> Basic:
        """The tree with the keys of ``rule`` replaced by their values,
        without evaluation (the constructors do not evaluate).

        Examples
        ========

        >>> from sympy_extras.core import symbols
        >>> x, y = symbols('x y')
        >>> (x + y).xreplace({x: 2}).args
        (2, y)
        >>> (x + y).xreplace({x: 2}).doit()
        y + 2
        """
        return self._xreplace(rule)

    def _xreplace(self, rule: Mapping[Basic, Basic]) -> Basic:
        if self in rule:
            return rule[self]
        if not self.args:
            return self
        args = tuple(a._xreplace(rule) for a in self.args)
        if all(a is b for a, b in zip(args, self.args)):
            return self
        return self._rebuild(args)

    def subs(self, *args: SubsArg) -> Basic:
        """Substitution with evaluation of the rebuilt nodes.

        Accepts ``subs(old, new)``, ``subs({old: new, ...})`` and
        ``subs([(old, new), ...])``, applied in sequence as SymPy does.
        A sum or a product among the keys is replaced in sums or products
        which contain all of its terms or factors.

        Examples
        ========

        >>> from sympy_extras.core import symbols
        >>> x, y, z = symbols('x y z')
        >>> (x + 1).subs(x, 1)
        2
        >>> (x*y*z).subs({x: 2, y: 3})
        6*z
        >>> (x + y + 1).subs(x + y, z)
        z + 1
        """
        from .sympify import sympify
        pairs: list[tuple[Basic, Basic]] = []
        if len(args) == 2:
            old, new = args
            if isinstance(old, (Basic, int, float, Fraction)) and isinstance(new, (Basic, int, float, Fraction)):
                pairs.append((sympify(old), sympify(new)))
            else:
                raise TypeError("subs(old, new) takes two expressions")
        elif len(args) == 1:
            rules = args[0]
            if isinstance(rules, Mapping):
                for k, v in rules.items():
                    pairs.append((sympify(k), sympify(v)))
            elif isinstance(rules, Iterable) and not isinstance(rules, (Basic, int, float, Fraction)):
                for k, v in rules:
                    pairs.append((sympify(k), sympify(v)))
            else:
                raise TypeError("subs takes (old, new), a mapping or pairs")
        else:
            raise TypeError("subs takes (old, new), a mapping or pairs")
        result: Basic = self
        for old, new in pairs:
            result = result._subs(old, new)
        return result

    def _subs(self, old: Basic, new: Basic) -> Basic:
        if self == old:
            return new
        if not self.args:
            return self
        replaced = self._subs_partial(old, new)
        if replaced is not None:
            return replaced
        args = tuple(a._subs(old, new) for a in self.args)
        if all(a is b for a, b in zip(args, self.args)):
            return self
        return self._evaluated(args)

    def _subs_partial(self, old: Basic, new: Basic) -> Optional[Basic]:
        """Replacement of ``old`` among the arguments of an associative
        node; ``None`` when not applicable (overridden by ``Add``, ``Mul``)."""
        return None

    def doit(self) -> Basic:
        """The tree evaluated bottom up with the evaluating functions.

        Examples
        ========

        >>> from sympy_extras.core import Symbol, Add, Mul
        >>> x = Symbol('x')
        >>> Add(x, x, Mul(2, x)).doit()
        4*x
        """
        if not self.args:
            return self
        args = tuple(a.doit() for a in self.args)
        return self._evaluated(args)

    def is_same(self, other: Basic) -> bool:
        """Structural identity, the same as ``==``."""
        return self == other


class Atom(Basic):
    """A leaf of the tree: a node without arguments."""

    __slots__ = ()

    is_Atom = True

    @classmethod
    def class_key(cls) -> ClassKey:
        return (2, 0, cls.__name__)

    def sort_key(self) -> SortKey:
        return (self.class_key(), (1, ()), str(self), (), _ONE_KEY)


def preorder_traversal(node: Basic) -> Iterator[Basic]:
    """The nodes of the tree, parents before children.

    Examples
    ========

    >>> from sympy_extras.core import symbols, preorder_traversal
    >>> x, y = symbols('x y')
    >>> list(preorder_traversal(x + 2*y))
    [x + 2*y, x, 2*y, 2, y]
    """
    yield node
    for a in node.args:
        yield from preorder_traversal(a)

