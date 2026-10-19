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
``xreplace``, ``subs``, ``doit``, ``count``, ``sort_key``, ``match``,
``replace``, ``find``, ``as_dummy``, ``dummy_eq``, ``rcall``, ...), with
one difference: ``xreplace`` is purely structural (the constructors do
not evaluate, so the result is not simplified) while ``subs`` and
``replace`` evaluate the nodes they rebuild, as SymPy's do.

References
==========

.. [1] SymPy Development Team, *SymPy documentation: Basic*,
       https://docs.sympy.org/latest/modules/core.html#basic
.. [2] O. Čertík et al., *SymPy: symbolic computing in Python*, PeerJ
       Computer Science 3:e103, 2017, section 2 (the expression tree).
"""
from __future__ import annotations

from fractions import Fraction
from typing import (TYPE_CHECKING, Callable, Iterable, Iterator, Literal, Mapping, Optional,
    Protocol, Union, overload)

from .kind import Kind, UndefinedKind
from .sorting import ArgsKey, ClassKey, CompareItem, CompareKey, SortKey, class_rank

if TYPE_CHECKING:
    from sympy.core.basic import Basic as SymPyBasic

    from .expr import Expr
    from .logic import Boolean
    from .function import FunctionClass
    from .printing import StrPrinter
    from .symbol import Dummy, Symbol

__all__ = ['Basic', 'Atom', 'preorder_traversal', 'SubsValue', 'SubsMapping', 'SubsRules',
    'SubsArg', 'ReplacementRule', 'MatchDict', 'Query', 'Replacement']

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
#: the bindings of a successful match: the wilds of the pattern mapped
#: to subtrees of the expression
MatchDict = dict['Basic', 'Basic']
#: what :meth:`Basic.replace`, :meth:`Basic.find` and :meth:`Basic.count`
#: look for: a pattern with wilds, a class, a function object, a predicate
Query = Union['Basic', type['Basic'], 'FunctionClass', Callable[['Basic'], bool]]
#: what :meth:`Basic.replace` puts in place of a match: an expression
#: with the wilds of the pattern, a function object, or a callable taking
#: the match (by name) or the arguments or the node, see the method
Replacement = Union['Basic', 'FunctionClass', Callable[..., 'Basic']]

_ONE_KEY = Fraction(1)


class ReplacementRule(Protocol):
    """What :meth:`Basic.xreplace` accepts: a mapping or a
    :class:`~sympy_extras.core.rules.Transform`."""

    def __contains__(self, item: Basic) -> bool: ...

    def __getitem__(self, item: Basic) -> Basic: ...


#: an assumption for ``refine``: a boolean node, a Python bool or a SymPy boolean
RefineArg = Union['Boolean', bool, 'SymPyBasic']


def _as_boolean(b: object) -> 'Boolean':
    from .logic import to_boolean
    return to_boolean(b)


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

    __slots__ = ('_hash', '_compare_key_cache', '_sympy_cache')

    # Class flags of SymPy, usable without importing the classes.
    is_Atom: bool = False
    is_Symbol: bool = False
    is_Dummy: bool = False
    is_Wild: bool = False
    is_Number: bool = False
    is_Rational: bool = False
    is_Float: bool = False
    is_NumberSymbol: bool = False
    is_AlgebraicNumber: bool = False
    is_Add: bool = False
    is_Mul: bool = False
    is_Pow: bool = False
    is_Function: bool = False
    is_Derivative: bool = False
    is_Order: bool = False
    is_Relational: bool = False
    is_Equality: bool = False
    is_Boolean: bool = False
    is_Not: bool = False
    is_Lambda: bool = False
    is_Tuple: bool = False
    is_Matrix: bool = False
    is_MatAdd: bool = False
    is_MatMul: bool = False
    is_Vector: bool = False
    is_Point: bool = False
    is_Poly: bool = False
    is_Piecewise: bool = False
    is_Indexed: bool = False
    #: whether the node is a scalar expression (``Expr``)
    is_scalar: bool = False
    #: whether the node is a symbol (``Symbol``, ``Dummy``, ``Wild``)
    is_symbol: bool = False

    def __init__(self) -> None:
        self._hash: int = hash((type(self).__name__, self._hashable_content()))
        self._compare_key_cache: Optional[CompareKey] = None
        self._sympy_cache: Optional[SymPyBasic] = None
        self._after_init()

    def _after_init(self) -> None:
        """A hook run at the end of the construction (``Expr`` resets its
        cache of facts here)."""

    @property
    def is_Integer(self) -> bool:
        """Whether the node is an integer (``Rational(2, 1)`` is one)."""
        return False

    @property
    def is_comparable(self) -> bool:
        """Whether the node is a number with a real value which can be
        evaluated (``pi`` is, ``oo`` and ``2 + I`` are not)."""
        return False

    @property
    def kind(self) -> Kind:
        """The kind of the node (:mod:`~sympy_extras.core.kind`)."""
        return UndefinedKind

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

    @classmethod
    def fromiter(cls, args: Iterable[Basic]) -> Basic:
        """The node of this class with the given arguments.

        Examples
        ========

        >>> from sympy_extras.core import symbols, Add
        >>> x, y = symbols('x y')
        >>> Add.fromiter([x, y])
        x + y
        """
        return cls(*tuple(args))

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

    def copy(self) -> Basic:
        """A node equal to this one (``func(*args)``)."""
        return self._rebuild(self.args)

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

    def _compare_name(self) -> str:
        """The name of the class in SymPy, for the ``compare`` order."""
        return type(self).__name__

    def _compare_content(self) -> tuple[CompareItem, ...]:
        """The content of the node in the ``compare`` order: the keys of
        the arguments by default (SymPy's ``_hashable_content``)."""
        return tuple(a.compare_key() for a in self.args)

    def compare_key(self) -> CompareKey:
        """The key of SymPy's ``compare`` order: the rank of the class,
        then the length of the content, then the content; the order of
        the arguments of sums and products. Computed once per node.

        Examples
        ========

        >>> from sympy_extras.core import symbols
        >>> x, y = symbols('x y')
        >>> x.compare_key() < y.compare_key() < (x**2).compare_key() < (x*y).compare_key()
        True
        """
        cached = self._compare_key_cache
        if cached is None:
            content = self._compare_content()
            cached = (class_rank(self._compare_name()), len(content), content)
            self._compare_key_cache = cached
        return cached

    def compare(self, other: Basic) -> int:
        """``-1``, ``0`` or ``1`` as ``self`` sorts before, with, or after
        ``other`` in the order of the arguments of sums and products (the
        order of SymPy's ``Basic.compare``: by class, then by content).

        Examples
        ========

        >>> from sympy_extras.core import symbols, Symbol
        >>> x, y = symbols('x y')
        >>> x.compare(y), y.compare(x), x.compare(x), x.compare(x**2), Symbol('n', integer=True).compare(y)
        (-1, 1, 0, -1, 1)
        """
        if self is other:
            return 0
        a, b = self.compare_key(), other.compare_key()
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

    @property
    def bound_symbols(self) -> tuple[Symbol, ...]:
        """The symbols bound by this node (the variables of a ``Lambda``
        or a ``Subs``); empty for the other nodes."""
        return ()

    @property
    def canonical_variables(self) -> dict[Symbol, Dummy]:
        """The bound symbols of the node mapped to canonical dummies
        ``_0``, ``_1``, ... (see :meth:`as_dummy`).

        Examples
        ========

        >>> from sympy_extras.core import symbols, Lambda
        >>> x, y = symbols('x y')
        >>> Lambda(x, x + y).canonical_variables
        {x: _0}
        """
        from .symbol import Dummy
        return {s: Dummy(str(i)) for i, s in enumerate(self.bound_symbols)}

    def as_dummy(self) -> Basic:
        """The tree with its bound symbols renamed to canonical dummies.

        Examples
        ========

        >>> from sympy_extras.core import symbols, Lambda
        >>> x, y = symbols('x y')
        >>> Lambda(x, x + y).as_dummy()
        Lambda(_0, _0 + y)
        """
        from .symbol import Dummy
        return self._as_dummy(lambda i: Dummy(str(i)))

    def _as_dummy(self, make: Callable[[int], Symbol]) -> Basic:
        node: Basic = self
        bound = node.bound_symbols
        if bound:
            rule: dict[Basic, Basic] = {s: make(i) for i, s in enumerate(bound)}
            node = node._xreplace_evaluated(rule)  # rebuilt canonically, as SymPy's xreplace does
        if not node.args:
            return node
        args = tuple(a._as_dummy(make) for a in node.args)
        if all(a is b for a, b in zip(args, node.args)):
            return node
        return node._rebuild(args)

    def dummy_eq(self, other: Basic, symbol: Optional[Symbol] = None) -> bool:
        """Structural equality up to the names of the bound symbols (and
        of ``symbol``, when given, compared with any symbol of ``other``).

        Examples
        ========

        >>> from sympy_extras.core import symbols, Lambda
        >>> x, y = symbols('x y')
        >>> Lambda(x, x).dummy_eq(Lambda(y, y)), Lambda(x, x).dummy_eq(Lambda(y, y + 1))
        (True, False)
        """
        from .symbol import Symbol
        placeholders = [Symbol('_dummy_eq_%d' % i) for i in range(16)]

        def make(i: int) -> Symbol:
            return placeholders[i] if i < len(placeholders) else Symbol('_dummy_eq_%d' % i)
        a, b = self._as_dummy(make), other._as_dummy(make)
        if symbol is not None:
            for s in other.free_symbols:
                if a == b._xreplace({s: symbol}):
                    return True
            return False
        return a == b

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

    def has_free(self, *patterns: Basic) -> bool:
        """Whether one of the ``patterns`` occurs free: its symbols are free
        symbols of the tree and substituting it changes the tree (so
        ``x + y`` occurs in ``x + y + z``, as for ``subs``).

        Examples
        ========

        >>> from sympy_extras.core import symbols, Lambda
        >>> x, y, z = symbols('x y z')
        >>> (x + y).has_free(x), Lambda(x, x + y).has_free(x), Lambda(x, x + y).has_free(y)
        (True, False, True)
        >>> (x + y + z).has_free(x + y), (x + y + z).has_xfree({x + y})
        (True, False)
        """
        from .symbol import Dummy
        free = self.free_symbols
        for pattern in patterns:
            if pattern.free_symbols and not pattern.free_symbols <= free:
                continue
            if self._subs(pattern, Dummy('free')) != self:
                return True
        return False

    def has_xfree(self, patterns: set[Basic]) -> bool:
        """Whether one of the ``patterns`` (a set) is a free subtree, see
        :meth:`has_free`."""
        for node, scope in _scoped_traversal(self, frozenset()):
            if node in patterns and not (node.free_symbols & scope):
                return True
        return False

    def count(self, query: Query) -> int:
        """The number of subtrees matching ``query`` (see :meth:`find`).

        Examples
        ========

        >>> from sympy_extras.core import symbols, Wild
        >>> x, y = symbols('x y')
        >>> (x*y + 2*x).count(x), (x*y + 2*x).count(Wild('a')*x)
        (2, 5)
        >>> (x*y + 2*x).find(Wild('a')*x) == {x, 2*x, x*y, 2}
        True
        """
        return sum(1 for node in preorder_traversal(self) if _query_matches(node, query))

    def find(self, query: Query) -> set[Basic]:
        """The subtrees matching ``query``: a pattern with wilds, a class, a
        function object (``sin`` or ``Function('f')``) or a predicate.

        Examples
        ========

        >>> from sympy_extras.core import symbols, Wild, Pow
        >>> x, y = symbols('x y')
        >>> (x**2 + x**3).find(Pow) == {x**2, x**3}
        True
        >>> (x**2 + x**3).find(x**Wild('a')) == {x**2, x**3, x}
        True
        >>> (x + 2).find(lambda e: e.is_Integer)
        {2}
        """
        return {node for node in preorder_traversal(self) if _query_matches(node, query)}

    # ------------------------------------------------------------------
    # pattern matching

    def match(self, pattern: Basic, old: bool = False) -> Optional[MatchDict]:
        """The bindings of the wilds of ``pattern`` for which it equals
        ``self``, ``None`` when there are none.

        Examples
        ========

        >>> from sympy_extras.core import symbols, Wild
        >>> x, y = symbols('x y')
        >>> a, b = Wild('a'), Wild('b')
        >>> (2*x + 3).match(a*x + b) == {a: 2, b: 3}
        True
        >>> (x**2).match(a*x**b) == {a: 1, b: 2}
        True
        >>> x.match(a + x) == {a: 0}
        True
        >>> (x**2).match(a*x + b) == {a: x, b: 0}
        True
        >>> a, b = Wild('a', exclude=[x]), Wild('b', exclude=[x])
        >>> (x**2 + 3).match(a*x + b)
        """
        from .matching import match
        return match(self, pattern)

    def matches(self, expr: Basic, repl_dict: Optional[MatchDict] = None, old: bool = False) -> Optional[MatchDict]:
        """``expr.match(self)`` extended with the bindings ``repl_dict``."""
        from .matching import match
        return match(expr, self, repl_dict)

    @overload
    def replace(self, query: Query, value: Replacement, map: Literal[False] = False,
                simultaneous: bool = True, exact: Optional[bool] = None) -> Basic: ...

    @overload
    def replace(self, query: Query, value: Replacement, map: Literal[True],
                simultaneous: bool = True, exact: Optional[bool] = None) -> tuple[Basic, dict[Basic, Basic]]: ...

    def replace(self, query: Query, value: Replacement, map: bool = False,
                simultaneous: bool = True, exact: Optional[bool] = None) -> Union[Basic, tuple[Basic, dict[Basic, Basic]]]:
        """The tree with the subtrees matching ``query`` replaced, bottom
        up, by ``value``; the rebuilt nodes are evaluated.

        ``query`` and ``value`` go together as in SymPy:

        * a pattern with wilds and an expression with the same wilds (or a
          callable receiving the bindings as keyword arguments by name);
        * a class or a function object (``sin``, ``Function('f')``) and a
          function object or a callable receiving the arguments;
        * a predicate on nodes and a callable receiving the node.

        With ``map=True`` the mapping of the replaced subtrees is returned
        too. With ``exact=True`` a wild does not match the identity of a
        sum or a product (``0`` or ``1``). ``simultaneous`` is accepted for
        compatibility: a replaced subtree is never examined again.

        Examples
        ========

        >>> from sympy_extras.core import symbols, Wild, Function
        >>> x, y = symbols('x y')
        >>> a = Wild('a')
        >>> f, g = Function('f'), Function('g')
        >>> (x*y + 2*x).replace(a*x, a)
        y + 2/x
        >>> (f(x) + f(y)).replace(f, g)
        g(x) + g(y)
        >>> (x + 2).replace(lambda e: e.is_Integer, lambda e: 2*e)
        x + 4
        >>> (f(x) + f(y)).replace(f(a), a + 1)
        x + y + 2
        """
        mapping: dict[Basic, Basic] = {}
        result = self._replace(query, value, mapping, bool(exact))
        if map:
            return (result, mapping)
        return result

    def _replace(self, query: Query, value: Replacement, mapping: dict[Basic, Basic], exact: bool) -> Basic:
        node: Basic = self
        if node.args:
            args = tuple(a._replace(query, value, mapping, exact) for a in node.args)
            if not all(a is b for a, b in zip(args, node.args)):
                node = node._evaluated(args)
        replacement = _apply_query(node, query, value, exact)
        if replacement is None:
            return node
        mapping[node] = replacement
        return replacement

    # ------------------------------------------------------------------
    # replacement

    def xreplace(self, rule: Union[ReplacementRule, SubsMapping]) -> Basic:
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
        return self._xreplace(_as_rule(rule))

    def _xreplace(self, rule: ReplacementRule) -> Basic:
        if self in rule:
            return rule[self]
        if not self.args:
            return self
        args = tuple(a._xreplace(rule) for a in self.args)
        if all(a is b for a, b in zip(args, self.args)):
            return self
        return self._rebuild(args)

    def _xreplace_evaluated(self, rule: ReplacementRule) -> Basic:
        """``xreplace`` with the rebuilt nodes evaluated."""
        if self in rule:
            return rule[self]
        if not self.args:
            return self
        args = tuple(a._xreplace_evaluated(rule) for a in self.args)
        if all(a is b for a, b in zip(args, self.args)):
            return self
        return self._evaluated(args)

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

    def rcall(self, *args: Basic) -> Basic:
        """The callables of the tree (``Lambda``) applied to ``args``.

        Examples
        ========

        >>> from sympy_extras.core import symbols, Lambda, Integer, Tuple
        >>> x, y = symbols('x y')
        >>> Tuple(Lambda(x, x)(x), Lambda(y, y**2)).rcall(Integer(2))
        (x, 4)
        """
        if not self.args:
            return self
        new_args = tuple(a.rcall(*args) for a in self.args)
        return self._evaluated(new_args)

    def is_same(self, other: Basic) -> bool:
        """Structural identity, the same as ``==``."""
        return self == other

    # ------------------------------------------------------------------
    # through SymPy

    def simplify(self) -> Basic:
        """SymPy's ``simplify`` of the node (converted and back).

        Examples
        ========

        >>> from sympy_extras.core import Symbol, Eq
        >>> x = Symbol('x')
        >>> Eq(x + x, 2*x).simplify()
        True
        """
        import sympy
        from .interop import through_sympy
        return through_sympy(self, sympy.simplify)

    def rewrite(self, *args: Union[Basic, FunctionClass, type]) -> Basic:
        """SymPy's ``rewrite`` of the node, in terms of the given functions.

        Examples
        ========

        >>> import sympy
        >>> from sympy_extras.core import Symbol, SymPyFunction
        >>> x = Symbol('x')
        >>> SymPyFunction(sympy.sin)(x).rewrite(sympy.exp)
        -I*(exp(I*x) - exp(-I*x))/2
        """
        from .interop import rewrite_targets, through_sympy
        targets = rewrite_targets(args)
        return through_sympy(self, lambda s: s.rewrite(*targets))

    def refine(self, assumption: RefineArg = True) -> Basic:
        """The node simplified under the assumption (and the global ones):
        :func:`~sympy_extras.core.refine.refine`. A SymPy assumption is
        accepted too.

        Examples
        ========

        >>> from sympy_extras.core import Symbol, sqrt, Q
        >>> x = Symbol('x')
        >>> sqrt(x**2).refine(Q.positive(x)), sqrt(x**2).refine(Q.real(x))
        (x, Abs(x))
        """
        from .refine import refine
        if isinstance(assumption, (Basic, bool)):
            return refine(self, _as_boolean(assumption))
        from .interop import from_sympy
        return refine(self, _as_boolean(from_sympy(assumption)))

    def count_ops(self, visual: bool = False) -> Basic:
        """The number of operations of the tree (:func:`~sympy_extras.core.function.count_ops`)."""
        from .function import count_ops
        return count_ops(self, visual)


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


def _scoped_traversal(node: Basic, scope: frozenset[Symbol]) -> Iterator[tuple[Basic, frozenset[Symbol]]]:
    """The nodes of the tree with the symbols bound around each of them."""
    yield (node, scope)
    inner = scope | frozenset(node.bound_symbols)
    for a in node.args:
        yield from _scoped_traversal(a, inner)


def _as_rule(rule: Union[ReplacementRule, SubsMapping]) -> ReplacementRule:
    if isinstance(rule, Mapping):
        from .sympify import sympify
        return {k: sympify(v) for k, v in rule.items()}
    return rule


def _query_matches(node: Basic, query: Query) -> bool:
    from .function import FunctionClass
    if isinstance(query, Basic):
        return node.match(query) is not None
    if isinstance(query, type):
        return isinstance(node, query)
    if isinstance(query, FunctionClass):
        return node.func == query
    return bool(query(node))


def _apply_query(node: Basic, query: Query, value: Replacement, exact: bool) -> Optional[Basic]:
    """``value`` applied to ``node`` when ``query`` matches it, else ``None``."""
    from .function import FunctionClass
    from .matching import match
    if isinstance(query, Basic):
        bindings = match(node, query, exact=exact)
        if bindings is None:
            return None
        if isinstance(value, Basic):
            return value._xreplace_evaluated(bindings)
        if isinstance(value, FunctionClass):
            return value(*_expr_args(node))
        return value(**{_wild_name(k): v for k, v in bindings.items()})
    if isinstance(query, (type, FunctionClass)):
        found = isinstance(node, query) if isinstance(query, type) else node.func == query
        if not found:
            return None
        if isinstance(value, Basic):
            return value
        if isinstance(value, FunctionClass):
            return value(*_expr_args(node))
        return value(*node.args)
    if not query(node):
        return None
    if isinstance(value, Basic):
        return value
    if isinstance(value, FunctionClass):
        return value(*_expr_args(node))
    return value(node)


def _expr_args(node: Basic) -> list[Expr]:
    from .expr import as_expr
    return [as_expr(a) for a in node.args]


def _wild_name(wild: Basic) -> str:
    from .function import WildFunction
    from .symbol import Wild
    if isinstance(wild, (Wild, WildFunction)):
        return wild.name
    raise TypeError("a wild is expected, got %s" % (wild,))
