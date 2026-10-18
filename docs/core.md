# An experimental core: unevaluating classes, evaluating functions

`sympy_extras.core` is a proposal for a replacement of `sympy.core` written
from scratch. It keeps the names, the methods and the evaluated forms of
SymPy's core, and changes one thing: **the classes build the node they are
asked for and nothing else**, while **the evaluation of SymPy's constructors
is done by functions** of the same name in lower case.

```python
>>> from sympy_extras.core import Symbol, Add, Mul, Pow, add, mul, pow
>>> x = Symbol('x')
>>> Add(x, x)            # a node with the arguments (x, x)
x + x
>>> Add(x, x).args
(x, x)
>>> add(x, x)            # SymPy's Add(x, x)
2*x
>>> x + x                # the operators call the functions
2*x
>>> Mul(2, x, 0), mul(2, x, 0), Pow(x, 0), pow(x, 0)
(2*x*0, 0, x**0, 1)
>>> Add(x, Mul(2, x)).doit()   # doit rebuilds the tree with the functions
3*x

```

This is the rule already followed by `sympy.matrices.expressions`
(`MatAdd(A, A)` is not `2*A`) and by every class of this package (AGENTS.md,
rule 7): a constructor whose return type is the class can be typed, a
constructor which returns a different object in trivial cases cannot. The
whole package is checked by mypy in strict mode, with no `Any`, no `cast`
and no `# type: ignore`: `Add.args` is a `tuple[Expr, ...]`, `Pow.base` an
`Expr`, `Rational.p` an `int`, `symbols` a `tuple[Symbol, ...]` or a
`Symbol` (SymPy's convention, kept; a typed caller uses `Symbol` for one).

## What is the same as `sympy.core`

Everything a user of SymPy's core relies on keeps its name and its meaning:

| `sympy.core` | here | notes |
| --- | --- | --- |
| `Basic`, `Atom`, `Expr`, `AtomicExpr` | the same | `args`, `func`, `atoms`, `free_symbols`, `has`, `count`, `xreplace`, `subs`, `doit`, `sort_key`, `compare`, the `is_Add`/`is_Mul`/... flags |
| `Symbol`, `Dummy`, `symbols` | the same | assumptions as keywords; `Symbol('x', positive=True) != Symbol('x')` |
| `Integer`, `Rational`, `Float`, `pi`, `E`, `I`, `oo`, `zoo`, `nan`, `S` | the same | `Rational(2, 4)` is `1/2` (a representation, not an evaluation); `Float` keeps its binary value and precision |
| `Add`, `Mul`, `Pow` | unevaluating classes + `add`, `mul`, `pow` | also `sub`, `div`, `neg`, `sqrt`, `cbrt`, `root` |
| `Function('f')`, `AppliedUndef`, `Lambda`, `Derivative`, `diff`, `expand` | the same | `Function` is a factory function, the base class of applications is `Application` |
| `Eq`, `Ne`, `Lt`, `Le`, `Gt`, `Ge` (and the long names) | unevaluating classes + `eq`, `ne`, `lt`, `le`, `gt`, `ge` | `x < 1` calls `lt` |
| `S.true`, `S.false` | `true`, `false` | atoms with a truth value |
| `sympify`, `SympifyError`, `Tuple`, `evalf`, `default_sort_key`, `ordered` | the same | strings are parsed by SymPy's parser |
| the `is_*` assumptions | the same subset: `commutative`, `real`, `integer`, `rational`, `positive`, `negative`, `nonnegative`, `nonpositive`, `zero`, `nonzero`, `even`, `odd`, `prime`, `finite`, `infinite`, `imaginary`, `complex`, `extended_real`, `irrational` | a table of implications and structural rules on sums, products and powers |
| `str`, `srepr` | the same notation | `x**2 + 2*x + 1`, `-x*y/(2*z)`, `sqrt(x)`, `1/x`, `Eq(x, y)`, `x < 1` |

The evaluated forms are SymPy's: `add`, `mul` and `pow` flatten, collect
the terms with the same non-numerical part (`x**y*x**(2*y)` is
`x**(3*y)`, `x**y*x**z` stays), multiply numerical bases with the same
exponent (`sqrt(2)*sqrt(3)` is `sqrt(6)`), distribute a number alone over
a sum (`2*(x + y)`), extract perfect powers (`sqrt(12)` is `2*sqrt(3)`,
`(-8)**(1/3)` is `2*(-1)**(1/3)`), rationalise (`(1/3)**(1/2)` is
`sqrt(3)/3`), reduce powers of `-1` and `I`, keep noncommutative factors in
order, and absorb into the infinities what SymPy absorbs (`oo + p` is `oo`
for a real `p`, `zoo*(1 + I)` is `zoo`). The test
`core/tests/test_core_known.py` checks them against SymPy on a table of
known forms and on random expressions, including derivatives, expansions,
substitutions and numerical values.

```python
>>> from sympy_extras.core import sqrt, Rational, I, oo, Symbol, S
>>> x, y = Symbol('x'), Symbol('y')
>>> sqrt(8), sqrt(2)*sqrt(6), Rational(1, 3)**S.Half, (-8)**Rational(1, 3), (-4)**S.Half
(2*sqrt(2), 2*sqrt(3), sqrt(3)/3, 2*(-1)**(1/3), 2*I)
>>> 2**x*3**x, x**y*x**(2*y), (2*x)**2, (4*x)**S.Half, -(x + y), x*(x + y)
(6**x, x**(3*y), 4*x**2, 2*sqrt(x), -x - y, x*(x + y))
>>> A, B = Symbol('A', commutative=False), Symbol('B', commutative=False)
>>> A*B - B*A, A*A*B, (A*B)**2
(A*B - B*A, A**2*B, (A*B)**2)
>>> oo + x, oo + Symbol('p', positive=True), oo - oo, 1/S.Zero
(x + oo, oo, nan, zoo)

```

## Where it differs

* **The classes never evaluate.** `Add(x, x)` is `x + x`; `Pow(x, 0)` is
  `x**0`; `Eq(x, x)` is a relation, not `True`. SymPy's `evaluate=False`
  flag does not exist, it is the default; `expr.doit()` is the opposite.
  `xreplace` is purely structural (it rebuilds with the constructors), while
  `subs` evaluates the nodes it rebuilds, as SymPy's `subs` does:

  ```python
  >>> from sympy_extras.core import Symbol, Integer
  >>> x, y = Symbol('x'), Symbol('y')
  >>> (x + y).xreplace({x: Integer(2)}).args, (x + y).subs(x, 2)
  ((2, y), y + 2)

  ```

* **`Rational(2, 1)` is a `Rational` instance**, equal to `Integer(2)`, with
  the same hash; the evaluating functions always return an `Integer` for an
  integral value. `Integer(2) == 2`, `Rational(1, 2) == Fraction(1, 2)`,
  `hash(Integer(2)) == hash(2)`, and, as in SymPy 1.14, `Float(2.0)` is not
  `Integer(2)`.
* **Two symbols with the same name and the same closure of assumptions are
  equal**, as in SymPy; the closure is computed by the implications of
  `core/assumptions.py` and `assumptions0` shows it.
* **`Function('f')` returns an `UndefinedFunction` object**, not a class
  (SymPy makes a class per name); `f(x)` is an `AppliedUndef`, whose `func`
  is `f`. The functions of SymPy which have no class here (`sin`, `exp`,
  `log`, `gamma`, ...) are reached through `SymPyFunction`:
  `SymPyFunction(sympy.sin)(x)` is an unevaluated node, `doit`, `diff`,
  `evalf` and the assumptions of such a node go through SymPy. This is also
  how `from_sympy` represents them, so that the round trip is exact.

  ```python
  >>> import sympy
  >>> from sympy_extras.core import SymPyFunction, Symbol, diff, sympify
  >>> x = Symbol('x')
  >>> sin = SymPyFunction(sympy.sin)
  >>> sin(0), sin(0).doit(), diff(sin(x**2), x), sympify('x*exp(x)').diff(x)
  (sin(0), 0, 2*x*cos(x**2), x*exp(x) + exp(x))

  ```

* **`diff` of an undefined function stays a `Derivative`** (SymPy writes
  `Subs(Derivative(f(_xi), _xi), _xi, 2*x)` for `f(2*x)`; there is no `Subs`
  here).
* **`Float` holds its value as a dyadic rational with its precision in bits**
  (what a binary float is); mpmath is used for printing and for the
  transcendental operations. `Float(2.5)` prints `2.50000000000000` alone
  and `2.5` inside an expression, as SymPy's printer does.
* **Two quirks of SymPy are not reproduced**: `1/(2 + I)` gives
  `2/5 - I/5` here where SymPy prints `(2 - I)/5` (an undistributed product
  SymPy's own `Mul` would distribute), and SymPy's occasional nested
  products (`3*sqrt(10)*(-I**(3/2))/40`) come out flat.
* **Not included**: `Wild` and pattern matching, `Order`, `Mod`, `Subs`,
  `AlgebraicNumber`, the cache, `evaluate()` as a context manager, and the
  `Q`/`ask` system (the old assumptions are kept, with a sound subset of
  their rules).

## Converting to and from SymPy

`to_sympy(expr)` builds the SymPy expression (evaluated by SymPy's
constructors, or with `evaluate=False` to keep the structure) and
`from_sympy(expr)` builds the tree of a SymPy expression; `sympify` accepts
SymPy objects and strings. The rest of SymPy stays usable on the converted
expressions.

```python
>>> import sympy
>>> from sympy_extras.core import Symbol, Add, to_sympy, from_sympy, srepr
>>> x = Symbol('x')
>>> to_sympy(x**2 + 1).is_polynomial(), sympy.integrate(to_sympy(x**2), to_sympy(x))
(True, x**3/3)
>>> to_sympy(Add(x, x)), to_sympy(Add(x, x), evaluate=False)
(2*x, x + x)
>>> t = from_sympy(sympy.Symbol('p', positive=True)**2 + sympy.sin(sympy.Symbol('x')))
>>> t, t.is_real
(p**2 + sin(x), None)
>>> srepr(t)
Add(Pow(Symbol('p', positive=True), Integer(2)), sin(Symbol('x')))

```

## Layout

```
sympy_extras/core/
    basic.py         Basic, Atom: the tree, equality, hashing, traversal, xreplace, subs, doit
    expr.py          Expr: the operators (calling add, mul, pow, lt, ...), the decompositions, the is_* queries
    sorting.py       the canonical order (sort_key, default_sort_key, ordered)
    numbers.py       Integer, Rational, Float, the infinities, pi, E, I; number_add, number_mul
    symbol.py        Symbol, Dummy, symbols
    assumptions.py   the implications of the assumptions, the structural rules, is_number
    add.py           Add and add
    mul.py           Mul and mul, div, neg
    power.py         Pow and pow, sqrt, cbrt, root; the powers of numbers
    function.py      Function, AppliedUndef, SymPyFunction, Lambda, Derivative, diff, expand
    relational.py    Eq, Ne, Lt, Le, Gt, Ge and eq, ne, lt, le, gt, ge
    logic.py         true, false
    containers.py    Tuple
    singleton.py     S
    sympify.py       sympify
    printing.py      sstr, srepr, the printing order of the terms
    evalf.py         evalf, float(), complex()
    interop.py       to_sympy, from_sympy
    tests/           test_core.py (the design, regressions), test_core_known.py (against SymPy)
```

## Open points

* Performance has not been a goal: every `is_*` query recomputes the facts
  of the subtree, `add` and `mul` sort with a Python key, and the exact
  number fields of SymPy's `Poly` are not used. A measurement against
  `sympy.core` on the usual benchmarks is the next step.
* The assumptions are a sound subset of SymPy's; `Q`/`ask` integration,
  `extended_*` predicates, `algebraic`, `transcendental`, `hermitian` are
  missing.
* `Subs`, `Wild`/`match`, `Order`, `Mod` and `AlgebraicNumber` would be
  needed for a full replacement, in the same style (a class which does not
  evaluate, a function which does).
