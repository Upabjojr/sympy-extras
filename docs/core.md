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
| `Mod` | unevaluating class + `mod` | `x % 3` calls `mod`; SymPy's reductions (`Mod(2*n, 2)` is `0`, `Mod(x + 3, 3)` is `Mod(x, 3)`) |
| `Function('f')`, `AppliedUndef`, `Lambda`, `Derivative`, `Subs`, `diff`, `expand`, `expand_*`, `count_ops`, `nfloat`, `arity`, `FunctionClass` | the same | `Function` is a factory function, the base class of applications is `Application`, the base class of function objects is `FunctionClass`; `f(2*x).diff(x)` gives the `Subs` SymPy gives |
| `Order` (`O`) | unevaluating class + `order` | sums and products with an order term are evaluated by SymPy; `series`, `removeO`, `getO`, `getn` |
| `Wild`, `WildFunction`, `match`, `matches`, `replace`, `find`, `count` | the same | a backtracking matcher with SymPy's conventions (sums and products as multisets, solving for a lone wild) |
| `Eq`, `Ne`, `Lt`, `Le`, `Gt`, `Ge` (and the long names), `Rel` | unevaluating classes + `eq`, `ne`, `lt`, `le`, `gt`, `ge` | `x < 1` calls `lt`; `canonical`, `negated`, `reversed` |
| `S.true`, `S.false` | `true`, `false` | atoms with a truth value |
| `And`, `Or`, `Not`, `Xor`, `Implies`, `Equivalent`, `ITE`, `Nand`, `Nor`, `Xnor`, `LatticeOp`, `BooleanFunction` | unevaluating classes + `and_`, `or_`, `not_`, `xor`, `implies`, `equivalent`, `ite` | the operators `&`, `\|`, `~`, `^`, `>>`, `<<` of symbols, relations and predicates call the functions; `And(y, x) == And(x, y)`; the printing (`z & (x \| y)`, `~(x & y)`, `Implies(x, y)`) is SymPy's |
| `Q`, `ask`, `Predicate`, `AppliedPredicate`, `AssumptionsContext`, `global_assumptions`, `assuming`, `refine` | the same | `Q.positive(x)` never evaluates; `ask` reads conjunctive assumptions as facts and answers by the old assumptions, the rest is asked to SymPy's solver; handlers with `Q.positive.register(cls)`; `refine` is SymPy's with this `Q` |
| `EulerGamma`, `Catalan`, `GoldenRatio`, `TribonacciConstant`, `AlgebraicNumber`, `UnevaluatedExpr`, `Dict`, `RealNumber` | the same | `AlgebraicNumber` keeps the coefficients as given (SymPy reduces them modulo the minimal polynomial) |
| `sympify`, `SympifyError`, `Tuple`, `evalf`, `N`, `comp`, `seterr`, `default_sort_key`, `ordered` | the same | strings are parsed by SymPy's parser |
| the `is_*` assumptions | all of SymPy's predicates: the ones above and `extended_positive`, ..., `extended_nonzero`, `hermitian`, `antihermitian`, `algebraic`, `transcendental`, `composite`, `noninteger`, `polar` | a table of implications and structural rules on sums, products and powers; the facts of a node are cached on the node; `assumptions`, `check_assumptions`, `failing_assumptions`, `common_assumptions` |
| the methods of `Expr` | the same: `as_coeff_Mul`, `as_base_exp`, `as_independent`, `as_coeff_exponent`, `as_coefficient`, `as_content_primitive`, `extract_additively`, `extract_multiplicatively`, `equals`, `as_real_imag`, `conjugate`, `adjoint`, `transpose`, `as_terms`, `round`, `is_polynomial`, `is_rational_function`, `is_algebraic_expr`, `is_constant`, `is_comparable`, `as_dummy`, `dummy_eq`, `has_free`, `rcall`, `as_leading_term`, `leadterm`, `removeO`, `getO`, `getn`, ... | the structural ones are written here (the leading term of a polynomial, rational or algebraic expression too); `series`, `nseries`, `lseries`, `aseries`, `limit`, `simplify`, `factor`, `cancel`, `apart`, `collect`, `together`, `powsimp`, `radsimp`, `ratsimp`, `trigsimp`, `nsimplify`, `integrate`, `as_poly`, `invert`, `rewrite`, `fps`, `fourier_series` and the leading term of a function convert to SymPy, call it and convert back |
| `evaluate`, `distribute`, `global_parameters`, `cacheit`, `clear_cache`, `Transform`, the kinds, `preorder_traversal`, `postorder_traversal`, `bottom_up`, `use`, `iterargs`, `iterfreeargs`, `igcd`, `ilcm`, `igcdex`, `mod_inverse`, `integer_nthroot`, `integer_log`, `num_digits`, `trailing`, `factor_terms`, `gcd_terms`, `factor_nc`, `vectorize`, `var`, `greeks`, `call_highest_priority`, `_op_priority`, the exceptions | the same | `with evaluate(False): x + x` gives `Add(x, x)`; `cacheit` keeps the signature of the function; `factor_terms` and `gcd_terms` are written here (`factor_nc` and the radicals are SymPy's) |
| `Basic.compare`, the order of `args` | the same | sums and products store their arguments in the order of SymPy's `compare` (the numbers, then by class, then by content: `(x + y + n).args` is `(x, y, n)` for an integer `n`), `sort_key` is the order of `default_sort_key` |
| `str`, `srepr` | the same notation | `x**2 + 2*x + 1`, `-x*y/(2*z)`, `sqrt(x)`, `1/x`, `Eq(x, y)`, `x < 1`, `O(x**2)`, `Subs(f(x), x, 2)`, `2*(Mod(x, 3))`, `a_` |

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

```python
>>> from sympy_extras.core import Wild, Mod, Order, Function, evaluate, count_ops, Symbol, sympify, S, sqrt
>>> x, y, n = Symbol('x'), Symbol('y'), Symbol('n', integer=True)
>>> a, b = Wild('a'), Wild('b')
>>> (2*x + 3).match(a*x + b) == {a: 2, b: 3}, (x*y + 2*x).replace(a*x, a)
(True, y + 2/x)
>>> x % 3, Mod(x, 3).args, (2*n) % 2, Mod(7, 3).doit(), (x + 3) % 3
(Mod(x, 3), (x, 3), 0, 1, Mod(x, 3))
>>> sympify('sin(x)').series(x, 0, 5), (x + Order(x**2)).removeO(), Order(x + x**2).doit()
(x - x**3/6 + O(x**5), x, O(x))
>>> f = Function('f')
>>> f(2*x).diff(x), (sympify('sin(x)**2 + cos(x)**2')).simplify(), (x**2 + 2*x + 1).factor()
(2*Subs(Derivative(f(_xi_1), _xi_1), _xi_1, 2*x), 1, (x + 1)**2)
>>> with evaluate(False):
...     e = x + x
>>> e.args, (n**2 + 1).is_positive, (n + S.Half).is_noninteger, sqrt(2).is_algebraic, count_ops(x**2 + 2*x + 1, visual=True)
((x, x), True, True, True, 2*ADD + MUL + POW)

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
* **`E**x` is `exp(x)`**, as in SymPy: `pow(E, a)` is SymPy's `exp(a)`
  (an `AppliedSymPyFunction`), whose `as_base_exp` is `(E, a)`, so that
  products collect exponentials as SymPy does (`exp(x)*exp(-x)` is `1`,
  `exp(x)*exp(2*x)` is `exp(3*x)`, `exp(x)*exp(y)` stays).
* **Every object of SymPy converts.** What has a class here is converted to
  it; a function of SymPy becomes an `AppliedSymPyFunction`; anything else
  (`Integral`, `Sum`, `Piecewise`, `And`, a set, ...) becomes a generic
  `SymPyExprNode` or `SymPyNode` which keeps the class of SymPy and the
  converted arguments, prints as SymPy prints it, and whose `doit` and
  assumptions are SymPy's. The methods of `Expr` which are algorithms of
  other packages of SymPy (`series`, `limit`, `simplify`, `factor`,
  `integrate`, `as_poly`, ...) are forwarded this way, in one line each.
* **Pattern matching** follows SymPy's conventions, with structural matches
  searched before solved ones: `(2*x + 3).match(a*x + b)` is `{a: 2, b: 3}`,
  `x.match(a + x)` is `{a: 0}`, `(x**3).match(a*x**2)` is `{a: x}`,
  `S(2).match(a*x)` is `{a: 2/x}`, `(2/x).match(a*x)` is `None`,
  `(x**3).match(x**a*x**b)` is `{a: 0, b: 3}`. In the few ambiguous cases
  where SymPy's search order gives an odd binding
  (`(x**2 + 2*x).match(a*x**2 + b*x)` is `{a: 2/x, b: x}` in SymPy), the
  natural one is found here (`{a: 1, b: 2}`); reproducing that order would
  mean reproducing SymPy's algorithm. SymPy's test on a solved binding is
  applied (the value must not have more operations than the subject:
  `(-E).match(a*x)` is `{a: -E/x}`, `(2/x).match(a*x)` is `None`), and
  `S(1).match(a**b)` is `{b: 0}` with `a` free, as in SymPy.
* **The conventions of the methods are SymPy's**, found by comparing
  dozens of methods on random expressions: `srepr` lists the arguments in
  the printing order and splits `-1` from a coefficient (`as_ordered_factors`
  does too); `sort_key` has SymPy's shape and `ordered` sorts by the number
  of nodes first; `could_extract_minus_sign` of a sum counts the negative
  terms and breaks a tie by the sort key (`1 - x` and `-x + y` do, `x - y`
  does not); `Integer(6).primitive()` is `(6, 1)`; `count_ops` counts `E`
  as `EXP` and a sum of negative terms as `NEG + SUB`; `evalf` keeps an
  integer exponent, a coefficient `-1`, `I` and a function of a symbolic
  argument (`-p + 2*pi*x` becomes `-p + 6.28*x`); `extract_multiplicatively`
  divides an integer coefficient only by a divisor and a fraction by any
  rational; `as_numer_denom` sums the terms with the same denominator first;
  `exp` has `NumberKind` and `sin` has `UndefinedKind`; the derivative of a
  function of SymPy is SymPy's (`Abs(x)` gives its `re`/`im` form) and the
  derivative of a power keeps `x**(pi/2)/x` apart; `expand` applies the
  `log` hint by default and writes `p/(2*(x - y))` as `p/(2*x - 2*y)`.
  The facts of a numerical sum, product or power are decided numerically at
  two precisions as SymPy decides them (`11 - 3*E` is positive,
  `(-1 - I)**-2` is imaginary); here a few more facts are known than in
  SymPy (`p + cos(4) + 3` is not zero, `(-E)**-2 + Abs(n)` is positive),
  and `equals(0)` answers `False` where SymPy answers `None`.
* **The logic and the new assumptions.** `And`, `Or`, `Not`, `Xor`,
  `Implies`, `Equivalent` and `ITE` are unevaluating classes (`And` and
  `Or` derive from `LatticeOp`: equal up to the order of the arguments)
  with the evaluating functions `and_`, `or_`, `not_`, `xor`, `implies`,
  `equivalent`, `ite`, which give what SymPy's constructors give;
  `Q.positive(x)` is an `AppliedPredicate` which never evaluates;
  `ask(proposition, assumptions)` reads a conjunction of applied predicates
  as facts about their arguments (closed under the implications, checked
  for consistency), replaces the arguments by dummies carrying those facts
  and answers with the old assumptions: `ask(Q.positive(x + 1), Q.positive(x))`
  is `(_x + 1).is_positive`. The handlers registered on a predicate are
  consulted first; what stays unanswered (and every question under
  non-conjunctive assumptions such as `Q.positive(x) | Q.negative(x)`) is
  asked to SymPy's `ask`, so that the answers are SymPy's. `refine` is
  SymPy's, with this `Q`. Unlike SymPy, the facts of a symbol with
  assumptions are used under any assumptions, an inconsistency among the
  assumptions about any argument of the proposition raises (SymPy ignores
  the ones about other expressions, and its message differs in the cases
  its solver finds). As in SymPy, `x & ~x` and `Q.positive(x) & ~Q.positive(x)`
  are kept, while a relation and its negation give `false`.
* **Small differences**: SymPy's occasional nested products
  (`3*sqrt(10)*(-I**(3/2))/40`) come out flat; `factor_terms(2*x + 4*y)` is
  the undistributed product `Mul(2, x + 2*y)` here as in SymPy, which `doit`
  distributes (SymPy's `Mul` would too); the variable of a `Subs` made by
  `diff` is the one SymPy makes when its cache is empty (SymPy returns the
  first equal `Subs` it built, with whatever dummy it had).
* **Not included**: the `FactKB` machinery (the implications are a plain
  table) and the SAT solver of `sympy.assumptions` (SymPy's is called).

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
    basic.py         Basic, Atom: the tree, equality, hashing, traversal, xreplace, subs, doit, match, replace, find, as_dummy
    expr.py          Expr: the operators (calling add, mul, pow, mod, lt, ...), the decompositions, the is_* queries,
                     the methods forwarded to SymPy; UnevaluatedExpr
    sorting.py       the two orders: compare (the arguments of sums and products) and sort_key (default_sort_key, ordered)
    numbers.py       Integer, Rational, Float, the infinities, pi, E, EulerGamma, Catalan, GoldenRatio,
                     TribonacciConstant, I, AlgebraicNumber; number_add, number_mul, comp, seterr
    symbol.py        Symbol, Dummy, Wild, symbols, var
    matching.py      the matcher behind Basic.match
    assumptions.py   the implications of the assumptions, the structural rules, the cache of facts, is_number,
                     check_assumptions, failing_assumptions, common_assumptions
    operations.py    AssocOp, the base of Add and Mul; LatticeOp, the base of And and Or
    add.py           Add and add
    mul.py           Mul and mul, div, neg, prod
    power.py         Pow and pow, sqrt, cbrt, root; the powers of numbers
    mod.py           Mod and mod
    order.py         Order and order
    function.py      FunctionClass, Function, AppliedUndef, SymPyFunction, WildFunction, Lambda, Derivative, Subs,
                     diff, expand and the expand_* hints, count_ops, nfloat, arity
    relational.py    Eq, Ne, Lt, Le, Gt, Ge, Rel and eq, ne, lt, le, gt, ge
    logic.py         Boolean, true, false, And, Or, Not, Xor, Implies, Equivalent, ITE and and_, or_, not_, xor,
                     implies, equivalent, ite, Nand, Nor, Xnor
    assume.py        Predicate, AppliedPredicate, AssumptionsContext, global_assumptions, assuming
    ask.py           Q (AssumptionKeys) and ask
    refine.py        refine (SymPy's, with this Q)
    containers.py    Tuple, Dict
    singleton.py     S
    sympify.py       sympify
    printing.py      sstr, srepr, the printing order of the terms
    evalf.py         evalf, N, float(), complex()
    interop.py       to_sympy, from_sympy, through_sympy, the generic wrappers SymPyNode/SymPyExprNode
    parameters.py    global_parameters, evaluate, distribute
    cache.py         cacheit, clear_cache
    kind.py          NumberKind, BooleanKind, UndefinedKind, TupleKind
    rules.py         Transform
    traversal.py     postorder_traversal, bottom_up, use, iterargs, iterfreeargs
    intfunc.py       igcd, ilcm, igcdex, mod_inverse, integer_nthroot, integer_log, num_digits, trailing
    exprtools.py     factor_terms, gcd_terms (written here), factor_nc (through SymPy)
    decorators.py    call_highest_priority
    multidimensional.py  vectorize
    random.py        random_complex_number, verify_numerically, test_derivative_numerically
    coreerrors.py    the exceptions
    alphabets.py     greeks
    tests/           test_core.py (the design, regressions), test_core_known.py (against SymPy)
```

## Performance

Measured against `sympy.core` (SymPy 1.14, one process, the script
`bench_core.py` of the session which wrote this; times in seconds):

| operation | `sympy.core` | here | ratio |
| --- | --- | --- | --- |
| a sum of 300 terms built term by term | 0.150 | 0.180 | 1.2 |
| a product of 200 factors | 0.027 | 0.039 | 1.4 |
| `expand((x + y + z)**8)` | 0.0020 | 0.0035 | 1.7 |
| `diff` of a 21-term polynomial, 10 times | 0.024 | 0.024 | 1.0 |
| `subs` of 20 different values in that polynomial | 0.038 | 0.019 | 0.5 |
| `is_positive` of 200 fresh expressions | 0.009 | 0.022 | 2.4 |
| `match` of `a*x**2 + b*x`, 200 times | 0.90 | 0.54 | 0.6 |
| 300 random expressions of depth 4 (the fuzz recipe) | 0.042 | 0.070 | 1.7 |
| `series` of `sin(x)*exp(x)`, 5 times (through SymPy) | 0.215 | 0.208 | 1.0 |
| `str` of an 84-term polynomial, 5 times | 0.017 | 0.045 | 2.6 |
| `factor_terms(2*x*y + 4*x*z + 6*x + i)`, 200 times | 0.123 | 0.101 | 0.8 |
| `as_leading_term` of a cubic, 200 times | 0.26 | 0.05 | 0.2 |
| `ask(Q.positive(x**2 + i), Q.real(x))`, 50 times | 0.84 | 0.009 | 0.01 |

The arithmetic is within a factor of two of SymPy's on everything but the
facts and the printing (a sum keeps the terms it does not change, a power
of a sum is expanded by the multinomial theorem, the closure of the facts
is read from a table of the consequences of each fact, the hash of a
number, the facts, the `compare` key and the SymPy form of a node are
computed once and cached on it, the hot paths import nothing); the
structural algorithms written here are faster than SymPy's, which pay for
their generality; the methods forwarded to SymPy cost one conversion of
the result (`from_sympy` keeps a table of what it converted, emptied by
`clear_cache`). What remains slower is the generic machinery of a Python
tree without SymPy's global cache of every constructor call.

## Open points

* The structural rules of the assumptions are a sound subset of SymPy's;
  a question they do not answer gives `None` where SymPy may know (`ask`
  then asks SymPy, the `is_*` properties do not).
* The methods forwarded to SymPy (`series`, `simplify`, `integrate`, the
  leading term of a function, `refine`, the SAT part of `ask`) are the
  algorithms of the other packages of SymPy; a core which replaced
  `sympy.core` would have those packages work on its own tree.
* `Mod(n*(n + 1)*(n + 2), 3)` and the like are not reduced, as in SymPy
  (only the modulus 2 is).
