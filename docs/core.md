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

## Functions of the core

The functions of SymPy were wrapped (`SymPyFunction(sympy.sin)`), their
evaluation run by SymPy. They are being replaced by typed classes in
`sympy_extras.functions`, on the base class
`sympy_extras.core.function.DefinedFunction`: the class (`Exp`, `Log`)
builds the unevaluated node, the classmethod `eval` holds the evaluation
rules, `apply` (what the lower case `exp`, `log` call) evaluates, `fdiff`
gives the partial derivatives (the core's `diff` applies the chain rule),
`numeric` the value with mpmath, and `_eval_facts` the assumptions. The
conversion maps `sympy.exp` and `sympy.log` to `Exp` and `Log` in both
directions, and `E**x` builds `exp(x)` natively.

The classes never evaluate and the functions do (`Sin(0)` is the node
`sin(0)`, `sin(0)` is `0`), and the nodes are immutable: the attributes
set by the constructors are `Final`, so mypy rejects any later
assignment (a runtime check in `__setattr__` cost 30-80 % of the speed
of the core and was not kept).

The trigonometric and hyperbolic families (`Sin`, `Cos`, `Tan`, `Cot`,
`Sec`, `Csc`, `Sinh`, `Cosh`, `Tanh`, `Coth`, `Sech`, `Csch` with `sin`,
..., `csch`) are ported from SymPy's `trigonometric.py` and
`hyperbolic.py`: symmetry, the exchange `sin(I*x) = I*sinh(x)` and back,
the exact values at rational multiples of `pi` (SymPy's reductions,
product formulas, Chebyshev values and half-angle roots), the multiples
of `pi` and `I*pi` peeled off sums, the reciprocal functions through
their partners, derivatives, facts. On 2904 values (rational multiples
of `pi` with denominators up to 30, imaginary arguments, shifted sums)
the results are SymPy's character for character, compared without
letting SymPy re-evaluate them. The inverse functions (`Asin`, `Acos`,
`Atan`, `Acot`, `Asec`, `Acsc`, `Asinh`, `Acosh`, `Atanh`, `Acoth`,
`Asech`, `Acsch` with `asin`, ..., `acsch`) are ported with SymPy's
tables of known values and the compositions with the direct functions
(`asin(sin(3)) = pi - 3`, reduced into the principal range): on 3744
values the results are SymPy's except the three where SymPy answers with
`AccumBounds` (`atan(zoo)`, `atanh(zoo)`, `asech(zoo)`), not ported yet,
like `sin(oo)`.

The complex parts (`Re`, `Im`, `Abs`, `Sign`, `Arg`, `Conjugate`,
`Atan2` with `re`, `im`, `abs_`, `sign`, `arg`, `conjugate`, `atan2`;
Python's `abs()` of a core expression calls `abs_`) are ported from
SymPy's `complexes.py`, together with the complex expansion
`as_real_imag` and the conjugation of every class of the core (SymPy's
`as_real_imag` and `_eval_conjugate` methods of `Expr`, `Add`, `Mul`,
`Pow`, `Symbol` and of the ported functions): `Expr.as_real_imag`,
`Expr.conjugate` and `abs()` no longer go through SymPy. The functions
which are not holomorphic (`re`, `im`, `Abs`, `arg`, `conjugate`) have
their own derivative (`DefinedFunction._eval_derivative`). The results
are SymPy's on the compared values; what is left: `Abs` applies only
`Abs(-z) = Abs(z)` of SymPy's `signsimp`, the `Heaviside` and
`Piecewise` answers of `atan2` are ported where they evaluate, and the
complex expansion of `(-p)**(1 + I*r)` differs because the core's
`expand` does not split `(-p)**(I*r)` as SymPy's does. The comparison
also brought two of SymPy's arithmetic rules into the core: a sum with
exactly one infinite term is infinite, and zero times an infinite
factor is `nan`.

The integer parts (`Floor`, `Ceiling`, `Frac` with `floor`, `ceiling`,
`frac`) are ported from SymPy's `integers.py`: numbers and number
symbols (`floor(pi) = 3`, the numerical value decided at two
precisions), integer terms of a sum taken out, imaginary arguments, and
the bounds decided from signs (`floor(p/(p + 1)) = 0`). The `//`
operator of the core uses them. `frac(oo)` stays unevaluated, because
SymPy's answer is `AccumBounds(0, 1)`, which is not ported.

`Min` and `Max` (the nodes, with any number of arguments kept as given)
and `min_`, `max_` (evaluating) are ported from SymPy's
`miscellaneous.py`: flattening, identity and zero, the removal of
redundant arguments, common arguments factored, and arguments whose
order is decided from the facts of their difference reduced to the
extreme one. `DefinedFunction.variadic` marks such functions. The
derivative is SymPy's, with `Heaviside`.
One rule is deliberately stricter than SymPy's: `Max(0, x)` is not
claimed finite and nonnegative from its argument `0`, since `x` may be
`oo`; the "one argument suffices" facts need all arguments finite.

`Piecewise` (the node, its pairs kept as given) and `piecewise`
(evaluating) are ported from SymPy's `piecewise.py` with `ExprCondPair`:
false conditions and the pairs after a true one dropped, a single true
pair giving its expression, relations made canonical with the
coefficient divided out (`-2*x + 1 < 0` is `x > 1/2`), repeated and
redundant conditions removed, nested `Piecewise` with the same
condition collapsed, successive pairs with the same expression merged.
Substitution re-evaluates; the derivative and the conjugate act on the
expressions. `piecewise_fold`, `piecewise_simplify` and the interval and
integration methods are not ported. For it the core's relations gained
SymPy's `canonical` and `reversedsign`, and the comparisons `lt`, `le`,
`gt`, `ge` are now decided from the facts of the difference
(`gt(p, 0)` is `true` for a positive `p`) and for an extended real
against `oo` or `-oo`. A known gap: `ge(k, 1)` for a positive integer
`k` stays unevaluated, because the core's sums do not derive that
`k - 1` is nonnegative.

The combinatorial functions `Factorial`, `Binomial`, `RisingFactorial`,
`FallingFactorial` with `factorial`, `binomial`, `rf`, `ff` are ported
from SymPy's `factorials.py` (in `sympy_extras.functions.combinatorial`):
exact integer values, `factorial(-1) = zoo`, the special cases of
`binomial`, the product formula for a numerical `n` and integer `k`,
and `rf`, `ff` expanded for integer `k`. SymPy's answers through `gamma`
(`binomial(5, 1/2) = 512/(63*pi)`) and the derivatives with `polygamma`
use the gamma functions below.

The gamma functions (`Gamma`, `LogGamma`, `PolyGamma`, `DiGamma`,
`TriGamma` with `gamma`, `loggamma`, `polygamma`, `digamma`, `trigamma`,
in `sympy_extras.functions.special`) are ported from SymPy's
`gamma_functions.py`: `gamma` at the integers and half-integers,
`loggamma` at the integers and positive half-integers, `polygamma(0, n)`
with the exact harmonic number, `polygamma(n, z)` for odd `n` at the
integers and `1/2` with the exact `zeta` at even arguments
(`trigamma(2) = pi**2/6 - 1`). `lowergamma`, `uppergamma` and the
series are not ported. Every ported function of floats is now evaluated
numerically at their precision (`sin(0.5) = 0.479425538604203`), as
SymPy's `Function.__new__` does.

`DiracDelta` and `Heaviside` (with the evaluating `dirac_delta` and
`heaviside`) are ported from SymPy's `delta_functions.py`: zero delta at
a nonzero argument, the sign of a negative coefficient taken out with
`(-1)**k`, the step decided from the sign of its argument, `H0 = 1/2` by
default and not printed, non-real arguments refused. With them `sign`,
`Min` and `Max` have SymPy's derivatives (`2*DiracDelta(r)`,
`Heaviside(x - r)`). The expansion of the delta of a polynomial and the
rewrites are not ported.

The zeta functions (`Zeta`, `DirichletEta`, `PolyLog` with `zeta`,
`dirichlet_eta`, `polylog`) are ported from SymPy's `zeta_functions.py`,
with the Bernoulli numbers (SymPy's `B_1 = 1/2`) and polynomials and the
harmonic numbers they need: `zeta` at the nonpositive and even
integers, the Hurwitz zeta at positive integers, `dirichlet_eta`
through `zeta`, `polylog` at `0`, `1`, `-1`, the dilogarithm table and
`s = 0, -1`. With them `polygamma(n, z)` has all of SymPy's values at the
integers and `1/2`, and `polygamma(0, p/q)` (q <= 6) is expanded by
Gauss' digamma theorem. A function without its own partial derivative
now differentiates as SymPy's `Function.fdiff` does (`Derivative(f(x),
x)`, or `Subs` for a compound argument). Not ported: `lerchphi`,
`stieltjes`, `riemann_xi`, the polar arguments of `polylog`.

The error functions (`Erf`, `Erfc`, `Erfi`, `Erfinv`, `Erfcinv` with
`erf`, `erfc`, `erfi`, `erfinv`, `erfcinv`) are ported from SymPy's
`error_functions.py`: the values at `0` and the infinities, the
compositions with the inverses, the odd symmetries, derivatives,
conjugates, SymPy's complex parts and facts. One deliberate difference:
`erfcinv` of a float is evaluated, where SymPy, without it in mpmath,
leaves the node. `could_extract_minus_sign` now counts an extended
negative coefficient, as SymPy's `_coeff_isneg` does (`-oo*I`). Not
ported: `erf2`, `erf2inv`, the Fresnel, exponential and trigonometric
integrals.

`LambertW` (with `lambertw`, in `exponential.py`) is ported with its
special values on the branches 0 and -1, the pole of the other branches
at 0, the derivative, facts and mpmath values; `root`, `cbrt` and
`real_root` (functions, as in SymPy, in `miscellaneous.py`) build the
powers, `real_root(x, n)` SymPy's `Piecewise`, and `real_root(e)`
replaces the odd roots of negative numbers by their real values.

`beta` (`Beta`, in `special/beta_functions.py`) and the Catalan numbers
`catalan` (`Catalan`, in `combinatorial/numbers.py`; not the constant
`Catalan` of the core) are ported from SymPy: `beta(x)` is `beta(x, x)`,
numbers follow SymPy's `doit` rules (`beta(2, 3) = 1/12` through
`catalan`), `catalan(n)` is the gamma quotient at the nonnegative
integers and negative non-integers, with derivatives, facts and mpmath
values. `betainc` and the other numbers of SymPy's `numbers.py` are not
ported.

The exponential family is the first one, ported from SymPy's
`sympy/functions/elementary/exponential.py` (with the `PORTED FROM SYMPY`
notice): the special values, `exp(log(x))`, the multiples of `pi*I`, a
single log factor in a product, sums split term by term; `log` of
rational numbers, of negative numbers, of multiples of `I`, of `exp` of
a real argument, `log(x, b)` with the powers of an integer base taken
out; the derivatives; the facts (real, positive, zero, finite, rational
and algebraic, the last by Lindemann–Weierstrass). The rules which need
`logcombine`, `re`/`im` or `AccumBounds` are not ported yet and leave the
node unevaluated (the module docstring lists them).

```python
>>> from sympy_extras.core import Symbol, Rational, I, pi
>>> from sympy_extras.functions import exp, log, Exp, sin, cos, Sin, sinh
>>> x = Symbol('x')
>>> exp(2*log(x)), exp(Rational(7, 3)*pi*I), log(-3), log(24, 2)
(x**2, exp(I*pi/3), log(3) + I*pi, log(3)/log(2) + 3)
>>> Sin(0), sin(0), cos(pi/5), sin(x + pi), sin(I*x)
(sin(0), 0, 1/4 + sqrt(5)/4, -sin(x), I*sinh(x))
>>> Exp(x).diff(x), (x*log(x)).diff(x)
(exp(x), log(x) + 1)

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
