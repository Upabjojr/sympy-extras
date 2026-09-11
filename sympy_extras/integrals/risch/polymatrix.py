# Ported from SymPy (sympy.polys.polymatrix) and from Aaron Meurer's pull
# requests to SymPy: sympy/sympy#30180 and #30221 ("Risch: Implement most
# remaining exp-log cases from Bronstein", branch risch-rde-cancellation
# @d7dafa43) and #30292 ("Hypertangent cases in the Risch algorithm", branch
# risch-hypertangent @66bc72d6, the version taken here), by Claude Fable 5.1
# for sympy-extras. The algorithm is Bronstein's (Symbolic Integration I:
# Transcendental Functions, 2nd edition, Springer 2005) as implemented by
# Aaron Meurer; only the imports were changed, so that the subpackage is
# self-contained on the released SymPy. SymPy's licence (BSD 3-clause,
# copyright the SymPy Development Team) applies to this file: see
# LICENSE-SymPy in this directory.
# Type annotations for sympy-extras (strict mypy), after Aaron Meurer's branch
# risch-typing, sympy/sympy#30282 (the overloads of __getitem__).
from __future__ import annotations

from typing import Callable, Iterator, Sequence, Union, overload

from sympy.core.expr import Expr
from sympy.core.symbol import Dummy
from sympy.core.sympify import _sympify
from sympy.matrices.dense import MutableDenseMatrix

from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.polytools import Poly, parallel_poly_from_expr
from sympy.polys.domains import QQ
from sympy.polys.domains.domain import Domain
from sympy.polys.domains.polynomialring import PolynomialRing
from sympy.polys.rings import PolyElement

from sympy.polys.matrices import DomainMatrix
from sympy.polys.matrices.domainscalar import DomainScalar


class MutablePolyDenseMatrix:
    """
    A mutable matrix of objects from poly module or to operate with them.

    Examples
    ========

    >>> from sympy.polys.polymatrix import PolyMatrix
    >>> from sympy import Symbol, Poly
    >>> x = Symbol('x')
    >>> pm1 = PolyMatrix([[Poly(x**2, x), Poly(-x, x)], [Poly(x**3, x), Poly(-1 + x, x)]])
    >>> v1 = PolyMatrix([[1, 0], [-1, 0]], x)
    >>> pm1*v1
    PolyMatrix([
    [    x**2 + x, 0],
    [x**3 - x + 1, 0]], ring=QQ[x])

    >>> pm1.ring
    ZZ[x]

    >>> v1*pm1
    PolyMatrix([
    [ x**2, -x],
    [-x**2,  x]], ring=QQ[x])

    >>> pm2 = PolyMatrix([[Poly(x**2, x, domain='QQ'), Poly(0, x, domain='QQ'), Poly(1, x, domain='QQ'), \
            Poly(x**3, x, domain='QQ'), Poly(0, x, domain='QQ'), Poly(-x**3, x, domain='QQ')]])
    >>> v2 = PolyMatrix([1, 0, 0, 0, 0, 0], x)
    >>> v2.ring
    QQ[x]
    >>> pm2*v2
    PolyMatrix([[x**2]], ring=QQ[x])

    """

    #: the matrix over the polynomial ring
    _dm: DomainMatrix
    #: the polynomial ring K[gens]
    ring: PolynomialRing
    #: the ground domain K
    domain: Domain
    #: the generators
    gens: tuple[Expr, ...]

    def __new__(cls, *args: object, ring: Union[Domain, str, None] = None) -> MutablePolyDenseMatrix:

        rows: int
        cols: int
        items: list[object]
        gens: Sequence[object]
        if not args:
            # PolyMatrix(ring=QQ[x])
            if ring is None:
                raise TypeError("The ring needs to be specified for an empty PolyMatrix")
            rows, cols, items, gens = 0, 0, [], ()
        elif isinstance(args[0], list):
            elements, gens = args[0], args[1:]
            if not elements:
                # PolyMatrix([])
                rows, cols, items = 0, 0, []
            elif isinstance(elements[0], (list, tuple)):
                # PolyMatrix([[1, 2]], x)
                rows, cols = len(elements), len(elements[0])
                items = [e for row in elements for e in row]
            else:
                # PolyMatrix([1, 2], x)
                rows, cols = len(elements), 1
                items = list(elements)
        elif len(args) >= 3 and isinstance(args[0], int) and isinstance(args[1], int) \
                and isinstance(args[2], list):
            # PolyMatrix(2, 2, [1, 2, 3, 4], x)
            rows, cols, items, gens = args[0], args[1], args[2], args[3:]
        elif len(args) >= 3 and isinstance(args[0], int) and isinstance(args[1], int) \
                and callable(args[2]):
            # PolyMatrix(2, 2, lambda i, j: i+j, x)
            rows, cols, func, gens = args[0], args[1], args[2], args[3:]
            items = [func(i, j) for i in range(rows) for j in range(cols)]
        else:
            raise TypeError("Invalid arguments")

        # PolyMatrix([[1]], x, y) vs PolyMatrix([[1]], (x, y))
        if len(gens) == 1 and isinstance(gens[0], tuple):
            gens = gens[0]
            # gens is now a tuple (x, y)

        return cls.from_list(rows, cols, items, gens, ring)

    @classmethod
    def from_list(cls, rows: int, cols: int, items: Sequence[object], gens: Sequence[object],
                  ring: Union[Domain, str, None]) -> MutablePolyDenseMatrix:

        # items can be Expr, Poly, or a mix of Expr and Poly
        sympified = [_sympify(item) for item in items]
        if sympified and all(isinstance(item, Poly) for item in sympified):
            polys = True
        else:
            polys = False

        # Identify the ring for the polys
        found: Domain
        if ring is not None:
            # Parse a domain string like 'QQ[x]'
            if isinstance(ring, str):
                found = Poly(0, Dummy(), domain=ring).domain
            else:
                found = ring
        elif polys:
            p = sympified[0]
            for p2 in sympified[1:]:
                p, _ = p.unify(p2)
            found = p.domain[p.gens]
        else:
            sympified, info = parallel_poly_from_expr(sympified, gens, field=True)
            found = info['domain'][info['gens']]
            polys = True
        if not isinstance(found, PolynomialRing):
            raise TypeError("a polynomial ring is expected, got %s" % (found,))
        the_ring: PolynomialRing = found

        # Efficiently convert when all elements are Poly
        if polys:
            p_ring = Poly(0, the_ring.symbols, domain=the_ring.domain)
            to_ring = the_ring.ring.from_list
            elements = [to_ring(p.unify(p_ring)[0].rep.to_list()) for p in sympified]
        else:
            convert_expr = the_ring.from_sympy
            elements = [convert_expr(e.as_expr()) for e in sympified]

        # Convert to domain elements and construct DomainMatrix
        elements_lol = [[elements[i*cols + j] for j in range(cols)] for i in range(rows)]
        dm = DomainMatrix(elements_lol, (rows, cols), the_ring)
        return cls.from_dm(dm)

    @classmethod
    def from_dm(cls, dm: DomainMatrix) -> MutablePolyDenseMatrix:
        obj = super().__new__(cls)
        dm = dm.to_sparse()
        R = dm.domain
        if not isinstance(R, PolynomialRing):
            raise TypeError("a matrix over a polynomial ring is expected, got %s" % (R,))
        obj._dm = dm
        obj.ring = R
        ground = R.domain
        symbols = R.symbols
        if not isinstance(ground, Domain) or not isinstance(symbols, tuple):
            raise TypeError("a polynomial ring over a domain is expected, got %s" % (R,))
        obj.domain = ground
        obj.gens = tuple(symbols)
        return obj

    def to_Matrix(self) -> MutableDenseMatrix:
        matrix = self._dm.to_Matrix()
        if not isinstance(matrix, MutableDenseMatrix):
            raise TypeError("a dense matrix is expected, got %s" % type(matrix))
        return matrix

    @classmethod
    def from_Matrix(cls, other: MutableDenseMatrix, *gens: object,
                    ring: Union[Domain, str, None] = None) -> MutablePolyDenseMatrix:
        return cls(*other.shape, other.flat(), *gens, ring=ring)

    def set_gens(self, gens: object) -> MutablePolyDenseMatrix:
        return self.from_Matrix(self.to_Matrix(), gens)

    def __repr__(self) -> str:
        if self.rows * self.cols:
            return 'Poly' + repr(self.to_Matrix())[:-1] + f', ring={self.ring})'
        else:
            return f'PolyMatrix({self.rows}, {self.cols}, [], ring={self.ring})'

    @property
    def shape(self) -> tuple[int, int]:
        rows, cols = self._dm.shape
        return int(rows), int(cols)

    @property
    def rows(self) -> int:
        return self.shape[0]

    @property
    def cols(self) -> int:
        return self.shape[1]

    def __len__(self) -> int:
        return self.rows * self.cols

    def __iter__(self) -> Iterator[Poly]:
        for k in range(len(self)):
            yield self[k]

    def _to_poly(self, v: PolyElement) -> Poly:
        ground = self.ring.domain
        gens = self.ring.symbols
        return Poly(v.to_dict(), gens, domain=ground)

    @overload
    def __getitem__(self, key: slice) -> list[Poly]: ...

    @overload
    def __getitem__(self, key: Union[int, tuple[int, int]]) -> Poly: ...

    @overload
    def __getitem__(self, key: Union[tuple[Union[int, slice], slice],
                                     tuple[slice, Union[int, slice]]]) -> MutablePolyDenseMatrix: ...

    def __getitem__(self, key: Union[slice, int, tuple[Union[int, slice], Union[int, slice]]]
                    ) -> Union[list[Poly], Poly, MutablePolyDenseMatrix]:
        dm = self._dm

        if isinstance(key, slice):
            items = dm.flat()[key]
            return [self._to_poly(item) for item in items]
        elif isinstance(key, int):
            row, col = divmod(key, self.cols)
            e = dm[row, col]
            return self._to_poly(e.element)

        i, j = key
        if isinstance(i, int) and isinstance(j, int):
            return self._to_poly(dm[i, j].element)
        else:
            return self.from_dm(dm[i, j])

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MutablePolyDenseMatrix):
            return NotImplemented
        return bool(self._dm == other._dm)

    def __hash__(self) -> int:
        return hash((self.shape, self.ring))

    def __add__(self, other: object) -> MutablePolyDenseMatrix:
        if isinstance(other, MutablePolyDenseMatrix):
            return self.from_dm(self._dm + other._dm)
        return NotImplemented

    def __sub__(self, other: object) -> MutablePolyDenseMatrix:
        if isinstance(other, MutablePolyDenseMatrix):
            return self.from_dm(self._dm - other._dm)
        return NotImplemented

    def __mul__(self, other: object) -> MutablePolyDenseMatrix:
        if isinstance(other, MutablePolyDenseMatrix):
            return self.from_dm(self._dm * other._dm)
        elif isinstance(other, int):
            other = _sympify(other)
        if isinstance(other, Expr):
            Kx = self.ring
            try:
                other_ds = DomainScalar(Kx.from_sympy(other), Kx)
            except (CoercionFailed, ValueError):
                other_ds = DomainScalar.from_sympy(other)
            dm = self._dm * other_ds
            if not dm.domain.is_PolynomialRing:
                # a fallback scalar over a ground domain (e.g. EX) can
                # drag the product's domain below a polynomial ring,
                # which from_dm cannot represent
                dm = dm.convert_to(dm.domain[self.gens])
            return self.from_dm(dm)
        return NotImplemented

    def __rmul__(self, other: object) -> MutablePolyDenseMatrix:
        if isinstance(other, int):
            other = _sympify(other)
        if isinstance(other, Expr):
            other_ds = DomainScalar.from_sympy(other)
            dm = other_ds * self._dm
            if not dm.domain.is_PolynomialRing:
                dm = dm.convert_to(dm.domain[self.gens])
            return self.from_dm(dm)
        return NotImplemented

    def __truediv__(self, other: object) -> MutablePolyDenseMatrix:

        if isinstance(other, Poly):
            other = other.as_expr()
        elif isinstance(other, int):
            other = _sympify(other)
        if not isinstance(other, Expr):
            return NotImplemented

        element = self.domain.from_sympy(other)
        inverse = self.ring.convert_from(1/element, self.domain)
        scalar = DomainScalar(inverse, self.ring)
        dm = self._dm * scalar
        return self.from_dm(dm)

    def __neg__(self) -> MutablePolyDenseMatrix:
        return self.from_dm(-self._dm)

    def transpose(self) -> MutablePolyDenseMatrix:
        return self.from_dm(self._dm.transpose())

    def row_join(self, other: MutablePolyDenseMatrix) -> MutablePolyDenseMatrix:
        dm = DomainMatrix.hstack(self._dm, other._dm)
        return self.from_dm(dm)

    def col_join(self, other: MutablePolyDenseMatrix) -> MutablePolyDenseMatrix:
        dm = DomainMatrix.vstack(self._dm, other._dm)
        return self.from_dm(dm)

    def applyfunc(self, func: Callable[[Expr], Expr]) -> MutablePolyDenseMatrix:
        M = self.to_Matrix().applyfunc(func)
        return self.from_Matrix(M, self.gens)

    @classmethod
    def eye(cls, n: int, gens: object) -> MutablePolyDenseMatrix:
        return cls.from_dm(DomainMatrix.eye(n, QQ[gens]))

    @classmethod
    def zeros(cls, m: int, n: int, gens: object) -> MutablePolyDenseMatrix:
        return cls.from_dm(DomainMatrix.zeros((m, n), QQ[gens]))

    def rref(self, simplify: str = 'ignore',
             normalize_last: str = 'ignore') -> tuple[MutablePolyDenseMatrix, tuple[int, ...]]:
        # If this is K[x] then computes RREF in ground field K.
        if not (self.domain.is_Field and all(p.is_ground for p in self)):
            raise ValueError("PolyMatrix rref is only for ground field elements")
        dm = self._dm
        dm_ground = dm.convert_to(self.domain)
        dm_rref, pivots = dm_ground.rref()
        dm_rref = dm_rref.convert_to(dm.domain)
        return self.from_dm(dm_rref), tuple(int(p) for p in pivots)

    def nullspace(self) -> list[MutablePolyDenseMatrix]:
        # If this is K[x] then computes nullspace in ground field K.
        if not (self.domain.is_Field and all(p.is_ground for p in self)):
            raise ValueError("PolyMatrix nullspace is only for ground field elements")
        dm = self._dm
        K, Kx = self.domain, self.ring
        dm_null_rows = dm.convert_to(K).nullspace(divide_last=True).convert_to(Kx)
        dm_null = dm_null_rows.transpose()
        dm_basis = [dm_null[:,i] for i in range(dm_null.shape[1])]
        return [self.from_dm(dmvec) for dmvec in dm_basis]

    def rank(self) -> int:
        return self.cols - len(self.nullspace())

MutablePolyMatrix = MutablePolyDenseMatrix
PolyMatrix = MutablePolyDenseMatrix
