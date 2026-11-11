"""Cylindrical algebraic decomposition (CAD).

The package exports nothing: import the modules. The bottom layer is
native (typed polynomials of :mod:`sympy_extras.polys` and symbols of the
core, no SymPy): the projection phase in
:mod:`sympy_extras.polys.cad.projection`, the exact real algebraic sample
points in :mod:`sympy_extras.polys.cad.samplepoints` and the lifting
phase, with the entry point ``cylindrical_algebraic_decomposition``, in
:mod:`sympy_extras.polys.cad.lifting`. Their SymPy faces, on SymPy's
``Poly`` and ``CRootOf``, are the modules ``projection_sympy``,
``samplepoints_sympy`` and ``lifting_sympy``. Quantifier elimination and
decision of formulas built on top of them are in
:mod:`sympy_extras.polys.cad.qe`, the cylindrical descriptions in
:mod:`sympy_extras.polys.cad.cylindrical` (both on SymPy).
"""
