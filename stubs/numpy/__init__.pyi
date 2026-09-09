# A deliberately empty stub for numpy, which nothing in sympy-extras imports.
#
# SymPy imports numpy inside a few modules, so mypy reaches the stubs shipped
# with numpy; those of numpy 2.2 and later use `type` statements, which only
# parse when analysing for Python 3.12, while this package is analysed for its
# minimum, 3.10. The parse error is fatal and stops the whole check. Shadowing
# the stubs through `mypy_path` (see pyproject.toml) makes numpy an opaque
# module of type Any, which is exactly how it is used here: not at all.

from typing import Any

def __getattr__(name: str) -> Any: ...
