"""Three-valued logic: ``fuzzy_and``, ``fuzzy_or``, ``fuzzy_not``,
``fuzzy_group`` on ``True``, ``False`` and ``None`` (unknown).

References
==========

.. [1] SymPy Development Team, *sympy.core.logic*,
       https://github.com/sympy/sympy/blob/master/sympy/core/logic.py
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/core/logic.py (fuzzy_and, fuzzy_or, fuzzy_not,
# _fuzzy_group) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: typed (Optional[bool] values only).
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Iterable, Optional

__all__ = ['fuzzy_and', 'fuzzy_or', 'fuzzy_not', 'fuzzy_group']


# PORTED FROM SYMPY: sympy/core/logic.py:fuzzy_and
def fuzzy_and(args: Iterable[Optional[bool]]) -> Optional[bool]:
    """``True`` when all are ``True``, ``False`` when one is ``False``,
    else ``None``.

    >>> from sympy_extras.core.fuzzy import fuzzy_and
    >>> fuzzy_and([True, True]), fuzzy_and([True, None]), fuzzy_and([None, False])
    (True, None, False)
    """
    rv: Optional[bool] = True
    for a in args:
        if a is None:
            if rv is True:
                rv = None
        elif a is False:
            return False
    return rv


# PORTED FROM SYMPY: sympy/core/logic.py:fuzzy_not
def fuzzy_not(v: Optional[bool]) -> Optional[bool]:
    """``not v``, ``None`` for ``None``."""
    return None if v is None else not v


# PORTED FROM SYMPY: sympy/core/logic.py:fuzzy_or
def fuzzy_or(args: Iterable[Optional[bool]]) -> Optional[bool]:
    """``True`` when one is ``True``, ``False`` when all are ``False``,
    else ``None``.

    >>> from sympy_extras.core.fuzzy import fuzzy_or
    >>> fuzzy_or([False, True]), fuzzy_or([False, None]), fuzzy_or([False, False])
    (True, None, False)
    """
    return fuzzy_not(fuzzy_and(fuzzy_not(a) for a in args))


# PORTED FROM SYMPY: sympy/core/logic.py:_fuzzy_group
def fuzzy_group(args: Iterable[Optional[bool]], quick_exit: bool = False) -> Optional[bool]:
    """``True`` when all are ``True``, ``None`` when one is ``None``, else
    ``False`` (``None`` at a second ``False`` with ``quick_exit``).

    >>> from sympy_extras.core.fuzzy import fuzzy_group
    >>> fuzzy_group([False, False, True]), fuzzy_group([False, False, True], quick_exit=True)
    (False, None)
    """
    saw_other = False
    for a in args:
        if a is True:
            continue
        if a is None:
            return None
        if quick_exit and saw_other:
            return None
        saw_other = True
    return not saw_other
