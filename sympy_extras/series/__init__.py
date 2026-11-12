"""Series tools on the typed core (SymPy's ``sympy.series``): ``residue``,
and the limits of sequences ``limit_seq`` (with ``difference_delta`` and
``dominant``)."""
from .limitseq import difference_delta, dominant, limit_seq
from .residues import residue

__all__ = ['residue', 'limit_seq', 'difference_delta', 'dominant']
