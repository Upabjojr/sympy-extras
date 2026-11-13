"""Series tools on the typed core (SymPy's ``sympy.series``): ``residue``,
the limits of sequences ``limit_seq`` (with ``difference_delta`` and
``dominant``), the formal power series ``fps`` and the Fourier series
``fourier_series``, with the sequences they use (``sequence``)."""
from .formal import (FormalPowerSeries, FormalPowerSeriesCompose, FormalPowerSeriesInverse, FormalPowerSeriesProduct,
                     fps)
from .fourier import FiniteFourierSeries, FourierSeries, fourier_series
from .limitseq import difference_delta, dominant, limit_seq
from .residues import residue
from .sequences import EmptySequence, SeqAdd, SeqFormula, SeqMul, SeqPer, sequence

__all__ = ['residue', 'limit_seq', 'difference_delta', 'dominant', 'fps', 'FormalPowerSeries',
           'FormalPowerSeriesProduct', 'FormalPowerSeriesCompose', 'FormalPowerSeriesInverse', 'fourier_series',
           'FourierSeries', 'FiniteFourierSeries', 'sequence', 'SeqFormula', 'SeqPer', 'SeqAdd', 'SeqMul',
           'EmptySequence']
