"""Suite-wide fixtures/environment.

torch (via the mace tests) and the LAMMPS wheel each bundle their own
libomp on macOS; loading both in one pytest process aborts with the
duplicate-OpenMP-runtime error unless KMP_DUPLICATE_LIB_OK is set — and
even then the two runtimes DEADLOCK at default thread counts (measured:
suite hangs at 0% CPU). OMP_NUM_THREADS=1 removes the contention; the
suite runs in ~35 s single-threaded. Both must be set before either
library is imported, hence conftest. Examples never load torch and LAMMPS
in the same process, so this is test-only.
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
