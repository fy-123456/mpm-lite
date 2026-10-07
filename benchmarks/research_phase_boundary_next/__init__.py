"""Deterministic CLI environment matching the authenticated parent coordinates."""
import os
import sys
if 'numpy' in sys.modules and os.environ.get('OPENBLAS_NUM_THREADS')!='1':
    raise RuntimeError('Import before NumPy, or launch with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1; sealed coordinates require the parent BLAS environment')
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
