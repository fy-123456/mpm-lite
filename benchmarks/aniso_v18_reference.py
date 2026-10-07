"""Unaccelerated v16 AVF with only its residual tolerance tightened.

The original module is immutable. The audited source is executed in a private
namespace; exactly one convergence expression changes. Used to separate
algebraic-solver error from equivalent fast-path error in long moving bridges.
"""
import hashlib
from pathlib import Path
from engine.aniso_phase1 import carrier_avf
text=Path(carrier_avf.__file__).read_text()
assert text.count('rn<=max(1e-14,dt*1e-9)')==1
source=text.replace('rn<=max(1e-14,dt*1e-9)','rn<=1e-17')
namespace=dict(__name__='engine.aniso_phase1.strict_reference',__package__='engine.aniso_phase1')
exec(compile(source,'<v16-with-strict-residual>','exec'),namespace)
StrictReference=namespace['CarrierAVFSolver']
SOURCE_SHA256=hashlib.sha256(text.encode()).hexdigest()
