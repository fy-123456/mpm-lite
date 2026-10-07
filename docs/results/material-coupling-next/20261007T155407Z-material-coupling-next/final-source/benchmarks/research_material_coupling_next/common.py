from pathlib import Path
import numpy as np
from benchmarks.research_formal_pressure_next.geometry_check import write
from engine.aniso_phase1.research_absolute_state_next.state import MaterialStateModel
from engine.aniso_phase1.research_formal_pressure_next.state import State

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'docs/results/formal-pressure-next/20261007T145423Z-formal-pressure-next'
def unit_boundary(m):
    q=m.rest().q;unit=np.zeros_like(q);right=m.r.fixed[m.space.carrier_X[m.space.fixed_scalar_ids,0]>=.75];unit[right,0]=1
    return unit,right

def samples(m):
    rng=np.random.default_rng(7117)
    with np.load(BASE/'S4/closed/checkpoint.npz',allow_pickle=False) as d:inherited=d['q'].copy()
    mixed=inherited.copy();mixed[m.r.free]+=rng.normal(size=(len(m.r.free),3))*1e-6
    full=np.zeros((m.space.ndof,3));full[m.space.n:]=rng.normal(size=(m.space.ndof-m.space.n,3))*1e-5
    local=m.r.project(full)
    direction=np.zeros_like(local);direction[m.r.free]=rng.normal(size=(len(m.r.free),3))*1e-6
    return dict(inherited=inherited,mixed=mixed,local=local),direction
