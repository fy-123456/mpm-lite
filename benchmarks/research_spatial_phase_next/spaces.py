"""Explicit material-space loading; no implicit replacement of old coordinates."""
from pathlib import Path
from .provenance import APP,PARENT,read,sha
from benchmarks.research_sequential_next.model_package import load_reduction


def load_selected(entry=None):
    if entry is None:return load_reduction(PARENT),dict(selected='original144',mass_order=5,full_order=7)
    path=Path(entry['path'])
    if sha(path)!=entry['sha256']:raise ValueError('selected space package changed')
    if entry.get('cache') is not None:
        from engine.aniso_phase1.research_spatial_phase_next.performance import load
        package=read(path)
        if sha(path.parent/'space.npz')!=package['data_sha256'] or sha(APP/'Q1/R3/space-package.json')!=package['reference_sha256']:raise ValueError('cached space source dependencies changed')
        r=load(entry['cache'],entry)
        if r.signature!=package['reduction_sha256']:raise ValueError('cached space package identity mismatch')
        return r,package
    return reopen_candidate(path.parent)


def reopen_candidate(folder, reference_reduction=None):
    import numpy as np
    from benchmarks.research_reference_next.reference_study import reopen
    from engine.aniso_phase1.research_spatial_phase_next.space import combine
    folder=Path(folder);p=read(folder/'space-package.json')
    if sha(folder/'space.npz')!=p['data_sha256'] or sha(APP/'Q1/R3/space-package.json')!=p['reference_sha256']:
        raise ValueError('candidate dependencies changed')
    r=reference_reduction if reference_reduction is not None else reopen(APP/'Q1/R3')[0]
    with np.load(folder/'space.npz',allow_pickle=False) as z:
        cr,T=combine(r,z['C'],p['name'])
        if cr.signature!=p['reduction_sha256'] or not np.array_equal(cr.original_mass,z['M']):
            raise ValueError('candidate reconstructed identity changed')
    return cr,p
