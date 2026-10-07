"""Authenticated numeric archive for an already qualified solid space.

No pickle, no importable objects or executable payload. The physical space and
all full operators are bit-preserved; this removes repeated R3 construction.
"""
from pathlib import Path
import json,hashlib
import numpy as np
import scipy.sparse as sp
from engine.aniso_phase1.research_d.common_space import CommonSpace
from engine.aniso_phase1.research_sequential.condensation import Condensation
from engine.aniso_phase1.types import AnisotropicMaterialParams

ARRAYS=('carrier_X','Ks','A','oldA','transform','reference','free_scalar_ids','fixed_scalar_ids','lift','q0')
SCALARS=('n','p','ndof','nfree_carrier')
SHAPES=('shape','oldshape','q_shape')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def freeze_array(a):
    a=np.array(a,copy=True);a.setflags(write=False);return a


def save(reduction,folder,parent_entry):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False);s=reduction.parent
    known=set(ARRAYS+SCALARS+SHAPES)|{'edges','oldedges','prolong','raw','signature','metadata','boundary','params','folder','test_vectors'}
    if set(vars(s))-known:raise ValueError('space archive needs explicit review for new fields: '+str(set(vars(s))-known))
    arrays={key:getattr(s,key) for key in ARRAYS};arrays.update(M=reduction.original_mass,K=reduction.original_stiffness)
    for i in range(3):arrays['edge'+str(i)]=s.edges[i];arrays['oldedge'+str(i)]=s.oldedges[i]
    for key,value in s.test_vectors.items():arrays['test_'+key]=value
    np.savez_compressed(folder/'arrays.npz',**arrays);sp.save_npz(folder/'raw.npz',s.raw)
    for i,p in enumerate(s.prolong):sp.save_npz(folder/f'prolong{i}.npz',p)
    meta=dict(schema='qualified-space-array-cache-v1',source_package_sha256=parent_entry['sha256'],space_sha256=s.signature,reduction_sha256=reduction.signature,
        scalars={k:int(getattr(s,k)) for k in SCALARS},shapes={k:list(getattr(s,k)) for k in SHAPES},metadata=s.metadata,boundary=s.boundary,
        original_folder=str(s.folder),test_keys=sorted(s.test_vectors),parameters=dict(mu=s.params.mu,lam=s.params.lam,k_f=s.params.k_f,fiber_direction=list(s.params.fiber_direction)),
        files={p.name:sha(p) for p in sorted(folder.iterdir()) if p.is_file()},construction='exact arrays from qualified slow reconstruction; no approximation or changed basis')
    (folder/'cache.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    return dict(path=str((folder/'cache.json').resolve()),sha256=sha(folder/'cache.json'))


def load(entry,parent_entry):
    path=Path(entry['path'])
    if sha(path)!=entry['sha256']:raise ValueError('space cache manifest changed')
    m=json.loads(path.read_text());folder=path.parent
    if m['schema']!='qualified-space-array-cache-v1' or m['source_package_sha256']!=parent_entry['sha256']:raise ValueError('cache belongs to another space package')
    expected={'arrays.npz','raw.npz','prolong0.npz','prolong1.npz','prolong2.npz'}
    if set(m['files'])!=expected:raise ValueError('unexpected space cache members')
    for name,h in m['files'].items():
        if sha(folder/name)!=h:raise ValueError('space cache data changed: '+name)
    s=CommonSpace.__new__(CommonSpace)
    with np.load(folder/'arrays.npz',allow_pickle=False) as z:
        for k in ARRAYS:setattr(s,k,freeze_array(z[k]))
        s.edges=tuple(freeze_array(z['edge'+str(i)]) for i in range(3));s.oldedges=tuple(freeze_array(z['oldedge'+str(i)]) for i in range(3))
        s.test_vectors={k:freeze_array(z['test_'+k]) for k in m['test_keys']};M=z['M'].copy();K=z['K'].copy()
    for k in SCALARS:setattr(s,k,m['scalars'][k])
    for k in SHAPES:setattr(s,k,tuple(m['shapes'][k]))
    s.raw=sp.load_npz(folder/'raw.npz').tocsr();s.prolong=tuple(sp.load_npz(folder/f'prolong{i}.npz').tocsr() for i in range(3))
    for a in (s.raw,*s.prolong):
        for x in (a.data,a.indices,a.indptr):x.setflags(write=False)
    s.signature=m['space_sha256'];s.metadata=m['metadata'];s.boundary=m['boundary'];s.folder=Path(m['original_folder']);s.params=AnisotropicMaterialParams(**m['parameters'])
    if s.raw.shape!=(int(np.prod(s.shape)),s.transform.shape[0]) or s.ndof!=s.n+s.transform.shape[1]:raise ValueError('cached space dimensions inconsistent')
    r=Condensation(s,M,K)
    if r.signature!=m['reduction_sha256']:raise ValueError('cached reduction identity differs')
    return r
