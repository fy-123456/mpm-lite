"""A loaded, immutable reduction shared only within one model construction tree."""
import numpy as np
import scipy.sparse as sp
from engine.aniso_phase1.research_d.identity import digest

def freeze_arrays(obj):
    for value in vars(obj).values():
        if isinstance(value,np.ndarray):value.setflags(write=False)
        elif sp.issparse(value):
            for array in (value.data,value.indices,value.indptr):array.setflags(write=False)
        elif isinstance(value,tuple):
            for item in value:
                if sp.issparse(item):
                    for array in (item.data,item.indices,item.indptr):array.setflags(write=False)
                elif isinstance(item,np.ndarray):item.setflags(write=False)

class SharedReduction:
    def __init__(self,entry,reduction,package,device):
        freeze_arrays(reduction);freeze_arrays(reduction.parent)
        self.reduction=reduction;self.package=dict(package)
        self.key=digest(dict(entry=entry,device=device,reduction=reduction.signature,version='immutable-reduction-v1'))
    def acquire(self,entry,device):
        key=digest(dict(entry=entry,device=device,reduction=self.reduction.signature,version='immutable-reduction-v1'))
        if key!=self.key:raise ValueError('foreign shared space/device identity')
        return self.reduction,dict(self.package)
