"""Bounded exact-state linearizations; immutable public responses and tokens."""
from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
from pathlib import Path
import uuid
import numpy as np
from ..research_d.stage2.contracts import array_digest
from ..research_d.identity import digest
from .model import readonly_response
from .segmented import SegmentedModel


@dataclass(frozen=True)
class PreparedState:
    owner: str
    key: str
    serial: int


class LinearizationCache:
    def __init__(self,operator,*,max_bytes=3<<30,max_entries=4):
        if max_bytes<=0 or max_entries<1:raise ValueError('positive bounded cache limits required')
        self.operator=operator;self.max_bytes=int(max_bytes);self.max_entries=int(max_entries)
        self._owner=uuid.uuid4().hex;self._entries=OrderedDict();self._serial=0
        self.used_bytes=0;self.peak_bytes=0
        self.stats=dict(hits=0,misses=0,evictions=0,tangent_actions=0,prepared_points=0,tangent_points=0)
        source=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self.signature=digest(dict(operator=operator.signature,device=str(operator.device),dtype='float64',cache_source=source))
        self.estimate=operator.count*21*8+256*1024
        if self.estimate>self.max_bytes:raise MemoryError('cache cannot contain even one complete linearization')

    def _key(self,q):
        q=self.operator.space._check(q,self.operator.space.q_shape,'linearization full state')
        return digest(dict(cache=self.signature,state=array_digest(q)))

    @staticmethod
    def _bytes(value):
        return int(value.capacity if hasattr(value,'capacity') else value.nbytes)

    def _evict(self):
        _,(_,_,size)=self._entries.popitem(last=False)
        self.used_bytes-=size;self.stats['evictions']+=1

    def prepare(self,q):
        key=self._key(q)
        if key in self._entries:
            token,lin,size=self._entries[key];self._entries.move_to_end(key);self.stats['hits']+=1
            return token,False
        self.stats['misses']+=1
        while self._entries and (self.used_bytes+self.estimate>self.max_bytes or len(self._entries)>=self.max_entries):self._evict()
        lin=self.operator.prepare(q)
        lin.response=readonly_response(lin.response)
        size=sum(self._bytes(v) for v in (lin.F,lin.Q,lin.c))+sum(v.nbytes for v in lin.response.values() if isinstance(v,np.ndarray))
        if size>self.max_bytes:raise MemoryError('actual linearization exceeds cache byte limit')
        while self._entries and self.used_bytes+size>self.max_bytes:self._evict()
        self._serial+=1;token=PreparedState(self._owner,key,self._serial)
        self._entries[key]=(token,lin,size);self.used_bytes+=size;self.peak_bytes=max(self.peak_bytes,self.used_bytes)
        self.stats['prepared_points']+=self.operator.count
        return token,True

    def _get(self,token,q):
        if not isinstance(token,PreparedState) or token.owner!=self._owner or token.key!=self._key(q):
            raise ValueError('foreign operator, rule, source, device, or state linearization')
        entry=self._entries.get(token.key)
        if entry is None or entry[0]!=token:raise ValueError('expired or evicted linearization')
        self._entries.move_to_end(token.key)
        return entry[1]

    def response(self,token,q):
        return readonly_response(self._get(token,q).response)

    def action(self,token,q,direction):
        lin=self._get(token,q)
        d=self.operator.space._check(direction,self.operator.space.q_shape,'linearization full direction')
        out=self.operator.action(lin,d)
        self.stats['tangent_actions']+=1;self.stats['tangent_points']+=self.operator.count
        return out

    def clear(self):
        self._entries.clear();self.used_bytes=0

    def report(self):
        return dict(self.stats,used_bytes=self.used_bytes,peak_bytes=self.peak_bytes,max_bytes=self.max_bytes,
                    entries=len(self._entries),max_entries=self.max_entries)


class CachedModel(SegmentedModel):
    def __init__(self,reduction,*,order=7,device='cuda:0',hold=None,cache_bytes=3<<30,cache_entries=4):
        super().__init__(reduction,order=order,device=device,hold=hold)
        self.linearizations=LinearizationCache(self.operator,max_bytes=cache_bytes,max_entries=cache_entries)

    def evaluate(self,q,direction=None):
        full=self.reduction.expand(q);token,created=self.linearizations.prepare(full)
        raw=self.linearizations.response(token,full)
        force=raw['full_force']
        out=dict(raw,force=self.reduction.P.T@force,material_force=self.reduction.P.T@raw['material_force'],
            original_full_force=force,algebraic_force=self.reduction.N.T@force,
            material_calls=self.operator.count if created else 0)
        if direction is not None:
            d=self.reduction.velocity(direction)
            action=self.linearizations.action(token,full,d)
            out['tangent_action']=self.reduction.P.T@action
            out['material_calls']+=self.operator.count
        return readonly_response(out)
