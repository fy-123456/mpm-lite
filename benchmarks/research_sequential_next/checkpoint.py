"""Immutable checkpoint generations; only CURRENT.json publishes a commit."""
from __future__ import annotations
import json
import os
from pathlib import Path
import uuid
import numpy as np
from engine.aniso_phase1.research_d.common_state import CommonState
from .provenance import digest, read, sha, sync_dir, write


def decode_state(payload):
    p=dict(payload)
    p['q']=np.array(p['q'],dtype=np.float64);p['velocity']=np.array(p['velocity'],dtype=np.float64)
    if p['predictor'] is not None:p['predictor']=np.array(p['predictor'],dtype=np.float64)
    state=CommonState(**p)
    if state.q.ndim!=2 or state.q.shape[1]!=3 or state.velocity.shape!=state.q.shape:
        raise ValueError('invalid checkpoint coefficient shapes')
    state.digest()  # validates every nested numerical history entry
    return state


class GenerationStore:
    def __init__(self,folder,identity):
        self.folder=Path(folder);self.identity=identity
        self.folder.mkdir(parents=True,exist_ok=True)
        self.pointer=self.folder/'CURRENT.json'
        self.generations=self.folder/'generations';self.generations.mkdir(exist_ok=True)

    def _generation(self,pointer):
        name=pointer['generation']
        if Path(name).name!=name or not name.startswith('g-'):raise ValueError('invalid generation name')
        folder=self.generations/name
        if sha(folder/'manifest.json')!=pointer['sha256']:raise ValueError('generation manifest mismatch')
        manifest=read(folder/'manifest.json')
        if manifest['identity']!=self.identity:raise ValueError('checkpoint model/protocol/numerical source mismatch')
        for filename,expected in manifest['files'].items():
            if Path(filename).name!=filename or sha(folder/filename)!=expected:raise ValueError('generation content mismatch')
        if not {'state.json','ledger.json'}<=set(manifest['files']):raise ValueError('incomplete generation')
        state_payload=read(folder/'state.json');state=decode_state(state_payload['payload'])
        if state.digest()!=state_payload['digest']:raise ValueError('state digest mismatch')
        rows=read(folder/'ledger.json')
        if state.step!=manifest['step'] or state.time!=manifest['time'] or len(rows)!=manifest['ledger_length']:
            raise ValueError('generation metadata differs from payload')
        if rows and (rows[-1]['step']!=state.step or abs(rows[-1]['time']-state.time)>1e-12):
            raise ValueError('state and ledger disagree')
        return folder,manifest,state,rows

    def load(self,validator=None):
        if not self.pointer.exists():return None
        pointer=read(self.pointer)
        folder,manifest,state,rows=self._generation(pointer)
        if validator is not None:validator(state)
        return dict(state=state,rows=rows,manifest=manifest,pointer=pointer,folder=folder)

    def history(self):
        """Verify the published chain, never adopt an orphan after a crash."""
        if not self.pointer.exists():return []
        pointer=read(self.pointer);chain=[];seen=set()
        while pointer is not None:
            if pointer['generation'] in seen:raise ValueError('cyclic generation chain')
            seen.add(pointer['generation'])
            folder,manifest,state,rows=self._generation(pointer)
            chain.append(dict(folder=folder,manifest=manifest,state=state,rows=rows))
            pointer=manifest['previous']
        chain.reverse()
        for a,b in zip(chain[:-1],chain[1:]):
            if b['state'].step!=a['state'].step+1 or b['state'].time<=a['state'].time:
                raise ValueError('noncontiguous checkpoint chain')
            if b['rows'][:-1]!=a['rows']:raise ValueError('rewritten ledger history')
        return chain

    def save(self,state,rows,*,frame=None,inject=None):
        # The caller must hold the global research lock. A complete generation
        # is durable before the pointer changes. Failed writes remain orphans.
        previous=read(self.pointer) if self.pointer.exists() else None
        if previous is not None:
            _,_,old,old_rows=self._generation(previous)
            if state.step!=old.step+1 or state.time<=old.time or rows[:-1]!=old_rows or len(rows)!=len(old_rows)+1:
                raise ValueError('duplicate/stale/noncontiguous generation publication')
        elif rows:
            raise ValueError('initial generation must have an empty local ledger')
        name=f'g-{state.step:06d}-{uuid.uuid4().hex}'
        folder=self.generations/name;folder.mkdir()
        def event(where):
            if inject is not None:inject(where)
        write(folder/'state.json',dict(payload=state.to_dict(),digest=state.digest()))
        event('after_state')
        write(folder/'ledger.json',rows)
        event('after_ledger')
        files=['state.json','ledger.json']
        if frame is not None:
            if not all(np.isfinite(v).all() for v in frame.values()):raise ValueError('nonfinite frame')
            with (folder/'frame.npz').open('wb') as f:
                np.savez_compressed(f,**frame);f.flush();os.fsync(f.fileno())
            files.append('frame.npz')
        event('after_frame')
        manifest=dict(schema='atomic-generation-v1',identity=self.identity,step=state.step,time=state.time,
            ledger_length=len(rows),files={name:sha(folder/name) for name in files},previous=previous)
        write(folder/'manifest.json',manifest);sync_dir(folder);sync_dir(self.generations)
        event('before_pointer')
        write(self.pointer,dict(generation=name,sha256=sha(folder/'manifest.json')))
        event('after_pointer')
        return name
