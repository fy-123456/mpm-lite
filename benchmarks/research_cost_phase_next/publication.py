"""Audit or inherit a sealed cost-phase release without overwriting it."""
from pathlib import Path
import argparse
from .provenance import APP,APP_SHA,ROOT,read,write,sha,verify,all_sources,check,audit_parent,freeze,serial_lock

def audit_release(run,full=True):
    run=Path(run);verify(run);pub=read(run/'release.json')
    if pub.get('schema')!='cost-phase-practical-v1' or pub['application_parent_release_sha256']!=APP_SHA:
        raise ValueError('unsupported release lineage')
    if pub['input_lock_sha256']!=sha(run/'input-lock.json'):raise ValueError('release input changed')
    for key,entry in pub.items():
        if isinstance(entry,dict) and 'path' in entry and 'sha256' in entry:
            if sha(run/entry['path'])!=entry['sha256']:raise ValueError('release entry changed: '+key)
    sources=read(run/pub['sources']['path'])
    if sources!=all_sources():raise ValueError('sealed source inventory differs')
    counts=dict(sources=check(ROOT,sources),snapshots=check(run/'final-source',sources))
    if full:counts['artifacts']=check(run,read(run/pub['artifacts']['path']))
    docs=read(run/'documentation.json')
    for key in ('progress','current_plan'):
        if sha(ROOT/docs[key+'_path'])!=docs[key+'_sha256']:raise ValueError('release document changed: '+key)
    counts['ancestors']=audit_parent(full)[2]
    return pub,counts

def fork(parent):
    parent=Path(parent).resolve();pub,counts=audit_release(parent)
    run=freeze();lock=read(run/'input-lock.json')
    lock['continuation_parent']=dict(path=str(parent),release_sha256=sha(parent/'release.json'),counts=counts)
    lock['direct_application_parent']=str(parent)
    write(run/'input-lock.json',lock);write(run/'input-lock-sha256.json',dict(sha256=sha(run/'input-lock.json')))
    write(run/'P0/version-audit.json',dict(status='passed',**lock))
    for source,target in [('selected-space.json','selected-space.json'),('P1/performance-decision.json','P1/performance-decision.json'),
                          ('P5/qualification-final.json','P5/qualification-final.json'),('P5/qualification-peak0075.json','P5/qualification-peak0075.json')]:
        p=run/target;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((parent/source).read_bytes())
    return run

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['audit','fork']);p.add_argument('--from-release',type=Path,required=True);a=p.parse_args()
    with serial_lock():
        if a.command=='fork':print(fork(a.from_release),flush=True)
        else:print(audit_release(a.from_release)[1],flush=True)

if __name__=='__main__':main()
