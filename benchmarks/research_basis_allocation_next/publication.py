"""Audit or inherit a sealed phase-stress release without overwriting it."""
from pathlib import Path
import argparse
from .provenance import APP,APP_SHA,ROOT,read,write,sha,verify,all_sources,check,audit_parent,freeze,serial_lock

def audit_release(run,full=True):
    run=Path(run);verify(run);pub=read(run/'release.json')
    if pub.get('schema')!='basis-allocation-practical-v1' or pub['application_parent_release_sha256']!=APP_SHA:
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

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['audit']);p.add_argument('--from-release',type=Path,required=True);a=p.parse_args()
    with serial_lock():print(audit_release(a.from_release)[1],flush=True)
if __name__=='__main__':main()
