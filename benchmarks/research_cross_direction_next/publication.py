"""Create a new authenticated child run or audit its immutable release."""
from pathlib import Path
import argparse
from .provenance import ROOT,APP,APP_SHA,read,sha,check,all_sources,audit_parent,freeze,serial_lock,verify

def audit_release(run,full=True):
    run=Path(run);verify(run);pub=read(run/'release.json')
    if pub['schema']!='cross-direction-practical-v1' or pub['application_parent_release_sha256']!=APP_SHA:raise ValueError('unsupported lineage')
    if pub['input_lock_sha256']!=sha(run/'input-lock.json'):raise ValueError('input changed')
    for entry in pub.values():
        if isinstance(entry,dict) and 'path' in entry and 'sha256' in entry:
            if sha(run/entry['path'])!=entry['sha256']:raise ValueError('release entry changed')
    sources=read(run/pub['sources']['path'])
    if sources!=all_sources():raise ValueError('source inventory changed')
    counts=dict(sources=check(ROOT,sources),snapshots=check(run/'final-source',sources))
    if full:counts['artifacts']=check(run,read(run/pub['artifacts']['path']))
    docs=read(run/'documentation.json')
    for key in ('progress','current_plan'):
        if sha(ROOT/docs[key+'_path'])!=docs[key+'_sha256']:raise ValueError('document changed')
    counts['ancestors']=audit_parent(full)[2]
    return pub,counts

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['create','audit']);p.add_argument('--from-release',type=Path,required=True);a=p.parse_args()
    if a.command=='create':
        if a.from_release.resolve()!=APP.resolve() or sha(a.from_release/'release.json')!=APP_SHA:raise ValueError('explicit latest parent required')
        with serial_lock():print(freeze(),flush=True)
    else:
        with serial_lock():print(audit_release(a.from_release)[1],flush=True)
if __name__=='__main__':main()
