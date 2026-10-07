"""Audit the descendant and all sealed input content without mutating it."""
from pathlib import Path
import argparse
from .provenance import *

def audit_release(run,full=True):
    run=Path(run);verify(run);pub=read(run/'release.json')
    if pub['schema']!='coupled-reference-practical-v1' or pub['application_parent_release_sha256']!=APP_SHA:raise ValueError('wrong child release')
    if pub['input_lock_sha256']!=sha(run/'input-lock.json'):raise ValueError('input lock differs')
    for entry in pub.values():
        if isinstance(entry,dict) and 'path' in entry and 'sha256' in entry:
            if sha(run/entry['path'])!=entry['sha256']:raise ValueError('release index differs')
    sources=read(run/pub['sources']['path'])
    if sources!=all_sources():raise ValueError('current source inventory differs')
    counts=dict(sources=check(ROOT,sources),snapshots=check(run/'final-source',sources))
    if full:counts['artifacts']=check(run,read(run/pub['artifacts']['path']))
    docs=read(run/'documentation.json')
    for key in ('progress','current_plan'):
        if sha(ROOT/docs[key+'_path'])!=docs[key+'_sha256']:raise ValueError('sealed document changed')
    counts['ancestors']=audit_parent(full)[2];return pub,counts

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();print(audit_release(a.run)[1],flush=True)
