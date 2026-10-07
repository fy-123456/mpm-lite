"""Verify and fork a sealed reference-next release without mutating it."""
from pathlib import Path
import argparse
from .provenance import APP_SHA,all_sources,freeze,verify,audit_parent,check,sha,read,write,serial_lock


def audit_release(run,full=True):
    run=Path(run);verify(run);pub=read(run/'release.json')
    if pub.get('schema')!='reference-next-practical-v1' or pub.get('application_parent_release_sha256')!=APP_SHA:
        raise ValueError('unsupported reference-next release lineage')
    if pub['input_lock_sha256']!=sha(run/'input-lock.json'):raise ValueError('release input lock changed')
    for key in ('sources','artifacts','qualification','sensitive_qualification','scene'):
        x=pub[key]
        if sha(run/x['path'])!=x['sha256']:raise ValueError('release index changed: '+key)
    sources=read(run/pub['sources']['path'])
    if sources!=all_sources():raise ValueError('current source inventory differs from sealed release')
    counts=dict(sources=check(Path(__file__).resolve().parents[2],sources),snapshots=check(run/'final-source',sources))
    if full:counts['artifacts']=check(run,read(run/pub['artifacts']['path']))
    counts['ancestors']=audit_parent(full)[2]
    return pub,counts


def fork(parent):
    parent=Path(parent).resolve();pub,counts=audit_release(parent)
    run=freeze();lock=read(run/'input-lock.json')
    # The original application/math ancestry is retained. The new immediate
    # application parent is explicit and bound by the new input-lock hash.
    lock['continuation_parent']=dict(path=str(parent),release_sha256=sha(parent/'release.json'),counts=counts)
    lock['direct_application_parent']=str(parent)
    write(run/'input-lock.json',lock);write(run/'input-lock-sha256.json',dict(sha256=sha(run/'input-lock.json')))
    write(run/'Q0/baseline-audit.json',dict(status='passed',**lock))
    for key,target in [('qualification','qualification.json'),('sensitive_qualification','qualification-peak0075.json')]:
        (run/'Q4'/target).write_bytes((parent/pub[key]['path']).read_bytes())
    return run


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['audit','fork']);p.add_argument('--from-release',type=Path,required=True);a=p.parse_args()
    with serial_lock():
        if a.command=='fork':print(fork(a.from_release),flush=True)
        else:print(audit_release(a.from_release)[1],flush=True)


if __name__=='__main__':main()
