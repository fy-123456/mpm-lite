"""Separate h and p verification on the final completed reference mesh."""
import time
from benchmarks.aniso_v21_common import *

def main(summary_name='reference-extension-summary.json', previous_q3='reference/local4-q3.npz', previous_q4='reference/local3-q4.npz'):
    while not (OUT/summary_name).exists() or not load(OUT/summary_name)['completed']:time.sleep(10)
    extension=load(OUT/summary_name);latest=OUT/extension['latest_reference'];last4=read_field(latest);last3=read_field(latest.with_name(latest.name.replace('-q4','-q3')));pairs={}
    for name,a,b in [('h_q3',read_field(OUT/previous_q3),last3),('h_q4',read_field(OUT/previous_q4),last4)]:
        r=compare(a,b);pairs[name]=r;print(name,r['regions'],flush=True);write(OUT/f'reference-{name}.json',r)
    lastkey=latest.stem.split('-q')[0];cross=extension['pairs'][lastkey+'-cross'];pairs['p_q3_q4']=cross
    checks={r:dict(stress_passed=all(p['regions'][r]['stress_relative']<.02 for p in pairs.values()),fiber_passed=all(p['regions'][r]['fiber_strain_relative']<.02 for p in pairs.values())) for r in ('global','grip','interior','deep_interior')}
    write(OUT/'reference-self-checks.json',dict(completed=True,pairs=pairs,regions=checks,all_stress_and_fiber_passed=all(v['stress_passed'] and v['fiber_passed'] for v in checks.values()),previous_q3=previous_q3,previous_q4=previous_q4,latest_reference=str(latest.relative_to(ROOT)),latest_reference_sha256=sha(latest),continuum_error_bound_proved=False,resource_limit_reached=extension['resource_limit_reached']))
if __name__=='__main__':main()
