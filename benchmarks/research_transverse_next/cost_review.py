"""Include first candidate compilation when checking setup recovery."""
import argparse,re
from .provenance import *

def main(run):
    run=Path(run);mutable(run);pairs=read(run/'S4/paired-performance.json');text=(run/'processes/candidate-static.log').read_text()
    matches=re.findall(r'Module engine\.aniso_phase1\.research_transverse_next\.device_volume .*?took ([0-9.]+) ms\s+\(compiled\)',text)
    jit=sum(float(v)/1000 for v in matches);rows=[]
    for pair in pairs['records']:
        a,b=pair['D3'],pair['DV'];saving=a['advance_s']-b['advance_s']
        # Installation CPU cost is charged even if unrelated build noise made
        # total candidate setup shorter. Count initial module compilation once.
        extra=max(pair['additional_total_setup_s'],b['additional_setup_s'])+jit
        recovery=extra/saving if saving>0 else None
        rows.append(dict(input_step=pair['input_step'],positive_extra_setup_s=extra,new_kernel_cold_compile_s=jit,step_saving_s=saving,recovery_steps=recovery,passed=recovery is not None and recovery<=16))
    passed=all(r['passed'] for r in rows);decision=read(run/'S4/backend-decision.json')
    if decision['selected'] and not passed:
        decision.update(status='retained',backend='D3',selected=False,reason='cold compile inclusive setup recovery exceeds16steps');write(run/'S4/backend-decision.json',decision)
        write(run/'S4/continuous-check.json',dict(status='not_triggered',reason=decision['reason'],new_steps=0))
    write(run/'S4/setup-cost.json',dict(status='passed_scoped' if passed else 'limited',records=rows,cold_compilation_log='processes/candidate-static.log',timing_includes_model_build_probe_and_IO=True,cold_q_cache=True,compile_charged_once=True,criterion='<=16 steps',passed=passed))
    print('SETUP',rows,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):main(a.run)
