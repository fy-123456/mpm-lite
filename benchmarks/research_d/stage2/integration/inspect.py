"""Inspect sealed handoff availability without importing live foreign code."""
from pathlib import Path
import json

def compatibility(repository):
    root=Path(repository);rows={}
    for group in 'BCE':
        result=root/'docs/results/parallel-v22-stage2'/group
        folders=sorted(p for p in result.glob('*') if p.is_dir())
        # Discovery is not acceptance: a producer must supply an explicitly
        # pinned contract + source/input closure before any cross-group import.
        packets=[p/'handoff.json' for p in folders if (p/'handoff.json').is_file()]
        rows[group]=dict(status='awaiting_pinned_handoff' if not packets else 'requires_explicit_packet_validation',
            observed_directories=[str(p.relative_to(root)) for p in folders],
            observed_packets=[str(p.relative_to(root)) for p in packets],accepted=False)
    rows['A']=dict(status='parent_space_verified',accepted=True,scope='unchanged parent instance only')
    rows['requirements']=dict(B=['fixed material rule','material/source identity','unchanged mass identity','state signature'],
        C=['equation','mass','lift and lift velocity','boundary work','transaction revision','read-only operator'],
        E=['independent 2D space','mixed_general','true residual','coupled_physics evidence'])
    rows['dynamic_gpu']=dict(accepted=False,status='out_of_scope_until_C_equations_and_cycle_frozen')
    return rows

if __name__=='__main__':print(json.dumps(compatibility(Path.cwd()),indent=2))
