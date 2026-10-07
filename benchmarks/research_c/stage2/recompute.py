"""Verify a sealed C package and independently reconstruct its zero initial state."""
import argparse
import json
from pathlib import Path
import numpy as np
from engine.aniso_phase1.research_c.stage2.delivery import verify_delivery, recompute
from engine.aniso_phase1.research_c.stage2.model import DynamicModel, MassRankError, recover_mass
from engine.aniso_phase1.research_d.common_kinetic import PointInertia
from engine.aniso_phase1.research_d.common_state import CommonState


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder',type=Path)
    p.add_argument('--trusted-parent-data-root',type=Path,required=True)
    p.add_argument('--expected-manifest-sha256')
    p.add_argument('--manifest',default='handoff-package.json')
    p.add_argument('--require-dynamic',action='store_true')
    p.add_argument('--rebuild-mass',action='store_true')
    args=p.parse_args()
    root=Path(__file__).resolve().parents[3]
    s,result=verify_delivery(args.folder,root,trusted_parent_data_root=args.trusted_parent_data_root,
        expected_manifest_sha256=args.expected_manifest_sha256,require_dynamic=args.require_dynamic,
        manifest_name=args.manifest)
    with np.load(args.folder/'initial-dynamic-state.npz') as z:
        state=CommonState(z['q'],z['velocity'],float(z['time']),int(z['step']))
        points=[z[f'points{k}'] for k in range(3)]
        expected_F,expected_P=z['F'],z['PK1']
    with np.load(args.folder/'operators.npz') as z:
        mass,K=z['M5'],z['K']
    fields=recompute(s,state,points,mass)
    result.update(initial_F_max_error=float(np.max(abs(fields['F']-expected_F))),
        initial_PK1_max_error=float(np.max(abs(fields['PK1']-expected_P))),
        kinematic_fields_finite=all(np.isfinite(v).all() for v in fields.values()))
    try:
        DynamicModel(s,mass,rest_K=K)
    except MassRankError as exc:
        result.update(dynamic_admission_rejected=True,rejection=str(exc))
    else:
        result['dynamic_admission_rejected']=False
    if args.rebuild_mass:
        errors=[]
        with np.load(args.folder/'operators.npz') as saved:
            for order in (5,6):
                rebuilt=recover_mass(PointInertia(s,order=order))
                reference=saved[f'M{order}']
                errors.append(float(np.linalg.norm(rebuilt-reference)/np.linalg.norm(reference)))
        result['independent_full_mass_recovery_relative']=errors
    result['passed']=bool(max(result.get('independent_full_mass_recovery_relative',[0.]))<1e-10 and
        result['initial_F_max_error']<1e-12 and
        result['initial_PK1_max_error']<1e-10 and result['kinematic_fields_finite'] and
        result['dynamic_admission_rejected'])
    print(json.dumps(result,indent=2))
    if not result['passed']: raise SystemExit(1)


if __name__=='__main__': main()
