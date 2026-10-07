"""Material-only and regional metrics; near-zero absolute scales are explicit."""
import numpy as np


def compare(a,b,budget):
    aa=np.asarray(a);bb=np.asarray(b)
    difference=float(np.linalg.norm((aa-bb).ravel())); norm=float(np.linalg.norm(bb.ravel()))
    limit=budget['atol']+budget['rtol']*norm
    return dict(difference=difference,reference_norm=norm,candidate_norm=float(np.linalg.norm(aa.ravel())),
                limit=limit,relative=None if norm==0 else difference/norm,
                scaled=difference/max(norm,budget['atol']/budget['rtol']),passed=bool(difference<=limit))


def metrics(a,b,space,budgets):
    checks={}
    def put(name,x,y,kind):checks[name]=compare(x,y,budgets[kind])
    put('material_energy',a['material_energy_J'],b['material_energy_J'],'energy')
    put('stabilization_energy',a['stabilization_energy_J'],b['stabilization_energy_J'],'energy')
    groups={'all':np.arange(space.ndof),'free':space.free_scalar_ids,'fixed':space.fixed_scalar_ids,
            'left_grip':np.flatnonzero(space.carrier_X[:,0]<=space.boundary['left_x']),
            'right_grip':np.flatnonzero(space.carrier_X[:,0]>=space.boundary['right_x'])}
    for key in ('full_force','material_force'):
        for name,ids in groups.items():
            put(f'{key}/{name}',a[key][ids],b[key][ids],'force')
            if name in ('left_grip','right_grip'):put(f'{key}/{name}_net',a[key][ids].sum(axis=0),b[key][ids].sum(axis=0),'force')
    mid=.5*(space.edges[0][:-1]+space.edges[0][1:])
    regions={'global':np.ones(len(mid),bool),'left_clamp':mid<=.3125,'right_clamp':mid>=.6875,
             'interior':(mid>.3125)&(mid<.6875),'deep_interior':(mid>.375)&(mid<.625)}
    for name,mask in regions.items():
        put(f'weak/{name}',a['weak_moments'][mask].sum(axis=0),b['weak_moments'][mask].sum(axis=0),'weak')
        put(f'energy/{name}',a['slab_energy'][mask].sum(),b['slab_energy'][mask].sum(),'energy')
    for k in range(len(mid)):
        put(f'weak/slab{k}',a['weak_moments'][k],b['weak_moments'][k],'weak')
        put(f'energy/slab{k}',a['slab_energy'][k],b['slab_energy'][k],'energy')
    for direction,ta in a['tangents'].items():
        tb=b['tangents'][direction]
        for key in ('full','material'):
            for name,ids in groups.items():put(f'tangent/{direction}/{key}/{name}',ta[key][ids],tb[key][ids],'tangent')
        for k in range(len(mid)):put(f'tangent_work/{direction}/slab{k}',ta['slab_work'][k],tb['slab_work'][k],'work')
    failed=[k for k,v in checks.items() if not v['passed']]
    return dict(passed=not failed,failed=failed,checks=checks)


def save_response(path,r):
    arrays={k:v for k,v in r.items() if isinstance(v,np.ndarray)}
    scalars={k:r[k] for k in ('energy_J','material_energy_J','stabilization_energy_J','min_detF','point_count')}
    arrays.update({k:np.array(v) for k,v in scalars.items()})
    for d,values in r['tangents'].items():
        for k,v in values.items():arrays[f'tangent__{d}__{k}']=v
    np.savez_compressed(path,**arrays)


def load_response(path):
    with np.load(path,allow_pickle=False) as z:
        r={k:z[k].copy() for k in z.files if not k.startswith('tangent__')};r['tangents']={}
        for k in z.files:
            if k.startswith('tangent__'):
                _,d,n=k.split('__');r['tangents'].setdefault(d,{})[n]=z[k].copy()
        return r
