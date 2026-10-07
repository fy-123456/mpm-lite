"""Matched time/target progression shared by coupled and dry reference drivers."""
import numpy as np

def advance_series(step,state,dt,steps,target_at,accept=None):
    if steps not in (1,2,4) or not np.isfinite(dt) or dt<=0:raise ValueError('registered positive 1/2/4-step window required')
    rows=[]
    for _ in range(steps):
        target=target_at(state.time+dt);trial,row=step(state,dt,target)
        if trial.step!=state.step+1 or not np.isclose(trial.time,state.time+dt,rtol=0,atol=1e-12):raise ValueError('wrong committed time layer')
        state=trial;rows.append(row)
        if accept:accept(state,row)
    return state,rows
