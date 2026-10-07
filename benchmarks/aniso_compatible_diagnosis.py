"""Read-only v14 attribution: spatial energy, forces and shared-history defects.

The fixed reference Lite gradient is a *discrete compatibility diagnostic*,
not an exact continuum gradient. Patch-row energy allocation is explicitly a
bookkeeping convention; exterior support has no physical material volume.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from benchmarks.aniso_dynamic_check import CORNERS, interpolation, pk1
from engine.aniso_phase1.material_patch import carrier_map
from engine.aniso_phase1.diagnostics import energy_density
from engine.aniso_phase1.types import AnisotropicMaterialParams

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'docs/results/lite-aniso-mainline'
LO = np.array([.125, .375, .375])
HI = np.array([.875, .625, .625])


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def maps(points, nodes, h):
    """Independent reference/current center-average position and gradient maps."""
    centers = np.unique((np.floor(points / h - .5).astype(int)[:, None] + CORNERS).reshape(-1, 3), axis=0)
    lookup = {tuple(n): i for i, n in enumerate(nodes)}
    ids = np.array([[lookup[tuple(c + o)] for o in CORNERS] for c in centers])
    rows = np.repeat(np.arange(len(centers)), 8)
    H = sp.csr_matrix((np.full(len(rows), .125), (rows, ids.ravel())), shape=(len(centers), len(nodes)))
    D = [sp.csr_matrix((np.tile((2 * CORNERS[:, k] - 1) / (4 * h), len(centers)),
                       (rows, ids.ravel())), shape=H.shape) for k in range(3)]
    S = interpolation(points, centers, h)
    return S @ H, tuple(S @ d for d in D)


def gradient(G, Y):
    return np.stack([g @ Y for g in G], axis=2)


def zones(X, width=.0625):
    outside = np.any((X < LO - 1e-12) | (X > HI + 1e-12), axis=1)
    grip = (~outside) & ((X[:, 0] <= .25 + width + 1e-12) | (X[:, 0] >= .75 - width - 1e-12))
    return dict(exterior=outside, grip_transition=grip, interior=~(outside | grip))


def scalar_K(z):
    ids, P, w = z['patch_ids'], z['patch_P'], z['patch_weight']
    m = ids.shape[1]
    return sp.csr_matrix(((w[:, None, None] * (P.swapaxes(1, 2) @ P)).ravel(),
                         (np.repeat(ids, m, axis=1).ravel(), np.tile(ids, (1, m)).ravel())),
                        shape=(len(z['patch_X']),) * 2)


def audit(z, cfg, saved_row):
    h, dt = 1 / (cfg['grid'] - 1), cfg['dt']
    X, Y = z['patch_X'], z['marker_after']
    Xp, F, F0, V = (z[k] for k in ('particle_reference_x', 'particle_F_after', 'particle_F_before', 'particle_volume'))
    T0, G0 = maps(Xp, np.rint(X / h).astype(int), h)
    T, G = maps(z['particle_x_before'], z['native_nodes'], h)
    _, _, N = carrier_map(z['patch_origin'], z['native_nodes'], h)
    v = z['native_velocity']
    FY = gradient(G0, Y)
    FbeforeY = gradient(G0, z['patch_origin'])
    rate_F = gradient(G, v) @ F0
    rate_Y = gradient(G0, N @ v)
    defect = F - FY
    params = AnisotropicMaterialParams(10., 20., cfg['kf'])
    material = V * energy_density(F, z['particle_A0'], params)
    compatible_material = V * energy_density(FY, z['particle_A0'], params)
    r = np.einsum('cij,cja->cia', z['patch_P'], Y[z['patch_ids']])
    energy_rows = .5 * z['patch_weight'][:, None] * np.sum(r * r, axis=2)
    energy_nodes = np.zeros(len(Y))
    np.add.at(energy_nodes, z['patch_ids'].ravel(), energy_rows.ravel())
    carrier_force = np.zeros_like(Y)
    local_force = z['patch_weight'][:, None, None] * np.einsum('cji,cja->cia', z['patch_P'], r)
    np.add.at(carrier_force, z['patch_ids'].ravel(), local_force.reshape(-1, 3))
    fs = N.T @ carrier_force
    P = pk1(F, z['particle_A0'], cfg['kf'])
    B = [sum(G[j].multiply(F0[:, j, k, None]) for j in range(3)).tocsr() for k in range(3)]
    fm = sum(b.T @ (V[:, None] * P[:, :, k]) for k, b in enumerate(B))
    free = (z['native_nodes'][:, 0] * h > .25) & (z['native_nodes'][:, 0] * h < .75)
    norm = lambda a: float(np.linalg.norm(a))
    rms = lambda a, mask: float(np.sqrt(np.sum(V[mask, None, None] * a[mask]**2) / V[mask].sum())) if np.any(mask) else 0.
    particle_zones, marker_zones, node_zones = zones(Xp), zones(X), zones(z['native_nodes'] * h)
    regions = {}
    for key in particle_zones:
        pm, ym, nm = particle_zones[key], marker_zones[key], node_zones[key]
        regions[key] = dict(material_J=float(material[pm].sum()), stabilization_row_J=float(energy_nodes[ym].sum()),
            marker_count=int(ym.sum()), particle_count=int(pm.sum()), material_grid_force_l2_N=norm(fm[nm]),
            stabilization_grid_force_l2_N=norm(fs[nm]), net_free_force_l2_N=norm((fm+fs)[nm & free]),
            material_force_sum_N=fm[nm].sum(0).tolist(), stabilization_force_sum_N=fs[nm].sum(0).tolist(),
            history_F_rms=rms(defect, pm), strain_rms=rms(F-np.eye(3), pm),
            history_rate_rms_per_s=rms(rate_F-rate_Y, pm))
    allp = np.ones(len(F), bool)
    Us = float(energy_rows.sum()); Um = float(material.sum())
    errors = dict(energy_J=abs(Us - saved_row['stabilization_energy']),
        material_J=abs(Um - (saved_row['elastic'] - saved_row['stabilization_energy'])),
        carrier_commit=float(np.max(abs(z['patch_origin'] + dt * (N @ v) - Y))),
        particle_commit=float(np.max(abs(F0 + dt * rate_F - F))),
        defect_increment=float(np.max(abs(defect - (F0-FbeforeY) - dt*(rate_F-rate_Y)))),
        partition_J=abs(sum(a['stabilization_row_J'] for a in regions.values()) - Us))
    assert max(errors.values()) < 1e-10, errors
    # Project the marker displacement into material-visible and material-null
    # subspaces. This is an algebraic attribution, never a proposed reset.
    A = np.vstack([g.toarray() * np.sqrt(V[:, None]) for g in G0])
    _, sing, Vt = np.linalg.svd(A, full_matrices=False)
    keep = sing > 1e-10 * sing[0]
    u = Y-X; visible = Vt[keep].T @ (Vt[keep] @ u); invisible = u-visible
    K = scalar_K(z)
    visible_E = float(.5*np.sum(visible*(K@visible)))
    invisible_E = float(.5*np.sum(invisible*(K@invisible)))
    cross_E = float(np.sum(visible*(K@invisible)))
    return dict(material_J=Um, stabilization_J=Us, kinetic_J=saved_row['kinetic'],
        compatible_counterfactual_material_J=float(compatible_material.sum()),
        history_F_rms=rms(defect, allp), strain_rms=rms(F-np.eye(3), allp),
        history_rate_rms_per_s=rms(rate_F-rate_Y, allp),
        position_history_rms_m=float(np.sqrt(np.mean(np.sum((z['particle_x_after']-T0@Y)**2, axis=1)))),
        free_material_force_l2_N=norm(fm[free]), free_stabilization_force_l2_N=norm(fs[free]),
        free_net_force_l2_N=norm((fm+fs)[free]),
        free_force_cosine=float(np.sum(fm[free]*fs[free])/max(norm(fm[free])*norm(fs[free]), 1e-30)),
        regions=regions, errors=errors, gradient_rank=int(keep.sum()),
        marker_energy_decomposition_J=dict(material_visible=visible_E, material_null=invisible_E, cross=cross_E,
            closure=abs(visible_E+invisible_E+cross_E-Us))), dict(marker_X=X, marker_Y=Y,
            marker_energy_J=energy_nodes, carrier_force=carrier_force, grid_nodes=z['native_nodes']*h,
            material_force=fm, stabilization_force=fs, particle_X=Xp, material_energy_J=material, history_defect=defect)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--output', type=Path, default=BASE/'v15/diagnosis')
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((BASE/'v14/protocol.json').read_text()); records=[]; inputs={}
    for level in ('coarse', 'fine', 'finest', 'fourth'):
        cfg = protocol['configs']['material-'+level]; folder=BASE/'v14/cases'/('material-'+level)
        rows=[json.loads(line) for line in (folder/'steps.jsonl').read_text().splitlines()]
        for t in (.5, .85, 1.1, 1.2, 1.4, 1.6):
            step=round(t/cfg['dt']); path=folder/f'audit-{step:05d}.npz'
            with np.load(path) as f:z={k:f[k] for k in f.files}
            result, arrays=audit(z,cfg,rows[step-1]); result.update(level=level,time=t)
            records.append(result); inputs[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
            np.savez_compressed(a.output/f'{level}-{t:g}.npz', **arrays)
            print(level,t,'Us',result['stabilization_J'],'Fdefect',result['history_F_rms'],
                  'strain',result['strain_rms'],'cos',result['free_force_cosine'],flush=True)
    write(a.output/'summary.json',dict(completed=True,records=records,input_sha256=inputs,
        definition='fixed initial Lite gradient G0: F_Y=G0Y; position comparator T0Y; not a continuum error',
        regions='disjoint exterior of physical [.125,.875]x[.375,.625]^2, grip/transition x<=.3125 or x>=.6875, interior',
        energy_allocation='sum .5*w_c*|(P_c Y)_j|^2 at reference marker j; nonunique bookkeeping, not exterior material energy',
        counterfactual='evaluating material energy at G0Y does not authorize discarding particle history',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))


if __name__ == '__main__':
    main()
