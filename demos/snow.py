import numpy as np
np.random.seed(0)
import warp as wp
wp.config.enable_backward = False
import os, shutil
from icecream import ic
import math
import time
from utils.sampling_utils import *
from engine.solver3d import MPMSolver
from engine.types import *
from argparse import ArgumentParser
import trimesh
from utils.viser_utils import ParticleViewer

parser = ArgumentParser()
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--method', type=str, default='lite_explicit')
parser.add_argument('--dt', type=float, default=1e-4)
parser.add_argument('--dx', type=float, default=0.0063)
parser.add_argument('--v_tol', type=float, default=1e-4)
parser.add_argument('--max_iters', type=int, default=50)
parser.add_argument('--out', type=str, default='output/snow')
parser.add_argument('--gui', default=False, action='store_true',
                    help='deprecated; browser visualization is always enabled')
parser.add_argument('--viser-host', type=str, default='127.0.0.1')
parser.add_argument('--viser-port', type=int, default=8080)
parser.add_argument('--viser-max-points', type=int, default=250000)
parser.add_argument('--grid_size', type=int, nargs='+', default=[355])
parser.add_argument('--dim', type=int, default=3)
parser.add_argument('--ppc', type=float, default=8)
parser.add_argument('--print_every', type=int, default=10)
args = parser.parse_args()

OUTPUT_DIR = args.out
if os.path.isdir(OUTPUT_DIR): shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)
dirname = f"{os.path.dirname(__file__)}/.."
shutil.copy(__file__, os.path.join(OUTPUT_DIR, os.path.basename(__file__)))
shutil.copy(f'{dirname}/engine/solver3d.py', os.path.join(OUTPUT_DIR, 'solver3d.py'))
with open(os.path.join(OUTPUT_DIR, "args.txt"), 'w') as f:
    for k, v in vars(args).items():
        f.write(f"{k}: {v}\n")

wp.init()
device = args.device

solver = MPMSolver(
    grid_size=args.grid_size if len(args.grid_size) > 1 else [args.grid_size[0]] * args.dim,
    dx=args.dx,
    device=device,
    gravity=-2.0,
    ppc=args.ppc,
    solver_type=args.method,
    n_psi=4,
    enable_apic=False,
)

mat_dict = {
    'snow': 0, 
    'snow_slope': 1, 
    'snow_ground': 2,
    'snowman': 3,
}
mat_color = {
    'snow': [200, 50, 50], 
    'snow_slope': [50, 50, 200],
    'snow_ground': [50, 50, 200],
    'snowman': [200, 50, 50], 
}

density_snow_slope = 1
density_snow_ground = 1.5
density_ball   = 3
density_snowman   = 1.2

X_offset = 0.0
Y_offset = 0.0
Z_offset = 0.0

if solver.sim_steps == 0:
    solver.add_material(
        material_k=mat_dict['snow'],
        material=Material.snow13,
        E=1000, nu=.15,
        hardening = 10.0,
        theta_c = .01,
        theta_s = .005,
    )

    solver.add_material(
        material_k=mat_dict['snow_slope'],
        material=Material.snow13,
        E=50, nu=.15,
        hardening = 10.0,
        theta_c = .01,
        theta_s = .005,
    )

    solver.add_material(
        material_k=mat_dict['snow_ground'],
        material=Material.snow13,
        E=3000, nu=.15,
        hardening = 10.0,
        theta_c = .015,
        theta_s = .001,
    )

    solver.add_material(
        material_k=mat_dict['snowman'],
        material=Material.snow13,
        E=1000, nu=.2,
        hardening = 10.0,
        theta_c = .01,
        theta_s = .005,
    )

    solver.add_mesh(
        surface_mesh="assets/ramp_slope.obj",
        offset=[0.0+X_offset, 0.0+Y_offset, 0.06+Z_offset],
        use_material_k=mat_dict['snow_slope'],
        density=density_snow_slope,
        velocity=[0.0, 0.0, 0.0],
        color=mat_color['snow_slope'],
    )

    solver.add_mesh(
        surface_mesh="assets/ramp_bottom.obj",
        offset=[0.0+X_offset, 0.0+Y_offset, 0.06+Z_offset],
        use_material_k=mat_dict['snow_ground'],
        density=density_snow_ground,
        velocity=[0.0, 0.0, 0.0],
        color=mat_color['snow_ground'],
    )

    solver.add_mesh(
        surface_mesh="assets/snowman.obj",
        offset=[0.0+X_offset, 0.2+Y_offset, 0.05 - 0.14 + Z_offset],
        use_material_k=mat_dict['snowman'],
        density=density_snowman,
        velocity=[0.0, 0.0, 0.0],
        color=mat_color['snowman'],
    )
    ic(solver.n_ptc)
    ptc = solver.get_points()
    ic(ptc.min(0), ptc.max(0))

################################
# Boundary 
################################
bc_mesh_path = os.path.join("./assets", "ramp.obj")
print(bc_mesh_path)
m = trimesh.load(bc_mesh_path, force='mesh')
m.vertices += np.array([X_offset, Y_offset, Z_offset]).reshape(1, 3)
vg = m.voxelized(pitch=solver.dx).fill()
mesh_bc_ijk = np.rint(vg.points / solver.dx).astype(np.int32)
g = np.array(solver.grid_size, dtype=np.int32)
mesh_bc_ijk = np.clip(mesh_bc_ijk, 0, g - 1)
bc_ijk = mesh_bc_ijk
bc_ijk = np.unique(bc_ijk.astype(np.int32), axis=0)
boundary = np.ones(len(bc_ijk), dtype=np.int32) * 1
solver.paint_boundary(bc_ijk, boundary)

################################
# Main 
################################

# simulation setup
SIM_DT = args.dt
BG = (17/255.0, 47/255.0, 65/255.0)

def simulate():
    global solver, args
    print("Time step:", solver.sim_steps)
    dt = SIM_DT
    if solver.n_ptc > 0:
        vmax = solver.max_particle_speed()
        if vmax > 1e-12:
            cfl = 0.5 * solver.dx / vmax
            if dt > cfl:
                print(f"\033[33m[CFL WARNING] step={solver.sim_steps}  dt={dt:.3e} > 0.5*dx/vmax={cfl:.3e} (dx={solver.dx:.3e}, vmax={vmax:.3e})\033[0m")
                dt = cfl
        else:
            pass

        solver.set_dt(dt)

        solver.step(**vars(args))

        if solver.sim_steps == 600:
            solver.add_mesh(
                surface_mesh="assets/snowball.obj",
                offset=[-0.4+X_offset, 0.2+Y_offset, 0.2 - 0.12+Z_offset],
                scale=1.5,
                use_material_k=mat_dict['snow'],
                density=density_ball,
                velocity=[0.0, 0.0, 0.0],
                color=mat_color['snow'],
                ppc_scale=2.0,
            )

viewer = ParticleViewer(
    host=args.viser_host,
    port=args.viser_port,
    max_points=args.viser_max_points,
    point_size=0.005,
    title="MPM Lite · Snow",
)

cnt = 0

while True:
    simulate()

    ic(solver.sim_steps)

    if solver.sim_steps % 200 == 0 and solver.sim_steps >= 600:
        if solver.n_ptc > 0:
            viewer.update(solver.get_points(), solver.ptc_color, step=solver.sim_steps)

        cnt += 1

        if cnt == 450: exit(0)
