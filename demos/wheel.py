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
from utils.viser_utils import ParticleViewer

parser = ArgumentParser()
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--method', type=str, default='lite_implicit')
parser.add_argument('--dt', type=float, default=1e-3)
parser.add_argument('--dx', type=float, default=0.005)
parser.add_argument('--v_tol', type=float, default=1e-4)
parser.add_argument('--max_iters', type=int, default=10)
parser.add_argument('--out', type=str, default='output/wheel')
parser.add_argument('--gui', default=False, action='store_true',
                    help='deprecated; browser visualization is always enabled')
parser.add_argument('--viser-host', type=str, default='127.0.0.1')
parser.add_argument('--viser-port', type=int, default=8080)
parser.add_argument('--viser-max-points', type=int, default=250000)
parser.add_argument('--grid_size', type=int, nargs='+', default=[355])
parser.add_argument('--dim', type=int, default=3)
parser.add_argument('--ppc', type=float, default=24)
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
    gravity=-9.81,
    ppc=args.ppc,
    solver_type=args.method,
    n_psi=1,
)

mat_dict = {'metal': 0}
mat_color = {'metal': [237, 85, 59]}

density = 1000
E = 1e8
nu = 0.3
mu = E / (2 * (1 + nu))

if solver.sim_steps == 0:
    solver.add_material(
        material_k=mat_dict['metal'],
        material=Material.vonmises,
        E=E, nu=nu, yield_stress=0.005 * mu
    )

    solver.add_mesh(
        surface_mesh=os.path.join("./assets", "wheel.obj"),
        offset=[0.5,0.5,0.02],
        use_material_k=mat_dict['metal'],
        density=density,
        velocity=[0.0, 0.0, -0.0],
        color=mat_color['metal'],
        scale=1.0,
    )
    ic(solver.n_ptc)

################################
# Boundary Test 
################################
i = np.linspace(0, solver.grid_size[0]-1, solver.grid_size[0])
j = np.linspace(0, solver.grid_size[1]-1, solver.grid_size[1])
k = np.linspace(0, solver.grid_size[2]-1, solver.grid_size[2])
ii, jj, kk = np.meshgrid(i, j, k, indexing='ij')
bc_ijk = np.stack((ii, jj, kk),axis=-1).astype(np.int32).reshape(-1, 3)
bc_pts = bc_ijk * solver.dx
boundary_mask_x = np.logical_or(bc_pts[...,0] <= (5 * solver.dx), bc_pts[...,0] > (solver.grid_size[0] - 5)*solver.dx)
boundary_mask_y = np.logical_or(bc_pts[...,1] <= (5 * solver.dx), bc_pts[...,1] > (solver.grid_size[1] - 5)*solver.dx)
boundary_mask_z = np.logical_or(bc_pts[...,2] <= (5 * solver.dx), bc_pts[...,2] > (solver.grid_size[2] - 5)*solver.dx)
boundary_mask = np.logical_or(np.logical_or(boundary_mask_x, boundary_mask_y), boundary_mask_z)
mark_bc_ids = np.where(boundary_mask)
bc_ijk = bc_ijk[mark_bc_ids]
boundary = np.ones(len(bc_ijk), dtype=np.int32) * 1  # 1 for sticky, 2 for slippy
solver.paint_boundary(bc_ijk, boundary)

################################
# Main 
################################

# simulation setup
SIM_DT = args.dt
print(SIM_DT)
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

        # presser
        vel = 0.5
        if solver.sim_time < 0.6:
            solver.paint_hf_boundary(
                hf_bc_p=np.array([
                    [0.5, 0.5, 0.51 - vel * solver.sim_time],
                    [0.5, 0.5, 0.03],
                ]), 
                hf_bc_n=np.array([
                    [0.0, 0.0, -1.0],
                    [0.0, 0.0, 1.0],
                ]), 
                hf_bc_v=np.array([
                    [0.0, 0.0, -vel],
                    [0.0, 0.0, 0.0],
                ]),
                hf_bc_type=np.array([
                    1,
                    1
                ]),
            )
        else:
            solver.paint_hf_boundary(
                hf_bc_p=np.array([
                    [0.5, 0.5, 0.51 - vel * 0.6],
                    [0.5, 0.5, 0.03],
                ]), 
                hf_bc_n=np.array([
                    [0.0, 0.0, -1.0],
                    [0.0, 0.0, 1.0],
                ]), 
                hf_bc_v=np.array([
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ]),
                hf_bc_type=np.array([
                    1,
                    1
                ]),
            )

        solver.step(**vars(args))

viewer = ParticleViewer(
    host=args.viser_host,
    port=args.viser_port,
    max_points=args.viser_max_points,
    point_size=0.005,
    title="MPM Lite · Wheel",
)

cnt = 0

while True:
    simulate()

    ic(solver.sim_steps)

    if solver.sim_steps % 50 == 0:
        if solver.n_ptc > 0:
            viewer.update(solver.get_points(), solver.ptc_color, step=solver.sim_steps)

        cnt += 1

        if cnt == 500: exit(0)
