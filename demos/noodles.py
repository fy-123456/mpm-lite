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
import imageio, trimesh

parser = ArgumentParser()
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--method', type=str, default='lite_implicit')
parser.add_argument('--dt', type=float, default=1e-3)
parser.add_argument('--dx', type=float, default=0.008)
parser.add_argument('--v_tol', type=float, default=1e-4)
parser.add_argument('--max_iters', type=int, default=50)
parser.add_argument('--out', type=str, default='output/noodles')
parser.add_argument('--gui', default=False, action='store_true')
parser.add_argument('--grid_size', type=int, default=355)
parser.add_argument('--dim', type=int, default=3)
parser.add_argument('--ppc', type=float, default=16)
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
    grid_size=[args.grid_size] * args.dim,
    dx=args.dx,
    device=device,
    gravity=-9.81,
    ppc=args.ppc,
    solver_type=args.method,
    n_psi=1,
)

mat_dict = {'default': 0}
mat_color = {'default': [237, 85, 59]}

density = 1000
E = 5e6
nu = 0.3
mu = E / (2 * (1 + nu))

UP_OFFSET = 0.5

if solver.sim_steps == 0:
    solver.add_material(
        material_k=mat_dict['default'],
        material=Material.vonmises,
        E=E, nu=nu, yield_stress=0.005*mu,
    )

    solver.add_cylinder(
        center=[0.5,0.5,0.75 + UP_OFFSET],
        radius=0.245,
        axis=[0.0,0.0,1.0],
        axis_length=0.49,
        use_material_k=mat_dict['default'],
        density=density,
        velocity=[0.0, 0.0, -0.0],
        color=mat_color['default'],
    )
    ic(solver.n_ptc)

################################
# Boundary 
################################
bc_mesh_path = os.path.join("./assets", "sieve.obj")
m = trimesh.load(bc_mesh_path, force='mesh')
m.vertices[:, [1, 2]] = m.vertices[:, [2, 1]]
m.vertices[:, 2] += UP_OFFSET
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
                # dt = cfl
        else:
            pass

        solver.set_dt(dt)

        # presser
        vel = (0.4375 / 3)
        lower_bound = 1.0 - vel * 3.0 + UP_OFFSET
        if 1.0 - vel * solver.sim_time + UP_OFFSET > lower_bound:
            solver.paint_hf_boundary(
                hf_bc_p=np.array([
                    [0.5, 0.5, 1.0 - solver.sim_time * vel + UP_OFFSET],
                    [0.5, 0.5, 4 * solver.dx],
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
            ic(1.0 - vel * solver.sim_time + UP_OFFSET)
        else:
            solver.paint_hf_boundary(
                hf_bc_p=np.array([
                    [0.5, 0.5, lower_bound],
                    [0.5, 0.5, 4 * solver.dx],
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
            ic(1.0 - vel * solver.sim_time + UP_OFFSET)

        solver.step(**vars(args))


last_time = time.time()
ui_frame_cnt = 0

canvas = None
from vispy import app
app.use_app('glfw' if args.gui else 'egl')
from vispy import scene
w = 512
h = 512
canvas = scene.SceneCanvas(
    title="3D MPM",
    keys='interactive', show=args.gui, bgcolor=BG, size=(w, h))
view = canvas.central_widget.add_view()
view.camera = scene.cameras.TurntableCamera(
    fov=45,
    azimuth=0,
    elevation=0,
    distance=2.2,
    center=(0.5,0.5,0.5)
)
scatter = scene.visuals.Markers(
    antialias=0.0,
    scaling=True,
    spherical=True,
)
view.add(scatter)

last_eqframe = -1

t0 = None
cnt = 0

while True:
    simulate()

    ui_frame_cnt += 1
    now = time.time()
    fps = 1.0 / (now - last_time)
    last_time = now

    if solver.sim_steps % 10 == 0:
        if canvas and solver.n_ptc > 0:
            pts = solver.get_points()
            scatter.set_data(
                pos=pts,
                size=0.005,
                face_color=solver.ptc_color,
                edge_color=solver.ptc_color,
                edge_width=0,
            )

        if canvas and app:
            canvas.update()
            app.process_events()

        out_path = os.path.join(OUTPUT_DIR, f"{cnt:05d}.png")
        img = canvas.render()
        imageio.imwrite(out_path, img)

        cnt += 1

        if cnt == 200: exit(0)