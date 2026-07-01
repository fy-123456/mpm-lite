import numpy as np  # numpy for linear algebra
np.random.seed(0)
import warp as wp
import os, shutil
from icecream import ic
import time

wp.init()
device = 'cpu'
# device = 'cuda:0'

vec2 = wp.vec2
mat22 = wp.mat22
real = wp.float32
_05 = wp.constant(0.5)
_1 = wp.constant(1.0)
_0 = wp.constant(0.0)
_2 = wp.constant(2.0)
_n1 = wp.constant(-1.0)
s_min = wp.constant(1e-6)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output/mpmlite2d")
if os.path.isdir(OUTPUT_DIR): shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# simulation setup
FPS = 24
FRAME_DT = 1.0 / FPS
grid_size = 96 # background Eulerian grid's resolution, in 2D is [128, 128]
dx = 1.0 / grid_size # the domain size is [1m, 1m] in 2D, so dx for each cell is (1/128)m
dt = 1e-4  # time step size in second
ppc = 6 # average particles per cell
density = 400 # mass density, unit: kg / m^3
E, nu = 1e6, 0.3 # block's Young's modulus and Poisson's ratio
mu, lam = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu)) # Lame parameters
friction_angle_in_degrees = 35.0 # Drucker Prager friction angle
gravity = -9.8
GRAD_APIC = False


# uniformly sampling material particles
def uniform_grid(x0, y0, x1, y1, dx):
    xx, yy = np.meshgrid(
        np.arange(x0, x1 + dx, dx),
        np.arange(y0, y1 + dx, dx))
    return np.column_stack((xx.ravel(), yy.ravel()))


def poisson_disk_sampling(radius, domain_size, k=30):
    """Bridson's algorithm for Poisson-disk sampling in 2D"""
    cell_size = radius / np.sqrt(2)
    grid_shape = (
        int(domain_size[0] / cell_size) + 1,
        int(domain_size[1] / cell_size) + 1)
    grid = -np.ones(grid_shape, dtype=int)
    samples = []
    active_list = []

    def in_domain(p):
        return 0 <= p[0] < domain_size[0] and 0 <= p[1] < domain_size[1]

    def get_cell_coords(p):
        return int(p[0] / cell_size), int(p[1] / cell_size)

    def get_nearby_samples(p):
        i, j = get_cell_coords(p)
        neighbors = []
        for di in [-2, -1, 0, 1, 2]:
            for dj in [-2, -1, 0, 1, 2]:
                ni, nj = i + di, j + dj
                if 0 <= ni < grid.shape[0] and 0 <= nj < grid.shape[1]:
                    idx = grid[ni, nj]
                    if idx != -1:
                        neighbors.append(samples[idx])
        return neighbors

    # Start with a random point
    first_point = np.array([np.random.uniform(0, domain_size[0]), np.random.uniform(0, domain_size[1])])
    samples.append(first_point)
    active_list.append(0)
    grid[get_cell_coords(first_point)] = 0

    while active_list:
        idx = np.random.choice(active_list)
        base_point = samples[idx]
        found = False
        for _ in range(k):
            angle = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(radius, 2 * radius)
            new_point = base_point + r * np.array([np.cos(angle), np.sin(angle)])
            if in_domain(new_point):
                neighbors = get_nearby_samples(new_point)
                if all(np.linalg.norm(new_point - n) >= radius for n in neighbors):
                    samples.append(new_point)
                    active_list.append(len(samples) - 1)
                    grid[get_cell_coords(new_point)] = len(samples) - 1
                    found = True
        if not found:
            active_list.remove(idx)
    return np.array(samples)


# box sizes (width, height) in domain units
box1_size = np.array([0.2, 0.2], dtype=np.float64)
box2_size = np.array([0.2, 0.2], dtype=np.float64)
box1_offset = np.array([0.1, 0.30], dtype=np.float64)
box2_offset = np.array([0.4, 0.40], dtype=np.float64)
def place_box_samples(cell, size, offset):
    size = np.asarray(size, dtype=np.float64)
    offset = np.asarray(offset, dtype=np.float64)
    # safety: clamp so the box stays in [0,1]^2
    offset = np.clip(offset, 0.0, 1.0 - size)
    samp = poisson_disk_sampling(cell, size)  # returns points in [0,size]
    return samp + offset
box1_samples = place_box_samples(dx / np.sqrt(ppc), box1_size, box1_offset)
box2_samples = place_box_samples(dx / np.sqrt(ppc), box2_size, box2_offset)
box1_velocities = np.tile(np.array([ 2.0, 0.0], dtype=np.float64), (len(box1_samples), 1))
box2_velocities = np.tile(np.array([-2.0, 0.0], dtype=np.float64), (len(box2_samples), 1))  # <- fix
all_samples    = np.concatenate([box1_samples,    box2_samples   ], axis=0)
all_velocities = np.concatenate([box1_velocities, box2_velocities], axis=0)


# material particles data
N_particles = len(all_samples)
x = wp.from_numpy(all_samples, dtype=vec2, device=device)
v = wp.from_numpy(all_velocities, dtype=vec2, device=device)
vol = wp.zeros(N_particles, dtype=real, device=device)
vol0 = 0.2 * 0.4 / N_particles
vol.fill_(vol0)
m = wp.zeros(N_particles, dtype=real, device=device)
m.fill_(vol0 * density)
F = wp.from_numpy(np.tile(np.eye(2)*1.0, (N_particles, 1, 1)), dtype=mat22, device=device)

# Plastic deformation & particle grad v
particle_grad_v = wp.zeros(N_particles, dtype=mat22, device=device)
# Grid data
grid_m = wp.zeros((grid_size, grid_size), dtype=real, device=device)
grid_v = wp.zeros((grid_size, grid_size), dtype=vec2, device=device)
grid_v_new = wp.zeros((grid_size, grid_size), dtype=vec2, device=device)

# convergence (uses SEARCH direction pre-clip)
max_update = wp.zeros(1, dtype=real, device=device)

ptc_m = m
ptc_x = x
ptc_v = v
ptc_F = F
ptc_vol0 = vol
ptc_G = particle_grad_v
n_ptc = ptc_x.shape[0]

# cell center data
center_size = grid_size - 1
grid_shape = wp.vec2i(grid_size, grid_size)
center_shape = wp.vec2i(center_size, center_size)
center_m = wp.zeros((center_size, center_size), dtype=real, device=device)
center_v = wp.zeros((center_size, center_size), dtype=vec2, device=device)
center_dv = wp.zeros((center_size, center_size), dtype=vec2, device=device)
center_vol = wp.zeros((center_size, center_size), dtype=real, device=device)  # Σ V0 at centers
center_tau = wp.zeros((center_size, center_size), dtype=mat22, device=device)  # τ at centers (after normalization)
center_G = wp.zeros((center_size, center_size), dtype=mat22, device=device)  # NEW: ∇v at centers

@wp.func
def in_region(i: wp.int32, j: wp.int32, width: wp.int32, height: wp.int32) -> wp.int32:
    return wp.int32((i >= 0) and (i < width) and (j >= 0) and (j < height))

@wp.func
def is_bc(i: wp.int32, j: wp.int32, width: wp.int32, height: wp.int32) -> wp.int32:
    # returns 1 if on boundary band, else 0
    return wp.int32((i <= 5) or (i > width - 5) or (j <= 5) or (j > height - 5))

@wp.kernel
def max_particle_speed_kernel(
    v: wp.array(dtype=vec2),       # particle velocities
    vmax: wp.array(dtype=real), # length-1 array for result
):
    p = wp.tid()
    s = wp.length(v[p])
    wp.atomic_max(vmax, 0, s)

def max_particle_speed() -> float:
    # wrapper that behaves like your Taichi function
    vmax = wp.zeros(1, dtype=real, device=device)
    wp.launch(
        kernel=max_particle_speed_kernel,
        dim=v.shape[0],
        inputs=[v, vmax],
        device=device,
    )
    # TODO: synchronize?
    return float(vmax.numpy()[0])

# ANCHOR: reset_grid
def reset_grid():
    # after each transfer, the grid is reset
    # grid_m.fill_(0.0) # wiped in c2g
    # grid_v.fill_(vec2(0.0, 0.0)) # wiped in c2g
    grid_v_new.fill_(vec2(0.0, 0.0))

    center_m.zero_()
    center_v.zero_()
    center_dv.zero_()
    center_G.zero_()
    center_vol.zero_()
    center_tau.zero_()


################################
# Stvk Hencky Elasticity
################################
# ANCHOR: stvk
@wp.func
def StVK_Hencky_PK1_2D(F: wp.mat22) -> wp.mat22:
    # F = U diag(sigma) V^T
    U, sigma, V = wp.svd2(F)         # sigma: wp.vec2
    sig0 = wp.max(sigma[0], s_min)
    sig1 = wp.max(sigma[1], s_min)
    inv_sig = wp.diag(wp.vec2(1.0 / sig0, 1.0 / sig1))
    e = wp.diag(wp.vec2(wp.log(sig0), wp.log(sig1)))
    M = 2.0 * mu * (inv_sig @ e) + lam * (e[0, 0] + e[1, 1]) * inv_sig
    Pk1 = U @ M @ wp.transpose(V)
    return Pk1

@wp.func
def StVK_Hencky_tau_2D(F: wp.mat22) -> wp.mat22:
    Pk1 = StVK_Hencky_PK1_2D(F)
    tau = Pk1 @ wp.transpose(F)   # Kirchhoff stress
    return tau


@wp.kernel
def lite_p2c_kernel_1(
    ptc_x: wp.array(dtype=vec2),
    ptc_v: wp.array(dtype=vec2),
    ptc_F: wp.array(dtype=mat22), 
    ptc_vol0: wp.array(dtype=real),
    ptc_m: wp.array(dtype=real),
    ptc_G: wp.array(dtype=mat22),

    center_m: wp.array(dtype=real, ndim=2),
    center_v: wp.array(dtype=vec2, ndim=2),
    center_G: wp.array(dtype=mat22, ndim=2),
    center_vol: wp.array(dtype=real, ndim=2),
    center_tau: wp.array(dtype=mat22, ndim=2),

    center_size: wp.vec2i,
    dx: real,
):
    p = wp.tid()
    Fp = ptc_F[p]
    xp = ptc_x[p]

    pos_c = xp / dx - vec2(_05)
    base = wp.vec2i(int(wp.floor(pos_c[0])), int(wp.floor(pos_c[1])))
    fx = pos_c - vec2(real(base.x), real(base.y))
    # for speed up, linear kernel also works
    wx0 = _1 - fx[0]; wx1 = fx[0]
    wy0 = _1 - fx[1]; wy1 = fx[1]
    Vp = ptc_vol0[p]
    tau_p = StVK_Hencky_tau_2D(Fp)

    V0_tau = Vp * tau_p
    Gp = ptc_G[p]

    for i in range(2):
        for j in range(2):
            center = base + wp.vec2i(i, j)
            ci, cj = center.x, center.y
            if not in_region(ci, cj, center_size[0], center_size[1]): continue
            w = (wx0 if i == 0 else wx1) * (wy0 if j == 0 else wy1)
            dm = w * ptc_m[p]
            vel = ptc_v[p]

            wp.atomic_add(center_m, ci, cj, dm)
            wp.atomic_add(center_G, ci, cj, dm * Gp)
            if GRAD_APIC:
                xc = dx * vec2(real(center[0]) + _05, real(center[1]) + _05)
                vel = vel + Gp @ (xc - xp)
            wp.atomic_add(center_v, ci, cj, dm * vel)
            wp.atomic_add(center_tau, ci, cj, w * V0_tau)
            wp.atomic_add(center_vol, ci, cj, w * Vp)


@wp.kernel
def lite_p2c_kernel_2(
    center_m: wp.array(dtype=real, ndim=2),
    center_vol: wp.array(dtype=real, ndim=2),
    center_tau: wp.array(dtype=mat22, ndim=2),
    center_v: wp.array(dtype=vec2, ndim=2),
    center_G: wp.array(dtype=mat22, ndim=2),
):
    ci, cj = wp.tid()

    if center_m[ci, cj] <= _0: return
    invm = _1 / center_m[ci, cj]
    center_v[ci, cj] *= invm
    center_G[ci, cj] *= invm

    if center_vol[ci, cj] > _0:
        center_tau[ci, cj] *= _1 / center_vol[ci, cj]


def lite_p2c(
    ptc_x: wp.array(dtype=vec2),
    ptc_v: wp.array(dtype=vec2),
    ptc_F: wp.array(dtype=mat22), 
    ptc_vol0: wp.array(dtype=real),
    ptc_m: wp.array(dtype=real),
    ptc_G: wp.array(dtype=mat22),

    center_m: wp.array(dtype=real, ndim=2),
    center_v: wp.array(dtype=vec2, ndim=2),
    center_G: wp.array(dtype=mat22, ndim=2),
    center_vol: wp.array(dtype=real, ndim=2),
    center_tau: wp.array(dtype=mat22, ndim=2),

    center_size: wp.vec2i,
    dx: real,
    device: str,
):
    wp.launch(
        kernel=lite_p2c_kernel_1,
        dim=ptc_m.shape,
        inputs=[
            ptc_x, ptc_v, ptc_F, ptc_vol0, ptc_m, ptc_G,
            center_m, center_v, center_G, center_vol, center_tau,
            center_size, dx,
        ],
        device=device,
    )
    wp.launch(
        kernel=lite_p2c_kernel_2,
        dim=center_vol.shape,
        inputs=[
            center_m, center_vol, center_tau, center_v, center_G,
        ],
        device=device,
    )


# 1) scatter from centers to grid (PIC + optional GRAD_APIC)
@wp.kernel
def lite_c2g_kernel_1(
    center_m: wp.array(dtype=real, ndim=2),       # [center_size, center_size]
    center_v: wp.array(dtype=vec2, ndim=2),
    center_G: wp.array(dtype=mat22, ndim=2),
    center_vol: wp.array(dtype=real, ndim=2),
    center_tau: wp.array(dtype=mat22, ndim=2),
    grid_m: wp.array(dtype=real, ndim=2),         # [grid_size, grid_size]
    grid_v: wp.array(dtype=vec2, ndim=2),
    grid_v_new: wp.array(dtype=vec2, ndim=2),
    grid_size: wp.vec2i,  
    center_size: wp.vec2i,  
    gravity: real,       # y-direction gravity
    dx: real,
    dt: real,
):
    i, j = wp.tid()

    w = real(0.25)
    xi = vec2(real(i), real(j)) * dx
    grid_m[i, j] = _0
    grid_v[i, j] = vec2(_0)
    impulse = vec2(_0)
    inv_2dx =  _05 / dx
    for di in range(2):
        for dj in range(2):
            ci = i - di; cj = j - dj
            center = wp.vec2i(ci, cj)
            if not in_region(ci, cj, center_size[0], center_size[1]): continue

            cm = center_m[ci, cj]
            if cm <= _0: continue
            mom = cm * center_v[ci, cj]
            grid_m[i, j] += w * cm
            grid_v[i, j] += w * mom
            if GRAD_APIC:
                xc = vec2(real(ci) + _05, real(cj) + _05) * dx
                grid_v[i, j] += w * cm * center_G[ci, cj] @ (xi - xc)

            vol_c = center_vol[ci, cj]
            if vol_c <= _0: continue
            sx = _n1 if di == 0 else _1
            sy = _n1 if dj == 0 else _1
            grad_w = inv_2dx * vec2(sx, sy) 
            fi = -vol_c * (center_tau[ci, cj] @ grad_w) # (equals -V0 * tau_c ∇w)
            impulse += dt * fi

    mi = grid_m[i, j]
    if mi > _0:
        grid_v[i, j] = grid_v[i, j] / mi
        # apply boundary condition
        is_bc_cond = is_bc(i, j, grid_size[0], grid_size[1]) > 0
        if is_bc_cond:
            grid_v[i, j] = vec2(_0)
            grid_v_new[i, j] = vec2(_0)
        else:
            grid_v_new[i, j] = grid_v[i, j] + impulse / mi + vec2(_0, dt * gravity)
    else:
        grid_v[i, j] = vec2(_0)
        grid_v_new[i, j] = vec2(_0)


def lite_c2g(
    center_m: wp.array(dtype=real, ndim=2),       # [center_size, center_size]
    center_v: wp.array(dtype=vec2, ndim=2),
    center_G: wp.array(dtype=mat22, ndim=2),
    center_vol: wp.array(dtype=real, ndim=2),
    center_tau: wp.array(dtype=mat22, ndim=2),
    grid_m: wp.array(dtype=real, ndim=2),         # [grid_size, grid_size]
    grid_v: wp.array(dtype=vec2, ndim=2),
    grid_v_new: wp.array(dtype=vec2, ndim=2),
    grid_size: wp.vec2i,
    center_size: wp.vec2i,
    gravity: real,       # y-direction gravity
    dx: real,
    dt: real,
    device: str,
):
    wp.launch(
        kernel=lite_c2g_kernel_1,
        dim=grid_m.shape,
        inputs=[
            center_m, center_v, center_G, center_vol, center_tau,
            grid_m, grid_v, grid_v_new, grid_size, center_size,
            gravity, dx, dt,
        ],
        device=device,
    )


@wp.kernel
def lite_g2c_kernel(
    grid_v      : wp.array(dtype=vec2, ndim=2),
    grid_v_new  : wp.array(dtype=vec2, ndim=2),
    center_v    : wp.array(dtype=vec2, ndim=2),
    center_dv   : wp.array(dtype=vec2, ndim=2),
    center_G    : wp.array(dtype=mat22, ndim=2),
    dx          : real,
):
    ci, cj = wp.tid()

    v_c  = vec2(_0)
    dv_c = vec2(_0)
    grad_v = mat22(_0)

    inv_2dx = _05 / dx

    w = real(0.25)
    for di in range(2):
        for dj in range(2):
            node = wp.vec2i(ci + di, cj + dj)
            i, j = node.x, node.y

            v_node = grid_v_new[i, j]
            v_c  = v_c  + w * v_node
            dv_c = dv_c + w * (v_node - grid_v[i, j])
            sx = _n1 if di == 0 else _1
            sy = _n1 if dj == 0 else _1
            grad_w = inv_2dx * vec2(sx, sy) # ∇w at cell center
            grad_v += wp.outer(v_node, grad_w) # Σ v ⊗ ∇w  == ∇v at center

    center_v[ci, cj] = v_c
    center_dv[ci, cj] = dv_c
    center_G[ci, cj] = grad_v


@wp.kernel
def lite_c2p_kernel(
    ptc_x      : wp.array(dtype=vec2),
    ptc_v      : wp.array(dtype=vec2),
    ptc_F      : wp.array(dtype=mat22),
    ptc_G      : wp.array(dtype=mat22),

    center_m   : wp.array(dtype=real,  ndim=2),
    center_v   : wp.array(dtype=vec2,  ndim=2),
    center_dv  : wp.array(dtype=vec2,  ndim=2),
    center_G   : wp.array(dtype=mat22, ndim=2),

    center_size  : wp.vec2i,
    dx           : real,
    dt           : real,
    flip_ratio   : real,
):
    p = wp.tid()
    pos_c = ptc_x[p] / dx - vec2(_05)
    base = wp.vec2i(int(wp.floor(pos_c[0])), int(wp.floor(pos_c[1])))
    fx = pos_c - vec2(real(base.x), real(base.y))

    wx0 = _1 - fx[0]; wx1 = fx[0]
    wy0 = _1 - fx[1]; wy1 = fx[1]

    v_pic = vec2(_0)
    dv_acc = vec2(_0)
    Gc = mat22(_0)

    for i in range(2):
        for j in range(2):
            center = base + wp.vec2i(i, j)
            if not in_region(center.x, center.y, center_size[0], center_size[1]): continue
            ci, cj = center.x, center.y

            w = (wx0 if i == 0 else wx1) * (wy0 if j == 0 else wy1)

            if center_m[ci, cj] <= _0: continue

            v_pic += w * center_v[ci, cj]
            dv_acc += w * center_dv[ci, cj]
            Gc += w * center_G[ci, cj]
    
    # FLIP + PIC blend
    v_flip = ptc_v[p] + dv_acc
    ptc_v[p] = flip_ratio * v_flip + (_1 - flip_ratio) * v_pic
    ptc_G[p] = Gc

    ptc_F[p] = (wp.identity(2, real) + dt * Gc) @ ptc_F[p]

    # advect particle using PIC velocity
    ptc_x[p] += dt * v_pic
    
def step(
    flip_ratio=0.9,
):
    reset_grid()
    lite_p2c(
        ptc_x, ptc_v, ptc_F, ptc_vol0, ptc_m, ptc_G,
        center_m, center_v, center_G, center_vol, center_tau,
        center_shape, dx,
        device,
    )
    lite_c2g(
        center_m, center_v, center_G, center_vol, center_tau,
        grid_m, grid_v, grid_v_new,
        grid_shape, center_shape,
        gravity, dx, dt,
        device,
    )
    wp.launch(
        kernel=lite_g2c_kernel,
        dim=center_m.shape,
        inputs=[grid_v, grid_v_new, center_v, center_dv, center_G, dx],
        device=device,
    )
    wp.launch(
        kernel=lite_c2p_kernel,
        dim=n_ptc,
        inputs=[
            ptc_x, ptc_v, ptc_F, ptc_G,
            center_m, center_v, center_dv, center_G,
            center_shape, dx, dt, flip_ratio,
        ],
        device=device,
    )

################################
# Main 
################################
time_step = 1
sim_time = 0.0
frame_id = 0
next_frame_t = FRAME_DT  # first dump at 1/24 s

def simulate():
    global time_step, sim_time, frame_id, next_frame_t
    last = time.time()
    vmax = max_particle_speed()
    if vmax > 1e-12:
        cfl = 0.6 * dx / vmax
        if dt > cfl:
            print(f"[CFL WARNING] step={time_step}  dt={dt:.3e} > 0.6*dx/vmax={cfl:.3e}  "
                  f"(dx={dx:.3e}, vmax={vmax:.3e})")
    else:
        print(f"[CFL INFO] step={time_step}  vmax≈0 → no CFL restriction")

    # print("Time step:", time_step)
    step(0.9)

    # advance simulated time
    sim_time += dt

    time_step += 1

def run_app():
    from vispy import app
    app.use_app('glfw')
    from vispy import scene
    import imageio

    canvas = scene.SceneCanvas(
        title="2D MPM",
        keys='interactive', show=True, bgcolor='white', size=(512, 512))
    view = canvas.central_widget.add_view()
    view.camera = 'panzoom'

    scatter = scene.visuals.Markers()
    scatter.set_gl_state(depth_test=False, blend=False)
    view.add(scatter)

    last_time = time.time()
    ui_frame_cnt = 0

    def update(ev):
        nonlocal last_time, ui_frame_cnt
        global sim_time, next_frame_t, frame_id, OUTPUT_DIR
        simulate()
        pts = x.numpy().copy()[:len(all_samples)]
        scatter.set_data(pts, size=1.5, face_color=(1, 0, 0, 1), edge_color=None, edge_width_rel=0)
        ui_frame_cnt += 1
        now = time.time()
        fps = 1.0 / (now - last_time)
        last_time = now
        canvas.title = f"2D MPM (FPS: {fps:.2f})"

        # while sim_time + 1e-12 >= next_frame_t:
        #     out_path = os.path.join(OUTPUT_DIR, f"{frame_id:05d}.png")
        #     img = canvas.render()
        #     imageio.imwrite(out_path, img)
        #     frame_id += 1
        #     next_frame_t += FRAME_DT

    timer = app.Timer(interval=0, connect=update, start=True)
    app.run()
    return timer


run_app()

