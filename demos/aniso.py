"""Interactive Viser scenes for fixed-reference transverse-isotropic elasticity.

Run from the repository root: PYTHONPATH=. .venv/bin/python demos/aniso.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import json
import os
from pathlib import Path
from queue import Empty, SimpleQueue
import time

import numpy as np
import warp as wp

from engine.aniso_phase1 import (
    AnisotropicLiteImplicitSolver, AnisotropicMaterialParams,
    energy, pk1, select_lowest_memory_device,
)
from utils.resource_guard import StoragePaused, prepare_warp_cache


SCENES = {"fixed": "固定边界纤维块体", "affine": "均匀三维仿射场", "material": "材料点方向响应", "tensile": "速度控制拉伸与卸载", "beam": "预弯曲梁释放（小变形）"}
DATA_ROOT = "/mnt/0c18569c-b839-4255-bae0-6f48c9fc835b/yin/tmp"


@dataclass(frozen=True)
class Config:
    scene: str = "fixed"
    grid: int = 8
    dt: float = 0.001
    fiber_angle: float = 0.0
    kf: float = 200.0
    flip_ratio: float = .9
    cg_tol: float = 1e-4
    v_tol: float = 1e-10
    loading_speed: float = .01
    loading_time: float = .5
    density: float = 1.
    samples: int = 2
    residual_atol: float = 1e-10
    force_discretization: str = "variational"
    linear_solver: str | None = None
    history_mode: str = "particle_resample"
    loading_cycles: int = 1
    smooth_loading: bool = False
    direction_model: str = "mean_tensor"
    stabilization: str = "none"
    stabilization_strength: float = 1.
    fiber_field: str = "uniform"
    quadrature: str = "center"
    reaction_force_atol: float | None = None
    boundary_impulse_transfer: bool = False
    apic_transfer: str = "overwrite"
    history_consistency: str = "standard"
    affine_flip_ratio: float | None = None
    velocity_dissipation: str = "none"

    @property
    def params(self):
        angle = np.deg2rad(self.fiber_angle)
        return AnisotropicMaterialParams(10.0, 20.0, self.kf, [np.cos(angle), np.sin(angle), 0.0])


def lattice(values):
    return np.stack(np.meshgrid(values, values, values, indexing="ij"), axis=-1).reshape(-1, 3)


class Scene:
    """Simulation ownership stays on the main thread, including reset and CUDA."""

    def __init__(self, config: Config, device: str):
        self.config, self.device = config, device
        self.solver = None
        self.boundary = np.empty((0, 3))
        self.reference = lattice(np.linspace(0.2, 0.56, 4))
        self.material_F = np.eye(3)
        self.loading_rows = []
        self.loading_work = 0.
        self.fiber_directions=config.params.fiber_direction
        if config.scene == "material":
            return
        dx = 1.0 / (config.grid - 1)
        if config.scene == "beam":
            lo,hi=np.array([.25,.4375,.4375]),np.array([.75,.5625,.5625])
            counts=np.maximum(1,np.rint((hi-lo)/dx*config.samples).astype(int))
            axes=[lo[d]+(np.arange(counts[d])+.5)*(hi[d]-lo[d])/counts[d] for d in range(3)]
            self.reference=np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1).reshape(-1,3)
            gradient=np.zeros((3,3));nodes=lattice(np.arange(config.grid)).astype(np.int32)
            boundary=nodes[nodes[:,0]*dx<=.25];self.boundary=boundary*dx
        elif config.scene == "tensile":
            if (config.grid-1) % 4:
                raise ValueError("拉伸网格必须让 .25/.75 夹具位置对齐节点，请选择 9 或 17")
            lo, hi = np.array([.125, .375, .375]), np.array([.875, .625, .625])
            counts = np.maximum(1, np.rint((hi-lo)/dx*config.samples).astype(int))
            axes = [lo[d]+(np.arange(counts[d])+.5)*(hi[d]-lo[d])/counts[d] for d in range(3)]
            self.reference = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
            gradient = np.zeros((3, 3))
            nodes = lattice(np.arange(config.grid)).astype(np.int32)
            boundary = nodes[(nodes[:, 0]*dx <= .25) | (nodes[:, 0]*dx >= .75)]
            self.boundary_indices = boundary
            self.boundary = boundary*dx
        elif config.scene == "affine":
            self.reference = lattice((np.arange(1, config.grid - 2) + 0.5) * dx)
            gradient = np.array([[0.08, 0.02, 0.01], [0., -0.03, 0.015], [0., 0., 0.04]])
        else:
            # Match the validated fixed-block probe at grid=8. At finer grids,
            # position the clamp immediately to the left of the same block.
            gradient = np.diag([0.08, 0., 0.])
            plane = max(1, int(np.floor(0.2 / dx)))
            boundary = np.array([[plane, j, k] for j in range(1, config.grid - 1)
                                 for k in range(1, config.grid - 1)], dtype=np.int32)
            self.boundary = boundary * dx
        solver_cls=AnisotropicLiteImplicitSolver
        if config.quadrature=='particle':
            from engine.aniso_phase1.particle_quadrature import ParticleQuadratureImplicitSolver
            solver_cls=ParticleQuadratureImplicitSolver
        elif config.quadrature=='group4x8':
            from engine.aniso_phase1.grouped_quadrature import GroupedQuadratureImplicitSolver
            solver_cls=GroupedQuadratureImplicitSolver
        elif config.quadrature!='center':
            raise ValueError('unknown quadrature')
        if config.history_consistency == "projected_center":
            if config.quadrature != "center":
                raise ValueError("projected history requires center quadrature")
            from engine.aniso_phase1.projected_history import ProjectedHistoryLiteSolver
            solver_cls = ProjectedHistoryLiteSolver
        elif config.history_consistency == "residual_center":
            if config.quadrature != "center":
                raise ValueError("residual history retains the center transfer path")
            from engine.aniso_phase1.residual_history import ResidualHistoryLiteSolver
            solver_cls = ResidualHistoryLiteSolver
            if config.stabilization == "selective_patch":
                from engine.aniso_phase1.selective_patch import SelectiveHistoryLiteSolver
                solver_cls = SelectiveHistoryLiteSolver
            elif config.stabilization == "material_patch":
                from engine.aniso_phase1.material_patch import MaterialPatchLiteSolver
                solver_cls = MaterialPatchLiteSolver
            elif config.stabilization == "compatible_patch":
                from engine.aniso_phase1.compatible_patch import CompatiblePatchLiteSolver
                solver_cls = CompatiblePatchLiteSolver
        elif config.history_consistency != "standard":
            raise ValueError("unknown history consistency mode")
        self.solver = solver_cls(
            (config.grid,) * 3, params=config.params, dx=dx, device=device, gravity=0.0, ppc=1,
            flip_ratio=config.flip_ratio, energy_diagnostics=True,
            force_discretization=config.force_discretization, history_mode=config.history_mode,
            direction_model=config.direction_model, stabilization=config.stabilization,
            stabilization_strength=config.stabilization_strength,
            boundary_impulse_transfer=config.boundary_impulse_transfer, apic_transfer=config.apic_transfer,
            affine_flip_ratio=config.affine_flip_ratio, velocity_dissipation=config.velocity_dissipation,
        )
        volume = (.5*.125*.125 if config.scene=='beam' else .75*.25*.25)/len(self.reference) if config.scene in ('tensile','beam') else 1e-3
        positions=self.reference.copy();seed_options={}
        if config.scene=='beam':
            from engine.aniso_phase1.stabilization_probe import bent_beam_state
            positions,F=bent_beam_state(self.reference)
            seed_options.update(deformation_gradient=F,reference_positions=self.reference)
        if config.fiber_field!='uniform':
            angles=np.where(np.arange(len(positions))%2,90.,0.) if config.fiber_field=='crossed' else 90*(self.reference[:,0]-self.reference[:,0].min())/np.ptp(self.reference[:,0])
            angles=np.deg2rad(angles);seed_options['fiber_directions']=np.stack([np.cos(angles),np.sin(angles),np.zeros_like(angles)],axis=1)
            self.fiber_directions=seed_options['fiber_directions']
        self.solver.seed_particles(positions, density=config.density if config.scene in ('tensile','beam') else 1000., vol0=volume,
                                   velocity=self.reference @ gradient.T, velocity_gradient=gradient,**seed_options)
        if len(self.boundary):
            self.solver.paint_boundary(boundary, np.ones(len(boundary), dtype=np.int32))
        self.solver.set_dt(config.dt)
        self.solver.energy_ledger.begin(self.solver)

    def step(self):
        if self.solver is None:
            return True
        if self.config.scene == "tensile":
            from engine.aniso_phase1.tensile import loading_displacement
            t = self.solver.sim_time
            disp0 = loading_displacement(t, self.config.loading_speed, self.config.loading_time, self.config.loading_cycles, self.config.smooth_loading)
            disp1 = loading_displacement(t+self.config.dt, self.config.loading_speed, self.config.loading_time, self.config.loading_cycles, self.config.smooth_loading)
            velocity = (disp1-disp0)/self.config.dt
            from engine.aniso_phase1.tensile import set_grip_velocity
            set_grip_velocity(self.solver, self.boundary_indices, velocity)
        points = self.solver.ptc_x.numpy()
        predicted = points + self.config.dt * self.solver.ptc_v.numpy()
        margin = self.solver.dx * .25
        if not np.isfinite(predicted).all() or np.any(predicted < margin) or np.any(predicted > 1 - margin):
            raise RuntimeError("粒子接近计算域边界，已暂停。请重置场景。")
        solve_options = {} if self.config.linear_solver is None else {"linear_solver": self.config.linear_solver}
        if self.config.reaction_force_atol is not None:
            solve_options["reaction_force_atol"] = self.config.reaction_force_atol
        success = bool(self.solver.step(max_iters=16, print_every=0, v_tol=self.config.v_tol,
                                    cg_tol=self.config.cg_tol, cg_atol=1e-12, newton_atol=self.config.residual_atol,
                                    max_cg_iters=max(100, self.config.grid ** 3 * 3), **solve_options))
        if success and self.config.scene == "tensile":
            from engine.aniso_phase1.tensile import grip_reactions
            reaction = grip_reactions(self.solver)
            self.loading_work += reaction['right_force']*(disp1-disp0)
            moved = self.solver.ptc_x.numpy()-self.reference
            measured = moved[self.reference[:, 0] >= .75, 0].mean()-moved[self.reference[:, 0] <= .25, 0].mean()
            row = dict(time=self.solver.sim_time, displacement=disp1, measured_grip_displacement=float(measured),
                       loading_velocity=velocity, loading_work=self.loading_work,
                       effective_stiffness=reaction['right_force']/disp1 if abs(disp1)>1e-14 else 0.,
                       mechanical_minus_loading_work=self.solver.energy_ledger.rows[-1]["cumulative_delta"]-self.loading_work,
                       **reaction)
            self.loading_rows.append(row)
            self.solver.energy_ledger.rows[-1].update(row)
        return success

    def frame(self, strain=0.0, shear=0.0):
        if self.solver is None:
            self.material_F = np.array([[1.0 + strain, shear, 0.], [0., 1., 0.], [0., 0., 1.]])
            center = self.reference.mean(axis=0)
            points = (self.reference - center) @ self.material_F.T + center
            F = np.broadcast_to(self.material_F, (len(points), 3, 3))
        else:
            points, F = self.solver.ptc_x.numpy(), self.solver.ptc_F.numpy()
        return points, F

    def metrics(self):
        if self.solver is None:
            p = self.config.params
            return {"scene": "material", "det_F": float(np.linalg.det(self.material_F)),
                    "energy_density": float(energy(self.material_F, p.A0, p)),
                    "P11": float(pk1(self.material_F, p.A0, p)[0, 0])}
        s = self.solver
        # Copy only allocated active blocks, not the entire 64-block reserve.
        count = int(s.bcn) if s.sim_steps else 0
        min_det = 1.0
        if self.config.quadrature in ('particle','group4x8'):
            min_det = float(np.linalg.det(s.ptc_F.numpy()).min())
        elif count:
            valid = s.aniso_state_valid[:, :count].numpy()[0] > 0
            F = s.aniso_committed_F[:, :count].numpy()[0][valid]
            if len(F):
                min_det = float(np.min(np.linalg.det(F)))
        return {"scene": self.config.scene, "device": self.device, "steps": s.sim_steps,
                "quadrature": self.config.quadrature, "flip_ratio": self.config.flip_ratio,
                "sim_time": float(s.sim_time), "min_det_F": min_det,
                "max_displacement": float(np.linalg.norm(s.ptc_x.numpy() - self.reference, axis=1).max()),
                "particles": len(self.reference), **s.last_step_stats,
                **(s.energy_ledger.rows[-1] if s.energy_ledger.rows else {})}


def display_geometry(reference, points, F, a0, magnification, fiber_length=0.035):
    """Magnification affects positions only; fiber direction uses physical F a0."""
    shown = reference + magnification * (points - reference)
    directions = np.einsum("pij,j->pi" if np.ndim(a0)==1 else "pij,pj->pi", F, a0)
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-15)
    stride = max(1, int(np.ceil(len(points) / 400)))
    half = directions[::stride] * (fiber_length / 2)
    segments = np.stack((shown[::stride] - half, shown[::stride] + half), axis=1)
    displacement = np.linalg.norm(points - reference, axis=1)
    t = np.clip(displacement / 0.01, 0., 1.)[:, None]
    colors = ((1 - t) * np.array([50, 145, 245]) + t * np.array([245, 100, 50])).astype(np.uint8)
    return shown.astype(np.float32), segments.astype(np.float32), colors


def directional_curves(kf):
    strain = np.linspace(0, 0.2, 81)
    curves = []
    for angle in (0., 45., 90.):
        params = Config(kf=kf, fiber_angle=angle).params
        curves.append([pk1(np.diag([1 + e, 1., 1.]), params.A0, params)[0, 0] for e in strain])
    return strain, np.asarray(curves)


def curve_image(kf):
    # Pillow is already a Viser dependency. No browser plotting add-on needed.
    from PIL import Image, ImageDraw
    strain, curves = directional_curves(kf)
    img = Image.new("RGB", (680, 340), "#17212b")
    draw = ImageDraw.Draw(img)
    left, top, right, bottom = 55, 45, 650, 292
    ymax = max(float(curves.max()), 1e-10)
    draw.text((left, 12), "Uniaxial imposed F: P11 vs engineering strain (mu=10, lambda=20)", fill="white")
    for fraction in np.linspace(0, 1, 5):
        y = bottom - fraction * (bottom - top)
        draw.line([(left, y), (right, y)], fill="#394858")
        draw.text((4, y - 5), f"{ymax * fraction:.1f}", fill="white")
        x = left + fraction * (right - left)
        draw.text((x - 10, bottom + 8), f"{fraction * .2:.2f}", fill="white")
    for index, (curve, color) in enumerate(zip(curves, ("#ffad47", "#6ed6a5", "#69aaf8"))):
        xy = [(left + e / .2 * (right - left), bottom - v / ymax * (bottom - top))
              for e, v in zip(strain, curve)]
        draw.line(xy, fill=color, width=3)
        draw.text((left + index * 150, 320), f"fiber {index * 45} deg", fill=color)
    return np.asarray(img)


def energy_image(rows):
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (680, 340), "#17212b")
    draw = ImageDraw.Draw(img)
    draw.text((45, 10), "Energy vs physical time (s)", fill="white")
    if not rows:
        return np.asarray(img)
    times = np.array([r['time'] for r in rows])
    curves = np.array([[r[k] for r in rows] for k in ('kinetic', 'elastic', 'mechanical')])
    ymax = max(curves.max(), 1e-15)
    xmax = max(times.max(), 1e-15)
    for frac in np.linspace(0, 1, 5):
        y = 285-frac*240
        draw.line([(65, y), (650, y)], fill="#394858")
        draw.text((0, y-5), f"{ymax*frac:.3g}", fill="white")
        draw.text((65+frac*585, 290), f"{xmax*frac:.3g}", fill="white")
    for j, (name, color) in enumerate(zip(('K (APIC)', 'U (center)', 'E = K + U'), ('#69aaf8', '#6ed6a5', '#ffad47'))):
        xy = list(zip(65+times/xmax*585, 285-curves[j]/ymax*240))
        if len(xy) > 1:
            draw.line(xy, fill=color, width=3)
        draw.text((65+j*190, 320), name, fill=color)
    return np.asarray(img)


def loading_image(rows):
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (680, 340), "#17212b")
    draw = ImageDraw.Draw(img)
    draw.text((45, 10), "Right grip force vs commanded displacement (loading / unloading)", fill="white")
    if len(rows) < 2:
        return np.asarray(img)
    x = np.array([r['displacement'] for r in rows])
    y = np.array([r['right_force'] for r in rows])
    ymin, ymax = min(0., y.min()), max(0., y.max())
    span = max(ymax-ymin, 1e-15)
    for f in np.linspace(0, 1, 5):
        draw.text((0, 285-f*240), f"{ymin+f*span:.3g}", fill="white")
        draw.text((65+f*580, 300), f"{f*x.max():.3g}", fill="white")
    points = np.stack((65+x/max(x.max(), 1e-15)*580, 285-(y-ymin)/span*240), axis=1)
    for i in range(1, len(points)):
        draw.line([tuple(points[i-1]), tuple(points[i])], fill="#ffad47" if rows[i]['loading_velocity']>0 else "#69aaf8", width=2)
    return np.asarray(img)


class Demo:
    def __init__(self, args, device):
        import viser
        self.args, self.device = args, device
        self.commands = SimpleQueue()
        self.scene = None
        self.error = ""
        self.server = viser.ViserServer(host=args.viser_host, port=args.viser_port, label="MPM Lite · 各向异性弹性")
        self.server.scene.set_up_direction("+z")
        self.server.scene.add_box("/domain", dimensions=(1., 1., 1.), position=(.5, .5, .5),
                                  color=(100, 120, 140), wireframe=True)
        self.cloud = self.server.scene.add_point_cloud("/particles", points=np.zeros((1, 3)),
                                                       colors=(50, 145, 245), point_size=.017)
        self.ghost = self.server.scene.add_point_cloud("/reference", points=np.zeros((1, 3)),
                                                       colors=(115, 125, 135), point_size=.009)
        self.fibers = self.server.scene.add_line_segments("/fibers", points=np.zeros((1, 2, 3)),
                                                          colors=(255, 220, 80), thickness=.003)
        self.clamp = self.server.scene.add_point_cloud("/clamp", points=np.zeros((1, 3)),
                                                       colors=(230, 180, 90), point_size=.022)

        @self.server.on_client_connect
        def connect(client):
            client.camera.position = (1.65, -1.65, 1.25)
            client.camera.look_at = (.45, .45, .45)

        gui = self.server.gui
        gui.add_markdown("**固定参考方向 · 横向各向同性弹性**\n\n灰点：初始位置；黄线：当前纤维方向；金色点：固定网格节点。")
        self.preset = gui.add_dropdown("场景", options=tuple(SCENES.values()), initial_value=SCENES[args.scene])
        self.play = gui.add_checkbox("播放", initial_value=args.play)
        self.single = gui.add_button("单步")
        self.reset = gui.add_button("应用参数并重置")
        self.batch = gui.add_slider("每次显示的仿真步数", min=1, max=20, step=1, initial_value=1)
        with gui.add_folder("材料与仿真参数（重置后生效）"):
            self.angle = gui.add_slider("参考纤维角度 XY (°)", min=0., max=180., step=5., initial_value=args.fiber_angle)
            self.kf = gui.add_slider("纤维刚度 k_f", min=0., max=500., step=10., initial_value=args.kf)
            self.grid = gui.add_dropdown("网格边长", options=("8", "9", "12", "16", "17", "24"), initial_value=str(args.grid))
            self.dt = gui.add_number("时间步 (s)", initial_value=args.dt, min=1e-5, max=.005, step=.0001)
            gui.add_markdown("μ = 10，λ = 20。方向和刚度改变后点击重置；k_f = 0 为各向同性对照。")
        with gui.add_folder("拉伸加载"):
            self.load_speed = gui.add_number("加载速度", initial_value=args.loading_speed, min=0.0001, max=.1, step=.001)
            self.load_time = gui.add_number("加载时长 (随后等时卸载)", initial_value=args.loading_time, min=.01, max=10., step=.1)
            gui.add_markdown("选择 9 或 17 网格。夹具位于 x≤.25 与 x≥.75；密度 1，固定物理体积。反力包含惯性，CSV 同时记录弹性项。")
        with gui.add_folder("显示"):
            self.scale = gui.add_slider("显示位移放大（不改变仿真）", min=1., max=100., step=1., initial_value=20.)
            self.show_fibers = gui.add_checkbox("显示纤维", initial_value=True)
            self.show_reference = gui.add_checkbox("显示初始位置", initial_value=True)
            gui.add_markdown("颜色：真实位移 0（蓝）→ 0.01 及以上（橙）。黄线用真实 F·a₀ 归一化；放大仅作用于位置。粒子 F 是中心状态传输后的显示量。")
        self.loading = gui.add_folder("材料点：指定均匀变形", visible=args.scene == "material")
        with self.loading:
            self.strain = gui.add_slider("工程应变", min=0., max=.2, step=.005, initial_value=.1)
            self.shear = gui.add_slider("简单剪切 γ", min=0., max=.3, step=.01, initial_value=0.)
            self.unload = gui.add_button("卸载到参考状态")
            gui.add_markdown("此模式直接指定 F，展示本构响应；不进行 MPM 时间积分。")
        with gui.add_folder("方向响应对照", expand_by_default=False):
            self.plot = gui.add_image(curve_image(args.kf), label="0° / 45° / 90° 同应变应力")
            gui.add_markdown("三条曲线共享当前已应用的 k_f；侧向伸长固定为 1，采用第一 Piola 应力 P₁₁。")
        with gui.add_folder("机械能诊断"):
            self.energy_plot = gui.add_image(energy_image([]), label="APIC kinetic / elastic / mechanical")
            self.force_plot = gui.add_image(loading_image([]), label="Reaction vs displacement")
            self.export = gui.add_button("下载能量及加载 CSV")
            self.export.on_click(lambda event: self.commands.put(("export", event.client)))
            gui.add_markdown("动能包含两层 APIC 仿射表示；CSV 分列记录传输、边界投影、状态重建与求解变化。诊断会同步设备，不用于性能计时。")
        self.status = gui.add_markdown("正在初始化…")
        self.single.on_click(lambda _: self.commands.put("step"))
        self.reset.on_click(lambda _: self.commands.put("reset"))
        self.preset.on_update(lambda _: self.commands.put("reset"))
        self.unload.on_click(lambda _: self.commands.put("unload"))
        self._view_key = None
        self._last_storage_check = 0.
        self.rebuild()

    def rebuild(self):
        self.play.value = False
        self.error = ""
        key = next(k for k, label in SCENES.items() if label == self.preset.value)
        if key == "tensile" and int(self.grid.value) not in (9, 17):
            self.grid.value = "9"
        config = Config(key, int(self.grid.value), float(self.dt.value), float(self.angle.value), float(self.kf.value),
                        loading_speed=float(self.load_speed.value), loading_time=float(self.load_time.value),
                        history_mode=self.args.history_mode, loading_cycles=self.args.loading_cycles, smooth_loading=self.args.smooth_loading,
                        direction_model=self.args.direction_model,stabilization=self.args.stabilization,stabilization_strength=self.args.stabilization_strength,fiber_field=self.args.fiber_field,
                        quadrature=self.args.quadrature,flip_ratio=self.args.flip_ratio,linear_solver=self.args.linear_solver,reaction_force_atol=self.args.reaction_force_atol,boundary_impulse_transfer=self.args.boundary_impulse_transfer,apic_transfer=self.args.apic_transfer,history_consistency=self.args.history_consistency,affine_flip_ratio=self.args.affine_flip_ratio,velocity_dissipation=self.args.velocity_dissipation)
        self.status.content = "正在重置场景…"
        # Release the previous GPU allocations before constructing a new solver.
        self.scene = None
        gc.collect()
        self.scene = Scene(config, self.device)
        self.loading.visible = key == "material"
        self.play.disabled = self.single.disabled = key == "material"
        self.batch.disabled = key == "material"
        self.force_plot.visible = key == "tensile"
        self.scale.value = 1. if key == "material" else 20.
        self.ghost.points = self.scene.reference.astype(np.float32)
        self.clamp.visible = len(self.scene.boundary) > 0
        if len(self.scene.boundary):
            self.clamp.points = self.scene.boundary.astype(np.float32)
        self.plot.image = curve_image(config.kf)
        self.render()

    def guard_storage(self):
        if time.monotonic() - self._last_storage_check > 10:
            self.args.cache = prepare_warp_cache(self.args.cache, self.args.data_root)
            wp.config.kernel_cache_dir = self.args.cache
            self._last_storage_check = time.monotonic()

    def advance(self):
        self.guard_storage()
        if self.scene is None:
            raise RuntimeError("场景未初始化，请应用参数并重置")
        if not self.scene.step():
            self.play.value = False
            self.error = "本步未收敛，已回滚并暂停。可减小时间步后重置。"
            return False
        return True

    def render(self):
        if self.scene is None:
            raise RuntimeError("场景未初始化，请应用参数并重置")
        points, F = self.scene.frame(float(self.strain.value), float(self.shear.value))
        if not np.isfinite(points).all() or not np.isfinite(F).all():
            raise RuntimeError("场景出现非有限值，已暂停")
        shown, segments, colors = display_geometry(self.scene.reference, points, F,
                                                   self.scene.fiber_directions, self.scale.value)
        with self.server.atomic():
            self.cloud.points, self.cloud.colors = shown, colors
            self.fibers.points = segments
            self.fibers.visible = self.show_fibers.value
            self.ghost.visible = self.show_reference.value
        self.force_plot.image = loading_image(self.scene.loading_rows)
        if self.scene.solver is not None:
            self.energy_plot.image = energy_image(self.scene.solver.energy_ledger.rows)
        else:
            self.energy_plot.image = energy_image([])
        metrics = self.scene.metrics()
        if self.scene.solver is None:
            lines = [f"指定 F 的 det：`{metrics['det_F']:.6f}`",
                     f"能量密度：`{metrics['energy_density']:.6g}`；P₁₁：`{metrics['P11']:.6g}`"]
        else:
            lines = [f"设备：`{self.device}`；粒子数：`{metrics['particles']}`",
                     f"步数：`{metrics['steps']}`；仿真时间：`{metrics['sim_time']:.4f} s`",
                     f"真实最大位移：`{metrics['max_displacement']:.6g}`；最小 det(F)：`{metrics['min_det_F']:.6g}`",
                     f"K / U / E：`{metrics.get('kinetic', 0):.6g} / {metrics.get('elastic', 0):.6g} / {metrics.get('mechanical', 0):.6g}`；累计 ΔE：`{metrics.get('cumulative_delta', 0):.3g}`",
                     f"Newton / 线性迭代：`{metrics.get('newton_iterations', 0)} / {metrics.get('cg_iterations', 0)}`"]
        if self.scene.loading_rows:
            row = self.scene.loading_rows[-1]
            lines += [f"加载位移：`{row['displacement']:.5g}`；反力：`{row['right_force']:.5g}`；加载功：`{row['loading_work']:.5g}`"]
        p = self.scene.config
        if p.quadrature=='group4x8':
            lines += [f"实验分组积分：`{metrics.get('group_samples', 0)}` 个采样；步间重建 ΔU：`{metrics.get('group_rebuild_delta', 0):.3g}`"]
        direction_label=f"统一参考角度 {p.fiber_angle:g}°" if p.fiber_field=='uniform' else p.fiber_field
        lines += [f"已应用参数：方向场 `{direction_label}`，k_f `{p.kf:g}`",
                  f"方向表示：`{metrics.get('direction_model',p.direction_model)}`；稳定化：`{p.stabilization}`（系数 {p.stabilization_strength:g}）",
                  f"积分：`{p.quadrature}`；FLIP 比例：`{p.flip_ratio:g}`",
                  f"显示位移放大：`{self.scale.value:g}×`", self.error]
        self.status.content = "  \n".join(lines)

    def run(self):
        self.play.value = self.args.play and self.scene.solver is not None
        started = time.monotonic()
        try:
            while not self.args.duration or time.monotonic() - started < self.args.duration:
                changed = False
                try:
                    self.guard_storage()
                    while True:
                        try:
                            action = self.commands.get_nowait()
                        except Empty:
                            break
                        if isinstance(action, tuple) and action[0] == "export":
                            if self.scene.solver is not None and action[1] is not None:
                                action[1].send_file_download("energy.csv", self.scene.solver.energy_ledger.csv().encode("utf-8"))
                        elif action == "reset":
                            self.rebuild()
                        elif action == "unload":
                            self.strain.value = self.shear.value = 0.
                        elif action == "step":
                            self.play.value = False
                            self.advance()
                        changed = True
                    if self.play.value:
                        for _ in range(int(self.batch.value)):
                            if not self.advance():
                                break
                        changed = True
                    key = (self.scale.value, self.show_fibers.value, self.show_reference.value,
                           self.strain.value, self.shear.value)
                    if changed or key != self._view_key:
                        self.render()
                        self._view_key = key
                except (StoragePaused, RuntimeError, ValueError) as exc:
                    self.play.value = False
                    self.error = str(exc)
                    self.status.content = f"**已暂停**：{self.error}"
                time.sleep(.03)
        finally:
            self.server.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=tuple(SCENES), default="fixed")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--csv", type=Path, help="export headless energy history")
    parser.add_argument("--grid", type=int, choices=(8, 9, 12, 16, 17, 24), default=8)
    parser.add_argument("--dt", type=float, default=.001)
    parser.add_argument("--fiber-angle", type=float, default=0.)
    parser.add_argument("--kf", type=float, default=200.)
    parser.add_argument("--loading-speed", type=float, default=.01)
    parser.add_argument("--loading-time", type=float, default=.5)
    parser.add_argument("--loading-cycles", type=int, default=1)
    parser.add_argument("--smooth-loading", action="store_true")
    parser.add_argument("--history-mode", choices=("particle_resample","grid_locked"), default="particle_resample")
    parser.add_argument("--direction-model",choices=("mean_tensor","fourth_moment"),default="mean_tensor")
    parser.add_argument("--stabilization",choices=("none","supplemental","hourglass","corotated","quadratic","material_quadratic","selective_patch","material_patch","compatible_patch"),default="none")
    parser.add_argument("--stabilization-strength",type=float,default=1.)
    parser.add_argument("--fiber-field",choices=("uniform","crossed","smooth"),default="uniform")
    parser.add_argument("--quadrature",choices=("center","particle","group4x8"),default="center",
                        help="group4x8 is experimental frozen grouped material integration")
    parser.add_argument("--velocity-dissipation",choices=("none","null","weak"),default="none",
                        help="CPU residual_center prototype: affine-protected joint v/C kinetic dissipation")
    parser.add_argument("--flip-ratio",type=float,default=.9)
    parser.add_argument("--affine-flip-ratio",type=float,default=None,
                        help="independent incremental APIC affine return; 1 retains affine history; omitted shares flip-ratio")
    parser.add_argument("--linear-solver",choices=("pcg","pcg_projected"),default=None,
                        help="guarded PCG (default), or positive-tangent PCG from the start")
    parser.add_argument("--apic-transfer", choices=("overwrite","incremental"), default="overwrite",
                        help="APIC coefficient return; incremental requires boundary impulse transfer")
    parser.add_argument("--history-consistency", choices=("standard","projected_center","residual_center"), default="standard",
                        help="experimental center material map consistent with frozen particle updates")
    parser.add_argument("--boundary-impulse-transfer", action="store_true",
                        help="include initial boundary projection impulse in FLIP transfer (center quadrature)")
    parser.add_argument("--reaction-force-atol", type=float, default=None,
                        help="optional resultant force residual bound (N); scales stopping with dt")
    parser.add_argument("--viser-host", default="127.0.0.1")
    parser.add_argument("--viser-port", type=int, default=8080)
    parser.add_argument("--play", action="store_true", help="start playback immediately")
    parser.add_argument("--headless", action="store_true", help="run finite steps without a Viser server")
    parser.add_argument("--steps", type=int, default=20, help="number of headless steps")
    parser.add_argument("--duration", type=float, default=0., help="stop the Viser server after N seconds; 0 runs until Ctrl+C")
    parser.add_argument("--cache", default=os.environ.get("MPM_LITE_WARP_CACHE", "/tmp/mpm-lite-warp-cache"))
    parser.add_argument("--data-root", default=os.environ.get("MPM_LITE_DATA_ROOT", DATA_ROOT if Path(DATA_ROOT).is_dir() else None))
    args = parser.parse_args()
    if not (1e-5 <= args.dt <= .005 and 0 <= args.fiber_angle <= 180 and 0 <= args.kf <= 500):
        parser.error("require dt in [1e-5, .005], fiber-angle in [0, 180], kf in [0, 500]")
    if args.reaction_force_atol is not None and (not np.isfinite(args.reaction_force_atol) or args.reaction_force_atol <= 0):
        parser.error("reaction-force-atol must be finite and positive")
    if args.loading_cycles < 1 or args.loading_time <= 0:
        parser.error("loading-cycles and loading-time must be positive")
    if not np.isfinite(args.flip_ratio) or not 0 <= args.flip_ratio <= 1:
        parser.error("flip-ratio must be in [0, 1]")
    if args.affine_flip_ratio is not None and (args.apic_transfer != "incremental" or
            not np.isfinite(args.affine_flip_ratio) or not 0 <= args.affine_flip_ratio <= 1):
        parser.error("affine-flip-ratio requires incremental APIC and a value in [0, 1]")
    if args.quadrature == 'particle' and args.direction_model != 'mean_tensor':
        parser.error("particle quadrature uses individual fibers; omit --direction-model fourth_moment")
    if args.steps < 1 or args.duration < 0 or not np.isfinite(args.duration):
        parser.error("steps must be positive and duration must be finite and non-negative")
    try:
        args.cache = prepare_warp_cache(args.cache, args.data_root)
        wp.config.kernel_cache_dir = args.cache
        wp.init()
        device = select_lowest_memory_device(args.device)
        if args.device == "auto" and not wp.is_cuda_available():
            device = "cpu"
        if args.headless:
            scene = Scene(Config(args.scene, args.grid, args.dt, args.fiber_angle, args.kf,
                                 loading_speed=args.loading_speed, loading_time=args.loading_time,
                                 history_mode=args.history_mode, loading_cycles=args.loading_cycles, smooth_loading=args.smooth_loading,
                                 direction_model=args.direction_model,stabilization=args.stabilization,stabilization_strength=args.stabilization_strength,fiber_field=args.fiber_field,
                                 quadrature=args.quadrature,flip_ratio=args.flip_ratio,linear_solver=args.linear_solver,reaction_force_atol=args.reaction_force_atol,boundary_impulse_transfer=args.boundary_impulse_transfer,apic_transfer=args.apic_transfer,history_consistency=args.history_consistency,affine_flip_ratio=args.affine_flip_ratio,velocity_dissipation=args.velocity_dissipation), device)
            for _ in range(args.steps if scene.solver is not None else 1):
                args.cache = prepare_warp_cache(args.cache, args.data_root)
                wp.config.kernel_cache_dir = args.cache
                if not scene.step():
                    raise RuntimeError("step failed and rolled back")
                points, F = scene.frame(strain=.1)
                if not np.isfinite(points).all() or not np.isfinite(F).all():
                    raise RuntimeError("non-finite frame")
            if args.csv and scene.solver is not None:
                args.csv.parent.mkdir(parents=True, exist_ok=True)
                args.csv.write_text(scene.solver.energy_ledger.csv())
            print(json.dumps(scene.metrics(), sort_keys=True))
        else:
            Demo(args, device).run()
    except KeyboardInterrupt:
        pass
    except (StoragePaused, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
