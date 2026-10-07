"""Read-only Viser comparison of the same static beam in Q1 and MLS spaces."""
import argparse
import json
from pathlib import Path
import socket
import time

import numpy as np
import viser


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('docs/results/mechanics-comparison/v1'))
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8087)
    p.add_argument('--duration', type=float, default=0.)
    args=p.parse_args()
    if not np.isfinite(args.duration) or args.duration<0:p.error('duration must be finite and nonnegative')
    if not (args.data/'results.json').exists():p.error('run benchmarks.aniso_mechanics_comparison first')
    results=json.loads((args.data/'results.json').read_text())
    records={r['name']:r for r in results['records']}
    grids=sorted({r['grid'] for r in records.values() if f"q1-g{r['grid']}" in records and f"mls-g{r['grid']}" in records})
    if not grids:p.error('completed Q1 and MLS cases at the same grid are required')
    arrays={}
    for g in grids:
        for kind in ('q1','mls'):
            name=f'{kind}-g{g}'
            with np.load(args.data/(name+'.npz')) as a:arrays[name]={k:a[k] for k in a.files}
            if 'mode_displacements' not in arrays[name]:p.error('rerun benchmark to export common-mode arrays')
    try:
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            probe.bind((args.host,args.port))
    except OSError as exc:p.error(f'cannot bind {args.host}:{args.port}: {exc}')
    server=viser.ViserServer(host=args.host,port=args.port)
    server.scene.set_up_direction('+z')
    server.gui.add_markdown('## 同载荷梁：Q1、MLS 与细参考\n'
        '**小应变静态对照，不推进生产 MPM。**\n\n'
        '蓝色：一致 Q1 空间；橙色：准确积分 MLS；绿色：细网格 Q1 参考。'
        '灰色：未变形梁。位移放大仅用于显示。')
    grid=server.gui.add_dropdown('候选网格',options=[str(g) for g in grids],initial_value=str(grids[0]))
    modes={'伸长模式':'stretch','剪切模式':'shear','弯曲 y 模式':'bend_y','弯曲 z 模式':'bend_z','扭转模式':'twist'}
    mode=server.gui.add_dropdown('比较内容',options=['相同总载荷','相同端面平均位移',*modes],initial_value='相同总载荷')
    scale=server.gui.add_slider('位移显示放大',min=1.,max=200.,step=1.,initial_value=50.)
    show_q1=server.gui.add_checkbox('显示 Q1',initial_value=True)
    show_mls=server.gui.add_checkbox('显示 MLS',initial_value=True)
    show_ref=server.gui.add_checkbox('显示参考',initial_value=True)
    info=server.gui.add_markdown('')
    first=arrays[f'q1-g{grids[0]}'];X=first['reference']
    def cloud(name,color,size=.003):
        return server.scene.add_point_cloud(name,points=X.astype(np.float32),colors=color,point_size=size)
    q1=cloud('/q1',(45,140,235));mls=cloud('/mls',(240,150,40));ref=cloud('/reference',(55,190,105))
    rest=cloud('/undeformed',(145,145,145),.0015)
    @server.on_client_connect
    def connect(client):
        client.camera.position=(.9,-.25,.95)
        client.camera.look_at=(.5,.48,.5)
    start=time.monotonic();last=None
    try:
        while not args.duration or time.monotonic()-start<args.duration:
            key=(grid.value,mode.value,scale.value,show_q1.value,show_mls.value,show_ref.value)
            if key!=last:
                g=int(grid.value);a=arrays[f'q1-g{g}'];b=arrays[f'mls-g{g}']
                ra=records[f'q1-g{g}'];rb=records[f'mls-g{g}'];rr=results['reference'];X=a['reference']
                if mode.value in modes:
                    label=modes[mode.value];j=list(ra['mode_audit']['modes']).index(label)
                    exact=a['exact_modes'][:,:,j]
                    factor=.001*scale.value/max(np.linalg.norm(exact,axis=1).max(),1e-30)
                    ua=a['mode_displacements'][:,:,j]*factor;ub=b['mode_displacements'][:,:,j]*factor;ur=exact*factor
                    info.content=(f'共同物理模式 `{label}`。绿色在此显示解析模式，而非梁加载解。\n\n'
                        f"刚度比：Q1 {ra['mode_audit']['modes'][label]['stiffness_ratio']:.6f}；"
                        f"MLS {rb['mode_audit']['modes'][label]['stiffness_ratio']:.6f}。\n\n"
                        '模式以同一幅值归一化展示；这是五个受控模式之一，不是全空间最软模式或新的动态模拟。')
                else:
                    ua=a['displacement'].copy();ub=b['displacement'].copy();ur=a['reference_displacement'].copy()
                    if mode.value=='相同端面平均位移':
                        target=rr['target_mean_displacement']
                        ua*=target/ra['tip_displacement'];ub*=target/rb['tip_displacement'];ur*=target/rr['tip_displacement']
                    ua*=scale.value;ub*=scale.value;ur*=scale.value
                    info.content=(f"参考 grid {rr['grid']}；进一步加密后端面位移变化 {100*abs(results['reference_refinement_relative']):.3f}%。\n\n"
                        '| 指标 | Q1 | MLS |\n|---|---:|---:|\n'
                        f"| 相同载荷平均位移 | {ra['tip_displacement']:.7g} | {rb['tip_displacement']:.7g} |\n"
                        f"| 相对细参考位移差 | {100*ra['tip_relative_to_reference']:+.2f}% | {100*rb['tip_relative_to_reference']:+.2f}% |\n"
                        f"| 同平均位移支反力 | {ra['support_reaction_at_target_mean']:.7g} | {rb['support_reaction_at_target_mean']:.7g} |\n\n"
                        '同平均位移约束允许端面翘曲，不是刚性端夹具。整体位移接近不等于局部应力已收敛。')
                q1.points=(X+ua).astype(np.float32);mls.points=(X+ub).astype(np.float32);ref.points=(X+ur).astype(np.float32)
                rest.points=X.astype(np.float32)
                q1.visible=show_q1.value;mls.visible=show_mls.value;ref.visible=show_ref.value
                last=key
            time.sleep(.05)
    except KeyboardInterrupt:pass
    finally:server.stop()


if __name__=='__main__':main()
