"""Replay frozen template proposals and accepted nested-refinement dynamics."""
import argparse,json,socket,time
from pathlib import Path
import numpy as np
import viser


def edges(nodes,shape):
    p=nodes.reshape(*shape,3);lines=[]
    for d in range(3):
        lo=[slice(None)]*3;hi=lo.copy();lo[d]=slice(None,-1);hi[d]=slice(1,None)
        lines.append(np.stack((p[tuple(lo)].reshape(-1,3),p[tuple(hi)].reshape(-1,3)),axis=1))
    return np.concatenate(lines).astype(np.float32)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('docs/results/template-remap/v1'))
    p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=8088)
    p.add_argument('--duration',type=float,default=0.)
    a=p.parse_args()
    if not np.isfinite(a.duration) or a.duration<0:p.error('duration must be finite and nonnegative')
    if not (a.data/'results.json').exists():p.error('run benchmarks.aniso_template_remap first')
    results=json.loads((a.data/'results.json').read_text())['results']
    records={str(r['grid']):r for r in results};data={}
    for g in records:
        for kind in ('identity','refine_x','shift','coarsen_x','control','switched'):
            with np.load(a.data/f'g{g}-{kind}.npz') as z:data[(g,kind)]={k:z[k] for k in z.files}
    try:
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);probe.bind((a.host,a.port))
    except OSError as exc:p.error(f'cannot bind {a.host}:{a.port}: {exc}')
    server=viser.ViserServer(host=a.host,port=a.port);server.scene.set_up_direction('+z')
    server.gui.add_markdown('## 受控模板切换\n**独立材料 Q1 原型，生产默认未变。**\n\n'
        '静态页固定同一批粒子；橙色是新模板的重建提案。未通过时，该提案没有写回粒子。'
        '短程页只显示已通过的嵌套细分。')
    grid=server.gui.add_dropdown('源网格',options=list(records),initial_value='33' if '33' in records else next(iter(records)))
    view=server.gui.add_dropdown('内容',options=['静态切换提案','通过后的短程释放'])
    kinds={'嵌套细分 x':'refine_x','移动内部节点':'shift','沿 x 粗化':'coarsen_x','原模板重建':'identity'}
    kind=server.gui.add_dropdown('模板变化',options=list(kinds))
    scale=server.gui.add_slider('形状位移放大',min=1.,max=30.,step=1.,initial_value=10.)
    frame=server.gui.add_slider('释放帧',min=0,max=len(data[(grid.value,'control')]['positions'])-1,step=1,initial_value=0)
    show_old=server.gui.add_checkbox('显示旧模板',initial_value=True)
    show_new=server.gui.add_checkbox('显示新模板',initial_value=True)
    info=server.gui.add_markdown('')
    empty=np.zeros((1,3),np.float32)
    before=server.scene.add_point_cloud('/before',points=empty,colors=(45,140,235),point_size=.002)
    after=server.scene.add_point_cloud('/after',points=empty,colors=(235,155,40),point_size=.002)
    oldmesh=server.scene.add_line_segments('/old-template',points=np.zeros((1,2,3),np.float32),colors=(70,145,220),line_width=1.)
    newmesh=server.scene.add_line_segments('/new-template',points=np.zeros((1,2,3),np.float32),colors=(235,155,40),line_width=1.)
    @server.on_client_connect
    def connected(client):client.camera.position=(.9,-.25,.95);client.camera.look_at=(.5,.5,.5)
    start=time.monotonic();last=None
    try:
        while not a.duration or time.monotonic()-start<a.duration:
            key=(grid.value,view.value,kind.value,scale.value,frame.value,show_old.value,show_new.value)
            if key!=last:
                g=grid.value;r=records[g]
                if view.value=='静态切换提案':
                    k=kinds[kind.value];d=data[(g,k)];row=next(x for x in r['records'] if x['kind']==k)
                    X=d['reference'];before.points=(X+scale.value*(d['positions_before']-X)).astype(np.float32)
                    after.points=(X+scale.value*(d['proposed_positions']-X)).astype(np.float32)
                    oldmesh.points=edges(d['old_reference_nodes']+scale.value*(d['old_nodes']-d['old_reference_nodes']),d['old_shape'])
                    newmesh.points=edges(d['new_reference_nodes']+scale.value*(d['new_nodes']-d['new_reference_nodes']),d['new_shape'])
                    oldmesh.visible=show_old.value;newmesh.visible=show_new.value
                    info.content=(f"**{'兼容检查通过' if row['dynamic_accepted'] else '提案未通过，禁止提交'}**\n\n"
                        f"能量净变化 {100*row['remap_energy_relative']:+.5f}%；局部能量变化绝对值合计 {100*row['absolute_local_energy_change_relative']:.3f}%。\n\n"
                        f"应力相对差 {100*row['stress_relative']:.3f}%；最大模式增量差 {100*row['max_mode_increment_relative']:.3f}%。\n\n"
                        f"新模板 {row['nodes']} 节点；原粒子 {row['particles']} 个，位置、F、方向和质量均未写回修改。")
                else:
                    control=data[(g,'control')];switched=data[(g,'switched')];X=control['reference'];i=min(int(frame.value),len(control['positions'])-1)
                    before.points=(X+scale.value*(control['positions'][i]-X)).astype(np.float32)
                    after.points=(X+scale.value*(switched['positions'][i]-X)).astype(np.float32)
                    oldmesh.visible=False;newmesh.visible=False
                    info.content=(f"蓝色：不切换；橙色：第 1 步后嵌套细分。当前 t={control['times'][i]:.5f}。\n\n"
                        f"机械能：不切换 {control['energy'][i]:.7g}；切换 {switched['energy'][i]:.7g}。\n\n"
                        '切换瞬间没有改写粒子；后续轨迹可因空间加密和隐式耗散而不同。手动拖动帧滑条查看，显示放大不改变物理数据。')
                before.visible=show_old.value;after.visible=show_new.value;last=key
            time.sleep(.05)
    except KeyboardInterrupt:pass
    finally:server.stop()


if __name__=='__main__':main()
