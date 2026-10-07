"""Read-only quadrature comparison: static shape and weakest stiffness mode."""
import argparse,json,time
from pathlib import Path
import numpy as np
import viser


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,nargs='+',default=[Path('docs/results/quadrature-validation')/d for d in ('aligned-v1','aligned-v1-fine','adaptive-v1','particle-refinement-v1')])
    p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=8085);p.add_argument('--duration',type=float,default=0.)
    args=p.parse_args()
    if not np.isfinite(args.duration) or args.duration<0:p.error('duration must be finite and nonnegative')
    records={};paths={}
    for folder in args.data:
        source=folder/'results.json'
        if not source.exists():continue
        for result in json.loads(source.read_text()):
            for r in result['records']:
                if r['name'] not in records:records[r['name']]=r;paths[r['name']]=folder
    if not records:p.error('no completed static result files; run benchmarks.aniso_quadrature_validation first')
    server=viser.ViserServer(host=args.host,port=args.port);server.scene.set_up_direction('+z')
    server.gui.add_markdown('## 材料积分：形状与最差刚度模式\n**材料积分参考；直接生产接入未通过传递/历史兼容检查。**\n静态诊断，不推进生产模拟。蓝色为所选积分，绿色为同一 MLS 的稠密积分，橙色为独立全积分 Q1。\n\n最差模式表示相对稠密参考漏算最多刚度的方向，不是实际加载下的变形。')
    case=server.gui.add_dropdown('积分方案',options=list(records),initial_value='g17-gauss3' if 'g17-gauss3' in records else next(iter(records)))
    mode=server.gui.add_dropdown('显示内容',options=['梁加载形状','最差刚度模式'])
    scale=server.gui.add_slider('加载位移放大',min=1.,max=100.,step=1.,initial_value=50.)
    info=server.gui.add_markdown('')
    def cloud(name,color):return server.scene.add_point_cloud(name,points=np.zeros((1,3),np.float32),colors=color,point_size=.004)
    body=cloud('/candidate',(55,145,235));dense=cloud('/dense',(65,185,90));q1=cloud('/q1',(235,160,50));rest=cloud('/rest',(160,160,160))
    @server.on_client_connect
    def connected(client):client.camera.position=(.9,-.4,1.0);client.camera.look_at=(.48,.48,.48)
    last=None;start=time.monotonic()
    try:
        while not args.duration or time.monotonic()-start<args.duration:
            key=(case.value,mode.value,scale.value)
            if key!=last:
                r=records[case.value];folder=paths[case.value];is_mode=mode.value=='最差刚度模式' or r['tip_displacement'] is None
                if is_mode:
                    with np.load(folder/(case.value+'-mode.npz')) as a:
                        X=a['reference'];u=a['displacement'];amplitude=max(np.linalg.norm(u,axis=1).max(),1e-20)
                        body.points=(X+.02*u/amplitude).astype(np.float32);rest.points=X.astype(np.float32)
                    dense.visible=False;q1.visible=False
                else:
                    with np.load(folder/(case.value+'.npz')) as a:
                        X=a['reference'];body.points=(X+scale.value*a['displacement_mls']).astype(np.float32)
                        dense.points=(X+scale.value*a['dense_displacement']).astype(np.float32)
                        q1.points=(X+scale.value*a['reference_displacement']).astype(np.float32);rest.points=X.astype(np.float32)
                    dense.visible=True;q1.visible=True
                tip='未求解' if r['tip_displacement'] is None else f"{r['tip_displacement']:.7g}（相对同空间稠密参考 {100*r['tip_relative_to_dense']:+.4f}%）"
                info.content=(f"网格 {r['grid']}，规则 `{r['rule']}`，材料点 {r['samples']}。\n\n"
                    f"端面位移：{tip}\n\n刚度比范围：{r['rho_min']:.5g} – {r['rho_max']:.5g}。越接近 1 越好；位移接近不能替代最差模式检查。\n\n"
                    +(f"最差模式已归一化至最大位移 0.02，仅用于显示。" if is_mode else f"加载形状放大 {scale.value:g} 倍，不改变物理结果。")
                    +'\n\n本页不表示这些规则已经接入生产或具有整体性能优势。')
                last=key
            time.sleep(.05)
    except KeyboardInterrupt:pass
    finally:server.stop()


if __name__=='__main__':main()
