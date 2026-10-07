"""Read-only Viser comparison of static clamp/load experiments; no time stepping."""
import argparse,json,time
from pathlib import Path
import numpy as np
from PIL import Image
import viser


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,default=Path('docs/results/boundary-consistency'))
    p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=8084);p.add_argument('--duration',type=float,default=0.)
    args=p.parse_args()
    if not np.isfinite(args.duration) or args.duration<0:p.error('duration must be finite and nonnegative')
    results=json.loads((args.data/'results.json').read_text());records={r['name']:r for result in results for r in result['records']}
    server=viser.ViserServer(host=args.host,port=args.port);server.scene.set_up_direction('+z')
    server.gui.add_markdown('## 夹持与加载的静态对照\n蓝色为所选方案，绿色为相同面载荷的全积分 Q1 参考，灰色为未变形梁，红色为固定面检查点。\n\n这是线性静态结果查看器，不做时间推进。`global_blend` 与 `sparse_face` 是异常定位对照，尚未用于生产求解。')
    case=server.gui.add_dropdown('方案',options=list(records),initial_value='g17-local-patch_faces-mls' if 'g17-local-patch_faces-mls' in records else next(iter(records)))
    display=server.gui.add_dropdown('显示插值',options=['MLS','Lite'],initial_value='MLS')
    scale=server.gui.add_slider('显示位移放大',min=1.,max=100.,step=1.,initial_value=50.)
    info=server.gui.add_markdown('')
    server.gui.add_image(np.asarray(Image.open(args.data/'summary.png').convert('RGB')),label='Static displacement and clamp errors')
    def cloud(name,color,size=.005):return server.scene.add_point_cloud(name,points=np.zeros((1,3),dtype=np.float32),colors=color,point_size=size)
    body=cloud('/result',(60,145,230));reference=cloud('/reference',(80,180,90));ghost=cloud('/rest',(150,150,150));root=cloud('/clamp',(220,60,45),.004)
    @server.on_client_connect
    def connected(client):client.camera.position=(.9,-.4,1.0);client.camera.look_at=(.48,.48,.48)
    last=None;start=time.monotonic()
    try:
        while not args.duration or time.monotonic()-start<args.duration:
            key=(case.value,display.value,scale.value)
            if key!=last:
                r=records[case.value]
                with np.load(args.data/(case.value+'.npz')) as a:
                    X=a['reference'];u=a['displacement_'+display.value.lower()]
                    body.points=(X+scale.value*u).astype(np.float32)
                    reference.points=(X+scale.value*a['reference_displacement']).astype(np.float32);ghost.points=X.astype(np.float32)
                    root.points=(a['clamp_points']+scale.value*a['clamp_displacement']).astype(np.float32)
                info.content=(f"网格 {r['grid']}；材料场 `{r['field']}`；夹持 `{r['clamp']}`；加载 `{r['load']}`。\n\n"
                    f"加载共轭位移：{r['tip_displacement']:.6g}；相对 Q1：{100*r['relative_to_reference']:.2f}%。\n\n"
                    f"独立面检查最大位移：{r['root_mls_max']:.3g}；逐局部拟合面位移：{r['root_local_patch_max']:.3g}。\n\n"
                    f"夹具 y 反力：{r['reaction_y']:.6g}；总加载力：-0.0001。\n\n"
                    f"若端面平均位移固定为 -0.0005，所需加载力：{r['force_for_mean_displacement']:.6g}。\n\n"
                    '显示放大不改变物理结果。MLS/Lite 切换只改变结果的读取方式；表中位移仍按对应加载的共轭量定义。')
                last=key
            time.sleep(.05)
    except KeyboardInterrupt:pass
    finally:server.stop()


if __name__=='__main__':main()
