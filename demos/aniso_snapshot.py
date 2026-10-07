"""View frozen material snapshots and candidate integration points; no solver."""
import argparse,json,time
from pathlib import Path
import numpy as np
from PIL import Image
import viser
from engine.aniso_phase1.material_snapshot import MaterialSnapshot,center_support,moment_points
from engine.aniso_phase1.quadratic import GAUSS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('docs/results/material-snapshot'))
    p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=8083)
    p.add_argument('--duration',type=float,default=0.)
    args=p.parse_args();records=json.loads((args.data/'snapshots.json').read_text());lookup={r['name']:r for r in records}
    from engine.aniso_phase1.joint_sampling import METHODS,build_rule
    joint=all(name in records[0]['methods'] for name in METHODS)
    server=viser.ViserServer(host=args.host,port=args.port);server.scene.set_up_direction('+z')
    server.gui.add_markdown('## 固定快照：材料积分对照\n灰色为同一批粒子，黄色为当前纤维方向，红色为所选积分点。没有时间推进。\n\n`moment8` 匹配中心收到的粒子位置均值与协方差；只是诊断候选，未接入生产求解。能量不包含稳定化。')
    if joint:server.gui.add_markdown('当前加载配对采样结果：`paired` 保留代表粒子自身方向；`group` 对方向分组后保留各组空间矩与方向矩。代表点数量有限时可能丢失空间方差，请结合能量、应力和成本对照。')
    case=server.gui.add_dropdown('快照',options=list(lookup),initial_value='bend-smooth-g17-ppc2-r0' if joint else 'bend-uniform-g17-ppc2-r0')
    method=server.gui.add_dropdown('积分点',options=['center','grid8','moment8','particle']+(list(METHODS) if joint else []),initial_value='group4x8' if joint else 'moment8')
    show=server.gui.add_checkbox('显示积分点',initial_value=True)
    info=server.gui.add_markdown('');server.gui.add_image(np.asarray(Image.open(args.data/'summary.png').convert('RGB')),label='Frozen material comparisons')
    particles=server.scene.add_point_cloud('/particles',points=np.zeros((1,3),dtype=np.float32),colors=(130,150,170),point_size=.006)
    samples=server.scene.add_point_cloud('/integration',points=np.zeros((1,3),dtype=np.float32),colors=(230,65,45),point_size=.007)
    fibers=server.scene.add_line_segments('/fibers',points=np.zeros((1,2,3),dtype=np.float32),colors=(220,175,30),line_width=1)
    @server.on_client_connect
    def connected(client):client.camera.position=(.8,-.6,1.2);client.camera.look_at=(.5,.5,.5)
    start=time.monotonic();last=None
    try:
        while not args.duration or time.monotonic()-start<args.duration:
            key=(case.value,method.value,show.value)
            if key!=last:
                r=lookup[case.value];s=MaterialSnapshot.load(args.data/(case.value+'.npz'));h=1/(r['grid']-1)
                particles.points=s.x.astype(np.float32);groups=center_support(s,h);positions=[]
                if method.value=='particle':positions=s.x
                else:
                    for cell,ids,w in groups:
                        c=(cell+.5)*h
                        if method.value in METHODS:positions.extend(build_rule(s,ids,w,h,method.value)[0])
                        else:positions.extend([c] if method.value=='center' else c+h*GAUSS[1:] if method.value=='grid8' else moment_points(s.x[ids],w))
                samples.points=np.asarray(positions,dtype=np.float32);samples.visible=show.value
                ix=np.arange(0,len(s.x),max(1,len(s.x)//800));_,a=np.linalg.eigh(s.A[ix]);direction=np.einsum('pij,pj->pi',s.F[ix],a[:,:,-1])
                direction*=.006/np.linalg.norm(direction,axis=1)[:,None]
                fibers.points=np.stack((s.x[ix]-direction,s.x[ix]+direction),axis=1).astype(np.float32)
                lines=['| 方法 | 材料能偏差 | 中心平均 P 误差 |','|---|---:|---:|']
                for name in (('particle','moment8','reconstructed_particles')+METHODS if joint else ('particle','center','grid8','moment8','reconstructed_particles','decorrelated_particles')):
                    m=r['methods'][name];lines.append(f"| {name} | {100*m['energy_relative_error']:.4f}% | {100*m['center_P_rms_relative_error']:.4f}% |")
                info.content=f"粒子 {len(s.x)}；中心 {r['centers']}；当前积分点 {len(positions)}。\n\n"+'\n'.join(lines)+'\n\n应力误差按中心局部均值比较，可能掩盖中心内部变化；它不是反力误差。'
                last=key
            time.sleep(.04)
    except KeyboardInterrupt:pass
    finally:server.stop()


if __name__=='__main__':main()
