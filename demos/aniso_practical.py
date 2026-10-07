"""Viser overlay of the short nonuniform-Y consistent-PIC comparison."""
import argparse
import json
from pathlib import Path
import socket
import time
import numpy as np
import viser
from benchmarks.aniso_practical_validation import OUT, CASES, paths, load
from benchmarks.aniso_directional_reference import initial
from engine.aniso_phase1.convergence_reference import geometry, knots, tensor_rule


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=OUT)
    p.add_argument('--factor',type=int,default=None,help='defaults to the latest results.json time refinement')
    p.add_argument('--host',default='127.0.0.1')
    p.add_argument('--port',type=int,default=8087)
    p.add_argument('--duration',type=float,default=0.)
    args=p.parse_args()
    if not np.isfinite(args.duration) or args.duration<0:p.error('duration must be finite and nonnegative')
    results=args.data/'results.json'
    factor=args.factor if args.factor is not None else (json.loads(results.read_text())['factor'] if results.exists() else 2)
    frames={}
    for c in CASES:
        path=Path(str(paths(args.data,c,factor))+'-replay.npz')
        if not path.exists():p.error(f'missing {path}; run benchmarks.aniso_practical_validation first')
        with np.load(path) as f:frames[c]={k:f[k].copy() for k in f.files}
    dense_X=tensor_rule(knots(geometry(33)),2)[0]
    dense_initial=initial()['position'].evaluate(dense_X)[0]
    dense_final={c:load(args.data,c,factor)[0]['position'].evaluate(dense_X)[0] for c in CASES}
    first=frames[CASES[0]]
    for f in frames.values():
        np.testing.assert_array_equal(f['reference'],first['reference'])
        np.testing.assert_allclose(f['times'],first['times'],rtol=0,atol=1e-14)
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        probe.bind((args.host,args.port))
    server=viser.ViserServer(host=args.host,port=args.port)
    server.scene.set_up_direction('+z')
    server.gui.add_markdown('## 非均匀 Y 与一致传递\n蓝：局部 Y24；橙：均匀 Y24；绿：较细 Y48。\n'
        '同一原初态、完整一致质量 PIC、材料与粒子历史同源。\n'
        '0.125 ms 短程对照；Y48 是比较网格，尚非收敛真解。')
    frame=server.gui.add_slider('帧',min=0,max=len(first['times'])-1,step=1,initial_value=0)
    scale=server.gui.add_slider('位移放大',min=1.,max=100.,step=1.,initial_value=10.)
    mode=server.gui.add_dropdown('显示',options=('各方案释放位移','相对 Y48 的位置差'),initial_value='各方案释放位移')
    dense=server.gui.add_checkbox('加密采样终态（含目标条带）',initial_value=False)
    play=server.gui.add_checkbox('循环播放',initial_value=True)
    info=server.gui.add_markdown('')
    colors=((50,140,240),(240,150,40),(60,200,100))
    bodies={};visible={}
    for c,color in zip(CASES,colors):
        bodies[c]=server.scene.add_point_cloud('/'+c,points=frames[c]['positions'][0].astype(np.float32),colors=color,point_size=.003,precision='float32')
        visible[c]=server.gui.add_checkbox(c,initial_value=True)
    server.scene.add_point_cloud('/reference',points=first['reference'].astype(np.float32),colors=(150,150,150),point_size=.001,precision='float32')
    results=args.data/'results.json'
    if results.exists():
        r=json.loads(results.read_text())
        server.gui.add_markdown('终态局部/均匀差异比（相对 Y48，<1 表示局部更接近）：\n\n'+
            '\n'.join(f'- {k}: {r["quality"]["all"][k]["ratio"]:.4f}' for k in ('F','P','v')))
    @server.on_client_connect
    def connected(client):
        client.camera.position=(.9,-.3,.9);client.camera.look_at=(.5,.5,.5)
    start=tick=time.monotonic();last=None
    print(f'Viewer: http://{args.host}:{args.port}',flush=True)
    try:
        while not args.duration or time.monotonic()-start<args.duration:
            now=time.monotonic()
            if play.value and now-tick>.4:frame.value=(frame.value+1)%len(first['times']);tick=now
            key=(frame.value,scale.value,mode.value,dense.value,*[v.value for v in visible.values()])
            if key!=last:
                i=len(first['times'])-1 if dense.value else int(frame.value)
                for c,f in frames.items():
                    origin=dense_initial if dense.value else f['positions'][0]
                    current=dense_final[c] if dense.value else f['positions'][i]
                    probe=dense_final['fineY'] if dense.value else frames['fineY']['positions'][i]
                    displacement=current-(probe if mode.value=='相对 Y48 的位置差' else origin)
                    bodies[c].points=(origin+scale.value*displacement).astype(np.float32)
                    bodies[c].visible=visible[c].value
                info.content=f'物理时间 {first["times"][i]*1e3:.6f} ms；帧 {i}。仅放大显示，循环是回放重置。加密采样仅显示终态，覆盖目标 Y 条带。\n\n'+ '\n'.join(f'- {c} 机械能 {f["mechanical"][i]:.8g}' for c,f in frames.items())
                last=key
            time.sleep(.05)
    except KeyboardInterrupt:pass
    finally:server.stop()


if __name__=='__main__':main()
