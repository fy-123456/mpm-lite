"""Replay actual particle advection and rebuilt spatial grids in Viser."""
import argparse
import json
from pathlib import Path
import socket
import time
import numpy as np
import viser
from benchmarks.aniso_spatial_mpm import DT, H
from engine.aniso_phase1.consistent_transfer import CORNERS


def grid_edges(x,origin):
    cells=np.unique(np.floor((x-origin)/H).astype(int),axis=0)
    edges=[(i,j) for i in range(8) for j in range(i+1,8) if np.abs(CORNERS[i]-CORNERS[j]).sum()==1]
    endpoints=np.asarray([(c+CORNERS[i],c+CORNERS[j]) for c in cells for i,j in edges])
    return (origin+H*endpoints).astype(np.float32)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('docs/results/xz-mpm/v1/mpm/refined'))
    p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=8088)
    p.add_argument('--duration',type=float,default=0.)
    a=p.parse_args()
    if not np.isfinite(a.duration) or a.duration<0:p.error('duration must be finite and nonnegative')
    if not (a.data/'results.json').exists():p.error('missing results.json; run benchmarks.aniso_spatial_mpm first')
    result=json.loads((a.data/'results.json').read_text());factor=result['factor'];frames={};rows={}
    for switched,name in [(False,'fixed'),(True,'shifted')]:
        path=a.data/f'{name}-dt{DT/factor:.7f}.npz'
        with np.load(path) as f:frames[name]={k:f[k].copy() for k in f.files}
        rows[name]=result['runs'][f'{switched}-{factor}']['rows']
    np.testing.assert_array_equal(frames['fixed']['times'],frames['shifted']['times'])
    times=frames['fixed']['times']
    try:
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);probe.bind((a.host,a.port))
    except OSError as exc:p.error(str(exc))
    server=viser.ViserServer(host=a.host,port=a.port);server.scene.set_up_direction('+z')
    server.gui.add_markdown('## 实际空间网格 MPM\n蓝：固定网格原点；橙：每 2 ms 切换原点。432 个材料粒子，每步按当前位置重建插值。\n\n'
        '**F 提交和动能一致性通过；时间精度检查未通过。** 网格相位敏感，尚不能用作生产精度结论。')
    frame=server.gui.add_slider('帧',min=0,max=len(times)-1,step=1,initial_value=0)
    play=server.gui.add_checkbox('循环播放',initial_value=True)
    mode=server.gui.add_dropdown('显示',options=('实际位置与空间网格','去平移形变'),initial_value='实际位置与空间网格')
    selected=server.gui.add_dropdown('显示哪个网格',options=('fixed','shifted'),initial_value='shifted')
    scale=server.gui.add_slider('形变放大（仅去平移模式）',min=1.,max=50.,step=1.,initial_value=10.)
    info=server.gui.add_markdown('')
    clouds={name:server.scene.add_point_cloud('/'+name,points=f['positions'][0].astype(np.float32),
        colors=color,point_size=.003,precision='float32') for (name,f),color in zip(frames.items(),[(40,140,240),(245,150,40)])}
    grid=server.scene.add_line_segments('/grid',points=grid_edges(frames['shifted']['positions'][0],np.zeros(3)),colors=(155,155,155),line_width=.8)
    @server.on_client_connect
    def connected(client):
        client.camera.position=(.85,-.1,.85);client.camera.look_at=(.38,.5,.5)
    start=tick=time.monotonic();last=None
    print(f'Viewer: http://{a.host}:{a.port}',flush=True)
    try:
        while not a.duration or time.monotonic()-start<a.duration:
            now=time.monotonic()
            if play.value and now-tick>.15:frame.value=(frame.value+1)%len(times);tick=now
            key=(frame.value,mode.value,selected.value,scale.value)
            if key!=last:
                i=int(frame.value);physical=mode.value=='实际位置与空间网格'
                for name,f in frames.items():
                    positions=f['positions'][i]
                    if not physical:
                        delta=positions-f['positions'][0]-times[i]*np.array([1.,.15,0.])
                        positions=f['positions'][0]+scale.value*delta
                    clouds[name].points=positions.astype(np.float32)
                f=frames[selected.value];grid.visible=physical
                grid.points=grid_edges(f['positions'][max(i-1,0)],f['origins'][max(i-1,0)])
                details=[]
                if i:
                    for name in frames:
                        row=rows[name][i-1]
                        details.append(f"- {name}: F 误差 {row['prediction_commit_F_absolute']:.2g}，K 相对误差 {row['kinetic_relative']:.2g}，本步跨单元粒子 {row['physical_cell_crossings']}")
                info.content=f'时间 {times[i]*1000:.3f} ms，帧 {i}。灰线为本步求解使用的空间网格。循环为回放重置。\n\n'+'\n'.join(details)
                last=key
            time.sleep(.04)
    except KeyboardInterrupt:pass
    finally:server.stop()

if __name__=='__main__':main()
