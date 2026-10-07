"""Viser replay of the archived controlled-motion audits (no GPU needed).

Generate data: python -m benchmarks.aniso_history_reference --part migration
These are prescribed kinematic tests, not free dynamic simulations.
"""
import argparse,csv,time
from pathlib import Path
import numpy as np
import viser


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',choices=('translate','rotate','directions'),default='translate')
    p.add_argument('--data',type=Path,default=Path('docs/results/history-reference'))
    p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=8081)
    p.add_argument('--duration',type=float,default=0.)
    args=p.parse_args();records=[]
    for policy in ('grid_locked','particle_resample'):
        name=f'{args.scene}-g65-dt0.05-{policy}-exact'
        with (args.data/(name+'.csv')).open() as f:rows=list(csv.DictReader(f))
        data=np.load(args.data/(name+'.npz'),allow_pickle=False)
        records.append((policy,rows,data))
    count=len(records[0][1]);server=viser.ViserServer(host=args.host,port=args.port)
    server.scene.set_up_direction('+z')
    server.gui.add_markdown('## 中心历史迁移对照\n左：旧网格固定历史；右：粒子历史重采样。\n\n这是规定运动的短程回放。颜色表示该帧总体历史误差（蓝色小，红色大），短线为中心当前纤维主方向。')
    play=server.gui.add_checkbox('播放',initial_value=True)
    slider=server.gui.add_slider('帧',min=0,max=count-1,step=1,initial_value=0)
    speed=server.gui.add_slider('帧/秒',min=1,max=12,step=1,initial_value=4)
    status=server.gui.add_markdown('')
    handles=[]
    for j,(policy,rows,data) in enumerate(records):
        offset=np.array([1.05*j,0,0])
        cloud=server.scene.add_point_cloud('/'+policy+'/particles',points=(data['points_0']+offset).astype(np.float32),colors=(80,150,220),point_size=.008)
        centers=data['centers_0']+offset;d=data['directions_0']*.012
        fibers=server.scene.add_line_segments('/'+policy+'/fibers',points=np.stack([centers-d,centers+d],axis=1).astype(np.float32),colors=(80,150,220),line_width=2.)
        server.scene.add_label('/'+policy+'/label',text='旧模式' if j==0 else '重采样',position=(float(.5+offset[0]),.5,.65))
        handles.append((cloud,fibers,offset))
    @server.on_client_connect
    def connected(client):
        client.camera.position=(1.05,-.9,1.2);client.camera.look_at=(1.05,.5,.5)
    started=time.monotonic();last_frame=None;last_tick=started
    try:
        while not args.duration or time.monotonic()-started<args.duration:
            now=time.monotonic()
            if play.value and now-last_tick>=1/speed.value:
                slider.value=(int(slider.value)+1)%count;last_tick=now
            i=int(slider.value)
            if i!=last_frame:
                lines=[]
                for (policy,rows,data),(cloud,fibers,offset) in zip(records,handles):
                    row=rows[i];error=max(float(row['F_relative_error']),float(row['A_rms_error']))
                    t=np.clip(error/.08,0,1);color=tuple(((1-t)*np.array([60,155,235])+t*np.array([235,70,55])).astype(int))
                    cloud.points=(data[f'points_{i}']+offset).astype(np.float32);cloud.colors=np.tile(np.array(color,dtype=np.uint8),(len(cloud.points),1))
                    centers=data[f'centers_{i}']+offset;d=data[f'directions_{i}']*.012
                    fibers.points=np.stack([centers-d,centers+d],axis=1).astype(np.float32)
                    lines.append(f"**{policy}**：F 误差 {float(row['F_relative_error']):.3g}，方向张量误差 {float(row['A_rms_error']):.3g}，弹性能误差 {100*float(row['elastic_relative_error']):.3g}%")
                status.content=f"时间 {float(records[0][1][i]['time']):.3f} s\n\n"+'\n\n'.join(lines)
                last_frame=i
            time.sleep(.02)
    except KeyboardInterrupt:pass
    finally:
        server.stop()
        for _,_,data in records:data.close()


if __name__=='__main__':main()
