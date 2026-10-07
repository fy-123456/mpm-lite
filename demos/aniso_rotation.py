"""Viser replay of exact bent-beam rotations; archived experiment, no GPU."""
import argparse,csv,time
from pathlib import Path
import numpy as np
import viser
from PIL import Image,ImageDraw


def chart(records):
    im=Image.new('RGB',(650,250),'white');d=ImageDraw.Draw(im)
    colors={'none':(50,130,210),'supplemental':(215,70,50),'hourglass':(45,160,90),'corotated':(150,70,190),
        'quadratic':(30,120,210),'material_quadratic':(20,155,90)};series=[]
    for _,rows,E0,_ in records:
        series.append(np.array([(float(r['center_material_energy'])+float(r['stabilization_energy']))/E0-1 for r in rows]))
    lo=min(-.01,min(y.min() for y in series));hi=max(.01,max(y.max() for y in series))
    d.text((10,4),'Elastic energy change / initial energy (sampling angle)',fill='black')
    for y in np.linspace(lo,hi,5):
        h=215-(y-lo)/(hi-lo)*170;d.line((60,h,630,h),fill=(220,220,220));d.text((3,h-5),f'{100*y:.1f}%',fill='black')
    for index,((mode,rows,_,_),ys) in enumerate(zip(records,series)):
        color=colors[mode]
        xs=[float(r['stabilization_sampling_angle_degrees']) for r in rows]
        pts=[(60+x/90*570,215-(y-lo)/(hi-lo)*170) for x,y in zip(xs,ys)]
        d.line(pts,fill=color,width=3)
        label={'material_quadratic':'material P2','quadratic':'fixed P2'}.get(mode,mode)
        d.text((65+95*index,28),label,fill=color)
    d.text((60,229),'0 deg',fill='black');d.text((575,229),'90 deg',fill='black')
    return np.asarray(im)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('docs/results/rotation-dissipation'))
    p.add_argument('--corotated-data',type=Path,default=Path('docs/results/corotated'))
    p.add_argument('--quadratic-data',type=Path,default=Path('docs/results/quadratic'))
    p.add_argument('--modes',nargs='+',choices=('none','supplemental','hourglass','corotated','quadratic','material_quadratic'),default=('none','supplemental','hourglass'))
    p.add_argument('--grid',type=int,choices=(17,33),default=17);p.add_argument('--dt',type=float,choices=(.01,.005),default=.01)
    p.add_argument('--axis',choices=('z','y'),default='z')
    p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=8081)
    p.add_argument('--duration',type=float,default=0.)
    args=p.parse_args();records=[]
    for mode in dict.fromkeys(args.modes):
        folder=args.corotated_data if mode=='corotated' else args.data
        if mode in ('quadratic','material_quadratic'):folder=args.quadratic_data
        name=f'rotation-g{args.grid}-dt{args.dt}-{mode}'+('' if args.axis=='z' else '-y')
        if not (folder/(name+'.npz')).exists():p.error(f'missing archive {folder/name}; generate the corresponding rotation benchmark first')
        with (folder/(name+'.csv')).open() as f:rows=list(csv.DictReader(f))
        with (folder/(name+'-budget.csv')).open() as f:E0=float(next(csv.DictReader(f))['elastic'])
        records.append((mode,rows,E0,np.load(folder/(name+'.npz'),allow_pickle=False)))
    server=viser.ViserServer(host=args.host,port=args.port);server.scene.set_up_direction('+z')
    server.gui.add_markdown('## 预弯曲梁：精确刚体旋转\n各模式规定相同运动，形状重合是预期结果。黄色线表示当前纤维方向。请结合能量曲线观察，不能通过相同动画认定力学效果相同。\n\n曲线使用步初采样朝向；状态显示为步末。共旋模式仍可能有固定网格重建误差。此回放不是自由动力学。')
    play=server.gui.add_checkbox('播放',initial_value=True);count=len(records[0][3]['positions'])
    frame=server.gui.add_slider('帧',min=0,max=count-1,step=1,initial_value=0)
    speed=server.gui.add_slider('帧/秒',min=1,max=12,step=1,initial_value=4)
    server.gui.add_image(chart(records),label='Elastic energy vs orientation')
    info=server.gui.add_markdown('');handles=[]
    for i,(mode,_,_,data) in enumerate(records):
        offset=np.array([i*.8,0,0]);x=data['positions'][0]+offset
        cloud=server.scene.add_point_cloud('/'+mode+'/particles',points=x.astype(np.float32),colors=(65,150,220),point_size=.009)
        fibers=server.scene.add_line_segments('/'+mode+'/fibers',points=np.stack([x-.008*data['fibers'][0],x+.008*data['fibers'][0]],axis=1).astype(np.float32),colors=(240,180,40),line_width=2)
        server.scene.add_label('/'+mode+'/label',text=mode,position=(.5+offset[0],.5,.83))
        handles.append((cloud,fibers,offset))
    @server.on_client_connect
    def connected(client):
        center=.5+.4*(len(records)-1)
        client.camera.position=(center,-1.8,2.2);client.camera.look_at=(center,.5,.5)
    start=time.monotonic();tick=start;last=None
    try:
        while not args.duration or time.monotonic()-start<args.duration:
            now=time.monotonic()
            if play.value and now-tick>=1/speed.value:frame.value=(int(frame.value)+1)%count;tick=now
            j=int(frame.value)
            if j!=last:
                lines=[]
                for (mode,rows,E0,data),(cloud,fibers,offset) in zip(records,handles):
                    x=data['positions'][j]+offset;direction=data['fibers'][j];direction=direction/np.linalg.norm(direction,axis=1)[:,None]*.008
                    cloud.points=x.astype(np.float32);fibers.points=np.stack([x-direction,x+direction],axis=1).astype(np.float32)
                    if j:
                        r=rows[j-1];change=(float(r['center_material_energy'])+float(r['stabilization_energy']))/E0-1
                        lines.append(f"**{mode}**：总弹性能变化 {change*100:.2f}%；粒子材料能变化 {float(r['particle_energy_relative_change'])*100:.2g}%；稳定化能 {float(r['stabilization_energy']):.3g}")
                info.content=f"时间 {records[0][3]['times'][j]:.3f} s；绕 {args.axis} 轴\n\n"+'\n\n'.join(lines);last=j
            time.sleep(.02)
    except KeyboardInterrupt:pass
    finally:
        server.stop()
        for _,_,_,data in records:data.close()


if __name__=='__main__':main()
