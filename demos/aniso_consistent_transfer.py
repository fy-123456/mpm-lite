"""Replay the isolated material-carried Q1 closed-loop experiment in Viser."""
import argparse
from pathlib import Path
import socket
import time

import numpy as np
import viser


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('docs/results/consistent-transfer/v1/g17-p2-q3-dt0.001.npz'))
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8086)
    p.add_argument('--duration', type=float, default=0.)
    args = p.parse_args()
    if not np.isfinite(args.duration) or args.duration < 0:
        p.error('duration must be finite and nonnegative')
    if not args.data.exists():
        p.error('run python -m benchmarks.aniso_consistent_transfer first')
    # Viser may wait indefinitely if its background server cannot bind at all.
    try:
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind((args.host, args.port))
    except OSError as exc:
        p.error(f'cannot bind visualization address {args.host}:{args.port}: {exc}')
    with np.load(args.data) as data:
        X = data['reference']
        frames = data['positions']
        times = data['times']
        mechanical = data['mechanical']
    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.set_up_direction('+z')
    server.gui.add_markdown('## 一致传递原型：预弯曲梁释放\n'
                            '随材料移动的 Q1 模板、相容弹性历史、一致质量 PIC。'
                            '**独立 CPU 验证原型，不是生产 MLS/MPM，也不是 APIC。**\n\n'
                            '蓝色：当前粒子；橙色：初始弯曲状态；灰色：未变形参考。'
                            '显示放大不改变物理结果。')
    frame = server.gui.add_slider('帧', min=0, max=len(frames)-1, step=1, initial_value=0)
    scale = server.gui.add_slider('位移显示放大', min=1., max=20., step=1., initial_value=10.)
    play = server.gui.add_checkbox('循环播放', initial_value=True)
    info = server.gui.add_markdown('')
    body = server.scene.add_point_cloud('/current', points=frames[0].astype(np.float32), colors=(45, 140, 235), point_size=.004)
    initial = server.scene.add_point_cloud('/initial', points=frames[0].astype(np.float32), colors=(235, 165, 50), point_size=.002)
    server.scene.add_point_cloud('/reference', points=X.astype(np.float32), colors=(140, 140, 140), point_size=.002)
    @server.on_client_connect
    def connected(client):
        client.camera.position = (.9, -.3, .9)
        client.camera.look_at = (.5, .5, .5)
    start = tick = time.monotonic()
    last = None
    try:
        while not args.duration or time.monotonic()-start < args.duration:
            now = time.monotonic()
            if play.value and now-tick > .3:
                frame.value = (frame.value+1) % len(frames)
                tick = now
            key = (frame.value, scale.value)
            if key != last:
                i = int(frame.value)
                body.points = (X+scale.value*(frames[i]-X)).astype(np.float32)
                initial.points = (X+scale.value*(frames[0]-X)).astype(np.float32)
                info.content = (f'物理时间 {times[i]:.5f} s；机械能 {mechanical[i]:.8g}；'
                                f'相对初始变化 {100*(mechanical[i]/mechanical[0]-1):+.4f}%。\n\n'
                                '短程释放包含隐式积分耗散；历史恢复误差请查看同名 CSV。'
                                '循环切回第 0 帧是回放重置，不是模拟中的能量跳变。')
                last = key
            time.sleep(.05)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == '__main__':
    main()
