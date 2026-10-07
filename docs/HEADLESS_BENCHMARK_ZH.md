# 3D Demo 无头帧率测试记录

## 测试范围与环境

本次测试覆盖 `demos.wheel`、`demos.noodles`、`demos.snow` 三个 Viser 版 3D demo，不包含使用 VisPy/GLFW 的 `mpmlite2d.py`。

| 项目 | 配置 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3090，24 GiB |
| 仿真设备 | `cuda:0`，单 GPU |
| Warp | 1.10.1 |
| Warp 启动日志显示的 CUDA Toolkit / Driver | 12.8 / 13.0 |
| 可视化方式 | Viser 服务监听 `127.0.0.1`，不打开窗口、不连接浏览器 |
| 显示点数上限 | `--viser-max-points 1000`，仅限制可视化点数，不减少仿真粒子 |
| 时间限制 | 每次运行由 `timeout` 在约 30 秒时终止 |
| 仿真参数 | 除额外的 snow 缩小场景外，均使用 demo 默认参数 |

这里的“无头”指无图形窗口、无浏览器客户端；Viser 服务及其点云更新代码仍然运行，并非完全关闭可视化逻辑的纯求解器测试。

## 统计口径

**仿真步进 FPS = 已完成仿真步数 / 进程总墙钟时间。**

墙钟时间包含启动、粒子播种、边界初始化、内核加载、日志输出、Viser 服务及终止开销。结果是一次短时运行的全程平均吞吐量，不是排除预热后的稳态性能，也不是浏览器渲染 FPS。

完成步数依据日志中的 `solver.sim_steps` 或下一次 `Time step` 入口判断。`Time step: 31` 表示此前已完成 31 步；不把被超时打断的当前步计入。

Viser 更新次数根据完成步数及代码中的更新条件推算，没有测量浏览器接收或绘制速率。

## 测试结果

| Demo / 配置 | 粒子数 | 墙钟时间 | 完成步数 | 仿真步进 FPS | 推算 Viser 更新次数 | 平均更新频率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| wheel，默认参数 | 1,193,678 | 30.12 s | 35 | **1.162 步/s** | 0 | 0 次/s |
| noodles，默认参数 | 2,887,540 | 30.19 s | 31 | **1.027 步/s** | 3 | 0.099 次/s |
| snow，默认参数 | 7,723,969 | 30.21 s | 0 | **未获得有效测量** | 0 | 0 次/s |
| snow，缩小场景 | 1,930 → 2,009 | 30.20 s | 7,108 | **235.364 步/s** | 33 | 1.093 次/s |

### 结果说明

- **wheel：** 每 50 步更新一次点云，本次只完成 35 步，因此没有点云更新。
- **noodles：** 每 10 步更新一次点云，本次在第 10、20、30 步触发更新。
- **默认 snow：** 已播种约 772 万粒子，日志最后停在 `./assets/ramp.obj` 对应的边界初始化阶段，尚未进入仿真循环。不能将该结果解释为求解器 0 FPS，也不能据此断言程序卡死；需要延长运行时间才能得到默认场景的有效帧率。
- **缩小 snow：** 使用 `--grid_size 32 --dx 0.05 --ppc 1`。初始 1,930 粒子，第 600 步加入雪球后为 2,009 粒子。第 600 步起每 200 步更新点云，至第 7,000 步共 33 次。该设置改变分辨率、粒子数及计算域，仅作为额外的小规模运行测试，不能代替默认 snow 的性能结果或与其他默认场景直接比较。

四次运行均因预设超时返回退出码 `124`；检查日志未发现 traceback 或运行异常。这只说明测试时间窗口内的情况，不代表已验证完整仿真过程。

### 对此前聊天汇总的更正

重新核对原始日志与代码后，更正如下：

- noodles 已完成 **31 步**，平均 **1.027 步/s**；此前写成 30 步、0.994 步/s。
- 缩小 snow 的粒子数是 **1,930 → 2,009**；此前的 6,607 为误记。
- 缩小 snow 共触发 **33 次**点云更新，平均 **1.093 次/s**；此前的 1.060 次/s 少计了一次更新。

## 复现命令

在项目目录执行以下命令，依次测试三个默认场景。每次的输出目录使用时间戳区分，日志保存在对应目录下。

```bash
cd /home/yin/mpm-lite
bench_dir="output/headless-bench-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$bench_dir"

for demo in wheel noodles snow; do
  /usr/bin/time -f 'WALL_SECONDS=%e' \
    timeout -s TERM 30s env PYTHONUNBUFFERED=1 \
    uv run -m "demos.$demo" --device cuda:0 \
      --viser-host 127.0.0.1 --viser-port 18090 \
      --viser-max-points 1000 --out "$bench_dir/$demo" \
      > "$bench_dir/$demo.log" 2>&1
  result=$?
  echo "$demo: exit=$result"
done
```

接着运行额外的 snow 缩小场景：

```bash
/usr/bin/time -f 'WALL_SECONDS=%e' \
  timeout -s TERM 30s env PYTHONUNBUFFERED=1 \
  uv run -m demos.snow --device cuda:0 \
    --grid_size 32 --dx 0.05 --ppc 1 \
    --viser-host 127.0.0.1 --viser-port 18093 \
    --viser-max-points 1000 --out "$bench_dir/snow-small" \
    > "$bench_dir/snow-small.log" 2>&1
```

以上命令不会自动生成 FPS 汇总，需根据日志中的完成步数和 `WALL_SECONDS` 按上述口径计算。首次内核编译、缓存和机器负载均可能使复测结果不同。

## 本次原始日志

本次检查使用以下本机临时日志；`/tmp` 内容可能被系统清理：

- `/tmp/bench-wheel.log`
- `/tmp/bench-noodles.log`
- `/tmp/bench-snow.log`
- `/tmp/bench-snow-small.log`
