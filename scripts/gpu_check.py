#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gpu_check.py —— GPU 环境自检（第一课：确认你真的在用 GPU）。

在租来的 GPU 机器（天池 DSW / 阿里云 / 第三方）上跑：
    python scripts/gpu_check.py

它做三件事：
  1. 打印环境信息（Python / torch / CUDA 版本）。
  2. 硬检查 GPU 是否可用：不可用直接 **报错退出**（不静默退回 CPU），
     防止你误以为在用 GPU 却其实在跑 CPU。
  3. 跑一个矩阵乘法基准，直观对比 GPU vs CPU 的加速比，并打印显存占用。

看懂这三段输出，你就掌握了 GPU 编程最核心的几个动作：
  torch.cuda.is_available() / .to("cuda") / torch.cuda.synchronize() / nvidia-smi。
"""
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass


def main():
    print("=" * 56)
    print("第 1 段：环境信息")
    print("=" * 56)
    print("Python :", sys.version.split()[0], "|", sys.platform)

    try:
        import torch
    except ImportError:
        print("\n❌ 没装 torch。先装（DSW 的 GPU 镜像通常已预装，若没有）：")
        print("   pip install torch --index-url https://download.pytorch.org/whl/cu121")
        sys.exit(1)

    print("torch  :", torch.__version__)
    print("CUDA 编译版本:", torch.version.cuda)  # torch 编译时绑定的 CUDA 版本

    print("\n" + "=" * 56)
    print("第 2 段：GPU 可用性（硬检查）")
    print("=" * 56)
    if not torch.cuda.is_available():
        print("❌ torch.cuda.is_available() = False")
        print("   你现在【不在 GPU 机器上】，或驱动/CUDA 没装好。")
        print("   —— 本项目约定全程用 GPU，这里直接中止，避免误跑 CPU。")
        print("   排查：nvidia-smi 能否看到显卡？pip 装的是 CUDA 版 torch 吗？")
        sys.exit(2)

    n = torch.cuda.device_count()
    print("✅ 检测到 %d 张 GPU：" % n)
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        print("   [%d] %s | 显存 %.1f GB | 算力 sm_%d%d"
              % (i, p.name, p.total_memory / 1024**3, p.major, p.minor))
    dev = torch.device("cuda:0")

    print("\n" + "=" * 56)
    print("第 3 段：GPU vs CPU 基准（4096x4096 矩阵乘）")
    print("=" * 56)
    size = 4096

    # CPU 计时
    a_cpu = torch.randn(size, size)
    b_cpu = torch.randn(size, size)
    t0 = time.perf_counter()
    _ = a_cpu @ b_cpu
    cpu_dt = time.perf_counter() - t0
    print("CPU  一次矩阵乘: %.3f 秒" % cpu_dt)

    # GPU 计时（注意：CUDA 是异步的，必须 synchronize 才能测准）
    a_gpu = a_cpu.to(dev)
    b_gpu = b_cpu.to(dev)
    torch.cuda.synchronize()          # 等数据搬运完
    _ = a_gpu @ b_gpu                 # 预热一次（首次调用含 kernel 编译开销）
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):              # 跑 10 次取平均，更稳
        _ = a_gpu @ b_gpu
    torch.cuda.synchronize()          # 关键：不 sync 会在 kernel 还没跑完就停表
    gpu_dt = (time.perf_counter() - t0) / 10
    print("GPU  一次矩阵乘: %.3f 秒" % gpu_dt)
    if gpu_dt > 0:
        print("加速比 ≈ %.1fx" % (cpu_dt / gpu_dt))

    # 显存占用（等价于 nvidia-smi 里看到的那个数）
    alloc = torch.cuda.memory_allocated(dev) / 1024**2
    reserved = torch.cuda.memory_reserved(dev) / 1024**2
    print("\n当前显存：已分配 %.0f MB | 已预留 %.0f MB" % (alloc, reserved))
    print("（对照命令行 `nvidia-smi` 看到的显存占用应大致吻合）")

    print("\n✅ GPU 环境就绪。下一步就可以把 SapBERT 编码放到这张卡上跑了。")


if __name__ == "__main__":
    main()
