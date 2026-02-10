# plot_convergence.py
"""
绘制 PCG 迭代收敛曲线（相对残差 vs 迭代次数）
对比不同的自适应阈值设置
"""
import os
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from multiprocessing import Pool


# ============================================================
# 实验配置：按论文 Diff3D 设置
# ============================================================
@dataclass
class ExpConfig:
    # 网格：128^3
    n: int = 128

    # 相对残差收敛阈值（论文 Diff3D：1e-10，RHD3D：1e-12）
    tol_res: float = 1e-17
    max_iter: int = 30000

    # 工作精度（论文 Diff3D：fp64），低精度（fp32）
    working_dtype: torch.dtype = torch.float64
    low_dtype: torch.dtype = torch.float32

    # BJAC 参数（论文默认：nb=32, k=t=2）
    nb: int = 32
    k_inter: int = 2
    t_intra: int = 2

    # 自适应阈值（论文：hl=1e-5，lh=1e-10）
    adp_tol_hl: float = 1e-5
    adp_tol_lh: float = 1e-10

    # 设备（运行时会被 local_rank 覆盖成 cuda:local_rank）
    device: str = "cuda"


# ============================================================
# Diff3D 系数场 κ(x)
# ============================================================
def make_kappa(case_name: str, n: int, s: float, seed: int, device: str, dtype: torch.dtype):
    """生成扩散系数场，返回 kx, ky, kz，形状均为 (n,n,n)"""
    if case_name.startswith("Diff3D-Const"):
        kx = torch.ones((n, n, n), device=device, dtype=dtype)
        ky = torch.ones((n, n, n), device=device, dtype=dtype)
        kz = torch.ones((n, n, n), device=device, dtype=dtype)
        return kx, ky, kz

    if case_name.startswith("Diff3D-Ani"):
        kx = torch.ones((n, n, n), device=device, dtype=dtype)
        ky = torch.full((n, n, n), float(s), device=device, dtype=dtype)
        kz = torch.full((n, n, n), float(s), device=device, dtype=dtype)
        return kx, ky, kz

    if case_name.startswith("Diff3D-Dis"):
        kx = torch.ones((n, n, n), device=device, dtype=dtype)
        ky = torch.ones((n, n, n), device=device, dtype=dtype)
        kz = torch.ones((n, n, n), device=device, dtype=dtype)
        a = int(0.25 * n)
        b = int(0.75 * n)
        kx[a:b, a:b, a:b] = float(s)
        ky[a:b, a:b, a:b] = float(s)
        kz[a:b, a:b, a:b] = float(s)
        return kx, ky, kz

    if case_name.startswith("Diff3D-Rand"):
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        delta = torch.rand((n, n, n), device=device, dtype=dtype, generator=g)
        k = float(s) ** delta
        return k, k, k

    # RHD3D-1T: 辐射流体力学单温度模型
    # 根据论文Table 3，RHD3D-1T是弱多尺度问题（99.85%的行在[1,10)区间）
    # 这里构造一个近似的系数场来模拟RHD物理特性
    if case_name.startswith("RHD3D-1T"):
        g = torch.Generator(device=device)
        g.manual_seed(seed)

        # 基础系数（代表辐射扩散系数）
        base_coeff = 1.0

        # 添加小幅度随机扰动（模拟弱多尺度特性）
        # 99.85%的值在[1,10)区间，意味着大部分系数接近1，少数接近10
        perturbation = 1.0 + 8.0 * torch.rand((n, n, n), device=device, dtype=dtype, generator=g)

        # 随机选择0.15%的点设置为稍大的值（模拟多尺度行为）
        mask = torch.rand((n, n, n), device=device, dtype=dtype, generator=g) > 0.9985
        perturbation[mask] = perturbation[mask] * 2.0

        k = base_coeff * perturbation

        return k, k, k

    # RHD3D-3T: 三温度模型（强多尺度）
    if case_name.startswith("RHD3D-3T"):
        g = torch.Generator(device=device)
        g.manual_seed(seed)

        # 三温度耦合导致强多尺度特性
        # 根据论文Table 3，有很大范围的多尺度强度分布
        base = torch.ones((n, n, n), device=device, dtype=dtype)

        # 生成多尺度系数
        delta = torch.rand((n, n, n), device=device, dtype=dtype, generator=g)

        # 根据Table 3的分布构造多尺度系数
        k = torch.ones_like(delta)
        k[delta < 0.0824] = base[delta < 0.0824] * 1.5  # [1, 10)
        k[(delta >= 0.0824) & (delta < 0.1282)] = base[(delta >= 0.0824) & (delta < 0.1282)] * 50.0  # [10, 100)
        k[(delta >= 0.1282) & (delta < 0.2176)] = base[(delta >= 0.1282) & (delta < 0.2176)] * 500.0  # [100, 1000)
        k[(delta >= 0.2176) & (delta < 0.5063)] = base[(delta >= 0.2176) & (delta < 0.5063)] * 5000.0  # [1000, 10000)
        k[(delta >= 0.5063) & (delta < 0.5956)] = base[(delta >= 0.5063) & (delta < 0.5956)] * 50000.0  # [10000, 100000)
        k[(delta >= 0.5956) & (delta < 0.8131)] = base[(delta >= 0.5956) & (delta < 0.8131)] * 5e7  # [100000, 1e10)
        k[(delta >= 0.8131) & (delta < 0.8866)] = base[(delta >= 0.8131) & (delta < 0.8866)] * 1e12  # [1e10, 1e15)
        k[delta >= 0.8866] = base[delta >= 0.8866] * 1e18  # [1e15, +inf)

        return k, k, k

    raise ValueError(f"未知 case_name: {case_name}")


# ============================================================
# 全域 7 点 stencil
# ============================================================
def apply_A_full(u: torch.Tensor, kx: torch.Tensor, ky: torch.Tensor, kz: torch.Tensor) -> torch.Tensor:
    n = u.shape[0]
    up = torch.zeros((n + 2, n + 2, n + 2), device=u.device, dtype=u.dtype)
    up[1:-1, 1:-1, 1:-1] = u

    c  = up[1:-1, 1:-1, 1:-1]
    xm = up[0:-2, 1:-1, 1:-1]
    xp = up[2:  , 1:-1, 1:-1]
    ym = up[1:-1, 0:-2, 1:-1]
    yp = up[1:-1, 2:  , 1:-1]
    zm = up[1:-1, 1:-1, 0:-2]
    zp = up[1:-1, 1:-1, 2:  ]

    kxp = torch.zeros_like(up)
    kyp = torch.zeros_like(up)
    kzp = torch.zeros_like(up)
    kxp[1:-1, 1:-1, 1:-1] = kx
    kyp[1:-1, 1:-1, 1:-1] = ky
    kzp[1:-1, 1:-1, 1:-1] = kz

    kx_m = 0.5 * (kxp[1:-1, 1:-1, 1:-1] + kxp[0:-2, 1:-1, 1:-1])
    kx_p = 0.5 * (kxp[1:-1, 1:-1, 1:-1] + kxp[2:  , 1:-1, 1:-1])
    ky_m = 0.5 * (kyp[1:-1, 1:-1, 1:-1] + kyp[1:-1, 0:-2, 1:-1])
    ky_p = 0.5 * (kyp[1:-1, 1:-1, 1:-1] + kyp[1:-1, 2:  , 1:-1])
    kz_m = 0.5 * (kzp[1:-1, 1:-1, 1:-1] + kzp[1:-1, 1:-1, 0:-2])
    kz_p = 0.5 * (kzp[1:-1, 1:-1, 1:-1] + kzp[1:-1, 1:-1, 2:  ])

    Au = (kx_m * (c - xm) + kx_p * (c - xp) +
          ky_m * (c - ym) + ky_p * (c - yp) +
          kz_m * (c - zm) + kz_p * (c - zp))
    return Au


# ============================================================
# 向量化 BJAC
# ============================================================
def apply_A_block_batch(u_blk: torch.Tensor,
                        kx_blk: torch.Tensor,
                        ky_blk: torch.Tensor,
                        kz_blk: torch.Tensor) -> torch.Tensor:
    nb, bs, n, _ = u_blk.shape

    up = torch.zeros((nb, bs + 2, n + 2, n + 2), device=u_blk.device, dtype=u_blk.dtype)
    up[:, 1:-1, 1:-1, 1:-1] = u_blk

    c  = up[:, 1:-1, 1:-1, 1:-1]
    xm = up[:, 0:-2, 1:-1, 1:-1]
    xp = up[:, 2:  , 1:-1, 1:-1]
    ym = up[:, 1:-1, 0:-2, 1:-1]
    yp = up[:, 1:-1, 2:  , 1:-1]
    zm = up[:, 1:-1, 1:-1, 0:-2]
    zp = up[:, 1:-1, 1:-1, 2:  ]

    kxp = torch.zeros_like(up)
    kyp = torch.zeros_like(up)
    kzp = torch.zeros_like(up)
    kxp[:, 1:-1, 1:-1, 1:-1] = kx_blk
    kyp[:, 1:-1, 1:-1, 1:-1] = ky_blk
    kzp[:, 1:-1, 1:-1, 1:-1] = kz_blk

    kx_m = 0.5 * (kxp[:, 1:-1, 1:-1, 1:-1] + kxp[:, 0:-2, 1:-1, 1:-1])
    kx_p = 0.5 * (kxp[:, 1:-1, 1:-1, 1:-1] + kxp[:, 2:  , 1:-1, 1:-1])
    ky_m = 0.5 * (kyp[:, 1:-1, 1:-1, 1:-1] + kyp[:, 1:-1, 0:-2, 1:-1])
    ky_p = 0.5 * (kyp[:, 1:-1, 1:-1, 1:-1] + kyp[:, 1:-1, 2:  , 1:-1])
    kz_m = 0.5 * (kzp[:, 1:-1, 1:-1, 1:-1] + kzp[:, 1:-1, 1:-1, 0:-2])
    kz_p = 0.5 * (kzp[:, 1:-1, 1:-1, 1:-1] + kzp[:, 1:-1, 1:-1, 2:  ])

    Au = (kx_m * (c - xm) + kx_p * (c - xp) +
          ky_m * (c - ym) + ky_p * (c - yp) +
          kz_m * (c - zm) + kz_p * (c - zp))
    return Au


def diag_of_A_block_batch(kx_blk: torch.Tensor,
                          ky_blk: torch.Tensor,
                          kz_blk: torch.Tensor) -> torch.Tensor:
    nb, bs, n, _ = kx_blk.shape

    kxp = torch.zeros((nb, bs + 2, n + 2, n + 2), device=kx_blk.device, dtype=kx_blk.dtype)
    kyp = torch.zeros_like(kxp)
    kzp = torch.zeros_like(kxp)
    kxp[:, 1:-1, 1:-1, 1:-1] = kx_blk
    kyp[:, 1:-1, 1:-1, 1:-1] = ky_blk
    kzp[:, 1:-1, 1:-1, 1:-1] = kz_blk

    kx_m = 0.5 * (kxp[:, 1:-1, 1:-1, 1:-1] + kxp[:, 0:-2, 1:-1, 1:-1])
    kx_p = 0.5 * (kxp[:, 1:-1, 1:-1, 1:-1] + kxp[:, 2:  , 1:-1, 1:-1])
    ky_m = 0.5 * (kyp[:, 1:-1, 1:-1, 1:-1] + kyp[:, 1:-1, 0:-2, 1:-1])
    ky_p = 0.5 * (kyp[:, 1:-1, 1:-1, 1:-1] + kyp[:, 1:-1, 2:  , 1:-1])
    kz_m = 0.5 * (kzp[:, 1:-1, 1:-1, 1:-1] + kzp[:, 1:-1, 1:-1, 0:-2])
    kz_p = 0.5 * (kzp[:, 1:-1, 1:-1, 1:-1] + kzp[:, 1:-1, 1:-1, 2:  ])

    return kx_m + kx_p + ky_m + ky_p + kz_m + kz_p


def build_bjac_cache(kx: torch.Tensor, ky: torch.Tensor, kz: torch.Tensor, nb: int):
    n = kx.shape[0]
    assert n % nb == 0, f"n={n} 必须能被 nb={nb} 整除"
    bs = n // nb

    kx_blk = kx.view(nb, bs, n, n).contiguous()
    ky_blk = ky.view(nb, bs, n, n).contiguous()
    kz_blk = kz.view(nb, bs, n, n).contiguous()

    D_blk = diag_of_A_block_batch(kx_blk, ky_blk, kz_blk)
    invD_blk = 1.0 / D_blk
    return {"invD_blk": invD_blk}


def bjac_apply_vectorized(r: torch.Tensor,
                          kx: torch.Tensor, ky: torch.Tensor, kz: torch.Tensor,
                          nb: int, k_inter: int, t_intra: int,
                          invD_blk_cache: torch.Tensor) -> torch.Tensor:
    n = r.shape[0]
    assert n % nb == 0, f"n={n} 必须能被 nb={nb} 整除"
    bs = n // nb

    r_blk  = r.view(nb, bs, n, n).contiguous()
    kx_blk = kx.view(nb, bs, n, n).contiguous()
    ky_blk = ky.view(nb, bs, n, n).contiguous()
    kz_blk = kz.view(nb, bs, n, n).contiguous()

    z_blk = torch.zeros_like(r_blk)
    invD_blk = invD_blk_cache

    for _ in range(k_inter):
        res_blk = r_blk - apply_A_block_batch(z_blk, kx_blk, ky_blk, kz_blk)

        y_blk = torch.zeros_like(res_blk)
        for _ in range(t_intra):
            y_blk = y_blk + invD_blk * (res_blk - apply_A_block_batch(y_blk, kx_blk, ky_blk, kz_blk))

        z_blk = z_blk + y_blk

    return z_blk.reshape(n, n, n)


# ============================================================
# 预条件子
# ============================================================
def precond_uniform_bjac_vec(r, kx64, ky64, kz64, cfg: ExpConfig, cache64):
    return bjac_apply_vectorized(r, kx64, ky64, kz64,
                                 cfg.nb, cfg.k_inter, cfg.t_intra,
                                 invD_blk_cache=cache64["invD_blk"])


def precond_fmp_bjac_vec(r, kx32, ky32, kz32, cfg: ExpConfig, cache32, out_dtype):
    r32 = r.to(torch.float32)
    z32 = bjac_apply_vectorized(r32, kx32, ky32, kz32,
                                cfg.nb, cfg.k_inter, cfg.t_intra,
                                invD_blk_cache=cache32["invD_blk"])
    return z32.to(out_dtype)


def precond_amp_bjac_vec(r, relres, mode,
                         kx64, ky64, kz64, cache64,
                         kx32, ky32, kz32, cache32,
                         cfg: ExpConfig, out_dtype):
    if mode == "hl":
        use_high = (relres >= cfg.adp_tol_hl)
    elif mode == "lh":
        use_high = (relres < cfg.adp_tol_lh)
    else:
        raise ValueError(mode)

    if use_high:
        return precond_uniform_bjac_vec(r, kx64, ky64, kz64, cfg, cache64)
    else:
        return precond_fmp_bjac_vec(r, kx32, ky32, kz32, cfg, cache32, out_dtype=out_dtype)


# ============================================================
# PCG：修改版，记录每次迭代的残差
# ============================================================
@torch.no_grad()
def pcg_solve_with_history(case_name: str, method: str, cfg: ExpConfig,
                           kx64: torch.Tensor, ky64: torch.Tensor, kz64: torch.Tensor,
                           kx32: torch.Tensor, ky32: torch.Tensor, kz32: torch.Tensor,
                           cache64: Dict[str, torch.Tensor], cache32: Dict[str, torch.Tensor],
                           b: torch.Tensor) -> Tuple[List[float], int, float]:
    """
    返回：relres_history (每次迭代的相对残差列表), iters, time_sec
    """
    dev = cfg.device
    pw = cfg.working_dtype

    x = torch.zeros_like(b, dtype=pw, device=dev)
    r = b - apply_A_full(x, kx64, ky64, kz64)

    r0_norm = torch.linalg.norm(r).item()
    if r0_norm == 0.0:
        return [0.0], 0, 0.0

    relres = 1.0
    relres_history = [relres]  # 记录每次迭代的相对残差

    # 计时
    if dev.startswith("cuda"):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    else:
        t0 = time.perf_counter()

    # 初始预条件
    if method == "fp64-uniform-BJAC PCG":
        z = precond_uniform_bjac_vec(r, kx64, ky64, kz64, cfg, cache64)
    elif method == "fp32-fMP-BJAC PCG":
        z = precond_fmp_bjac_vec(r, kx32, ky32, kz32, cfg, cache32, out_dtype=pw)
    elif method == "fp32-aMP-BJAC(hl) PCG":
        z = precond_amp_bjac_vec(r, relres, "hl", kx64, ky64, kz64, cache64,
                                 kx32, ky32, kz32, cache32, cfg, out_dtype=pw)
    elif method == "fp32-aMP-BJAC(lh) PCG":
        z = precond_amp_bjac_vec(r, relres, "lh", kx64, ky64, kz64, cache64,
                                 kx32, ky32, kz32, cache32, cfg, out_dtype=pw)
    else:
        raise ValueError(f"未知 method: {method}")

    p = z.clone()
    rz_old = torch.sum(r * z)

    iters = 0
    for iters in range(cfg.max_iter):
        Ap = apply_A_full(p, kx64, ky64, kz64)
        alpha = rz_old / torch.sum(p * Ap)

        x = x + alpha * p
        r = r - alpha * Ap

        relres = torch.linalg.norm(r).item() / r0_norm
        relres_history.append(relres)  # 记录当前迭代的残差

        if relres < cfg.tol_res:
            break

        # 预条件更新
        if method == "fp64-uniform-BJAC PCG":
            z = precond_uniform_bjac_vec(r, kx64, ky64, kz64, cfg, cache64)
        elif method == "fp32-fMP-BJAC PCG":
            z = precond_fmp_bjac_vec(r, kx32, ky32, kz32, cfg, cache32, out_dtype=pw)
        elif method == "fp32-aMP-BJAC(hl) PCG":
            z = precond_amp_bjac_vec(r, relres, "hl", kx64, ky64, kz64, cache64,
                                     kx32, ky32, kz32, cache32, cfg, out_dtype=pw)
        elif method == "fp32-aMP-BJAC(lh) PCG":
            z = precond_amp_bjac_vec(r, relres, "lh", kx64, ky64, kz64, cache64,
                                     kx32, ky32, kz32, cache32, cfg, out_dtype=pw)

        rz_new = torch.sum(r * z)
        beta = rz_new / rz_old
        p = z + beta * p
        rz_old = rz_new

    # 结束计时
    if dev.startswith("cuda"):
        end.record()
        torch.cuda.synchronize()
        time_sec = start.elapsed_time(end) / 1000.0
    else:
        time_sec = time.perf_counter() - t0

    return relres_history, iters + 1, float(time_sec)


# ============================================================
# 绘图函数
# ============================================================
def plot_convergence_curves(convergence_data: Dict[str, List[float]],
                            case_name: str,
                            save_path: str = None):
    """
    绘制收敛曲线

    参数:
        convergence_data: {method_name: [relres_history]}
        case_name: 问题名称（用于标题）
        save_path: 保存路径
    """
    plt.figure(figsize=(14, 9))

    # 定义颜色和线型（红蓝绿实线 + 红蓝绿虚线）
    styles = {
        "fp32-aMP-BJAC(hl) PCG, adp_tol:10.0": {"color": "red", "linestyle": "-", "linewidth": 3.0},  # 红色实线
        "fp32-aMP-BJAC(hl) PCG, adp_tol:1.0": {"color": "blue", "linestyle": "-", "linewidth": 3.0},  # 蓝色实线
        "fp32-aMP-BJAC(hl) PCG, adp_tol:10^-1": {"color": "green", "linestyle": "-", "linewidth": 3.0},  # 绿色实线
        "fp32-aMP-BJAC(lh) PCG, adp_tol:10.0": {"color": "red", "linestyle": "--", "linewidth": 3.0},  # 红色虚线
        "fp32-aMP-BJAC(lh) PCG, adp_tol:1.0": {"color": "blue", "linestyle": "--", "linewidth": 3.0},  # 蓝色虚线
        "fp32-aMP-BJAC(lh) PCG, adp_tol:10^-1": {"color": "green", "linestyle": "--", "linewidth": 3.0},  # 绿色虚线
    }

    # 绘制每条曲线
    for method_name, history in convergence_data.items():
        style = styles.get(method_name, {"color": "black", "linestyle": "-", "linewidth": 2})
        iterations = range(len(history))
        plt.semilogy(iterations, history, label=method_name, **style)

    # 设置坐标轴
    plt.xlabel("Iteration", fontsize=16, fontweight='bold')
    plt.ylabel("RelResNorm", fontsize=16, fontweight='bold')
    plt.title(case_name, fontsize=18, fontweight='bold')
    plt.grid(True, alpha=0.3, which='both', linestyle='-', linewidth=0.5)
    plt.legend(fontsize=11, loc="upper right", framealpha=0.95, edgecolor='black')

    # 设置 y 轴范围（适配收敛阈值1e-15）
    plt.ylim([1e-16, 1e2])

    # 设置刻度标签大小
    plt.tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ 图片已保存到: {save_path}")
    else:
        plt.show()


def plot_convergence_curves_detail(convergence_data: Dict[str, List[float]],
                                   case_name: str,
                                   save_path: str = None,
                                   xlim: tuple = None,
                                   ylim: tuple = None):
    """
    绘制收敛曲线的局部细节图

    参数:
        convergence_data: {method_name: [relres_history]}
        case_name: 问题名称（用于标题）
        save_path: 保存路径
        xlim: x轴范围 (min, max)
        ylim: y轴范围 (min, max)
    """
    plt.figure(figsize=(14, 9))

    # 定义颜色和线型（红蓝绿实线 + 红蓝绿虚线）
    styles = {
        "fp32-aMP-BJAC(hl) PCG, adp_tol:10.0": {"color": "red", "linestyle": "-", "linewidth": 3.0},  # 红色实线
        "fp32-aMP-BJAC(hl) PCG, adp_tol:1.0": {"color": "blue", "linestyle": "-", "linewidth": 3.0},  # 蓝色实线
        "fp32-aMP-BJAC(hl) PCG, adp_tol:10^-1": {"color": "green", "linestyle": "-", "linewidth": 3.0},  # 绿色实线
        "fp32-aMP-BJAC(lh) PCG, adp_tol:10.0": {"color": "red", "linestyle": "--", "linewidth": 3.0},  # 红色虚线
        "fp32-aMP-BJAC(lh) PCG, adp_tol:1.0": {"color": "blue", "linestyle": "--", "linewidth": 3.0},  # 蓝色虚线
        "fp32-aMP-BJAC(lh) PCG, adp_tol:10^-1": {"color": "green", "linestyle": "--", "linewidth": 3.0},  # 绿色虚线
    }

    # 绘制每条曲线
    for method_name, history in convergence_data.items():
        style = styles.get(method_name, {"color": "black", "linestyle": "-", "linewidth": 2})
        iterations = range(len(history))
        # 简化标签，去掉前缀
        simplified_label = method_name.replace("fp32-aMP-BJAC(hl) PCG, adp_tol:", "aMP-BJAC(hl) PCG, adp_tol:")
        simplified_label = simplified_label.replace("fp32-aMP-BJAC(lh) PCG, adp_tol:", "aMP-BJAC(lh) PCG, adp_tol:")
        plt.semilogy(iterations, history, label=simplified_label, **style)

    # 设置坐标轴
    plt.xlabel("Iteration", fontsize=16, fontweight='bold')
    plt.ylabel("RelResNorm", fontsize=16, fontweight='bold')
    plt.title(case_name, fontsize=18, fontweight='bold')
    plt.grid(True, alpha=0.3, which='both', linestyle='-', linewidth=0.5)
    plt.legend(fontsize=11, loc="upper right", framealpha=0.95, edgecolor='black')

    # 设置坐标轴范围
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)

    # 设置刻度标签大小
    plt.tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 细节图已保存到: {save_path}")
    else:
        plt.show()


# ============================================================
# 单个配置的求解函数（用于多GPU并行）
# ============================================================
def solve_single_config(args):
    """
    在指定GPU上求解单个配置

    参数:
        args: (gpu_id, base_method, adp_tol_val, mode, cfg_dict, kx64, ky64, kz64, kx32, ky32, kz32, b, case_name)
    """
    gpu_id, base_method, adp_tol_val, mode, cfg_dict, case_name = args

    # 设置当前进程使用的GPU
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)

    # 重建配置对象
    cfg = ExpConfig(**cfg_dict)
    cfg.device = device

    # 重新生成数据（每个进程独立生成，避免跨进程传递大张量）
    print(f"[GPU {gpu_id}] 生成系数场: {case_name}...")

    # 根据问题类型设置参数
    if case_name.startswith("RHD3D"):
        s = 1.0  # RHD问题不需要强度参数
    else:
        s = 1000.0  # Diff3D问题的强度参数

    seed = 0
    kx64, ky64, kz64 = make_kappa(case_name, cfg.n, s, seed, device=device, dtype=cfg.working_dtype)
    kx32, ky32, kz32 = kx64.to(torch.float32), ky64.to(torch.float32), kz64.to(torch.float32)

    # 预条件缓存
    cache64 = build_bjac_cache(kx64, ky64, kz64, cfg.nb)
    cache32 = build_bjac_cache(kx32, ky32, kz32, cfg.nb)

    # RHS
    b = torch.ones((cfg.n, cfg.n, cfg.n), device=device, dtype=cfg.working_dtype)

    # 临时修改配置
    original_tol_hl = cfg.adp_tol_hl
    original_tol_lh = cfg.adp_tol_lh

    if mode == "hl":
        cfg.adp_tol_hl = adp_tol_val
        tol_str = f"{adp_tol_val}" if adp_tol_val >= 1 else f"10^{int(np.log10(adp_tol_val))}"
        method_label = f"{base_method}, adp_tol:{tol_str}"
    else:
        cfg.adp_tol_lh = adp_tol_val
        tol_str = f"{adp_tol_val}" if adp_tol_val >= 1 else f"10^{int(np.log10(adp_tol_val))}"
        method_label = f"{base_method}, adp_tol:{tol_str}"

    print(f"[GPU {gpu_id}] 运行: {method_label}")
    history, iters, time_sec = pcg_solve_with_history(
        case_name, base_method, cfg,
        kx64, ky64, kz64,
        kx32, ky32, kz32,
        cache64, cache32,
        b
    )

    print(f"[GPU {gpu_id}] 完成: {method_label}, iters={iters}, time={time_sec:.3f}s, final_relres={history[-1]:.3e}")

    # 恢复配置
    cfg.adp_tol_hl = original_tol_hl
    cfg.adp_tol_lh = original_tol_lh

    return method_label, history


# ============================================================
# 主程序：测试不同的自适应阈值（多GPU并行版本）
# ============================================================
def main():
    # 检查GPU数量
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("错误：没有检测到可用的GPU")
        return

    print(f"检测到 {num_gpus} 张GPU，将使用多GPU并行计算")

    # ========================================
    # 选择测试问题
    # ========================================
    print("\n" + "=" * 80)
    print("可选测试问题（根据论文Figure 3和Figure 4）：")
    print("=" * 80)
    print("1. Diff3D-Const        - 常数系数         (单尺度, ~148次迭代, 有收敛延迟)")
    print("2. Diff3D-Ani(1000)    - 各向异性         (强多尺度, ~327次迭代, 无收敛延迟)")
    print("3. Diff3D-Dis(1000)    - 不连续系数       (弱多尺度, ~185次迭代, 有收敛延迟)")
    print("4. Diff3D-Rand(1000)   - 随机系数         (强多尺度, ~226次迭代, 无收敛延迟)")
    print("5. RHD3D-1T            - 单温度辐射流体   (弱多尺度, ~628次迭代, 有收敛延迟) ⭐")
    print("6. RHD3D-3T            - 三温度辐射流体   (强多尺度, ~1305次迭代, 无收敛延迟)")
    print("=" * 80)

    # 修改此处来选择不同的测试问题
    case_name = "Diff3D-Ani(1000)"  # ⭐ 修改这里来切换问题
    display_name = case_name

    print(f"\n✓ 当前测试问题: {case_name}")

    # 基础配置
    cfg = ExpConfig(device="cuda:0")  # 临时设备，会在子进程中重新设置

    # 注意：论文中RHD3D使用fp80精度，但PyTorch GPU不支持
    # 这里使用fp64作为近似
    if case_name.startswith("RHD3D"):
        print("\n⚠️  重要说明：")
        print("    - 论文中RHD3D使用 fp80 (long double) 精度")
        print("    - PyTorch GPU 不支持 float128，当前使用 fp64 作为近似")
        print("    - 迭代次数可能与论文略有差异（但收敛行为应该相似）")

    # 将配置转换为字典（用于跨进程传递）
    cfg_dict = {
        'n': cfg.n,
        'tol_res': cfg.tol_res,
        'max_iter': cfg.max_iter,
        'working_dtype': cfg.working_dtype,
        'low_dtype': cfg.low_dtype,
        'nb': cfg.nb,
        'k_inter': cfg.k_inter,
        't_intra': cfg.t_intra,
        'adp_tol_hl': cfg.adp_tol_hl,
        'adp_tol_lh': cfg.adp_tol_lh,
        'device': 'cuda:0'  # 占位符
    }

    # 测试配置列表
    test_configs = [
        # aMP-BJAC(hl) 不同阈值
        ("fp32-aMP-BJAC(hl) PCG", 10.0, "hl"),
        ("fp32-aMP-BJAC(hl) PCG", 1.0, "hl"),
        ("fp32-aMP-BJAC(hl) PCG", 0.1, "hl"),
        # aMP-BJAC(lh) 不同阈值
        ("fp32-aMP-BJAC(lh) PCG", 10.0, "lh"),
        ("fp32-aMP-BJAC(lh) PCG", 1.0, "lh"),
        ("fp32-aMP-BJAC(lh) PCG", 0.1, "lh"),
    ]

    # 为每个任务分配GPU
    tasks = []
    for idx, (base_method, adp_tol_val, mode) in enumerate(test_configs):
        gpu_id = idx % num_gpus  # 循环分配GPU
        tasks.append((gpu_id, base_method, adp_tol_val, mode, cfg_dict, case_name))

    print(f"\n开始并行计算，共 {len(tasks)} 个任务...")
    print("=" * 80)

    # 使用进程池并行执行
    with Pool(processes=min(num_gpus, len(tasks))) as pool:
        results = pool.map(solve_single_config, tasks)

    # 收集结果
    convergence_data = {}
    for method_label, history in results:
        convergence_data[method_label] = history

    print("\n" + "=" * 80)
    print("所有任务完成！")

    # 绘制收敛曲线（全局图）
    os.makedirs("out", exist_ok=True)
    plot_convergence_curves(
        convergence_data,
        case_name=display_name,
        save_path=f"out/{display_name}_convergence.png"
    )

    # 绘制收敛曲线（局部细节图）- 适配收敛阈值1e-15
    # 根据问题类型自动调整细节图范围
    if case_name.startswith("RHD3D-1T"):
        # RHD3D-1T: 观察收敛后期的详细行为
        max_iters = max(len(history) for history in convergence_data.values())
        detail_xlim = (max_iters * 2 // 3, max_iters)  # 观察最后1/3的迭代
        detail_ylim = (1e-18, 1e-10)
    elif case_name.startswith("RHD3D-3T"):
        max_iters = max(len(history) for history in convergence_data.values())
        detail_xlim = (max_iters * 2 // 3, max_iters)
        detail_ylim = (1e-18, 1e-10)
    else:
        # Diff3D问题
        max_iters = max(len(history) for history in convergence_data.values())
        detail_xlim = (max_iters * 2 // 3, max_iters)
        detail_ylim = (1e-18, 1e-10)

    plot_convergence_curves_detail(
        convergence_data,
        case_name=display_name,
        save_path=f"out/{display_name}_convergence_detail.png",
        xlim=detail_xlim,
        ylim=detail_ylim
    )


if __name__ == "__main__":
    # 多进程必须在 if __name__ == "__main__" 保护下启动
    main()
