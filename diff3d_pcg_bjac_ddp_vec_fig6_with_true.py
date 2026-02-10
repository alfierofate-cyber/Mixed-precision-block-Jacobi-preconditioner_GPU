#!/usr/bin/env python3
# diff3d_pcg_bjac_ddp_vec_fig6_with_true.py
# 完整版：同时记录 RelResNorm 和 true RelResNorm（支持多GPU并行）
import os
import time
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import torch
import torch.distributed as dist

# ============================================================
# 实验配置
# ============================================================
@dataclass
class ExpConfig:
    n: int = 128
    tol_res: float = 1e-18
    max_iter: int = 30000
    working_dtype: torch.dtype = torch.float64
    low_dtype: torch.dtype = torch.float32
    nb: int = 32
    k_inter: int = 2
    t_intra: int = 2
    adp_tol_hl: float = 1e-5
    adp_tol_lh: float = 1e-10
    device: str = "cuda"


# ============================================================
# 系数场生成
# ============================================================
def make_kappa(case_name: str, n: int, s: float, seed: int, device: str, dtype: torch.dtype):
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

    raise ValueError(f"未知 case_name: {case_name}")


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


def apply_A_block_batch(u_blk: torch.Tensor, kx_blk: torch.Tensor, ky_blk: torch.Tensor, kz_blk: torch.Tensor) -> torch.Tensor:
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


def diag_of_A_block_batch(kx_blk: torch.Tensor, ky_blk: torch.Tensor, kz_blk: torch.Tensor) -> torch.Tensor:
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


def bjac_apply_vectorized(r: torch.Tensor, kx: torch.Tensor, ky: torch.Tensor, kz: torch.Tensor,
                          nb: int, k_inter: int, t_intra: int, invD_blk_cache: torch.Tensor) -> torch.Tensor:
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


def precond_uniform_bjac_vec(r, kx64, ky64, kz64, cfg: ExpConfig, cache64):
    return bjac_apply_vectorized(r, kx64, ky64, kz64, cfg.nb, cfg.k_inter, cfg.t_intra,
                                 invD_blk_cache=cache64["invD_blk"])


def precond_fmp_bjac_vec(r, kx32, ky32, kz32, cfg: ExpConfig, cache32, out_dtype):
    r32 = r.to(torch.float32)
    z32 = bjac_apply_vectorized(r32, kx32, ky32, kz32, cfg.nb, cfg.k_inter, cfg.t_intra,
                                invD_blk_cache=cache32["invD_blk"])
    return z32.to(out_dtype)


# ============================================================
# 步骤 1：计算"真实解" x_true
# ============================================================
@torch.no_grad()
def compute_true_solution(case_name: str, cfg: ExpConfig,
                         kx64: torch.Tensor, ky64: torch.Tensor, kz64: torch.Tensor,
                         cache64: Dict[str, torch.Tensor], b: torch.Tensor) -> torch.Tensor:
    """
    用 fp64-uniform + 极严格收敛阈值计算"真实解"
    """
    print(f"  计算真实解 (tol=1e-20, fp64)...")

    dev = cfg.device
    pw = cfg.working_dtype

    x = torch.zeros_like(b, dtype=pw, device=dev)
    r = b - apply_A_full(x, kx64, ky64, kz64)
    r0_norm = torch.linalg.norm(r).item()

    if r0_norm == 0.0:
        return x

    z = precond_uniform_bjac_vec(r, kx64, ky64, kz64, cfg, cache64)
    p = z.clone()
    rz_old = torch.sum(r * z)

    # 使用极严格阈值求解
    tol_true = 1e-20
    max_iter_true = 50000

    for iters in range(max_iter_true):
        Ap = apply_A_full(p, kx64, ky64, kz64)
        alpha = rz_old / torch.sum(p * Ap)

        x = x + alpha * p
        r = r - alpha * Ap

        relres = torch.linalg.norm(r).item() / r0_norm
        if relres < tol_true:
            print(f"    真实解收敛: {iters+1} iters, relres={relres:.3e}")
            break

        z = precond_uniform_bjac_vec(r, kx64, ky64, kz64, cfg, cache64)
        rz_new = torch.sum(r * z)
        beta = rz_new / rz_old
        p = z + beta * p
        rz_old = rz_new

    return x


# ============================================================
# 步骤 2：PCG 求解，同时记录 RelResNorm 和 true RelResNorm
# ============================================================
@torch.no_grad()
def pcg_solve_with_true_history(case_name: str, method: str, cfg: ExpConfig,
                                kx64: torch.Tensor, ky64: torch.Tensor, kz64: torch.Tensor,
                                kx32: torch.Tensor, ky32: torch.Tensor, kz32: torch.Tensor,
                                cache64: Dict[str, torch.Tensor], cache32: Dict[str, torch.Tensor],
                                b: torch.Tensor, x_true: torch.Tensor) -> Dict[str, Any]:
    """
    返回：{"iters", "relres_end", "time_sec", "relres_history", "true_relres_history"}
    """
    dev = cfg.device
    pw = cfg.working_dtype

    x = torch.zeros_like(b, dtype=pw, device=dev)
    r = b - apply_A_full(x, kx64, ky64, kz64)

    r0_norm = torch.linalg.norm(r).item()
    x_true_norm = torch.linalg.norm(x_true).item()

    if r0_norm == 0.0:
        return {"iters": 0, "relres_end": 0.0, "time_sec": 0.0,
                "relres_history": [0.0], "true_relres_history": [0.0]}

    relres = 1.0
    true_relres = torch.linalg.norm(x - x_true).item() / x_true_norm

    relres_history = [relres]
    true_relres_history = [true_relres]

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

        # 计算两种残差
        relres = torch.linalg.norm(r).item() / r0_norm
        true_relres = torch.linalg.norm(x - x_true).item() / x_true_norm

        relres_history.append(relres)
        true_relres_history.append(true_relres)

        if relres < cfg.tol_res:
            break

        # 预条件更新
        if method == "fp64-uniform-BJAC PCG":
            z = precond_uniform_bjac_vec(r, kx64, ky64, kz64, cfg, cache64)
        elif method == "fp32-fMP-BJAC PCG":
            z = precond_fmp_bjac_vec(r, kx32, ky32, kz32, cfg, cache32, out_dtype=pw)

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

    return {
        "iters": iters + 1,
        "relres_end": float(relres),
        "time_sec": float(time_sec),
        "relres_history": relres_history,
        "true_relres_history": true_relres_history
    }


# ============================================================
# 分布式初始化
# ============================================================
def init_distributed():
    """
    返回：(rank, world_size, local_rank, is_distributed)
    """
    # 如果没有 torchrun 环境变量，就当单进程跑
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return 0, 1, 0, False

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if not dist.is_initialized():
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")

    return rank, world_size, local_rank, True


def gather_results(results_local: Dict, world_size: int, rank: int) -> Dict:
    """收集所有 rank 的结果"""
    if world_size == 1:
        return results_local

    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, results_local)

    # 合并结果
    if rank == 0:
        merged = {}
        for part in gathered:
            if part is not None:
                merged.update(part)
        return merged
    return {}


# ============================================================
# 主程序
# ============================================================
def main():
    # 初始化分布式
    rank, world_size, local_rank, is_dist = init_distributed()

    # 设置设备
    if torch.cuda.is_available():
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"

    cfg = ExpConfig(device=device)

    if rank == 0:
        print(f"运行模式: {'分布式 (world_size={})'.format(world_size) if is_dist else '单进程'}")
        print(f"设备: {device}\n")

    # Figure 6 需要的问题
    cases = [
        ("Diff3D-Ani(1000)", 1000.0, 0),
        ("Diff3D-Ani(100)", 100.0, 0),
        ("Diff3D-Ani(10)", 10.0, 0),
        ("Diff3D-Ani(4)", 4.0, 0),
        ("Diff3D-Ani(2)", 2.0, 0),
        ("Diff3D-Const", 1.0, 0),
    ]

    methods = [
        "fp64-uniform-BJAC PCG",
        "fp32-fMP-BJAC PCG",
    ]

    # 任务分配：每个 rank 负责不同的 case
    # 6 个问题分配到 world_size 个进程
    cases_for_rank = [cases[i] for i in range(len(cases)) if i % world_size == rank]

    if rank == 0:
        print(f"总共 {len(cases)} 个问题，分配到 {world_size} 个进程")
        for r in range(world_size):
            rank_cases = [cases[i][0] for i in range(len(cases)) if i % world_size == r]
            print(f"  Rank {r}: {rank_cases}")
        print()

    results_local = {}

    for case_name, s, seed in cases_for_rank:
        print(f"\n{'='*60}")
        print(f"[Rank {rank}] 运行 {case_name}")
        print(f"{'='*60}")
        results_local[case_name] = {}

        # 构造 kappa
        kx64, ky64, kz64 = make_kappa(case_name, cfg.n, s, seed, device=device, dtype=cfg.working_dtype)
        kx32, ky32, kz32 = kx64.to(torch.float32), ky64.to(torch.float32), kz64.to(torch.float32)

        # 预条件缓存
        cache64 = build_bjac_cache(kx64, ky64, kz64, cfg.nb)
        cache32 = build_bjac_cache(kx32, ky32, kz32, cfg.nb)

        # RHS
        b = torch.ones((cfg.n, cfg.n, cfg.n), device=device, dtype=cfg.working_dtype)

        # 步骤 1：计算真实解
        x_true = compute_true_solution(case_name, cfg, kx64, ky64, kz64, cache64, b)

        # 步骤 2：运行两种方法
        for method in methods:
            print(f"  {method}...")
            result = pcg_solve_with_true_history(
                case_name, method, cfg,
                kx64, ky64, kz64,
                kx32, ky32, kz32,
                cache64, cache32,
                b, x_true
            )
            results_local[case_name][method] = result
            print(f"    完成 | iters={result['iters']} relres={result['relres_end']:.3e} time={result['time_sec']:.3f}s")

    # 收集所有 rank 的结果
    if is_dist:
        dist.barrier()
        results = gather_results(results_local, world_size, rank)
    else:
        results = results_local

    # 只有 rank 0 保存结果
    if rank == 0:
        os.makedirs("out", exist_ok=True)
        with open("out/figure6_convergence_data_with_true.json", "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n✅ 结果已保存：out/figure6_convergence_data_with_true.json")

    # 清理分布式环境
    if is_dist:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
#(pytorch) chy@f10-17u-luoji-38:~/chy$ cd /home/chy/chy
#(pytorch) chy@f10-17u-luoji-38:~/chy$ ./run_fig6_multi_gpu.sh