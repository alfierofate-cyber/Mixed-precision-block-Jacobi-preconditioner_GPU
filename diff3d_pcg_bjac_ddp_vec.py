# diff3d_pcg_bjac_ddp_vec.py
import os # os 用于文件路径操作
import time # time 用于计时
from dataclasses import dataclass # dataclass 用于简化配置类定义
from typing import List, Dict, Any, Tuple # 类型注解
import pandas as pd # 用于数据处理
import torch # PyTorch 主库
import torch.distributed as dist # 分布式训练


# ============================================================
# 实验配置：按论文 Diff3D 设置
# ============================================================
@dataclass
class ExpConfig:
    # 网格：128^3
    n: int = 128

    # 相对残差收敛阈值（论文 Diff3D：1e-10）
    tol_res: float = 1e-10
    max_iter: int = 20000

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
# Diff3D 系数场 κ(x)：四个 case
# ============================================================
def make_kappa(case_name: str, n: int, s: float, seed: int, device: str, dtype: torch.dtype):
    """
    生成扩散系数场，返回 kx, ky, kz，形状均为 (n,n,n)
    """
    if case_name.startswith("Diff3D-Const"):
        kx = torch.ones((n, n, n), device=device, dtype=dtype)
        ky = torch.ones((n, n, n), device=device, dtype=dtype)
        kz = torch.ones((n, n, n), device=device, dtype=dtype)
        return kx, ky, kz

    if case_name.startswith("Diff3D-Ani"):
        # 各向异性：diag(1, s, s)
        kx = torch.ones((n, n, n), device=device, dtype=dtype)
        ky = torch.full((n, n, n), float(s), device=device, dtype=dtype)
        kz = torch.full((n, n, n), float(s), device=device, dtype=dtype)
        return kx, ky, kz

    if case_name.startswith("Diff3D-Dis"):
        # 不连续：中心立方体区域系数为 s，其余为 1
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
        # 随机：k = s^delta, delta ~ U[0,1]
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        delta = torch.rand((n, n, n), device=device, dtype=dtype, generator=g)
        k = float(s) ** delta
        return k, k, k

    raise ValueError(f"未知 case_name: {case_name}")


# ============================================================
# 全域 7 点 stencil：A(u) = -div(k grad u)，Dirichlet 边界（外部为 0）
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

    # 面系数：算术平均
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
# 向量化 BJAC：nb 块彻底并行
# 张量 reshape 为 (nb, bs, n, n)，一次性对 nb 做 stencil + Jacobi
# ============================================================
def apply_A_block_batch(u_blk: torch.Tensor,
                        kx_blk: torch.Tensor,
                        ky_blk: torch.Tensor,
                        kz_blk: torch.Tensor) -> torch.Tensor:
    """
    块内算子 A_bb(u)，块边界按块对角假设：块外邻居视为 0
    输入/输出形状： (nb, bs, n, n)
    """
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
    """
    计算块内对角 diag(A_bb)，输出形状 (nb, bs, n, n)
    """
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
    """
    为某个精度版本的 (kx,ky,kz) 预计算 invD_blk（块内对角逆）
    返回 dict：{"invD_blk": (nb,bs,n,n)}
    """
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
    """
    z ≈ M_bjac^{-1} r（向量化版本）
    invD_blk_cache: (nb,bs,n,n)，必须提供（避免重复算 diag）
    """
    n = r.shape[0]
    assert n % nb == 0, f"n={n} 必须能被 nb={nb} 整除"
    bs = n // nb

    r_blk  = r.view(nb, bs, n, n).contiguous()
    kx_blk = kx.view(nb, bs, n, n).contiguous()
    ky_blk = ky.view(nb, bs, n, n).contiguous()
    kz_blk = kz.view(nb, bs, n, n).contiguous()

    z_blk = torch.zeros_like(r_blk)
    invD_blk = invD_blk_cache

    # k 次块间迭代（块 Jacobi）
    for _ in range(k_inter):
        res_blk = r_blk - apply_A_block_batch(z_blk, kx_blk, ky_blk, kz_blk)

        # t 次块内 Jacobi 近似求解
        y_blk = torch.zeros_like(res_blk)
        for _ in range(t_intra):
            y_blk = y_blk + invD_blk * (res_blk - apply_A_block_batch(y_blk, kx_blk, ky_blk, kz_blk))

        z_blk = z_blk + y_blk

    return z_blk.reshape(n, n, n)


# ============================================================
# 预条件子：uniform / fMP / aMP(hl) / aMP(lh)
# 关键点：fp32 的 kappa 与 invD_blk 都要缓存，否则很难加速
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
# PCG：工作精度 fp64；预条件子按 method 选择（论文风格）
# ============================================================
@torch.no_grad()
def pcg_solve(case_name: str, method: str, cfg: ExpConfig,
              kx64: torch.Tensor, ky64: torch.Tensor, kz64: torch.Tensor,
              kx32: torch.Tensor, ky32: torch.Tensor, kz32: torch.Tensor,
              cache64: Dict[str, torch.Tensor], cache32: Dict[str, torch.Tensor],
              b: torch.Tensor) -> Tuple[int, float, float]:
    """
    返回：iters, relres_end, time_sec
    method:
      - "fp64-uniform-BJAC PCG"
      - "fp32-fMP-BJAC PCG"
      - "fp32-aMP-BJAC(hl) PCG"
      - "fp32-aMP-BJAC(lh) PCG"
    """
    dev = cfg.device
    pw = cfg.working_dtype

    x = torch.zeros_like(b, dtype=pw, device=dev)
    r = b - apply_A_full(x, kx64, ky64, kz64)

    r0_norm = torch.linalg.norm(r).item()
    if r0_norm == 0.0:
        return 0, 0.0, 0.0

    relres = 1.0

    # 计时（GPU event）
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

    return iters + 1, float(relres), float(time_sec)


# ============================================================
# 分布式初始化：既支持 torchrun，也支持单进程直接 python 运行
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


def gather_results(rows_local: List[Dict[str, Any]], world_size: int) -> List[Dict[str, Any]]:
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, rows_local)
    rows = []
    for part in gathered:
        rows.extend(part)
    return rows


# ============================================================
# 主程序：Diff3D 四个 case × 4 种 method
# ============================================================
def main():
    rank, world_size, local_rank, is_dist = init_distributed()

    # 设备
    if torch.cuda.is_available():
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"

    cfg = ExpConfig(device=device)

    # Diff3D 四个 case（论文：s=1000）
    cases = [
        ("Diff3D-Const", 1000.0, 0),
        ("Diff3D-Ani(1000)", 1000.0, 0),
        ("Diff3D-Dis(1000)", 1000.0, 0),
        ("Diff3D-Rand(1000)", 1000.0, 0),
    ]

    # 论文 4 种方法顺序（用于后续画图/对照）
    methods = [
        "fp64-uniform-BJAC PCG",
        "fp32-fMP-BJAC PCG",
        "fp32-aMP-BJAC(hl) PCG",
        "fp32-aMP-BJAC(lh) PCG",
    ]

    # 任务列表：case × method
    tasks = []
    for case_name, s, seed in cases:
        for m in methods:
            tasks.append((case_name, s, seed, m))

    # 分配到各 rank：按 index % world_size
    tasks_rank = [t for i, t in enumerate(tasks) if (i % world_size) == rank]

    rows_local = []
    for case_name, s, seed, method in tasks_rank:
        # 构造 kappa：工作精度 fp64
        kx64, ky64, kz64 = make_kappa(case_name, cfg.n, s, seed, device=device, dtype=cfg.working_dtype)

        # 低精度版本只转换一次（关键：避免迭代里反复 to(fp32)）
        kx32, ky32, kz32 = kx64.to(torch.float32), ky64.to(torch.float32), kz64.to(torch.float32)

        # 预条件缓存：invD_blk 只算一次（关键）
        cache64 = build_bjac_cache(kx64, ky64, kz64, cfg.nb)
        cache32 = build_bjac_cache(kx32, ky32, kz32, cfg.nb)

        # RHS：这里用全 1（用于性能/收敛对比）
        b = torch.ones((cfg.n, cfg.n, cfg.n), device=device, dtype=cfg.working_dtype)

        print(f"[rank{rank}] 运行 {case_name} / {method} ...", flush=True)
        iters, relres_end, time_sec = pcg_solve(
            case_name, method, cfg,
            kx64, ky64, kz64,
            kx32, ky32, kz32,
            cache64, cache32,
            b
        )

        rows_local.append({
            "problem": case_name,
            "method": method,
            "n": cfg.n,
            "s": s,
            "seed": seed,
            "nb": cfg.nb,
            "k": cfg.k_inter,
            "t": cfg.t_intra,
            "tol_res": cfg.tol_res,
            "adp_tol_hl": cfg.adp_tol_hl,
            "adp_tol_lh": cfg.adp_tol_lh,
            "working_precision": "fp64",
            "low_precision": "fp32",
            "iters": iters,
            "relres_end": relres_end,
            "time_sec": time_sec,
        })

        print(f"[rank{rank}] 完成 {case_name} / {method} | iters={iters} relres={relres_end:.3e} time={time_sec:.3f}s",
              flush=True)

    # 汇总并写 CSV（rank0）
    if is_dist:
        rows_all = gather_results(rows_local, world_size)
    else:
        rows_all = rows_local

    if rank == 0:
        os.makedirs("out", exist_ok=True)
        df = pd.DataFrame(rows_all)

        # 固定顺序：按论文 case 顺序 + method 顺序
        prob_order = [c[0] for c in cases]
        meth_order = methods
        df["problem"] = pd.Categorical(df["problem"], categories=prob_order, ordered=True)
        df["method"] = pd.Categorical(df["method"], categories=meth_order, ordered=True)
        df = df.sort_values(["problem", "method"]).reset_index(drop=True)

        out_csv = "out/diff3d_results_bjac_nb32_k2_t2_vec.csv"
        df.to_csv(out_csv, index=False)
        print(f"[rank0] 结果已保存：{out_csv}", flush=True)

    if is_dist:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
