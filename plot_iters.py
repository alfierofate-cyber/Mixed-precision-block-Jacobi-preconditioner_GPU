
"""
生成迭代次数对比柱状图（论文中图4的风格）
在RHD3D-1T和RHD3D-3T问题上运行实际实验
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 配置参数
# ============================================================================
n: int = 128  # 网格大小
tol_res: float = 1e-10  # 收敛阈值
max_iter: int = 20000# 最大迭代次数
num_gpus: int = 4# 可用GPU数量

# ============================================================================
# 矩阵生成函数（RHD方程）
# ============================================================================
def make_kappa(case_name: str, n: int, device: str, dtype: torch.dtype, seed: int = 42):
    """为RHD方程生成系数场"""
    g = torch.Generator(device=device).manual_seed(seed)

    # RHD3D-1T: 单温辐射流体力学（弱多尺度）
    if case_name.startswith("RHD3D-1T"):
        # 99.85%的行在[1,10)区间内，0.15%可达到~20
        base_coeff = 1.0
        perturbation = 1.0 + 8.0 * torch.rand((n, n, n), device=device, dtype=dtype, generator=g)
        mask = torch.rand((n, n, n), device=device, dtype=dtype, generator=g) > 0.9985
        perturbation[mask] = perturbation[mask] * 2.0
        k = base_coeff * perturbation
        return k, k, k

    # RHD3D-3T: 三温模型（强多尺度）
    if case_name.startswith("RHD3D-3T"):
        delta = torch.rand((n, n, n), device=device, dtype=dtype, generator=g)
        base = torch.rand((n, n, n), device=device, dtype=dtype, generator=g)
        base = base * 9.0 + 1.0  # [1, 10)

        k = torch.ones_like(delta)
        k[delta < 0.0824] = base[delta < 0.0824] * 1.5  # [1, 10)

        mask = (delta >= 0.0824) & (delta < 0.0824 + 0.0206)
        k[mask] = base[mask] * 15.0  # [1e1, 1e2)

        mask = (delta >= 0.0824 + 0.0206) & (delta < 0.0824 + 0.0206 + 0.0206)
        k[mask] = base[mask] * 150.0  # [1e2, 1e3)

        # 继续处理更高尺度直到1e18
        mask = (delta >= 0.0824 + 3*0.0206) & (delta < 0.0824 + 4*0.0206)
        k[mask] = base[mask] * 1.5e4  # [1e4, 1e5)

        mask = (delta >= 0.0824 + 4*0.0206) & (delta < 0.0824 + 5*0.0206)
        k[mask] = base[mask] * 1.5e5  # [1e5, 1e6)

        mask = (delta >= 0.0824 + 5*0.0206) & (delta < 0.0824 + 6*0.0206)
        k[mask] = base[mask] * 1.5e6  # [1e6, 1e7)

        mask = delta >= 0.0824 + 6*0.0206
        k[mask] = base[mask] * 1.5e17  # [1e17, 1e18)

        return k, k, k

    raise ValueError(f"Unknown case: {case_name}")

def build_A_mv(kx, ky, kz, n: int, h: float, device: str):
    """构建7点模板的矩阵-向量乘积函数"""
    def matvec(x):
        x3d = x.view(n, n, n)
        out = torch.zeros_like(x3d)
        h2 = h * h

        # 内部点
        out[1:-1, 1:-1, 1:-1] = (
            (kx[1:-1, 1:-1, 1:-1] + kx[2:, 1:-1, 1:-1]) * (x3d[2:, 1:-1, 1:-1] - x3d[1:-1, 1:-1, 1:-1]) / h2 -
            (kx[1:-1, 1:-1, 1:-1] + kx[:-2, 1:-1, 1:-1]) * (x3d[1:-1, 1:-1, 1:-1] - x3d[:-2, 1:-1, 1:-1]) / h2 +
            (ky[1:-1, 1:-1, 1:-1] + ky[1:-1, 2:, 1:-1]) * (x3d[1:-1, 2:, 1:-1] - x3d[1:-1, 1:-1, 1:-1]) / h2 -
            (ky[1:-1, 1:-1, 1:-1] + ky[1:-1, :-2, 1:-1]) * (x3d[1:-1, 1:-1, 1:-1] - x3d[1:-1, :-2, 1:-1]) / h2 +
            (kz[1:-1, 1:-1, 1:-1] + kz[1:-1, 1:-1, 2:]) * (x3d[1:-1, 1:-1, 2:] - x3d[1:-1, 1:-1, 1:-1]) / h2 -
            (kz[1:-1, 1:-1, 1:-1] + kz[1:-1, 1:-1, :-2]) * (x3d[1:-1, 1:-1, 1:-1] - x3d[1:-1, 1:-1, :-2]) / h2
        )
        return -out.view(-1)

    return matvec

def block_jacobi_preconditioner(kx, ky, kz, n: int, h: float, nb: int, device: str, dtype_prec: torch.dtype):
    """构建块Jacobi预条件子"""
    num_blocks_per_dim = n // nb
    h2 = h * h

    # 填充系数以便访问边界
    kx_pad = torch.zeros((n+1, n, n), device=device, dtype=kx.dtype)
    ky_pad = torch.zeros((n, n+1, n), device=device, dtype=ky.dtype)
    kz_pad = torch.zeros((n, n, n+1), device=device, dtype=kz.dtype)

    kx_pad[:-1, :, :] = kx
    kx_pad[-1, :, :] = kx[-1, :, :]
    ky_pad[:, :-1, :] = ky
    ky_pad[:, -1, :] = ky[:, -1, :]
    kz_pad[:, :, :-1] = kz
    kz_pad[:, :, -1] = kz[:, :, -1]

    blocks_inv = []
    for i in range(num_blocks_per_dim):
        for j in range(num_blocks_per_dim):
            for k in range(num_blocks_per_dim):
                i_start, i_end = i * nb, (i + 1) * nb
                j_start, j_end = j * nb, (j + 1) * nb
                k_start, k_end = k * nb, (k + 1) * nb

                local_kx = kx_pad[i_start:i_end+1, j_start:j_end, k_start:k_end].to(dtype_prec)
                local_ky = ky_pad[i_start:i_end, j_start:j_end+1, k_start:k_end].to(dtype_prec)
                local_kz = kz_pad[i_start:i_end, j_start:j_end, k_start:k_end+1].to(dtype_prec)

                D_block = torch.zeros((nb, nb, nb), device=device, dtype=dtype_prec)
                for ii in range(nb):
                    for jj in range(nb):
                        for kk in range(nb):
                            D_block[ii, jj, kk] = (
                                (local_kx[ii, jj, kk] + local_kx[ii+1, jj, kk]) / h2 +
                                (local_ky[ii, jj, kk] + local_ky[ii, jj+1, kk]) / h2 +
                                (local_kz[ii, jj, kk] + local_kz[ii, jj, kk+1]) / h2
                            )

                D_block_inv = 1.0 / D_block
                blocks_inv.append(D_block_inv.to(torch.float32))

    def apply_preconditioner(r):
        r3d = r.view(n, n, n)
        z3d = torch.zeros_like(r3d)
        block_idx = 0
        for i in range(num_blocks_per_dim):
            for j in range(num_blocks_per_dim):
                for k in range(num_blocks_per_dim):
                    i_start, i_end = i * nb, (i + 1) * nb
                    j_start, j_end = j * nb, (j + 1) * nb
                    k_start, k_end = k * nb, (k + 1) * nb
                    z3d[i_start:i_end, j_start:j_end, k_start:k_end] = (
                        r3d[i_start:i_end, j_start:j_end, k_start:k_end] * blocks_inv[block_idx]
                    )
                    block_idx += 1
        return z3d.view(-1)

    return apply_preconditioner

# ============================================================================
# PCG求解器
# ============================================================================
def pcg_solve(A_mv, M_inv, b, x0, tol: float, max_iter: int, adp_tol: float = None, mode: str = "hl", M_inv_alt=None):
    """预条件共轭梯度求解器

    参数:
        A_mv: 矩阵-向量乘积函数
        M_inv: 主预条件子（固定精度方法）或初始预条件子（自适应方法）
        b: 右侧向量
        x0: 初始解
        tol: 收敛容差
        max_iter: 最大迭代次数
        adp_tol: 自适应切换容差（仅用于aMP方法）
        mode: 切换模式 "hl"（高到低）或 "lh"（低到高）
        M_inv_alt: 备选预条件子（仅用于aMP方法）
    """
    x = x0.clone()
    r = b - A_mv(x)
    z = M_inv(r)
    p = z.clone()

    rz_old = torch.dot(r, z)

    for it in range(max_iter):
        Ap = A_mv(p)
        alpha = rz_old / torch.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap

        # 检查收敛性
        res_norm = torch.norm(r).item()
        if res_norm < tol:
            return x, it + 1

        # 自适应精度切换
        if adp_tol is not None and M_inv_alt is not None:
            # hl模式：高精度(M_inv)开始，残差变小后切换到低精度(M_inv_alt)
            if mode == "hl":
                if res_norm < adp_tol:
                    z = M_inv_alt(r)  # 切换到低精度
                else:
                    z = M_inv(r)      # 使用高精度
            # lh模式：低精度(M_inv)开始，残差变小后切换到高精度(M_inv_alt)
            elif mode == "lh":
                if res_norm < adp_tol:
                    z = M_inv_alt(r)  # 切换到高精度
                else:
                    z = M_inv(r)      # 使用低精度
            else:
                z = M_inv(r)
        else:
            # 固定精度方法
            z = M_inv(r)

        rz_new = torch.dot(r, z)
        beta = rz_new / rz_old
        p = z + beta * p
        rz_old = rz_new

    return x, max_iter

# ============================================================================
# 单一配置求解器
# ============================================================================
def solve_single_config(args):
    """求解单个配置并返回迭代次数"""
    gpu_id, problem, method, adp_tol_val, mode = args

    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)

    try:
        # 生成问题
        kx, ky, kz = make_kappa(problem, n, device, torch.float64, seed=42)
        h = 1.0 / n

        # 构建矩阵-向量乘积
        A_mv = build_A_mv(kx, ky, kz, n, h, device)

        # 生成右侧向量
        g = torch.Generator(device=device).manual_seed(123)
        b = torch.randn(n*n*n, device=device, dtype=torch.float64, generator=g)
        x0 = torch.zeros_like(b)

        # 构建预条件子
        nb = 32

        if "aMP" in method:
            # 自适应混合精度方法：需要构建两个预条件子
            M_inv_high = block_jacobi_preconditioner(kx, ky, kz, n, h, nb, device, torch.float64)
            M_inv_low = block_jacobi_preconditioner(kx, ky, kz, n, h, nb, device, torch.float32)

            # 根据模式确定初始和备选预条件子
            if mode == "hl":
                # 高到低：从高精度开始，低精度作为备选
                _, iters = pcg_solve(A_mv, M_inv_high, b, x0, tol_res, max_iter, adp_tol_val, mode, M_inv_low)
            else:  # mode == "lh"
                # 低到高：从低精度开始，高精度作为备选
                _, iters = pcg_solve(A_mv, M_inv_low, b, x0, tol_res, max_iter, adp_tol_val, mode, M_inv_high)
        else:
            # 固定精度方法
            if "fp64" in method or "fp80" in method:
                dtype_prec = torch.float64
            elif "fp32" in method:
                dtype_prec = torch.float32
            else:
                dtype_prec = torch.float64

            M_inv = block_jacobi_preconditioner(kx, ky, kz, n, h, nb, device, dtype_prec)
            _, iters = pcg_solve(A_mv, M_inv, b, x0, tol_res, max_iter)

        result = (problem, method, iters)

    finally:
        # 清理GPU内存
        if 'kx' in locals():
            del kx, ky, kz
        if 'b' in locals():
            del b, x0
        torch.cuda.empty_cache()

    return result

# ============================================================================
# 主执行
# ============================================================================
def collect_iteration_data():
    """收集所有方法-问题组合的迭代次数"""

    problems = ["RHD3D-1T", "RHD3D-3T"]

    # 定义测试配置
    tasks = []
    gpu_id = 0

    # 对每个问题
    for problem in problems:
        # fp64-uniform（基准）
        tasks.append((gpu_id % num_gpus, problem, "fp64-uniform", None, None))
        gpu_id += 1

        # fp32-fMP（固定混合精度）
        tasks.append((gpu_id % num_gpus, problem, "fp32-fMP", None, None))
        gpu_id += 1

        # fp32-aMP-BJAC(hl)使用不同的容差
        for adp_tol in [10.0, 1.0, 0.1]:
            tasks.append((gpu_id % num_gpus, problem, f"fp32-aMP(hl)-{adp_tol}", adp_tol, "hl"))
            gpu_id += 1

        # fp32-aMP-BJAC(lh)使用不同的容差
        for adp_tol in [10.0, 1.0, 0.1]:
            tasks.append((gpu_id % num_gpus, problem, f"fp32-aMP(lh)-{adp_tol}", adp_tol, "lh"))
            gpu_id += 1

    print(f"Running {len(tasks)} experiments on {num_gpus} GPUs...")

    # 并行运行 - 限制为num_gpus个进程以防止内存溢出
    # 使用chunksize=1以确保任务在工作进程可用时逐个分配
    with Pool(processes=num_gpus) as pool:
        results = pool.map(solve_single_config, tasks, chunksize=1)

    return results

def plot_iteration_comparison(results: List[Tuple[str, str, int]], output_path: str = "out/iteration_comparison.png"):
    """生成比较迭代次数的柱状图"""

    # 组织数据
    data = {}
    for problem, method, iters in results:
        if problem not in data:
            data[problem] = {}
        data[problem][method] = iters

    # 定义方法显示顺序和标签
    method_groups = {
        "fp64-uniform": ("fp64-uniform", "#1f77b4", ""),
        "fp32-fMP": ("fp32-fMP", "#ff7f0e", ""),
        "fp32-aMP(hl)-10.0": ("aMP(hl)-10", "#2ca02c", ""),
        "fp32-aMP(hl)-1.0": ("aMP(hl)-1", "#d62728", ""),
        "fp32-aMP(hl)-0.1": ("aMP(hl)-0.1", "#9467bd", ""),
        "fp32-aMP(lh)-10.0": ("aMP(lh)-10", "#8c564b", "//"),
        "fp32-aMP(lh)-1.0": ("aMP(lh)-1", "#e377c2", "//"),
        "fp32-aMP(lh)-0.1": ("aMP(lh)-0.1", "#7f7f7f", "//"),
    }

    problems = sorted(data.keys())
    methods = list(method_groups.keys())

    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(problems))
    width = 0.1

    # 绘制柱状图
    for i, method in enumerate(methods):
        label, color, hatch = method_groups[method]
        iters = [data[prob].get(method, 0) for prob in problems]
        offset = (i - len(methods)/2 + 0.5) * width
        bars = ax.bar(x + offset, iters, width, label=label, color=color,
                     edgecolor='black', linewidth=0.8, hatch=hatch)

        # 在柱状图顶部添加数值标签
        for bar, iter_val in zip(bars, iters):
            if iter_val > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(iter_val)}',
                       ha='center', va='bottom', fontsize=8, rotation=0)

    # 格式化
    ax.set_xlabel('Problem', fontsize=12, fontweight='bold')
    ax.set_ylabel('Iteration Count', fontsize=12, fontweight='bold')
    ax.set_title('Iteration Comparison: RHD3D Problems (tol=1e-10)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(problems, fontsize=11)
    ax.legend(ncol=4, loc='upper left', fontsize=9, frameon=True)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()

    # 保存
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved figure to: {output_path}")
    plt.show()

# ============================================================================
# 主入口
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("Generating Iteration Comparison (Figure 4 Style)")
    print("=" * 70)
    print(f"Grid size: {n}x{n}x{n}")
    print(f"Convergence threshold: {tol_res}")
    print(f"Max iterations: {max_iter}")
    print(f"Using {num_gpus} GPUs")
    print("=" * 70)

    # 收集数据
    results = collect_iteration_data()

    # 打印结果
    print("\n" + "=" * 70)
    print("Iteration Count Results:")
    print("=" * 70)
    for problem, method, iters in sorted(results):
        print(f"{problem:15s} | {method:25s} | {iters:5d} iterations")

    # 生成图表
    plot_iteration_comparison(results)

    print("\n" + "=" * 70)
    print("✓ Complete!")
    print("=" * 70)
