#!/usr/bin/env python3
"""
绘制 Figure 6: PCG 收敛曲线（RelResNorm vs Iteration）
"""
import json
import matplotlib.pyplot as plt
import numpy as np

# 设置字体和样式（论文风格）
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 9
plt.rcParams['figure.dpi'] = 150

# 读取数据
with open('out/figure6_convergence_data.json', 'r') as f:
    results = json.load(f)

# 问题顺序（论文 Figure 6 的布局）
problems = [
    'Diff3D-Ani(1000)',
    'Diff3D-Ani(100)',
    'Diff3D-Ani(10)',
    'Diff3D-Ani(4)',
    'Diff3D-Ani(2)',
    'Diff3D-Const'
]

# 创建 3x2 子图布局
fig, axes = plt.subplots(3, 2, figsize=(12, 12))
axes = axes.flatten()

# 方法名称映射
method_labels = {
    'fp64-uniform-BJAC PCG': 'RelResNorm: fp64-uniform-BJAC PCG',
    'fp32-fMP-BJAC PCG': 'RelResNorm: fp32-fMP-BJAC PCG'
}

# 绘制每个子图
for idx, problem in enumerate(problems):
    ax = axes[idx]

    if problem not in results:
        ax.set_visible(False)
        continue

    prob_data = results[problem]

    # 绘制 fp64-uniform（蓝色实线）
    if 'fp64-uniform-BJAC PCG' in prob_data:
        history_uniform = prob_data['fp64-uniform-BJAC PCG']['relres_history']
        iterations_uniform = range(len(history_uniform))
        ax.semilogy(iterations_uniform, history_uniform,
                   'b-', linewidth=2,
                   label='RelResNorm: fp64-uniform-BJAC PCG')

    # 绘制 fp32-fMP（橙色虚线）
    if 'fp32-fMP-BJAC PCG' in prob_data:
        history_fmp = prob_data['fp32-fMP-BJAC PCG']['relres_history']
        iterations_fmp = range(len(history_fmp))
        ax.semilogy(iterations_fmp, history_fmp,
                   color='darkorange', linestyle='--', linewidth=2,
                   label='RelResNorm: fp32-fMP-BJAC PCG')

    # 绘制收敛阈值线（红色水平线 1e-10）
    ax.axhline(y=1e-10, color='red', linestyle=':', linewidth=1.5, alpha=0.8)
    ax.text(0.02, 1e-10, '1E-10', transform=ax.get_xaxis_transform(),
           ha='left', va='bottom', fontsize=8, color='red', fontweight='bold')

    # 设置坐标轴
    ax.set_xlabel('Iteration', fontsize=10, fontweight='bold')
    ax.set_ylabel('RelResNorm', fontsize=10, fontweight='bold')
    ax.set_title(problem, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(1e-20, 1e5)  # 扩展到 1e-20 以显示更深的收敛

    # 图例（只在第一个子图显示）
    if idx == 0:
        ax.legend(loc='upper right', fontsize=8, frameon=True)

    # 设置 X 轴范围（扩展以显示更多迭代）
    max_iter = max(
        len(prob_data.get('fp64-uniform-BJAC PCG', {}).get('relres_history', [])),
        len(prob_data.get('fp32-fMP-BJAC PCG', {}).get('relres_history', []))
    )
    ax.set_xlim(0, min(max_iter + 50, 1000))  # 扩展到 1000 以显示完整收敛

# 添加总标题
fig.suptitle('Fig. 6  RelResNorm curves of the PCG algorithm for solving the Diff3D-Ani(s) problem with s = 1000, 100, 10, 4, 2, 1',
            fontsize=11, fontweight='bold', y=0.995)

plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('figure6_convergence_curves.png', dpi=300, bbox_inches='tight')
print("✓ Figure 6 已保存: figure6_convergence_curves.png")
plt.close()


# ============================================================
# 生成统计信息
# ============================================================
print("\n" + "="*80)
print("Figure 6 收敛统计")
print("="*80)

for problem in problems:
    if problem not in results:
        continue

    print(f"\n【{problem}】")
    prob_data = results[problem]

    for method in ['fp64-uniform-BJAC PCG', 'fp32-fMP-BJAC PCG']:
        if method not in prob_data:
            continue

        data = prob_data[method]
        iters = data['iters']
        relres = data['relres_end']
        time_sec = data['time_sec']

        method_short = 'fp64-uniform' if 'uniform' in method else 'fp32-fMP'
        print(f"  {method_short:15s}: {iters:4d} iters, relres={relres:.3e}, time={time_sec:.3f}s")

print("\n" + "="*80)
