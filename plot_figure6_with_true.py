#!/usr/bin/env python3
"""
绘制完整的 Figure 6: 包含 RelResNorm 和 true RelResNorm
"""
import json
import matplotlib.pyplot as plt
import numpy as np

# 设置字体和样式
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 9
plt.rcParams['figure.dpi'] = 150

# 读取数据
with open('out/figure6_convergence_data_with_true.json', 'r') as f:
    results = json.load(f)

# 问题顺序
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

# 绘制每个子图
for idx, problem in enumerate(problems):
    ax = axes[idx]

    if problem not in results:
        ax.set_visible(False)
        continue

    prob_data = results[problem]

    # ========== fp64-uniform ==========
    if 'fp64-uniform-BJAC PCG' in prob_data:
        data_uniform = prob_data['fp64-uniform-BJAC PCG']

        # RelResNorm（蓝色实线）
        relres_hist = data_uniform['relres_history']
        iters = range(len(relres_hist))
        ax.semilogy(iters, relres_hist, 'b-', linewidth=2,
                   label='RelResNorm: fp64-uniform-BJAC PCG')

        # true RelResNorm（黄色虚线）
        true_relres_hist = data_uniform['true_relres_history']
        ax.semilogy(iters, true_relres_hist, color='gold', linestyle='--', linewidth=2,
                   label='true RelResNorm: fp64-uniform-BJAC PCG')

    # ========== fp32-fMP ==========
    if 'fp32-fMP-BJAC PCG' in prob_data:
        data_fmp = prob_data['fp32-fMP-BJAC PCG']

        # RelResNorm（橙色实线）
        relres_hist = data_fmp['relres_history']
        iters = range(len(relres_hist))
        ax.semilogy(iters, relres_hist, color='darkorange', linestyle='-', linewidth=2,
                   label='RelResNorm: fp32-fMP-BJAC PCG')

        # true RelResNorm（绿色虚线）
        true_relres_hist = data_fmp['true_relres_history']
        ax.semilogy(iters, true_relres_hist, color='lime', linestyle='--', linewidth=2,
                   label='true RelResNorm: fp32-fMP-BJAC PCG')

    # 绘制收敛阈值线（红色虚线）
    ax.axhline(y=1e-10, color='red', linestyle=':', linewidth=1.5, alpha=0.8)
    ax.text(0.02, 1e-10, '1E-10', transform=ax.get_xaxis_transform(),
           ha='left', va='bottom', fontsize=8, color='red', fontweight='bold')

    # 设置坐标轴
    ax.set_xlabel('Iteration', fontsize=10, fontweight='bold')
    ax.set_ylabel('RelResNorm', fontsize=10, fontweight='bold')
    ax.set_title(problem, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(1e-20, 1e5)

    # 图例（只在第一个子图显示）
    if idx == 0:
        ax.legend(loc='upper right', fontsize=7, frameon=True)

    # 设置 X 轴范围
    max_iter = 0
    for method_data in prob_data.values():
        max_iter = max(max_iter, len(method_data.get('relres_history', [])))
    ax.set_xlim(0, min(max_iter + 50, 1000))

# 添加总标题
fig.suptitle('Fig. 6  RelResNorm and true RelResNorm curves of the PCG algorithm for solving the Diff3D-Ani(s) problem with s = 1000, 100, 10, 4, 2, 1',
            fontsize=11, fontweight='bold', y=0.995)

plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('figure6_with_true_relresnorm.png', dpi=300, bbox_inches='tight')
print("✓ Figure 6 (完整版) 已保存: figure6_with_true_relresnorm.png")
plt.close()


# ============================================================
# 生成统计信息
# ============================================================
print("\n" + "="*80)
print("Figure 6 收敛统计（包含 true RelResNorm）")
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
        true_relres = data['true_relres_history'][-1]
        time_sec = data['time_sec']

        method_short = 'fp64-uniform' if 'uniform' in method else 'fp32-fMP'
        print(f"  {method_short:15s}: {iters:4d} iters, relres={relres:.3e}, true_relres={true_relres:.3e}, time={time_sec:.3f}s")

print("\n" + "="*80)
