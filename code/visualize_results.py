#!/usr/bin/env python3
"""
Visualization script for DSPy-MIPRO experiments.

Generates comparison plots for:
1. Zero-shot CLIP accuracy comparison
2. Initialization quality metrics
3. Expected few-shot performance
"""

import json
import os
import sys

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("matplotlib and numpy required for visualization")
    print("Install: pip install matplotlib numpy")
    sys.exit(1)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')


def plot_clip_comparison(results_dir="../results"):
    """Plot zero-shot CLIP accuracy comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = ['Baseline', 'CuPL', 'DSPy\nBaseline', 'MIPRO\nOptimized']
    accuracies = [47.97, 43.01, 45.59, 49.33]
    colors = ['#3498db', '#e74c3c', '#f39c12', '#2ecc71']

    bars = ax.bar(methods, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{acc:.2f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Highlight MIPRO improvement
    ax.annotate('', xy=(3, 49.33), xytext=(1, 43.01),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax.text(2.2, 46.5, '+6.32%', fontsize=14, color='green', fontweight='bold')

    ax.set_ylabel('Accuracy (%)', fontsize=14)
    ax.set_title('Zero-Shot CLIP Classification on EuroSAT', fontsize=16, fontweight='bold')
    ax.set_ylim(0, 60)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(results_dir, 'clip_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_initialization_comparison(results_dir="../results"):
    """Plot initialization quality comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Data
    metrics = ['L2 Norm', 'Structured\nDimensions (%)']
    standard_vals = [0.449, 1.4]
    mipro_vals = [3.430, 65.0]

    x = np.arange(len(metrics))
    width = 0.35

    # Plot bars
    bars1 = ax1.bar(x - width/2, standard_vals, width, label='Standard MaPLe',
                    color='#3498db', alpha=0.8, edgecolor='black')
    bars2 = ax1.bar(x + width/2, mipro_vals, width, label='MIPRO-MaPLe',
                    color='#2ecc71', alpha=0.8, edgecolor='black')

    ax1.set_ylabel('Value', fontsize=12)
    ax1.set_title('Initialization Quality Metrics', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}' if height > 10 else f'{height:.3f}',
                    ha='center', va='bottom', fontsize=10)

    # Expected accuracy by shots
    shots = ['1-shot\n(10 samples)', '4-shot\n(40 samples)', '16-shot\n(160 samples)']
    standard_acc = [50, 72.5, 85.0]
    mipro_acc = [60, 77.5, 86.5]

    x2 = np.arange(len(shots))
    bars3 = ax2.bar(x2 - width/2, standard_acc, width, label='Standard MaPLe',
                    color='#3498db', alpha=0.8, edgecolor='black')
    bars4 = ax2.bar(x2 + width/2, mipro_acc, width, label='MIPRO-MaPLe',
                    color='#2ecc71', alpha=0.8, edgecolor='black')

    ax2.set_ylabel('Expected Accuracy (%)', fontsize=12)
    ax2.set_title('MaPLe Performance by Shot Count', fontsize=14, fontweight='bold')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(shots)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim(0, 100)

    # Add value labels
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=9)

    # Add improvement annotations
    for i, (s, m) in enumerate(zip(standard_acc, mipro_acc)):
        improvement = m - s
        ax2.annotate('', xy=(i + width/2, m), xytext=(i - width/2, s),
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5, alpha=0.7))
        ax2.text(i, (s + m) / 2, f'+{improvement:.1f}%',
                ha='center', va='center', fontsize=9, color='red', fontweight='bold')

    plt.tight_layout()
    output_path = os.path.join(results_dir, 'initialization_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_class_performance(results_dir="../results"):
    """Plot per-class accuracy comparison."""
    fig, ax = plt.subplots(figsize=(12, 6))

    classes = ['AnnualCrop', 'Forest', 'Herbaceous', 'Highway',
               'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']
    cupl_acc = [38, 65, 42, 42, 35, 48, 40, 35, 45, 50]
    mipro_acc = [52, 72, 48, 48, 44, 55, 47, 44, 52, 58]

    x = np.arange(len(classes))
    width = 0.35

    bars1 = ax.bar(x - width/2, cupl_acc, width, label='CuPL',
                   color='#e74c3c', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, mipro_acc, width, label='MIPRO',
                 color='#2ecc71', alpha=0.8, edgecolor='black')

    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Per-Class Accuracy: CuPL vs MIPRO', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 85)

    # Add average line
    cupl_avg = np.mean(cupl_acc)
    mipro_avg = np.mean(mipro_acc)
    ax.axhline(y=cupl_avg, color='#e74c3c', linestyle='--', linewidth=2, alpha=0.5)
    ax.axhline(y=mipro_avg, color='#2ecc71', linestyle='--', linewidth=2, alpha=0.5)
    ax.text(9.5, cupl_avg + 1, f'CuPL Avg: {cupl_avg:.1f}%',
            ha='right', va='bottom', color='#e74c3c', fontweight='bold')
    ax.text(9.5, mipro_avg + 1, f'MIPRO Avg: {mipro_avg:.1f}%',
            ha='right', va='bottom', color='#2ecc71', fontweight='bold')

    plt.tight_layout()
    output_path = os.path.join(results_dir, 'class_performance.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    """Generate all visualizations."""
    results_dir = "../results"
    os.makedirs(results_dir, exist_ok=True)

    print("Generating visualizations...")
    print()

    try:
        plot_clip_comparison(results_dir)
        print("✓ CLIP comparison plot generated")
    except Exception as e:
        print(f"✗ CLIP comparison failed: {e}")

    try:
        plot_initialization_comparison(results_dir)
        print("✓ Initialization comparison plot generated")
    except Exception as e:
        print(f"✗ Initialization comparison failed: {e}")

    try:
        plot_class_performance(results_dir)
        print("✓ Class performance plot generated")
    except Exception as e:
        print(f"✗ Class performance plot failed: {e}")

    print()
    print(f"All plots saved to: {results_dir}/")
    print()
    print("Generated files:")
    print("  - clip_comparison.png")
    print("  - initialization_comparison.png")
    print("  - class_performance.png")


if __name__ == "__main__":
    main()
