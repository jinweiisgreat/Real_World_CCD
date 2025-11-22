import matplotlib.pyplot as plt
import numpy as np

# 设置全局字体和样式
# plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.2

# 数据准备
x = np.arange(1, 11)

# ========================= CIFAR100 =========================
cifar_all_baseline = [78.42, 81.68, 76.35, 66.64, 60.29, 55.05, 51.33, 48.14, 45.23, 39.15]
cifar_all_happy = [83.91, 82.25, 78.72, 74.33, 72.64, 68.10, 66.68, 63.28, 62.66, 59.61]
cifar_all_concept = [86.09, 80.67, 79.49, 75.87, 73.45, 72.60, 69.93, 66.49, 64.76, 62.35]

cifar_known_baseline = [85.22, 84.18, 80.35, 77.74, 70.29, 65.02, 56.56, 53.14, 50.23, 49.15]
cifar_known_happy = [86.78, 84.27, 82.65, 76.88, 73.21, 70.36, 67.51, 64.98, 64.07, 61.81]
cifar_known_concept = [85.64, 84.91, 81.63, 77.74, 75.31, 71.52, 70.99, 68.41, 67.24, 66.62]

cifar_new_baseline = [65.22, 63.18, 55.35, 49.74, 35.29, 34.02, 30.56, 29.14, 27.23, 30.15]
cifar_new_happy = [85.20, 84.00, 67.60, 69.20, 64.60, 60.20, 53.40, 52.40, 51.40, 49.80]
cifar_new_concept = [86.60, 70.00, 65.80, 72.60, 70.40, 64.80, 69.00, 65.80, 64.00, 66.20]

# ========================= CUB =========================
cub_all_baseline = [80.92, 73.15, 67.45, 59.32, 50.76, 48.84, 47.21, 44.67, 39.53, 37.18]
cub_all_happy = [79.54, 72.64, 68.98, 61.14, 61.42, 59.69, 59.49, 58.27, 56.37, 54.58]
cub_all_concept = [80.45, 75.42, 74.61, 70.02, 69.87, 66.92, 66.02, 65.71, 63.49, 62.34]

cub_known_baseline = [82.18, 72.38, 68.12, 60.45, 60.23, 55.12, 53.18, 47.54, 45.31, 43.06]
cub_known_happy = [81.28, 76.84, 75.91, 66.89, 64.85, 60.47, 59.94, 56.46, 51.27, 50.15]
cub_known_concept = [80.51, 74.98, 68.24, 68.03, 67.98, 66.40, 63.37, 62.07, 61.59, 62.04]

cub_new_baseline = [78.89, 79.24, 70.67, 59.52, 50.94, 45.17, 42.45, 37.83, 35.16, 33.74]
cub_new_happy = [80.75, 81.14, 78.23, 70.19, 61.91, 56.96, 53.28, 51.05, 49.2, 48.01]
cub_new_concept = [80.87, 80.13, 75.14, 71.47, 65.06, 64.83, 63.76, 59.3, 58.26, 56.8]

# SCI专业配色
colors = {
    'Baseline': '#1f77b4',  # 专业蓝
    'Happy': '#d62728',  # 专业红
    'CAPE': '#2ca02c'  # 专业绿
}

# 创建2行3列的子图
fig, axes = plt.subplots(2, 3, figsize=(16, 5.5), dpi=100)
fig.subplots_adjust(wspace=0.25, hspace=0.35)

# 数据集配置
datasets = [
    # 第一行：CIFAR100
    [
        ('ALL', cifar_all_baseline, cifar_all_happy, cifar_all_concept),
        ('Known', cifar_known_baseline, cifar_known_happy, cifar_known_concept),
        ('New', cifar_new_baseline, cifar_new_happy, cifar_new_concept)
    ],
    # 第二行：CUB
    [
        ('ALL', cub_all_baseline, cub_all_happy, cub_all_concept),
        ('Known', cub_known_baseline, cub_known_happy, cub_known_concept),
        ('New', cub_new_baseline, cub_new_happy, cub_new_concept)
    ]
]

dataset_names = ['CIFAR100', 'CUB']

# 绘制子图
for row in range(2):
    for col in range(3):
        ax = axes[row, col]
        title, baseline, happy, concept = datasets[row][col]

        # 设置淡蓝灰色背景
        ax.set_facecolor('#fafbfc')

        # 绘制线条
        ax.plot(x, baseline, marker='o', linewidth=2, markersize=6,
                color=colors['Baseline'], label='Baseline', alpha=0.8)
        ax.plot(x, happy, marker='s', linewidth=2, markersize=6,
                color=colors['Happy'], label='Happy', alpha=0.8)
        ax.plot(x, concept, marker='^', linewidth=2, markersize=6,
                color=colors['CAPE'], label='CAPE(Ours)', alpha=0.8)

        # 第一行显示title（ALL/Known/New），第二行不显示
        if row == 0:
            ax.set_title(title, fontsize=12, fontweight='bold', pad=10)

        # 设置X轴
        ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        ax.set_xlim(0.5, 10.5)

        # 第二行显示x轴标签，第一行不显示
        if row == 1:
            ax.set_xlabel('Online session', fontsize=11)

        # 只在每行第一个子图显示Y轴标签
        if col == 0:
            ax.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')

        # 设置Y轴范围
        ax.set_ylim(15, 90)

        # 网格线
        ax.grid(True, linestyle='--', alpha=0.4, linewidth=0.8)

        # 设置刻度
        ax.tick_params(direction='out', length=4, width=1.2)

        # 只在最后一个子图（右下角）添加图例
        if row == 1 and col == 2:
            ax.legend(loc='upper right', frameon=True, fancybox=False,
                      shadow=False, framealpha=0.9, edgecolor='black',
                      fontsize=10, ncol=1)

# 调整布局
plt.tight_layout()
plt.show()