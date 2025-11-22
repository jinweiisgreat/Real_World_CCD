import matplotlib.pyplot as plt
import numpy as np

# 设置全局字体和样式 - 单栏适配
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.2

# 数据准备 - 3个shuffle的Novel数据
shuffle1_baseline = [51.24, 46.34, 40.2, 39.32, 32.25]
shuffle1_happy = [50.24, 51.25, 46.39, 45.64, 39.8]
shuffle1_concept = [66.1, 60.24, 53.86, 52.22, 47.26]

shuffle2_baseline = [50.43, 47.22, 43.63, 33.14, 29.12]
shuffle2_happy = [66.54, 63.45, 57.24, 58.22, 56.43]
shuffle2_concept = [73.02, 68, 65.12, 62.02, 60.12]

shuffle3_baseline = [43.64, 37.22, 38.92, 29.42, 25.42]
shuffle3_happy = [55.52, 56.12, 49.32, 51.25, 47.41]
shuffle3_concept = [69.02, 57.35, 59.52, 50.23, 53.33]

# 组织数据：对于每个session，收集3个shuffle的结果
sessions = 5
baseline_data = []
happy_data = []
concept_data = []

for i in range(sessions):
    baseline_data.append([shuffle1_baseline[i], shuffle2_baseline[i], shuffle3_baseline[i]])
    happy_data.append([shuffle1_happy[i], shuffle2_happy[i], shuffle3_happy[i]])
    concept_data.append([shuffle1_concept[i], shuffle2_concept[i], shuffle3_concept[i]])

# SCI专业配色改为IEEE专业配色
colors = {
    'Baseline': '#00429d',      # IEEE深蓝
    'Happy': '#93003a',         # IEEE深红
    'ConceptCCD': '#2e7d32'     # IEEE深绿
}

# 创建图形 - 单栏宽度
fig, ax = plt.subplots(figsize=(7, 5), dpi=100)
ax.set_facecolor('#fafbfc')

# 设置箱形图的位置
positions_baseline = np.arange(1, sessions + 1) - 0.25
positions_happy = np.arange(1, sessions + 1)
positions_concept = np.arange(1, sessions + 1) + 0.25

# 绘制箱形图
box_width = 0.2

bp1 = ax.boxplot(baseline_data, positions=positions_baseline, widths=box_width,
                 patch_artist=True, showfliers=True,
                 boxprops=dict(facecolor=colors['Baseline'], alpha=0.7, linewidth=1.5),
                 medianprops=dict(color='black', linewidth=2),
                 whiskerprops=dict(color=colors['Baseline'], linewidth=1.5),
                 capprops=dict(color=colors['Baseline'], linewidth=1.5),
                 flierprops=dict(marker='o', markerfacecolor=colors['Baseline'],
                               markersize=6, alpha=0.5))

bp2 = ax.boxplot(happy_data, positions=positions_happy, widths=box_width,
                 patch_artist=True, showfliers=True,
                 boxprops=dict(facecolor=colors['Happy'], alpha=0.7, linewidth=1.5),
                 medianprops=dict(color='black', linewidth=2),
                 whiskerprops=dict(color=colors['Happy'], linewidth=1.5),
                 capprops=dict(color=colors['Happy'], linewidth=1.5),
                 flierprops=dict(marker='s', markerfacecolor=colors['Happy'],
                               markersize=6, alpha=0.5))

bp3 = ax.boxplot(concept_data, positions=positions_concept, widths=box_width,
                 patch_artist=True, showfliers=True,
                 boxprops=dict(facecolor=colors['ConceptCCD'], alpha=0.7, linewidth=1.5),
                 medianprops=dict(color='black', linewidth=2),
                 whiskerprops=dict(color=colors['ConceptCCD'], linewidth=1.5),
                 capprops=dict(color=colors['ConceptCCD'], linewidth=1.5),
                 flierprops=dict(marker='^', markerfacecolor=colors['ConceptCCD'],
                               markersize=7, alpha=0.5))

# 设置坐标轴 - 单栏适配
ax.set_xlabel('Online Session', fontsize=11, fontweight='bold')
ax.set_ylabel('Accuracy (%) on "Novel" Classes', fontsize=11, fontweight='bold')
ax.set_title('Robustness Analysis Across Different Shuffles', fontsize=12, fontweight='bold', pad=12)

ax.set_xticks(np.arange(1, sessions + 1))
ax.set_xticklabels([str(i) for i in range(1, sessions + 1)])
ax.set_xlim(0.5, sessions + 0.5)
ax.set_ylim(20, 90)

# 网格线
ax.grid(True, linestyle='--', alpha=0.4, linewidth=0.8, color='white', axis='y')

# 图例
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=colors['Baseline'], alpha=0.7, edgecolor='black', label='Baseline'),
    Patch(facecolor=colors['Happy'], alpha=0.7, edgecolor='black', label='Happy'),
    Patch(facecolor=colors['ConceptCCD'], alpha=0.7, edgecolor='black', label='CAPE(Ours)')
]
ax.legend(handles=legend_elements, loc='upper right', frameon=True,
         fancybox=False, shadow=False, framealpha=0.9, edgecolor='black', fontsize=10)

# 添加说明文本 - 单栏适配
textstr = 'Box: IQR (25th-75th percentile) | Line: Median | Whiskers: Min-Max'
ax.text(0.5, 0.02, textstr, transform=ax.transAxes, fontsize=8,
        verticalalignment='bottom', horizontalalignment='center',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.show()