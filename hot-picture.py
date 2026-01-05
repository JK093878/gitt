import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据
train = pd.read_csv("TrainingData.csv")
val = pd.read_csv("ValidationData.csv")

# 创建交叉表
train_counts = train.groupby(['BUILDINGID', 'FLOOR']).size().unstack(fill_value=0)
val_counts = val.groupby(['BUILDINGID', 'FLOOR']).size().unstack(fill_value=0)

# 画图
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(train_counts, annot=True, fmt="d", cmap="Blues", ax=axes[0])
axes[0].set_title("Training set sample distribution")
axes[0].set_xlabel("Floor")
axes[0].set_ylabel("Building")

sns.heatmap(val_counts, annot=True, fmt="d", cmap="Greens", ax=axes[1])
axes[1].set_title("Validation set sample distribution")
axes[1].set_xlabel("Floor")
axes[1].set_ylabel("Building")

plt.tight_layout()
plt.savefig("uji_distribution_heatmap.png", dpi=300)
plt.show()
