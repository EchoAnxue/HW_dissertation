import torch
from torch_geometric.datasets import Reddit
from torch_geometric.data import Data

# 1️⃣ 加载 Reddit 数据集
dataset = Reddit(root="./data/Reddit")

# 2️⃣ 获取数据
data = dataset[0]

# 3️⃣ 重新创建 Planetoid 格式的数据
reddit_data = Data(
    x=data.x,                   # 特征矩阵 [232,965 x 602]
    edge_index=data.edge_index,  # 边索引 [2 x 114,615,892]
    y=data.y,                   # 标签 [232,965]
    train_mask=data.train_mask,  # 训练集 mask
    val_mask=data.val_mask,      # 验证集 mask
    test_mask=data.test_mask     # 测试集 mask
)

print(reddit_data)
