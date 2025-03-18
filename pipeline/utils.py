import os,csv

import numpy as np



def save_to_single_row(filepath, model_info, accuracy, f1_macro, f1_micro, auc, loss_values):
    # 如果文件不存在，则创建并写入表头
    if not os.path.exists(filepath):
        with open(filepath, mode='w', newline='') as file:
            writer = csv.writer(file)
            # 写入表头：包括模型信息、准确率、400个loss值
            header = ["Model Info", "Accuracy","f1_macro", "f1_micro", "auc"] + [f"Loss_{i + 1}" for i in range(len(loss_values))]
            writer.writerow(header)

    # 追加写入每一行数据
    with open(filepath, mode='a', newline='') as file:
        writer = csv.writer(file)
        row = [model_info, accuracy, f1_macro, f1_micro, auc] + loss_values
        writer.writerow(row)


from sklearn.metrics import f1_score, roc_auc_score
import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
# reddit
# def loader(data,batch_size = 1024):
#     train_loader = NeighborLoader(
#         data,
#         num_neighbors=[3, 1],
#         batch_size=batch_size,
#         input_nodes=torch.where(data.train_mask)[0],
#         shuffle=True,  # 随机采样
#         num_workers=3,  # 使用多线程加速数据加载
#         persistent_workers=True,  # 让 worker 持续运行，提高效率
#         pin_memory=True
#     )

#     val_loader = NeighborLoader(
#         data,
#         num_neighbors=[3, 1],
#         batch_size=batch_size,
#         input_nodes=torch.where(data.val_mask)[0],
#         shuffle = True,  # 随机采样
#         num_workers=3,  # 使用多线程加速数据加载
#         persistent_workers=True,  # 让 worker 持续运行，提高效率
#         pin_memory=True
#     )
#     # 用 NeighborLoader 进行测试集采样
#     test_loader = NeighborLoader(
#         data,
#         num_neighbors=[3, 1],  # 采样所有邻居（全局传播）
#         batch_size=batch_size,
#         input_nodes=torch.where(data.test_mask)[0],
#         shuffle=True,  # 随机采样
#         num_workers=3,  # 使用多线程加速数据加载
#         persistent_workers=True,  # 让 worker 持续运行，提高效率
#         pin_memory=True

#     )
#     return train_loader, val_loader, test_loader

# pubmed
def loader(data,batch_size = 1024):
    train_loader = NeighborLoader(
        data,
        num_neighbors=[10, 5],
        batch_size=batch_size,
        input_nodes=torch.where(data.train_mask)[0],
        shuffle=True,  # 随机采样
        num_workers=8,  # 使用多线程加速数据加载
        persistent_workers=True,  # 让 worker 持续运行，提高效率
        pin_memory=True
    )

    val_loader = NeighborLoader(
        data,
        num_neighbors=[10, 5],
        batch_size=batch_size,
        input_nodes=torch.where(data.val_mask)[0],
        shuffle = True,  # 随机采样
        num_workers=8,  # 使用多线程加速数据加载
        persistent_workers=True,  # 让 worker 持续运行，提高效率
        pin_memory=True
    )
    # 用 NeighborLoader 进行测试集采样
    test_loader = NeighborLoader(
        data,
        num_neighbors=[-1, -1],  # 采样所有邻居（全局传播）
        batch_size=batch_size,
        input_nodes=torch.where(data.test_mask)[0],
        shuffle=True,  # 随机采样
        num_workers=8,  # 使用多线程加速数据加载
        persistent_workers=True,  # 让 worker 持续运行，提高效率
        pin_memory=True

    )
    return train_loader, val_loader, test_loader


# sage
# def loader(data,batch_size = 1024):
#     train_loader = NeighborLoader(
#         data,
#         num_neighbors=[5, 3],
#         batch_size=batch_size,
#         input_nodes=torch.where(data.train_mask)[0],
#         shuffle=True,  # 随机采样
#         num_workers=3,  # 使用多线程加速数据加载
#         persistent_workers=True,  # 让 worker 持续运行，提高效率
#         pin_memory=True
#     )

#     val_loader = NeighborLoader(
#         data,
#         num_neighbors=[5, 3],
#         batch_size=batch_size,
#         input_nodes=torch.where(data.val_mask)[0],
#         shuffle = True,  # 随机采样
#         num_workers=3,  # 使用多线程加速数据加载
#         persistent_workers=True,  # 让 worker 持续运行，提高效率
#         pin_memory=True
#     )
#     # 用 NeighborLoader 进行测试集采样
#     test_loader = NeighborLoader(
#         data,
#         num_neighbors=[-1, -1],  # 采样所有邻居（全局传播）
#         batch_size=batch_size,
#         input_nodes=torch.where(data.test_mask)[0],
#         shuffle=True,  # 随机采样
#         num_workers=3,  # 使用多线程加速数据加载
#         persistent_workers=True,  # 让 worker 持续运行，提高效率
#         pin_memory=True

#     )
#     return train_loader, val_loader, test_loader
def test_batches(model,test_loader):
    model.eval()

    all_preds, all_trues, all_probs = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(next(model.parameters()).device)
            logits = model(batch.x, batch.edge_index)

            pred = logits.max(1)[1].cpu().numpy()  # 预测标签
            true = batch.y.cpu().numpy()  # 真实标签
            probs = torch.softmax(logits, dim=1).cpu().numpy()  # 概率分布

            all_preds.append(pred)
            all_trues.append(true)
            all_probs.append(probs)

    # 合并所有 batch
    all_preds = (
        np.concatenate(all_preds))
    all_trues = np.concatenate(all_trues)
    all_probs = np.concatenate(all_probs)

    # 计算准确率
    acc = (all_preds == all_trues).sum() / len(all_trues)

    # 计算 F1-score
    f1_macro = f1_score(all_trues, all_preds, average='macro', zero_division=0)
    f1_micro = f1_score(all_trues, all_preds, average='micro', zero_division=0)

    # 计算 AUC-ROC（适用于多分类）
    try:
        auc = roc_auc_score(all_trues, all_probs, multi_class='ovr')
    except ValueError:
        auc = 0.0  # 计算 AUC 失败时返回 0.0

    return acc, f1_macro, f1_micro, auc
def train_batches(model, optimizer, train_loader):
    model.train()
    total_loss = 0
    device = next(model.parameters()).device
    for batch in train_loader:  # 遍历每个 mini-batch
        print("batch size:",batch.size())
        optimizer.zero_grad()

        batch = batch.to(device)  # 确保数据在正确的设备上

        out = model(batch.x, batch.edge_index)  # 仅计算当前 batch
        loss = F.nll_loss(out[batch.train_mask], batch.y[batch.train_mask])

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss/len(train_loader)


def validate_batches(model, val_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(next(model.parameters()).device)
            logits = model(batch.x, batch.edge_index)
            loss = F.nll_loss(logits[batch.val_mask], batch.y[batch.val_mask])
            total_loss += loss.item()

    return total_loss/len(val_loader)