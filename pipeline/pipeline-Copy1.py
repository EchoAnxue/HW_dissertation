import torch
import torch.nn.functional as F

from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import torch.optim as optim

from GNNModel import GNNModel
from utils import save_to_single_row, loader, test_batches, train_batches, validate_batches


# 训练函数
def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


# **验证函数（计算 val loss）**
def validate(model, data):
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        loss = F.nll_loss(logits[data.val_mask], data.y[data.val_mask])
    return loss.item()


from sklearn.metrics import f1_score, roc_auc_score


def test(model, data):
    model.eval()
    logits = model(data.x, data.edge_index)

    # 取出测试集的 mask
    mask = data.test_mask

    # 计算预测结果
    pred = logits[mask].max(1)[1].cpu().numpy()  # 预测标签
    true = data.y[mask].cpu().numpy()  # 真实标签
    probs = torch.softmax(logits[mask], dim=1).cpu().detach().numpy()  # 概率分布

    # 计算准确率（Accuracy）
    acc = (pred == true).sum() / mask.sum().item()

    # 计算 F1-score（macro 和 micro）
    f1_macro = f1_score(true, pred, average='macro', zero_division=0)
    f1_micro = f1_score(true, pred, average='micro', zero_division=0)

    # 计算 AUC-ROC（仅适用于二分类或多标签分类）
    if len(set(true)) > 1:  # 需要至少两个类别才能计算 AUC
        try:
            auc = roc_auc_score(true, probs, multi_class='ovr')
        except ValueError:
            auc = 0.0  # 计算 AUC 时遇到问题
    else:
        auc = 0.0

    return acc, f1_macro, f1_micro, auc


from torch_geometric.datasets import Reddit
from torch_geometric.data import Data
import torch.autograd.profiler as profiler
# **主流程**
from torch.utils.tensorboard import SummaryWriter
import os

log_dir = os.path.join('/root/tf-logs', 'train')
train_writer = SummaryWriter(log_dir=log_dir)
# 将 test accuracy 保存到 "tensorboard/test" 文件夹
log_dir = os.path.join('/root/tf-logs', 'val')
val_writer = SummaryWriter(log_dir=log_dir)
log_dir = os.path.join('/root/tf-logs', 'test')
test_writer = SummaryWriter(log_dir=log_dir)

import random
import numpy as np


def set_seed(seed=42):
    random.seed(seed)  # Python 内置随机数
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed_all(seed)  # PyTorch GPU（如果使用多个GPU）
    torch.backends.cudnn.deterministic = True  # 确保 CuDNN 结果可复现
    torch.backends.cudnn.benchmark = False  # 禁用优化，确保一致性


# set_seed(42)


def run_pipeline(dataset_name="Cora",
                 # model_types=["GCN", "GraphSAGE", "GAT", "JKNet","resGCN","GINConv","mlp","mamba3","mambawores"，“mambawosort”,"mambawomamba"],
                 model_types=["mamba3"],
                 # 2,8,32
                 num_layers_list=[2,8,32], batch=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if dataset_name == "Reddit":
        # "C:/Users/lenovo/PycharmProjects/DropEdge/src/pipeline/data"
        # /content/gdrive/MyDrive/Colab Notebooks/GNN/pipeline/data/Reddit
        dataset = Reddit(root="./data/Reddit")
        print(dataset, dataset.num_node_features, dataset.num_classes)

    else:
        dataset = Planetoid(root=f"./data/{dataset_name}", name=dataset_name)
    # data = dataset[0].to(device)
    data = dataset[0]
    # bs 2048
    train_loader, val_loader, test_loader = loader(data, batch_size=512)
    results = {}
    parameter = []

    for model_type in model_types:
        for num_layers in num_layers_list:
            train_losses, val_losses = [], []
            print(f"Training {model_type} with {num_layers} layers...")

            model = GNNModel(model_type, dataset.num_node_features, dataset.num_classes, num_layers=num_layers).to(
                device)
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
            parameter.append(f"{trainable_params}_{model_type}_{non_trainable_params}")
            print(f"可训练参数: {trainable_params}_{model_type}")
            print(f"不可训练参数: {non_trainable_params}")

            optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=0.0005)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
            best_val_loss = float('inf')  # 设定初始最小值为无穷大
            best_model_path = "best_model.pth"
            for epoch in range(200):
                if batch == False:
                    data = data.to(device)
                    train_loss = train(model, optimizer, data)
                    val_loss = validate(model, data)
                else:

                    # with profiler.profile(use_cuda=True) as prof:

                    train_loss = train_batches(model, optimizer, train_loader)
                    val_loss = validate_batches(model, val_loader)
                    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

                if val_loss < best_val_loss:  # 发现更优模型  2 layers有效 8，32layers还是取最后epoch
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), best_model_path)  # 保存当前最优权重
                    print(f"✅ Epoch {epoch}: New best model saved with val_loss = {val_loss:.4f}")
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                train_writer.add_scalar(f"Loss/{model_type}_{num_layers}_{dataset_name}/Train&Val", train_loss, epoch)
                val_writer.add_scalar(f"Loss/{model_type}_{num_layers}_{dataset_name}/Train&Val", val_loss, epoch)
                scheduler.step(val_loss)
                if batch == False:
                    accs, f1_macro, f1_micro, auc = test(model, data)
                    test_writer.add_scalar(f"Loss/{model_type}_{num_layers}_{dataset_name}/test", accs, epoch)
                    print(f"Epoch {epoch}, accuracy: {accs:.4f}")
                else:
                    accs, f1_macro, f1_micro, auc = test_batches(model, test_loader)
                    test_writer.add_scalar(f"Loss/{model_type}_{num_layers}_{dataset_name}/test", accs, epoch)
                    print(f"Epoch {epoch}, accuracy: {accs:.4f}")
                if epoch % 2 == 0:
                    print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # 训练结束后，加载最佳模型
            model.load_state_dict(torch.load(best_model_path))
            print("Loaded best model with val_loss =", best_val_loss)
            #         test
            if batch == False:
                accs, f1_macro, f1_micro, auc = test(model, data)
            else:
                accs, f1_macro, f1_micro, auc = test_batches(model, test_loader)

            results[f"{model_type}_{num_layers}_{dataset_name}"] = accs
            print(f"test :accuracy {accs:.4f}, f1_macro: {f1_macro:.4f}, f1_micro: {f1_micro:.4f}, auc: {auc:.4f}")
            # filepath, model_info, accuracy, f1_macro, f1_micro, auc, loss_values
            save_to_single_row("./training_log.csv", f"train_{model_type}_{num_layers}_{dataset_name}", accuracy=accs,
                               f1_macro=f1_macro, f1_micro=f1_micro, auc=auc, loss_values=train_losses)
            save_to_single_row("./training_log.csv", f"val_{model_type}_{num_layers}_{dataset_name}", accuracy=accs,
                               f1_macro=f1_macro, f1_micro=f1_micro, auc=auc, loss_values=val_losses)
            # 记录测试指标到 TensorBoard
            # writer.add_scalar(f"Metrics/{model_type}_{num_layers}/Accuracy", accs, 0)
            # writer.add_scalar(f"Metrics/{model_type}_{num_layers}/F1_Macro", f1_macro, 0)
            # writer.add_scalar(f"Metrics/{model_type}_{num_layers}/F1_Micro", f1_micro, 0)
            # writer.add_scalar(f"Metrics/{model_type}_{num_layers}/AUC", auc, 0)
    train_writer.close()
    val_writer.close()
    test_writer.close()
    return results, parameter


# **运行Pipeline**
if __name__ == "__main__":


    m = 3  # 设定 baseline 运行次数
    all_results = {}  # 用于存储所有实验结果

    for i in range(m):
        print(f"========== Running baseline {i+1}/{m} ==========")
        results, parameters = run_pipeline(dataset_name="Citeseer",
            
                 model_types=["mamba3"],
                 # 2,8,32
                 num_layers_list=[2,8,32], batch=False)
        print("Final Results:", results, parameters)
        # 记录结果
        for key, acc in results.items():
            if key not in all_results:
                all_results[key] = []
            all_results[key].append(acc)

    # 计算均值和标准差
    final_results = {}
    for key, acc_list in all_results.items():
        mean_acc = np.mean(acc_list)
        std_acc = np.std(acc_list, ddof=1)  # 样本标准差
        final_results[key] = (mean_acc, std_acc)
        print(f"{key}: Accuracy = {mean_acc:.4f} ± {std_acc:.4f}")


    m = 3  # 设定 baseline 运行次数
    all_results = {}  # 用于存储所有实验结果

    for i in range(m):
        print(f"========== Running baseline {i+1}/{m} ==========")
        results, parameters = run_pipeline(dataset_name="Pubmed",
            
                 model_types=["mamba3"],
                 # 2,8,32
                 num_layers_list=[2,8,32], batch=True)
        print("Final Results:", results, parameters)
        # 记录结果
        for key, acc in results.items():
            if key not in all_results:
                all_results[key] = []
            all_results[key].append(acc)

    # 计算均值和标准差
    final_results = {}
    for key, acc_list in all_results.items():
        mean_acc = np.mean(acc_list)
        std_acc = np.std(acc_list, ddof=1)  # 样本标准差
        final_results[key] = (mean_acc, std_acc)
        print(f"{key}: Accuracy = {mean_acc:.4f} ± {std_acc:.4f}")