# training/trainer.py
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import psutil
from tqdm import tqdm
import os

from training.utils import train_epoch, validate_epoch


def train_model(model, train_loader, val_loader, criterion, optimizer,
                floor_loss_fn, coord_mean, coord_std, scheduler,
                num_epochs, patience, device, flood_level=0.3):
    """训练模型主函数"""
    model = model.to(device)
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0

    # 记录历史
    history = {
        'train_building': [],
        'val_building': [],
        'train_floor': [],
        'val_floor': []
    }

    # EMA参数
    ema_decay = 0.999
    shadow_params = {name: param.data.clone() for name, param in model.named_parameters()}

    for epoch in tqdm(range(num_epochs), desc="Training Progress"):
        # 训练
        train_metrics = train_epoch(
            model, train_loader, criterion, floor_loss_fn,
            optimizer, device, coord_mean, coord_std, flood_level=flood_level
        )
        train_losses.append(train_metrics['loss'])

        # 验证
        val_metrics = validate_epoch(
            model, val_loader, criterion, floor_loss_fn,
            device, coord_mean, coord_std
        )
        val_losses.append(val_metrics['loss'])

        scheduler.step()

        # 记录指标
        history['train_building'].append(train_metrics['building_acc'])
        history['val_building'].append(val_metrics['building_acc'])
        history['train_floor'].append(train_metrics['floor_acc'])
        history['val_floor'].append(val_metrics['floor_acc'])

        # 更新EMA
        for name, param in model.named_parameters():
            if name in shadow_params:
                shadow_params[name].mul_(ema_decay).add_(param.data, alpha=1 - ema_decay)

        # 打印信息
        current_lr = optimizer.param_groups[0]['lr']
        print(f"当前学习率: {current_lr:.6f}")
        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'Train Loss: {train_metrics["loss"]:.4f}')
        print(f'Val Loss: {val_metrics["loss"]:.4f}')
        print(f'Building Acc: {train_metrics["building_acc"]:.4f}/{val_metrics["building_acc"]:.4f}')
        print(f'Floor Acc: {train_metrics["floor_acc"]:.4f}/{val_metrics["floor_acc"]:.4f}')
        print(f'Coord MAE: {train_metrics["coord_mae"]:.2f}/{val_metrics["coord_mae"]:.2f}')

        # 内存监控
        mem = psutil.virtual_memory()
        print(f"Memory usage: {mem.percent}%")

        # 早停和保存模型
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0

            # 保存普通模型
            torch.save(model.state_dict(), 'best_model_normal.pth')

            # 保存EMA模型
            backup = {name: param.data.clone() for name, param in model.named_parameters()}
            for name, param in model.named_parameters():
                if name in shadow_params:
                    param.data.copy_(shadow_params[name])
            torch.save(model.state_dict(), 'best_model.pth')
            print("✅ EMA平滑参数已保存为 best_model.pth")

            # 恢复原参数
            for name, param in model.named_parameters():
                if name in backup:
                    param.data.copy_(backup[name])
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch + 1} epochs')
                break

    # 绘制图表
    plot_training_results(train_losses, val_losses, history, val_metrics)

    return train_losses, val_losses


def plot_training_results(train_losses, val_losses, history, val_metrics):
    """绘制训练结果图表"""
    # 损失曲线
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid()
    plt.savefig('loss_curve.png')
    print("✅ 损失曲线已保存到 loss_curve.png")

    # 精度曲线
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_building'], 'b-', label='Train')
    plt.plot(history['val_building'], 'r--', label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Building Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.subplot(1, 2, 2)
    plt.plot(history['train_floor'], 'b-', label='Train')
    plt.plot(history['val_floor'], 'r--', label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Floor Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ 训练曲线已保存为 training_curves.png")

    # CDF图
    if 'errors' in val_metrics and val_metrics['errors'] is not None:
        val_errors = val_metrics['errors']
        errors_np = np.sort(np.array(val_errors))
        cdf = np.arange(len(errors_np)) / len(errors_np)

        plt.figure(figsize=(8, 6))
        plt.plot(errors_np, cdf, label='CDF of Localization Error')
        plt.xlabel('Localization Error (meters)')
        plt.ylabel('Cumulative Probability')
        plt.title('CDF of Validation Localization Error')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig('val_error_cdf.png')
        print("✅ 验证集定位误差CDF图已保存为 val_error_cdf.png")

        # 保存误差数据
        np.save('val_errors.npy', np.array(val_errors))
        print("✅ 验证集误差数据已保存到 val_errors.npy")