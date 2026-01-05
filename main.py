import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import psutil
from sklearn.metrics import accuracy_score, classification_report
import os


def train_epoch(model, train_loader, criterion, floor_loss_fn, optimizer, device,
                coord_mean, coord_std, flood_level=0.3):
    model.train()
    total_loss = 0
    coord_mae = 0

    if not isinstance(coord_mean, torch.Tensor):
        coord_mean = torch.tensor(coord_mean, dtype=torch.float32, device=device)
    if not isinstance(coord_std, torch.Tensor):
        coord_std = torch.tensor(coord_std, dtype=torch.float32, device=device)

    all_true_building, all_pred_building = [], []
    all_true_floor, all_pred_floor = [], []

    for features, coordinates, hierarchical_labels, _, _ in train_loader:
        features = features.to(device)
        coordinates = coordinates.to(device)
        hierarchical_labels = hierarchical_labels.to(device)

        building_outputs, floor_outputs, coord_outputs = model(features)

        # 分类损失
        building_loss = criterion(
            building_outputs.reshape(-1, model.config['num_buildings']),
            hierarchical_labels[:, :, 0].reshape(-1)
        )
        floor_loss = floor_loss_fn(
            floor_outputs.reshape(-1, model.config['num_floors']),
            hierarchical_labels[:, :, 1].reshape(-1)
        )

        # 坐标回归损失
        coord_target = coordinates.mean(dim=1)
        mae_loss = nn.L1Loss()(coord_outputs, coord_target)
        mse_loss = nn.MSELoss()(coord_outputs, coord_target)
        coord_loss = 0.7 * mae_loss + 0.3 * torch.sqrt(mse_loss + 1e-6)

        # 坐标误差监控
        coord_outputs_real = coord_outputs * coord_std + coord_mean
        coord_target_real = coord_target * coord_std + coord_mean
        errors = torch.norm(coord_outputs_real - coord_target_real, dim=1)
        coord_mae += torch.mean(errors).item()

        # 总损失
        loss = 0.3 * building_loss + 1.0 * floor_loss + 0.7 * coord_loss

        if flood_level > 0:
            loss = (loss - flood_level).abs() + flood_level

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # 记录预测
        _, building_pred = torch.max(building_outputs[:, -1, :], dim=1)
        _, floor_pred = torch.max(floor_outputs[:, -1, :], dim=1)
        invalid_mask = (floor_pred == 4) & (building_pred != 2)
        floor_pred[invalid_mask] = 3

        all_true_building.extend(hierarchical_labels[:, -1, 0].cpu().numpy())
        all_pred_building.extend(building_pred.cpu().numpy())
        all_true_floor.extend(hierarchical_labels[:, -1, 1].cpu().numpy())
        all_pred_floor.extend(floor_pred.cpu().numpy())

    building_acc = accuracy_score(all_true_building, all_pred_building)
    floor_acc = accuracy_score(all_true_floor, all_pred_floor)

    return {
        'loss': total_loss / len(train_loader),
        'building_acc': building_acc,
        'floor_acc': floor_acc,
        'coord_mae': coord_mae / len(train_loader)
    }


def validate_epoch(model, val_loader, criterion, floor_loss_fn, device, coord_mean, coord_std):
    model.eval()
    total_loss = 0
    coord_mae = 0
    all_errors = []

    coord_mean = torch.tensor(coord_mean, dtype=torch.float32, device=device)
    coord_std = torch.tensor(coord_std, dtype=torch.float32, device=device)

    all_true_building, all_pred_building = [], []
    all_true_floor, all_pred_floor = [], []

    with torch.no_grad():
        for batch_idx, (features, coordinates, hierarchical_labels, _, _) in enumerate(val_loader):
            features = features.to(device)
            coordinates = coordinates.to(device)
            hierarchical_labels = hierarchical_labels.to(device)

            building_outputs, floor_outputs, coord_outputs = model(features)

            _, building_pred = torch.max(building_outputs[:, -1, :], dim=1)
            _, floor_pred = torch.max(floor_outputs[:, -1, :], dim=1)

            # Floor 4预测修正
            floor4_mask = (floor_pred == 4)
            if torch.any(floor4_mask):
                invalid_building = (building_pred[floor4_mask] != 2)
                floor_pred[floor4_mask] = torch.where(
                    invalid_building,
                    3,
                    floor_pred[floor4_mask]
                )

            all_true_building.extend(hierarchical_labels[:, -1, 0].cpu().numpy())
            all_pred_building.extend(building_pred.cpu().numpy())
            all_true_floor.extend(hierarchical_labels[:, -1, 1].cpu().numpy())
            all_pred_floor.extend(floor_pred.cpu().numpy())

            # 坐标误差计算
            coord_target = coordinates.mean(dim=1)
            coord_outputs_real = coord_outputs * coord_std + coord_mean
            coord_target_real = coord_target * coord_std + coord_mean
            errors = torch.norm(coord_outputs_real - coord_target_real, dim=1)
            coord_mae += torch.mean(errors).item()
            all_errors.extend(errors.cpu().numpy())

            # 损失计算
            building_loss = criterion(
                building_outputs.reshape(-1, model.config['num_buildings']),
                hierarchical_labels[:, :, 0].reshape(-1)
            )
            floor_loss = nn.CrossEntropyLoss()(
                floor_outputs.reshape(-1, model.config['num_floors']),
                hierarchical_labels[:, :, 1].reshape(-1)
            )
            coord_loss = nn.MSELoss()(coord_outputs, coord_target)
            total_loss += (0.3 * building_loss + 1.0 * floor_loss + 0.7 * coord_loss).item()

    print("\n📊 分类报告：")
    print(classification_report(all_true_floor, all_pred_floor,
                                target_names=["Floor 0", "Floor 1", "Floor 2", "Floor 3", "Floor 4"]))

    np.save('val_errors.npy', np.array(all_errors))
    print("✅ 验证集误差数据已保存到 val_errors.npy")

    return {
        'loss': total_loss / len(val_loader),
        'building_acc': accuracy_score(all_true_building, all_pred_building),
        'floor_acc': accuracy_score(all_true_floor, all_pred_floor),
        'coord_mae': coord_mae / len(val_loader),
        'errors': all_errors
    }


def train_model(model, train_loader, val_loader, criterion, optimizer,
                floor_loss_fn, coord_mean, coord_std, scheduler,
                num_epochs, patience, device, flood_level=0.3):
    model = model.to(device)
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0

    history = {
        'train_building': [],
        'val_building': [],
        'train_floor': [],
        'val_floor': []
    }

    ema_decay = 0.999
    shadow_params = {name: param.data.clone() for name, param in model.named_parameters()}

    for epoch in tqdm(range(num_epochs), desc="Training Progress"):
        train_metrics = train_epoch(model, train_loader, criterion, floor_loss_fn,
                                    optimizer, device, coord_mean, coord_std, flood_level=flood_level)
        train_losses.append(train_metrics['loss'])

        val_metrics = validate_epoch(model, val_loader, criterion, floor_loss_fn,
                                     device, coord_mean, coord_std)
        val_losses.append(val_metrics['loss'])
        scheduler.step()

        history['train_building'].append(train_metrics['building_acc'])
        history['val_building'].append(val_metrics['building_acc'])
        history['train_floor'].append(train_metrics['floor_acc'])
        history['val_floor'].append(val_metrics['floor_acc'])

        for name, param in model.named_parameters():
            if name in shadow_params:
                shadow_params[name].mul_(ema_decay).add_(param.data, alpha=1 - ema_decay)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"当前学习率: {current_lr:.6f}")

        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'Train Loss: {train_metrics["loss"]:.4f}')
        print(f'Val Loss: {val_metrics["loss"]:.4f}')
        print(f'Building Acc: {train_metrics["building_acc"]:.4f}/{val_metrics["building_acc"]:.4f}')
        print(f'Floor Acc: {train_metrics["floor_acc"]:.4f}/{val_metrics["floor_acc"]:.4f}')
        print(f'Coord MAE (Mean Positioning Error): {train_metrics["coord_mae"]:.2f}/{val_metrics["coord_mae"]:.2f}')

        mem = psutil.virtual_memory()
        print(f"Memory usage: {mem.percent}%")

        # 早停和模型保存
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0

            torch.save(model.state_dict(), 'best_model_normal.pth')

            backup = {name: param.data.clone() for name, param in model.named_parameters()}
            for name, param in model.named_parameters():
                if name in shadow_params:
                    param.data.copy_(shadow_params[name])
            torch.save(model.state_dict(), 'best_model.pth')
            print("✅ EMA平滑参数已保存为 best_model.pth")

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
    val_errors = val_metrics.get('errors', None)
    if val_errors is not None:
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