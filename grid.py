# 网格搜索αβγ.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import pandas as pd
import os
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
from wifi_localization.config import Config
from wifi_localization.data.preprocessing import load_and_preprocess_data
from wifi_localization.models.base_model import WiFiLocalizationModel
from wifi_localization.training.trainer import train_model, validate_epoch
# 设置设备
assert torch.cuda.is_available(), "CUDA is not available - GPU required!"
device = torch.device('cuda')
print(f"Using device: {device}")

def compute_composite_score(building_acc, floor_acc, coord_mae):
    """
    计算综合评分
    - 建筑和楼层准确率越高越好
    - 坐标误差越小越好
    """
    # 将坐标MAE转换为得分（误差越小得分越高）
    coord_score = 1.0 / (coord_mae + 1e-8)
    coord_score = min(coord_score, 0.5)  # 限制最大得分

    # 加权综合得分
    composite_score = (0.2 * building_acc +  # 建筑分类权重20%
                       0.5 * floor_acc +  # 楼层分类权重50%
                       0.3 * coord_score)  # 坐标回归权重30%

    return composite_score


def grid_search_weights():
    """执行损失权重网格搜索"""

    # 1. 加载数据
    TRAIN_PATH = 'TrainingData.csv'
    VAL_PATH = 'ValidationData.csv'

    train_loader, val_loader, num_buildings, num_floors, floor_counts, coord_mean, coord_std = load_and_preprocess_data(
        train_data_path=TRAIN_PATH,
        val_data_path=VAL_PATH,
        train_subset_ratio=1.0,
        device=device
    )

    # 2. 准备基础配置
    model_config = Config.MODEL_CONFIG.copy()
    model_config.update({
        'num_buildings': num_buildings,
        'num_floors': num_floors,
        'num_classes': num_buildings + num_floors
    })

    print(f"模型配置: 建筑={num_buildings}, 楼层={num_floors}")

    # 3. 定义要搜索的权重组合
    weight_configs = [
        (0.1, 1.0, 0.5),  # 组合1：强调楼层分类
        (0.3, 1.0, 0.7),  # 组合2：原始设置
        (0.5, 1.0, 1.0),  # 组合3：均衡权重
        (0.2, 1.0, 0.8),  # 组合4：适度强调坐标
        (0.1, 0.8, 1.0),  # 组合5：强调坐标回归
        (0.4, 1.0, 0.6),  # 组合6：建筑权重稍高
    ]

    # 4. 网格搜索
    best_composite_score = 0
    best_weights = None
    all_results = []

    for i, weights in enumerate(weight_configs):
        print(f"\n{'=' * 50}")
        print(f"🔬 测试权重组合 {i + 1}/{len(weight_configs)}: α={weights[0]}, β={weights[1]}, γ={weights[2]}")
        print(f"{'=' * 50}")

        model = WiFiLocalizationModel(model_config)

        # 准备损失函数
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        floor_weights_tensor = torch.tensor(1.0 / (floor_counts + 1e-7), dtype=torch.float32).to(device)
        floor_loss_fn = nn.CrossEntropyLoss(weight=floor_weights_tensor, label_smoothing=0.05)

        # 优化器和调度器
        optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-2)
        scheduler = StepLR(optimizer, step_size=15, gamma=0.5)

        # 使用简化的训练（减少epochs以加速搜索）
        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            floor_loss_fn=floor_loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            coord_mean=coord_mean,
            coord_std=coord_std,
            loss_weights=weights,
            flood_level=Config.FLOOD_LEVEL,
            device=device,
            save_plots=False  # 不保存每个组合的图表
        )

        # 评估当前权重组合的性能
        final_metrics = validate_epoch(
            model, val_loader, criterion, floor_loss_fn, device, coord_mean, coord_std
        )

        # 计算综合评分
        composite_score = compute_composite_score(
            final_metrics['building_acc'],
            final_metrics['floor_acc'],
            final_metrics['coord_mae']
        )

        # 记录结果
        result = {
            'weights': weights,
            'composite_score': composite_score,
            'building_acc': final_metrics['building_acc'],
            'floor_acc': final_metrics['floor_acc'],
            'coord_mae': final_metrics['coord_mae'],
            'val_loss': final_metrics['loss']
        }
        all_results.append(result)

        print(f"✅ 组合结果:")
        print(f"   建筑准确率: {result['building_acc']:.4f}")
        print(f"   楼层准确率: {result['floor_acc']:.4f}")
        print(f"   坐标MAE: {result['coord_mae']:.2f}m")
        print(f"   综合评分: {result['composite_score']:.4f}")

        # 更新最佳权重
        if composite_score > best_composite_score:
            best_composite_score = composite_score
            best_weights = weights
            print(f"🏆 新的最佳权重: {best_weights}")

    # 5. 输出网格搜索总结
    print(f"\n{'=' * 60}")
    print("🎯 网格搜索完成！结果总结:")
    print(f"{'=' * 60}")

    # 按综合评分排序
    all_results.sort(key=lambda x: x['composite_score'], reverse=True)

    for i, result in enumerate(all_results):
        α, β, γ = result['weights']
        rank = f"{i + 1}." if i < 3 else "   "
        print(f"{rank} 权重(α={α}, β={β}, γ={γ}): "
              f"建筑={result['building_acc']:.4f}, "
              f"楼层={result['floor_acc']:.4f}, "
              f"MAE={result['coord_mae']:.2f}m, "
              f"综合={result['composite_score']:.4f}")

    # 6. 使用最佳权重进行最终训练
    print(f"\n{'=' * 50}")
    print(f"🚀 使用最佳权重进行最终训练: α={best_weights[0]}, β={best_weights[1]}, γ={best_weights[2]}")
    print(f"{'=' * 50}")

    final_model = WiFiLocalizationModel(model_config)
    final_optimizer = optim.Adam(final_model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-2)
    final_scheduler = StepLR(final_optimizer, step_size=15, gamma=0.5)

    # 完整训练
    train_model(
        model=final_model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        floor_loss_fn=floor_loss_fn,
        optimizer=final_optimizer,
        scheduler=final_scheduler,
        num_epochs=Config.NUM_EPOCHS,
        patience=Config.PATIENCE,
        coord_mean=coord_mean,
        coord_std=coord_std,
        loss_weights=best_weights,
        flood_level=Config.FLOOD_LEVEL,
        device=device
    )

    return best_weights, all_results


def visualize_grid_results(all_results):
    """可视化网格搜索结果"""

    # 提取数据
    alphas = [r['weights'][0] for r in all_results]
    betas = [r['weights'][1] for r in all_results]
    gammas = [r['weights'][2] for r in all_results]
    composite_scores = [r['composite_score'] for r in all_results]
    building_accs = [r['building_acc'] for r in all_results]
    floor_accs = [r['floor_acc'] for r in all_results]
    coord_maes = [r['coord_mae'] for r in all_results]

    # 创建热力图数据
    unique_alphas = sorted(set(alphas))
    unique_gammas = sorted(set(gammas))
    heatmap_data = np.zeros((len(unique_alphas), len(unique_gammas)))

    for i, alpha in enumerate(unique_alphas):
        for j, gamma in enumerate(unique_gammas):
            # 找到对应的结果（假设beta固定为1.0）
            for result in all_results:
                if abs(result['weights'][0] - alpha) < 0.01 and abs(result['weights'][2] - gamma) < 0.01:
                    heatmap_data[i, j] = result['composite_score']
                    break

    plt.figure(figsize=(10, 8))
    plt.imshow(heatmap_data, cmap='viridis', aspect='auto')
    plt.colorbar(label='综合评分')
    plt.xticks(range(len(unique_gammas)), [f'{g:.1f}' for g in unique_gammas])
    plt.yticks(range(len(unique_alphas)), [f'{a:.1f}' for a in unique_alphas])
    plt.xlabel('γ (坐标权重)')
    plt.ylabel('α (建筑权重)')
    plt.title('损失权重网格搜索热力图 (β=1.0固定)')
    plt.tight_layout()
    plt.savefig('grid_search_heatmap.png', dpi=300)
    print("✅ 网格搜索热力图已保存为 grid_search_heatmap.png")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].bar(range(len(composite_scores)), composite_scores)
    axes[0, 0].set_xlabel('权重组合')
    axes[0, 0].set_ylabel('综合评分')
    axes[0, 0].set_title('各权重组合综合评分')
    axes[0, 0].set_xticks(range(len(composite_scores)))

    x = range(len(all_results))
    axes[0, 1].plot(x, building_accs, 'o-', label='建筑准确率')
    axes[0, 1].plot(x, floor_accs, 's-', label='楼层准确率')
    axes[0, 1].set_xlabel('权重组合')
    axes[0, 1].set_ylabel('准确率')
    axes[0, 1].set_title('分类准确率对比')
    axes[0, 1].legend()
    axes[0, 1].set_xticks(x)

    axes[1, 0].bar(x, coord_maes)
    axes[1, 0].set_xlabel('权重组合')
    axes[1, 0].set_ylabel('MAE (米)')
    axes[1, 0].set_title('坐标定位误差对比')
    axes[1, 0].set_xticks(x)

    scatter = axes[1, 1].scatter(alphas, gammas, c=composite_scores, s=100, cmap='viridis')
    axes[1, 1].set_xlabel('α (建筑权重)')
    axes[1, 1].set_ylabel('γ (坐标权重)')
    axes[1, 1].set_title('权重参数分布')
    plt.colorbar(scatter, ax=axes[1, 1], label='综合评分')

    plt.tight_layout()
    plt.savefig('grid_search_analysis.png', dpi=300)
    print("✅ 网格搜索结果分析图已保存为 grid_search_analysis.png")

    plt.close('all')


def main():
    """主函数"""
    print("=" * 60)
    print("损失权重网格搜索")
    print("=" * 60)

    try:
        # 执行网格搜索
        best_weights, all_results = grid_search_weights()

        # 可视化结果
        visualize_grid_results(all_results)

        # 保存结果到文件
        results_df = pd.DataFrame(all_results)
        results_df.to_csv('grid_search_results.csv', index=False)
        print("✅ 网格搜索结果已保存到 grid_search_results.csv")

        print(f"\n🎯 最佳权重组合: α={best_weights[0]}, β={best_weights[1]}, γ={best_weights[2]}")

    except Exception as e:
        logger.error(f"网格搜索过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()