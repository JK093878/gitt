import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


def calculate_positioning_errors(predictions, targets):
    """计算定位误差统计量"""
    errors = np.sqrt(np.sum((predictions - targets) ** 2, axis=1))

    stats = {
        'mean_error': np.mean(errors),
        'median_error': np.median(errors),
        'std_error': np.std(errors),
        '75th_percentile': np.percentile(errors, 75),
        '90th_percentile': np.percentile(errors, 90),
        'max_error': np.max(errors),
        'min_error': np.min(errors)
    }

    return errors, stats


def plot_confusion_matrix(y_true, y_pred, labels, title='Confusion Matrix'):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{title.lower().replace(" ", "_")}.png')
    plt.close()