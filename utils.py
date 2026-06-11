import psutil
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


def log_memory(start_time, end_time):
    """just a quick helper to check if we have memory leaks during training"""
    process = psutil.Process(os.getpid())
    print(f"Training Time: {end_time - start_time:.2f} seconds")
    print(f"Memory Consumption: {process.memory_info().rss / (1024 * 1024):.2f} MB")


def plot_metrics(losses, accuracies, title, save_path, loss_color='tab:red', acc_color='tab:green'):
    """plots the standard train vs epoch curves"""
    epochs = range(1, len(losses) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses, label='Loss', color=loss_color, marker='.')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{title} - Training Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracies, label='Accuracy', color=acc_color, marker='.')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'{title} - Training Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Metrics plot saved to: {save_path}")


def plot_metrics_test(accuracies, title, save_path, color='tab:blue'):
    """simple 1d plot for validation accuracy over time"""
    epochs = range(1, len(accuracies) + 1)
    plt.figure(figsize=(6, 5))

    plt.plot(epochs, accuracies, label='Test Accuracy', color=color, marker='.')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'{title} - Test Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Test metrics plot saved to: {save_path}")


def plot_accuracy_time_multi(model_name, accuracy, time_per_epoch,
                             save_path="accuracy_vs_time_plot.png",
                             data_file="model_data.json"):
    """
    this logs acc vs time to a shared json file so we can overlay multiple training
    runs on the same plot later
    """
    cumulative_time = [sum(time_per_epoch[:i + 1]) for i in range(len(time_per_epoch))]

    all_model_data = {}
    json_path = data_file if data_file.endswith('.json') else data_file + '_train.json'
    if os.path.exists(json_path) and os.path.isfile(json_path):
        with open(json_path, "r") as f:
            all_model_data = json.load(f)

    all_model_data[model_name] = {
        "cumulative_time": cumulative_time,
        "accuracy": accuracy
    }

    with open(json_path, "w") as f:
        json.dump(all_model_data, f, indent=4)

    plt.figure(figsize=(8, 6))
    for name, data in all_model_data.items():
        plt.plot(data["cumulative_time"], data["accuracy"], label=name, marker="o")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy vs Time")
    plt.legend()
    png_path = save_path if save_path.endswith('.png') else save_path + '_train_acc.png'
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    plt.savefig(png_path)
    plt.close()


def plot_kept_ratio(kept_ratios, title, save_path, color='tab:purple'):
    """tracks how much data is actually being retained by DBPD or SMRD each epoch"""
    epochs = range(1, len(kept_ratios) + 1)
    plt.figure(figsize=(6, 5))
    
    plt.plot(epochs, kept_ratios, label='Kept Ratio', color=color, marker='o')
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Full Dataset')
    
    plt.xlabel('Epochs')
    plt.ylabel('Ratio Kept')
    plt.ylim(0, 1.1)
    plt.title(f'{title} - Kept Ratio')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Kept Ratio plot saved to: {save_path}")


def plot_accuracy_time_multi_test(model_name, accuracy, time_per_epoch,
                                  samples_per_epoch, threshold,
                                  save_path="test_accuracy_vs_time_plot.png",
                                  data_file="model_data.json"):
    """same as the training one but for test/validation set results"""
    cumulative_time = [sum(time_per_epoch[:i + 1]) for i in range(len(time_per_epoch))]
    
    all_model_data = {}
    json_path = data_file if data_file.endswith('.json') else data_file + '_test.json'
    if os.path.exists(json_path) and os.path.isfile(json_path):
        with open(json_path, "r") as f:
            all_model_data = json.load(f)

    all_model_data[model_name] = {
        "cumulative_time": cumulative_time,
        "accuracy": accuracy,
        "samples_per_epoch": samples_per_epoch,
        "threshold": threshold
    }

    with open(json_path, "w") as f:
        json.dump(all_model_data, f, indent=4)

    plt.figure(figsize=(8, 6))
    for name, data in all_model_data.items():
        plt.plot(data["cumulative_time"], data["accuracy"], label=name, marker="o")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy vs Time")
    plt.legend()
    png_path = save_path if save_path.endswith('.png') else save_path + '_test_acc.png'
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    plt.savefig(png_path)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, title, save_path):
    """draws a standard confusion matrix with raw counts and percentages"""
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           ylabel='True Label', xlabel='Predicted Label',
           title=title)

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)",
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black',
                    fontsize=9)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")
