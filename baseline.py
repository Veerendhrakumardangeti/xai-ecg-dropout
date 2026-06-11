import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import time
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score as sklearn_f1_score
from utils import log_memory, plot_accuracy_time_multi, plot_accuracy_time_multi_test, plot_metrics, plot_metrics_test, plot_confusion_matrix


def compute_class_weights(cls_num_list):
    """implementation of CB loss weights based on the class frequencies"""
    num_classes = len(cls_num_list)
    beta = 0.9999

    effective_num = 1.0 - np.power(beta, cls_num_list)
    per_cls_weights = (1.0 - beta) / np.maximum(effective_num, 1e-8)
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * num_classes

    print(f"[CB Loss] beta={beta}, weights={per_cls_weights}")
    return torch.tensor(per_cls_weights, dtype=torch.float32), num_classes


class BaselineTrainer:
    """Standard supervised training loop without any data dropout/filtering"""

    def __init__(self, model_name, model, train_loader, val_loader, device,
                 epochs, save_path):
        self.model_name = model_name
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.save_path = save_path

        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_f1": [],
            "time_per_epoch": []
        }

    def _get_criterion(self):
        return nn.CrossEntropyLoss().to(self.device)

    def _validate(self, criterion):
        self.model.eval()
        test_correct = 0
        test_total = 0
        test_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating", leave=False):
                if len(batch) == 4:
                     x_meta, y_ecg, z_true, _ = batch
                else:
                     x_meta, y_ecg, z_true = batch
                x_meta = x_meta.to(self.device)
                y_ecg = y_ecg.to(self.device)
                z_true = z_true.to(self.device)

                outputs = self.model(x_meta, y_ecg)
                loss = criterion(outputs, z_true)

                test_loss += loss.item()
                preds = outputs.argmax(dim=1)
                test_correct += torch.sum(preds == z_true).item()
                test_total += z_true.size(0)

                all_preds.append(preds.cpu())
                all_labels.append(z_true.cpu())

        avg_loss = test_loss / len(self.val_loader)
        accuracy = test_correct / test_total if test_total > 0 else 0

        all_preds_np = torch.cat(all_preds, dim=0).numpy()
        all_labels_np = torch.cat(all_labels, dim=0).numpy()
        
        # macro F1: NORM class dominates, accuracy alone is misleading
        val_f1 = sklearn_f1_score(all_labels_np, all_preds_np,
                                  average="macro", zero_division=0)

        return avg_loss, accuracy, val_f1

    def _final_confusion_matrix(self):
        """evaluates on validation set and plots the confusion matrix"""
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Final CM Eval", leave=False):
                if len(batch) == 4:
                    x_meta, y_ecg, z_true, _ = batch
                else:
                    x_meta, y_ecg, z_true = batch
                x_meta = x_meta.to(self.device)
                y_ecg = y_ecg.to(self.device)
                z_true = z_true.to(self.device)

                outputs = self.model(x_meta, y_ecg)
                preds = outputs.argmax(dim=1)
                all_preds.append(preds.cpu())
                all_labels.append(z_true.cpu())

        all_preds_np = torch.cat(all_preds, dim=0).numpy()
        all_labels_np = torch.cat(all_labels, dim=0).numpy()

        class_names = None
        if hasattr(self.val_loader.dataset, 'label_names'):
            class_names = self.val_loader.dataset.label_names
        if class_names is None:
            unique_labels = sorted(set(all_labels_np.tolist()))
            class_names = [str(c) for c in unique_labels]

        save_cm_path = os.path.join(self.save_path, "confusion_matrix_baseline.png")
        plot_confusion_matrix(
            all_labels_np, all_preds_np,
            class_names=class_names,
            title=f"{self.model_name} - Baseline Confusion Matrix",
            save_path=save_cm_path
        )

    def train(self, batch_size):
        print("\n--- BASELINE TRAINING ---")
        self.model.to(self.device)
        criterion = self._get_criterion()
        optimizer = optim.AdamW(self.model.parameters(), lr=3e-4)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.98)

        ckpt_path = os.path.join(self.save_path, "model_baseline.pt")
        best_val_f1 = -np.inf
        start_time = time.time()
        total_samples_processed = 0

        for epoch in range(self.epochs):
            epoch_start = time.time()
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            print(f"Epoch [{epoch+1}/{self.epochs}]")
            pbar = tqdm(self.train_loader, desc="Training")
            for batch in pbar:
                if len(batch) == 4:
                     x_meta, y_ecg, z_true, _ = batch
                else:
                     x_meta, y_ecg, z_true = batch
                x_meta, y_ecg, z_true = x_meta.to(self.device), y_ecg.to(self.device), z_true.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(x_meta, y_ecg)
                loss = criterion(outputs, z_true)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                preds = outputs.argmax(dim=1)
                correct += (preds == z_true).sum().item()
                total += z_true.size(0)
                total_samples_processed += z_true.size(0)
                

            avg_train_loss = running_loss / len(self.train_loader)
            train_acc = correct / total if total > 0 else 0
            epoch_duration = time.time() - epoch_start

            val_loss, val_acc, val_f1 = self._validate(criterion)
            scheduler.step()

            self.history["train_loss"].append(avg_train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["val_f1"].append(val_f1)
            self.history["time_per_epoch"].append(epoch_duration)

            print(f"  Results: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"           Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(self.model.state_dict(), ckpt_path)
                print(f"  [*] Best Val F1 reached! Saved to {ckpt_path}")

            plot_metrics(self.history["train_loss"], self.history["train_acc"],
                         f"{self.model_name}_Baseline", os.path.join(self.save_path, "baseline_metrics.png"))
            plot_metrics_test(self.history["val_acc"],
                              f"{self.model_name}_Baseline", os.path.join(self.save_path, "baseline_test_metrics.png"))
            plot_accuracy_time_multi(self.model_name, self.history["train_acc"], self.history["time_per_epoch"],
                                     os.path.join(self.save_path, "accuracy_time.png"),
                                     os.path.join(self.save_path, "model_data.json"))
            plot_accuracy_time_multi_test(self.model_name, self.history["val_acc"], self.history["time_per_epoch"],
                                          [len(self.train_loader.dataset)] * len(self.history["val_acc"]),
                                          None,
                                          os.path.join(self.save_path, "test_accuracy_time.png"),
                                          os.path.join(self.save_path, "model_data.json"))

        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device, weights_only=True))
        
        wall_time = time.time() - start_time
        effective_epochs = total_samples_processed / len(self.train_loader.dataset)
        
        print(f"\nTraining Complete.")
        print(f"Baseline total time: {wall_time:.1f}s")
        print(f"Total samples processed: {total_samples_processed}")
        print(f"Effective Epochs: {effective_epochs:.2f}")

        self._final_confusion_matrix()

        return self.model
