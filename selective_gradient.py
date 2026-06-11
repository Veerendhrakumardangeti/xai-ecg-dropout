import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Subset
import time
import json
import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from sklearn.metrics import f1_score as sklearn_f1_score

from baseline import compute_class_weights
from utils import log_memory, plot_accuracy_time_multi, plot_accuracy_time_multi_test, plot_metrics, plot_metrics_test, plot_kept_ratio, plot_confusion_matrix
from gradcam_utils import compute_cam_scores, get_target_layer


class TrainRevision:
    

    def __init__(self, model_name, model, train_loader, val_loader, device,
                 epochs, save_path, loss_threshold):
        self.model_name = model_name
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.save_path = save_path
        self.loss_threshold = loss_threshold

        self.history = {
            "train_loss_used": [],
            "train_acc_used": [],
            "train_acc_all": [],
            "val_loss": [],
            "val_acc": [],
            "val_f1": [],
            "time_per_epoch": [],
            "kept_ratio": [],
            "samples_used": [],
            "train_size": []
        }

    def _get_criterion(self, cls_num_list=None):
        if cls_num_list is not None:
            weights, _ = compute_class_weights(cls_num_list)
            return nn.CrossEntropyLoss(weight=weights.to(self.device), reduction='none').to(self.device)
        return nn.CrossEntropyLoss(reduction='none').to(self.device)

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
                losses = criterion(outputs, z_true)
                loss = losses.mean()

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
        val_f1 = sklearn_f1_score(all_labels_np, all_preds_np,
                                  average="macro", zero_division=0)

        return avg_loss, accuracy, val_f1

    def _final_confusion_matrix(self, method_name):
        
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

        save_cm_path = os.path.join(self.save_path, f"confusion_matrix_{method_name}.png")
        plot_confusion_matrix(
            all_labels_np, all_preds_np,
            class_names=class_names,
            title=f"{self.model_name} - {method_name} Confusion Matrix",
            save_path=save_cm_path
        )

    def train_with_revision(self, start_revision, cls_num_list, batch_size, warm_up_epochs=5):
        """
        Original DBPD logic. We drop samples if the model is already confident on them
        (i.e., confidence > loss_threshold).
        """
        self.model.to(self.device)
        criterion = self._get_criterion(cls_num_list)
        optimizer = optim.AdamW(self.model.parameters(), lr=3e-4, weight_decay=0.01)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.98)

        ckpt_path = os.path.join(self.save_path, f"model_dbpd_percentile.pt")
        best_val_f1 = -np.inf
        total_samples_processed = 0

        survival_log = defaultdict(list)
        label_log = defaultdict(int)  # type: ignore[assignment]

        start_time = time.time()
        

        for epoch in range(self.epochs):
            epoch_start = time.time()
            self.model.train()
            
            
            
            train_loss_sum_used = 0.0
            train_correct_used = 0
            train_total_used = 0
            
            train_correct_all = 0
            train_total_all = 0

            samples_kept_in_epoch = 0 
            total_samples_in_epoch = 0
            
            pbar = tqdm(self.train_loader, desc=f"Epoch [{epoch+1}/{self.epochs}]")
            
            for batch in pbar:
                if len(batch) == 4:
                    x_meta, y_ecg, z_true, idx = batch
                else:
                    x_meta, y_ecg, z_true = batch
                    idx = None

                x_meta = x_meta.to(self.device)
                y_ecg = y_ecg.to(self.device)
                z_true = z_true.to(self.device).long()
                
                with torch.no_grad():
                    outputs = self.model(x_meta, y_ecg)
                    preds = torch.argmax(outputs, dim=1)
                    
                    if self.loss_threshold == 0:
                        mask = torch.ne(preds, z_true)
                    else:
                        prob = torch.softmax(outputs, dim=1)
                        correct_class = prob[torch.arange(z_true.size(0)), z_true]
                        mask = torch.lt(correct_class, self.loss_threshold)

                with torch.no_grad():
                    train_correct_all += torch.sum(preds == z_true).item()
                    train_total_all += z_true.size(0)

                if epoch < warm_up_epochs:
                    if idx is not None:
                        survival_log[epoch].extend(idx.cpu().tolist())
                    for label in z_true.tolist():
                        label_log[int(label)] += 1

                    optimizer.zero_grad()
                    outputs_full = self.model(x_meta, y_ecg)
                    losses = criterion(outputs_full, z_true)
                    final_loss = losses.mean()
                    final_loss.backward()
                    optimizer.step()

                    num_kept = outputs_full.size(0)
                    train_loss_sum_used += losses.sum().item()

                    with torch.no_grad():
                        preds = outputs_full.argmax(dim=1)
                        train_correct_used += torch.sum(preds == z_true).item()
                        train_total_used += num_kept

                elif epoch < start_revision:
                    if not mask.any():
                        total_samples_in_epoch += z_true.size(0)
                        continue

                    # class-aware rescue
                    for cls in z_true.unique():
                        cls_indices = (z_true == cls).nonzero(as_tuple=True)[0]
                        if not mask[cls_indices].any():
                            mask[cls_indices[0]] = True

                    x_meta_misclassified = x_meta[mask]
                    y_ecg_misclassified = y_ecg[mask]
                    z_true_misclassified = z_true[mask]

                    if idx is not None:
                        kept_indices = idx[mask.cpu()].tolist()
                        survival_log[epoch].extend(kept_indices)
                    for label in z_true_misclassified.tolist():
                        label_log[int(label)] += 1

                    optimizer.zero_grad()
                    outputs_misclassified = self.model(x_meta_misclassified, y_ecg_misclassified)
                    losses = criterion(outputs_misclassified, z_true_misclassified)
                    final_loss = losses.mean()
                    final_loss.backward()
                    optimizer.step()

                    num_kept = outputs_misclassified.size(0)
                    train_loss_sum_used += losses.sum().item()

                    with torch.no_grad():
                        preds_kept = outputs_misclassified.argmax(dim=1)
                        train_correct_used += torch.sum(preds_kept == z_true_misclassified).item()
                        train_total_used += num_kept

                else:
                    if idx is not None:
                        survival_log[epoch].extend(idx.cpu().tolist())
                    for label in z_true.tolist():
                        label_log[int(label)] += 1

                    optimizer.zero_grad()
                    outputs_full = self.model(x_meta, y_ecg)
                    losses = criterion(outputs_full, z_true)
                    final_loss = losses.mean()
                    final_loss.backward()
                    optimizer.step()

                    num_kept = outputs_full.size(0)
                    train_loss_sum_used += losses.sum().item()

                    with torch.no_grad():
                        preds = outputs_full.argmax(dim=1)
                        train_correct_used += torch.sum(preds == z_true).item()
                        train_total_used += num_kept

                samples_kept_in_epoch += num_kept
                total_samples_in_epoch += z_true.size(0)
                total_samples_processed += num_kept
                
                batch_ratio = num_kept / z_true.size(0) if z_true.size(0) > 0 else 0
                pbar.set_postfix({'loss_used': f"{final_loss.item():.4f}", 'ratio': f"{batch_ratio:.2f}"})

            epoch_loss_used = train_loss_sum_used / train_total_used if train_total_used > 0 else 0.0
            epoch_acc_used = train_correct_used / train_total_used if train_total_used > 0 else 0.0
            epoch_acc_all = train_correct_all / train_total_all if train_total_all > 0 else 0.0
            
            epoch_kept_ratio = samples_kept_in_epoch / total_samples_in_epoch if total_samples_in_epoch > 0 else 1.0
            
            val_loss, val_acc, val_f1 = self._validate(criterion)
            
            epoch_duration = time.time() - epoch_start
            scheduler.step()
            
            print(f"Epoch [{epoch+1}/{self.epochs}] "
                  f"Loss(Used): {epoch_loss_used:.4f} Acc(Used): {epoch_acc_used:.4f} Acc(All): {epoch_acc_all:.4f} "
                  f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f} Val F1: {val_f1:.4f} "
                  f"Kept: {epoch_kept_ratio:.2%}")
            
            self.history["train_loss_used"].append(epoch_loss_used)
            self.history["train_acc_used"].append(epoch_acc_used)
            self.history["train_acc_all"].append(epoch_acc_all)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["val_f1"].append(val_f1)
            self.history["time_per_epoch"].append(epoch_duration)
            self.history["kept_ratio"].append(epoch_kept_ratio)
            self.history["samples_used"].append(samples_kept_in_epoch)
            self.history["train_size"].append(total_samples_in_epoch)
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(self.model.state_dict(), ckpt_path)
                print(f"  [*] Best Val F1 reached! Saved to {ckpt_path}")

            plot_metrics(
                self.history["train_loss_used"],
                self.history["train_acc_used"],
                f"{self.model_name} (Used Samples)",
                os.path.join(self.save_path, "metrics_used.png")
            )
            
            plot_metrics(
                self.history["train_loss_used"],
                self.history["train_acc_all"],
                f"{self.model_name} (All Samples)",
                os.path.join(self.save_path, "metrics_all.png")
            )

            plot_kept_ratio(
                self.history["kept_ratio"],
                f"{self.model_name} Kept Ratio",
                os.path.join(self.save_path, "kept_ratio.png")
            )
            
            plot_metrics_test(
                self.history["val_acc"],
                f"{self.model_name}_Validation", 
                os.path.join(self.save_path, "val_metrics.png")
            )

        plot_accuracy_time_multi_test(self.model_name, self.history["val_acc"], self.history["time_per_epoch"],
                                      self.history["samples_used"], self.loss_threshold,
                                      os.path.join(self.save_path, "test_accuracy_time.png"),
                                      os.path.join(self.save_path, "model_data.json"))

        if os.path.exists(ckpt_path):
            self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device, weights_only=True))
        
        effective_epochs = total_samples_processed / len(self.train_loader.dataset)
        total_time = time.time() - start_time
        print(f"\nTraining Complete.")
        print(f"Total Training Time: {total_time/60:.2f} minutes")
        print(f"Total samples processed: {total_samples_processed}")
        print(f"Effective Epochs: {effective_epochs:.2f}")

        survival_log_path = os.path.join(self.save_path, "survival_log.json")
        with open(survival_log_path, "w") as f:
            json.dump({str(k): v for k, v in survival_log.items()}, f, indent=2)
        print(f"Survival log saved to {survival_log_path}")

        label_log_path = os.path.join(self.save_path, "label_log.json")
        with open(label_log_path, "w") as f:
            json.dump(dict(label_log), f, indent=2)
        print(f"Label log saved to {label_log_path}")

        with open(os.path.join(self.save_path, "dbpd_history.json"), 'w') as f:
            json.dump(self.history, f, indent=2)

        self._final_confusion_matrix("dbpd")

        return self.model, total_samples_processed

    def train_with_smrd_matched(self, start_revision, cls_num_list, batch_size, warm_up_epochs=5):
        """
        Random dropout baselines (SMRD). We figure out how many samples DBPD *would* have kept,
        and then just randomly pick that exactly many samples to keep instead.
        """
        self.model.to(self.device)
        criterion = self._get_criterion(cls_num_list)
        optimizer = optim.AdamW(self.model.parameters(), lr=3e-4, weight_decay=0.01)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.98)

        ckpt_path = os.path.join(self.save_path, "model_smrd.pt")
        best_val_f1 = -np.inf
        total_samples_processed = 0

        start_time = time.time()

        for epoch in range(self.epochs):
            epoch_start = time.time()
            self.model.train()

            running_loss = 0.0
            train_correct_used = 0
            train_total_used = 0

            train_correct_all = 0
            train_total_all = 0

            samples_kept_in_epoch = 0
            total_samples_in_epoch = 0

            pbar = tqdm(self.train_loader, desc=f"Epoch [{epoch+1}/{self.epochs}] SMRD")

            for batch in pbar:
                if len(batch) == 4:
                    x_meta, y_ecg, z_true, idx = batch
                else:
                    x_meta, y_ecg, z_true = batch

                x_meta = x_meta.to(self.device)
                y_ecg = y_ecg.to(self.device)
                z_true = z_true.to(self.device).long()
                current_batch_size = z_true.size(0)

                with torch.no_grad():
                    outputs = self.model(x_meta, y_ecg)
                    preds = torch.argmax(outputs, dim=1)
                    train_correct_all += torch.sum(preds == z_true).item()
                    train_total_all += current_batch_size

                    if self.loss_threshold == 0:
                        mask = torch.ne(preds, z_true)
                    else:
                        prob = torch.softmax(outputs, dim=1)
                        correct_class = prob[torch.arange(z_true.size(0)), z_true]
                        mask = torch.lt(correct_class, self.loss_threshold)

                    num_to_select = mask.sum().item()

                if epoch < warm_up_epochs:
                    optimizer.zero_grad()
                    outputs_full = self.model(x_meta, y_ecg)
                    loss = criterion(outputs_full, z_true).mean()
                    loss.backward()
                    optimizer.step()

                    num_kept = outputs_full.size(0)
                    running_loss += loss.item()

                    with torch.no_grad():
                        preds = outputs_full.argmax(dim=1)
                        train_correct_used += torch.sum(preds == z_true).item()
                        train_total_used += num_kept

                elif epoch < start_revision:
                    if num_to_select == 0:
                        total_samples_in_epoch += current_batch_size
                        continue

                    indices = torch.randperm(current_batch_size)[:num_to_select]
                    x_meta_sampled = x_meta[indices]
                    y_ecg_sampled = y_ecg[indices]
                    z_true_sampled = z_true[indices]

                    optimizer.zero_grad()
                    outputs_sampled = self.model(x_meta_sampled, y_ecg_sampled)
                    loss = criterion(outputs_sampled, z_true_sampled).mean()
                    loss.backward()
                    optimizer.step()

                    num_kept = outputs_sampled.size(0)
                    running_loss += loss.item()

                    with torch.no_grad():
                        preds_kept = outputs_sampled.argmax(dim=1)
                        train_correct_used += torch.sum(preds_kept == z_true_sampled).item()
                        train_total_used += num_kept

                else:
                    optimizer.zero_grad()
                    outputs_full = self.model(x_meta, y_ecg)
                    loss = criterion(outputs_full, z_true).mean()
                    loss.backward()
                    optimizer.step()

                    num_kept = outputs_full.size(0)
                    running_loss += loss.item()

                    with torch.no_grad():
                        preds = outputs_full.argmax(dim=1)
                        train_correct_used += torch.sum(preds == z_true).item()
                        train_total_used += num_kept

                samples_kept_in_epoch += num_kept
                total_samples_in_epoch += current_batch_size
                total_samples_processed += num_kept

                batch_ratio = num_kept / current_batch_size if current_batch_size > 0 else 0
                pbar.set_postfix({'loss': f"{loss.item():.4f}", 'ratio': f"{batch_ratio:.2f}"})

            epoch_loss_used = running_loss / len(self.train_loader)
            epoch_acc_used = train_correct_used / train_total_used if train_total_used > 0 else 0.0
            epoch_acc_all = train_correct_all / train_total_all if train_total_all > 0 else 0.0

            epoch_kept_ratio = samples_kept_in_epoch / total_samples_in_epoch if total_samples_in_epoch > 0 else 1.0

            val_loss, val_acc, val_f1 = self._validate(criterion)

            epoch_duration = time.time() - epoch_start
            scheduler.step()

            print(f"Epoch [{epoch+1}/{self.epochs}] "
                  f"Loss(Used): {epoch_loss_used:.4f} Acc(Used): {epoch_acc_used:.4f} Acc(All): {epoch_acc_all:.4f} "
                  f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f} Val F1: {val_f1:.4f} "
                  f"Kept: {epoch_kept_ratio:.2%}")

            self.history["train_loss_used"].append(epoch_loss_used)
            self.history["train_acc_used"].append(epoch_acc_used)
            self.history["train_acc_all"].append(epoch_acc_all)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["val_f1"].append(val_f1)
            self.history["time_per_epoch"].append(epoch_duration)
            self.history["kept_ratio"].append(epoch_kept_ratio)
            self.history["samples_used"].append(samples_kept_in_epoch)
            self.history["train_size"].append(total_samples_in_epoch)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(self.model.state_dict(), ckpt_path)
                print(f"  [*] Best Val F1 reached! Saved to {ckpt_path}")

            plot_metrics(
                self.history["train_loss_used"], self.history["train_acc_used"],
                f"{self.model_name}_SMRD (Used Samples)",
                os.path.join(self.save_path, "metrics_used.png"),
                loss_color='tab:orange', acc_color='teal'
            )
            plot_metrics(
                self.history["train_loss_used"], self.history["train_acc_all"],
                f"{self.model_name}_SMRD (All Samples)",
                os.path.join(self.save_path, "metrics_all.png"),
                loss_color='tab:orange', acc_color='teal'
            )
            plot_kept_ratio(
                self.history["kept_ratio"],
                f"{self.model_name} SMRD Kept Ratio",
                os.path.join(self.save_path, "kept_ratio.png"),
                color='tab:orange'
            )
            plot_metrics_test(
                self.history["val_acc"],
                f"{self.model_name}_SMRD_Validation",
                os.path.join(self.save_path, "val_metrics.png"),
                color='darkcyan'
            )

        plot_accuracy_time_multi_test(
            self.model_name, self.history["val_acc"], self.history["time_per_epoch"],
            self.history["samples_used"], self.loss_threshold,
            os.path.join(self.save_path, "test_accuracy_time.png"),
            os.path.join(self.save_path, "model_data.json")
        )

        if os.path.exists(ckpt_path):
            self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device, weights_only=True))

        effective_epochs = total_samples_processed / len(self.train_loader.dataset)
        total_time = time.time() - start_time
        print(f"\nSMRD Training Complete.")
        print(f"Total Training Time: {total_time/60:.2f} minutes")
        print(f"Total samples processed: {total_samples_processed}")
        print(f"Effective Epochs: {effective_epochs:.2f}")

        with open(os.path.join(self.save_path, "smrd_history.json"), 'w') as f:
            json.dump(self.history, f, indent=2)

        self._final_confusion_matrix("smrd")

        return self.model, total_samples_processed

    def train_with_srd_decay(self, start_revision, cls_num_list, batch_size, srd_decay=0.99, warm_up_epochs=5):
        """
        Static Random Dropout (SRD). This one just drops a fixed percentage of the dataset randomly
        every epoch, decaying by 0.99 each step so we drop more as training goes on.
        """
        self.model.to(self.device)
        criterion = self._get_criterion(cls_num_list)
        optimizer = optim.AdamW(self.model.parameters(), lr=3e-4, weight_decay=0.01)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.98)

        ckpt_path = os.path.join(self.save_path, "model_srd.pt")
        best_val_f1 = -np.inf
        total_samples_processed = 0

        start_time = time.time()

        for epoch in range(self.epochs):
            epoch_start = time.time()
            self.model.train()

            running_loss = 0.0
            train_correct_used = 0
            train_total_used = 0

            train_correct_all = 0
            train_total_all = 0

            samples_kept_in_epoch = 0
            total_samples_in_epoch = 0

            decay_factor = srd_decay ** epoch

            pbar = tqdm(self.train_loader, desc=f"Epoch [{epoch+1}/{self.epochs}] SRD (decay={decay_factor:.4f})")

            for batch in pbar:
                if len(batch) == 4:
                    x_meta, y_ecg, z_true, idx = batch
                else:
                    x_meta, y_ecg, z_true = batch

                x_meta = x_meta.to(self.device)
                y_ecg = y_ecg.to(self.device)
                z_true = z_true.to(self.device).long()
                current_batch_size = z_true.size(0)

                if epoch < warm_up_epochs:
                    optimizer.zero_grad()
                    outputs_full = self.model(x_meta, y_ecg)
                    loss = criterion(outputs_full, z_true).mean()
                    loss.backward()
                    optimizer.step()

                    num_kept = outputs_full.size(0)
                    running_loss += loss.item()

                    with torch.no_grad():
                        preds = outputs_full.argmax(dim=1)
                        train_correct_used += torch.sum(preds == z_true).item()
                        train_total_used += num_kept
                        train_correct_all += torch.sum(preds == z_true).item()
                        train_total_all += num_kept

                elif epoch < start_revision:
                    selected_count = int(decay_factor * current_batch_size)

                    if selected_count == 0:
                        total_samples_in_epoch += current_batch_size
                        continue

                    selected_indices = torch.randperm(current_batch_size)[:selected_count]
                    x_meta_selected = x_meta[selected_indices]
                    y_ecg_selected = y_ecg[selected_indices]
                    z_true_selected = z_true[selected_indices]

                    optimizer.zero_grad()
                    outputs_selected = self.model(x_meta_selected, y_ecg_selected)
                    loss = criterion(outputs_selected, z_true_selected).mean()
                    loss.backward()
                    optimizer.step()

                    num_kept = selected_count
                    running_loss += loss.item()

                    with torch.no_grad():
                        preds_all = torch.argmax(self.model(x_meta, y_ecg), dim=1)
                        train_correct_all += torch.sum(preds_all == z_true).item()
                        train_total_all += current_batch_size

                    with torch.no_grad():
                        preds_kept = outputs_selected.argmax(dim=1)
                        train_correct_used += torch.sum(preds_kept == z_true_selected).item()
                        train_total_used += num_kept

                else:
                    optimizer.zero_grad()
                    outputs_full = self.model(x_meta, y_ecg)
                    loss = criterion(outputs_full, z_true).mean()
                    loss.backward()
                    optimizer.step()

                    num_kept = outputs_full.size(0)
                    running_loss += loss.item()

                    with torch.no_grad():
                        preds = outputs_full.argmax(dim=1)
                        train_correct_used += torch.sum(preds == z_true).item()
                        train_total_used += num_kept
                        train_correct_all += torch.sum(preds == z_true).item()
                        train_total_all += num_kept

                samples_kept_in_epoch += num_kept
                total_samples_in_epoch += current_batch_size
                total_samples_processed += num_kept

                batch_ratio = num_kept / current_batch_size if current_batch_size > 0 else 0
                pbar.set_postfix({'loss': f"{loss.item():.4f}", 'ratio': f"{batch_ratio:.2f}"})

            epoch_loss_used = running_loss / len(self.train_loader)
            epoch_acc_used = train_correct_used / train_total_used if train_total_used > 0 else 0.0
            epoch_acc_all = train_correct_all / train_total_all if train_total_all > 0 else 0.0

            epoch_kept_ratio = samples_kept_in_epoch / total_samples_in_epoch if total_samples_in_epoch > 0 else 1.0

            val_loss, val_acc, val_f1 = self._validate(criterion)

            epoch_duration = time.time() - epoch_start
            scheduler.step()

            print(f"Epoch [{epoch+1}/{self.epochs}] "
                  f"Loss(Used): {epoch_loss_used:.4f} Acc(Used): {epoch_acc_used:.4f} Acc(All): {epoch_acc_all:.4f} "
                  f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f} Val F1: {val_f1:.4f} "
                  f"Kept: {epoch_kept_ratio:.2%} Decay: {decay_factor:.4f}")

            self.history["train_loss_used"].append(epoch_loss_used)
            self.history["train_acc_used"].append(epoch_acc_used)
            self.history["train_acc_all"].append(epoch_acc_all)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["val_f1"].append(val_f1)
            self.history["time_per_epoch"].append(epoch_duration)
            self.history["kept_ratio"].append(epoch_kept_ratio)
            self.history["samples_used"].append(samples_kept_in_epoch)
            self.history["train_size"].append(total_samples_in_epoch)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(self.model.state_dict(), ckpt_path)
                print(f"  [*] Best Val F1 reached! Saved to {ckpt_path}")

            plot_metrics(
                self.history["train_loss_used"], self.history["train_acc_used"],
                f"{self.model_name}_SRD (Used Samples)",
                os.path.join(self.save_path, "metrics_used.png"),
                loss_color='crimson', acc_color='olive'
            )
            plot_metrics(
                self.history["train_loss_used"], self.history["train_acc_all"],
                f"{self.model_name}_SRD (All Samples)",
                os.path.join(self.save_path, "metrics_all.png"),
                loss_color='crimson', acc_color='olive'
            )
            plot_kept_ratio(
                self.history["kept_ratio"],
                f"{self.model_name} SRD Kept Ratio",
                os.path.join(self.save_path, "kept_ratio.png"),
                color='goldenrod'
            )
            plot_metrics_test(
                self.history["val_acc"],
                f"{self.model_name}_SRD_Validation",
                os.path.join(self.save_path, "val_metrics.png"),
                color='navy'
            )

        plot_accuracy_time_multi_test(
            self.model_name, self.history["val_acc"], self.history["time_per_epoch"],
            self.history["samples_used"], None,
            os.path.join(self.save_path, "test_accuracy_time.png"),
            os.path.join(self.save_path, "model_data.json")
        )

        if os.path.exists(ckpt_path):
            self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device, weights_only=True))

        effective_epochs = total_samples_processed / len(self.train_loader.dataset)
        total_time = time.time() - start_time
        print(f"\nSRD Training Complete.")
        print(f"Total Training Time: {total_time/60:.2f} minutes")
        print(f"Total samples processed: {total_samples_processed}")
        print(f"Effective Epochs: {effective_epochs:.2f}")

        with open(os.path.join(self.save_path, "srd_history.json"), 'w') as f:
            json.dump(self.history, f, indent=2)

        self._final_confusion_matrix("srd")

        return self.model, total_samples_processed

    def train_with_xai_srd(self, start_revision, cls_num_list, batch_size,
                            cam_keep_frac=0.7, cam_start_epoch=3, srd_decay=0.99, warm_up_epochs=5):
        """
        XAI-SRD: Combine random dropout with Grad-CAM filtering. We randomly grab our subset,
        then run CAM and drop the ones where the model is looking at the wrong part of the ECG.
        """
        self.model.to(self.device)
        criterion = self._get_criterion(cls_num_list)
        optimizer = optim.AdamW(self.model.parameters(), lr=3e-4, weight_decay=0.01)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.98)

        ckpt_path = os.path.join(self.save_path, "model_xai_srd.pt")
        best_val_f1 = -np.inf
        total_samples_processed = 0
        total_cam_filtered = 0

        survival_log = defaultdict(list)
        label_log = defaultdict(int)  # type: ignore[assignment]

        target_layers = [get_target_layer(self.model)]

        start_time = time.time()

        for epoch in range(self.epochs):
            epoch_start = time.time()
            self.model.train()

            train_loss_sum_used = 0.0
            train_correct_used = 0
            train_total_used = 0

            train_correct_all = 0
            train_total_all = 0

            samples_kept_in_epoch = 0
            total_samples_in_epoch = 0
            cam_filtered_in_epoch = 0

            decay_factor = srd_decay ** epoch

            pbar = tqdm(self.train_loader, desc=f"Epoch [{epoch+1}/{self.epochs}] XAI-SRD (decay={decay_factor:.4f})")

            for batch in pbar:
                if len(batch) == 4:
                    x_meta, y_ecg, z_true, idx = batch
                else:
                    x_meta, y_ecg, z_true = batch
                    idx = None

                x_meta = x_meta.to(self.device)
                y_ecg = y_ecg.to(self.device)
                z_true = z_true.to(self.device).long()
                current_batch_size = z_true.size(0)

                with torch.no_grad():
                    outputs_all = self.model(x_meta, y_ecg)
                    preds_all = torch.argmax(outputs_all, dim=1)
                    prob_all = torch.softmax(outputs_all, dim=1)
                    correct_class = prob_all[torch.arange(current_batch_size), z_true]
                    train_correct_all += torch.sum(preds_all == z_true).item()
                    train_total_all += current_batch_size

                if epoch < warm_up_epochs:
                    if idx is not None:
                        survival_log[epoch].extend(idx.cpu().tolist())
                    for label in z_true.tolist():
                        label_log[int(label)] += 1

                    optimizer.zero_grad()
                    outputs_full = self.model(x_meta, y_ecg)
                    losses = criterion(outputs_full, z_true)
                    final_loss = losses.mean()
                    final_loss.backward()
                    optimizer.step()

                    num_kept = outputs_full.size(0)
                    train_loss_sum_used += losses.sum().item()

                    with torch.no_grad():
                        preds = outputs_full.argmax(dim=1)
                        train_correct_used += torch.sum(preds == z_true).item()
                        train_total_used += num_kept

                elif epoch < start_revision:
                    selected_count = int(decay_factor * current_batch_size)

                    if selected_count == 0:
                        total_samples_in_epoch += current_batch_size
                        continue

                    selected_indices = torch.randperm(current_batch_size)[:selected_count]
                    x_meta_selected = x_meta[selected_indices]
                    y_ecg_selected = y_ecg[selected_indices]
                    z_true_selected = z_true[selected_indices]

                    if epoch >= warm_up_epochs + cam_start_epoch and len(z_true_selected) > 1:
                        # combined = (1-conf) × cam_focus, all classes
                        cam_scores = compute_cam_scores(
                            self.model,
                            x_meta_selected, y_ecg_selected, z_true_selected,
                            target_layers, self.device
                        )
                        conf_selected = correct_class[selected_indices]
                        combined = (1.0 - conf_selected) * cam_scores
                        k = max(1, int(len(combined) * cam_keep_frac))
                        _, top_k = torch.topk(combined, k)
                        keep_mask = torch.zeros(len(z_true_selected), dtype=torch.bool, device=self.device)
                        keep_mask[top_k] = True
                        # class-aware rescue
                        for cls in z_true_selected.unique():
                            cls_pos = (z_true_selected == cls).nonzero(as_tuple=True)[0]
                            if not keep_mask[cls_pos].any():
                                best = cls_pos[combined[cls_pos].argmax()]
                                keep_mask[best] = True
                        cam_filtered_in_epoch += (~keep_mask).sum().item()
                        x_meta_selected = x_meta_selected[keep_mask]
                        y_ecg_selected = y_ecg_selected[keep_mask]
                        z_true_selected = z_true_selected[keep_mask]

                    if idx is not None:
                        survival_log[epoch].extend(idx[selected_indices].cpu().tolist())
                    for label in z_true_selected.tolist():
                        label_log[int(label)] += 1

                    optimizer.zero_grad()
                    outputs_selected = self.model(x_meta_selected, y_ecg_selected)
                    losses = criterion(outputs_selected, z_true_selected)
                    final_loss = losses.mean()
                    final_loss.backward()
                    optimizer.step()

                    num_kept = outputs_selected.size(0)
                    train_loss_sum_used += losses.sum().item()

                    with torch.no_grad():
                        preds_kept = outputs_selected.argmax(dim=1)
                        train_correct_used += torch.sum(preds_kept == z_true_selected).item()
                        train_total_used += num_kept

                else:
                    if idx is not None:
                        survival_log[epoch].extend(idx.cpu().tolist())
                    for label in z_true.tolist():
                        label_log[int(label)] += 1

                    optimizer.zero_grad()
                    outputs_full = self.model(x_meta, y_ecg)
                    losses = criterion(outputs_full, z_true)
                    final_loss = losses.mean()
                    final_loss.backward()
                    optimizer.step()

                    num_kept = outputs_full.size(0)
                    train_loss_sum_used += losses.sum().item()

                    with torch.no_grad():
                        preds = outputs_full.argmax(dim=1)
                        train_correct_used += torch.sum(preds == z_true).item()
                        train_total_used += num_kept

                samples_kept_in_epoch += num_kept
                total_samples_in_epoch += current_batch_size
                total_samples_processed += num_kept

                batch_ratio = num_kept / current_batch_size if current_batch_size > 0 else 0
                pbar.set_postfix({'loss': f"{final_loss.item():.4f}", 'ratio': f"{batch_ratio:.2f}"})

            epoch_loss_used = train_loss_sum_used / train_total_used if train_total_used > 0 else 0.0
            epoch_acc_used = train_correct_used / train_total_used if train_total_used > 0 else 0.0
            epoch_acc_all = train_correct_all / train_total_all if train_total_all > 0 else 0.0
            epoch_kept_ratio = samples_kept_in_epoch / total_samples_in_epoch if total_samples_in_epoch > 0 else 1.0

            val_loss, val_acc, val_f1 = self._validate(criterion)

            epoch_duration = time.time() - epoch_start
            scheduler.step()
            total_cam_filtered += cam_filtered_in_epoch

            print(f"Epoch [{epoch+1}/{self.epochs}] "
                  f"Loss(Used): {epoch_loss_used:.4f} Acc(Used): {epoch_acc_used:.4f} Acc(All): {epoch_acc_all:.4f} "
                  f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f} Val F1: {val_f1:.4f} "
                  f"Kept: {epoch_kept_ratio:.2%} Decay: {decay_factor:.4f} CAM-filtered: {cam_filtered_in_epoch}")

            self.history["train_loss_used"].append(epoch_loss_used)
            self.history["train_acc_used"].append(epoch_acc_used)
            self.history["train_acc_all"].append(epoch_acc_all)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["val_f1"].append(val_f1)
            self.history["time_per_epoch"].append(epoch_duration)
            self.history["kept_ratio"].append(epoch_kept_ratio)
            self.history["samples_used"].append(samples_kept_in_epoch)
            self.history["train_size"].append(total_samples_in_epoch)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(self.model.state_dict(), ckpt_path)
                print(f"  [*] Best Val F1 reached! Saved to {ckpt_path}")

            plot_metrics(
                self.history["train_loss_used"], self.history["train_acc_used"],
                f"{self.model_name}_XAI-SRD (Used)",
                os.path.join(self.save_path, "metrics_used.png"),
                loss_color='crimson', acc_color='olive'
            )
            plot_metrics(
                self.history["train_loss_used"], self.history["train_acc_all"],
                f"{self.model_name}_XAI-SRD (All)",
                os.path.join(self.save_path, "metrics_all.png"),
                loss_color='crimson', acc_color='olive'
            )
            plot_kept_ratio(
                self.history["kept_ratio"],
                f"{self.model_name} XAI-SRD Kept Ratio",
                os.path.join(self.save_path, "kept_ratio.png"),
                color='goldenrod'
            )
            plot_metrics_test(
                self.history["val_acc"],
                f"{self.model_name}_XAI-SRD_Validation",
                os.path.join(self.save_path, "val_metrics.png"),
                color='navy'
            )

        plot_accuracy_time_multi_test(
            self.model_name, self.history["val_acc"], self.history["time_per_epoch"],
            self.history["samples_used"], None,
            os.path.join(self.save_path, "test_accuracy_time.png"),
            os.path.join(self.save_path, "model_data.json")
        )

        if os.path.exists(ckpt_path):
            self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device, weights_only=True))

        effective_epochs = total_samples_processed / len(self.train_loader.dataset)
        total_time = time.time() - start_time
        print(f"\nXAI-SRD Training Complete.")
        print(f"Total Training Time: {total_time/60:.2f} minutes")
        print(f"Total samples processed: {total_samples_processed}")
        print(f"Effective Epochs: {effective_epochs:.2f}")
        print(f"Total CAM-filtered samples: {total_cam_filtered}")

        survival_log_path = os.path.join(self.save_path, "survival_log.json")
        with open(survival_log_path, "w") as f:
            json.dump({str(k): v for k, v in survival_log.items()}, f, indent=2)
        print(f"Survival log saved to {survival_log_path}")

        label_log_path = os.path.join(self.save_path, "label_log.json")
        with open(label_log_path, "w") as f:
            json.dump(dict(label_log), f, indent=2)
        print(f"Label log saved to {label_log_path}")

        with open(os.path.join(self.save_path, "xai_srd_history.json"), 'w') as f:
            json.dump(self.history, f, indent=2)

        self._final_confusion_matrix("xai_srd")

        return self.model, total_samples_processed

    def train_with_xai_dbpd(self, start_revision, cls_num_list, batch_size,
                             cam_keep_frac=0.7, cam_start_epoch=3, warm_up_epochs=5):
        """
        XAI-DBPD: First filter out easy samples based on confidence (DBPD),
        then filter the hard samples using Grad-CAM focused-class filtering.
        """
        self.model.to(self.device)
        criterion = self._get_criterion(cls_num_list)
        optimizer = optim.AdamW(self.model.parameters(), lr=3e-4, weight_decay=0.01)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.98)

        ckpt_path = os.path.join(self.save_path, "model_xai_dbpd.pt")
        best_val_f1 = -np.inf
        total_samples_processed = 0
        total_cam_filtered = 0

        survival_log = defaultdict(list)
        label_log = defaultdict(int)  # type: ignore[assignment]

        target_layers = [get_target_layer(self.model)]

        start_time = time.time()

        for epoch in range(self.epochs):
            epoch_start = time.time()
            self.model.train()

            train_loss_sum_used = 0.0
            train_correct_used = 0
            train_total_used = 0

            train_correct_all = 0
            train_total_all = 0

            samples_kept_in_epoch = 0
            total_samples_in_epoch = 0
            cam_filtered_in_epoch = 0

            pbar = tqdm(self.train_loader, desc=f"Epoch [{epoch+1}/{self.epochs}] XAI-DBPD")

            for batch in pbar:
                if len(batch) == 4:
                    x_meta, y_ecg, z_true, idx = batch
                else:
                    x_meta, y_ecg, z_true = batch
                    idx = None

                x_meta = x_meta.to(self.device)
                y_ecg = y_ecg.to(self.device)
                z_true = z_true.to(self.device).long()
                current_batch_size = z_true.size(0)

                with torch.no_grad():
                    outputs = self.model(x_meta, y_ecg)
                    preds = torch.argmax(outputs, dim=1)
                    prob = torch.softmax(outputs, dim=1)
                    correct_class = prob[torch.arange(z_true.size(0)), z_true]
                    if self.loss_threshold == 0:
                        mask = torch.ne(preds, z_true)
                    else:
                        mask = torch.lt(correct_class, self.loss_threshold)

                with torch.no_grad():
                    train_correct_all += torch.sum(preds == z_true).item()
                    train_total_all += current_batch_size

                if epoch < warm_up_epochs:
                    if idx is not None:
                        survival_log[epoch].extend(idx.cpu().tolist())
                    for label in z_true.tolist():
                        label_log[int(label)] += 1

                    optimizer.zero_grad()
                    outputs_full = self.model(x_meta, y_ecg)
                    losses = criterion(outputs_full, z_true)
                    final_loss = losses.mean()
                    final_loss.backward()
                    optimizer.step()

                    num_kept = outputs_full.size(0)
                    train_loss_sum_used += losses.sum().item()

                    with torch.no_grad():
                        preds = outputs_full.argmax(dim=1)
                        train_correct_used += torch.sum(preds == z_true).item()
                        train_total_used += num_kept

                elif epoch < start_revision:
                    if not mask.any():
                        total_samples_in_epoch += current_batch_size
                        continue

                    hard_indices = mask.nonzero(as_tuple=True)[0]
                    x_meta_hard = x_meta[hard_indices]
                    y_ecg_hard = y_ecg[hard_indices]
                    z_true_hard = z_true[hard_indices]
                    surviving_hard_indices = hard_indices.clone()

                    if epoch >= warm_up_epochs + cam_start_epoch and len(z_true_hard) > 1:
                        # combined = (1-conf) × cam_focus, all classes
                        cam_scores = compute_cam_scores(
                            self.model,
                            x_meta_hard, y_ecg_hard, z_true_hard,
                            target_layers, self.device
                        )
                        conf_hard = correct_class[hard_indices]
                        combined = (1.0 - conf_hard) * cam_scores
                        k = max(1, int(len(combined) * cam_keep_frac))
                        _, top_k = torch.topk(combined, k)
                        keep_mask = torch.zeros(len(z_true_hard), dtype=torch.bool, device=self.device)
                        keep_mask[top_k] = True
                        # class-aware rescue
                        for cls in z_true_hard.unique():
                            cls_pos = (z_true_hard == cls).nonzero(as_tuple=True)[0]
                            if not keep_mask[cls_pos].any():
                                best = cls_pos[combined[cls_pos].argmax()]
                                keep_mask[best] = True
                        cam_filtered_in_epoch += (~keep_mask).sum().item()
                        x_meta_hard = x_meta_hard[keep_mask]
                        y_ecg_hard = y_ecg_hard[keep_mask]
                        z_true_hard = z_true_hard[keep_mask]
                        surviving_hard_indices = surviving_hard_indices[keep_mask]

                    if idx is not None:
                        survival_log[epoch].extend(idx[surviving_hard_indices.cpu()].tolist())
                    for label in z_true_hard.tolist():
                        label_log[int(label)] += 1

                    optimizer.zero_grad()
                    outputs_hard = self.model(x_meta_hard, y_ecg_hard)
                    losses = criterion(outputs_hard, z_true_hard)
                    final_loss = losses.mean()
                    final_loss.backward()
                    optimizer.step()

                    num_kept = outputs_hard.size(0)
                    train_loss_sum_used += losses.sum().item()

                    with torch.no_grad():
                        preds_kept = outputs_hard.argmax(dim=1)
                        train_correct_used += torch.sum(preds_kept == z_true_hard).item()
                        train_total_used += num_kept

                else:
                    if idx is not None:
                        survival_log[epoch].extend(idx.cpu().tolist())
                    for label in z_true.tolist():
                        label_log[int(label)] += 1

                    optimizer.zero_grad()
                    outputs_full = self.model(x_meta, y_ecg)
                    losses = criterion(outputs_full, z_true)
                    final_loss = losses.mean()
                    final_loss.backward()
                    optimizer.step()

                    num_kept = outputs_full.size(0)
                    train_loss_sum_used += losses.sum().item()

                    with torch.no_grad():
                        preds = outputs_full.argmax(dim=1)
                        train_correct_used += torch.sum(preds == z_true).item()
                        train_total_used += num_kept

                samples_kept_in_epoch += num_kept
                total_samples_in_epoch += current_batch_size
                total_samples_processed += num_kept

                batch_ratio = num_kept / current_batch_size if current_batch_size > 0 else 0
                pbar.set_postfix({'loss': f"{final_loss.item():.4f}", 'ratio': f"{batch_ratio:.2f}"})

            epoch_loss_used = train_loss_sum_used / train_total_used if train_total_used > 0 else 0.0
            epoch_acc_used = train_correct_used / train_total_used if train_total_used > 0 else 0.0
            epoch_acc_all = train_correct_all / train_total_all if train_total_all > 0 else 0.0
            epoch_kept_ratio = samples_kept_in_epoch / total_samples_in_epoch if total_samples_in_epoch > 0 else 1.0

            val_loss, val_acc, val_f1 = self._validate(criterion)

            epoch_duration = time.time() - epoch_start
            scheduler.step()
            total_cam_filtered += cam_filtered_in_epoch

            print(f"Epoch [{epoch+1}/{self.epochs}] "
                  f"Loss(Used): {epoch_loss_used:.4f} Acc(Used): {epoch_acc_used:.4f} Acc(All): {epoch_acc_all:.4f} "
                  f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f} Val F1: {val_f1:.4f} "
                  f"Kept: {epoch_kept_ratio:.2%} CAM-filtered: {cam_filtered_in_epoch}")

            self.history["train_loss_used"].append(epoch_loss_used)
            self.history["train_acc_used"].append(epoch_acc_used)
            self.history["train_acc_all"].append(epoch_acc_all)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["val_f1"].append(val_f1)
            self.history["time_per_epoch"].append(epoch_duration)
            self.history["kept_ratio"].append(epoch_kept_ratio)
            self.history["samples_used"].append(samples_kept_in_epoch)
            self.history["train_size"].append(total_samples_in_epoch)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(self.model.state_dict(), ckpt_path)
                print(f"  [*] Best Val F1 reached! Saved to {ckpt_path}")

            plot_metrics(
                self.history["train_loss_used"], self.history["train_acc_used"],
                f"{self.model_name}_XAI-DBPD (Used)",
                os.path.join(self.save_path, "metrics_used.png"),
                loss_color='darkred', acc_color='darkgreen'
            )
            plot_metrics(
                self.history["train_loss_used"], self.history["train_acc_all"],
                f"{self.model_name}_XAI-DBPD (All)",
                os.path.join(self.save_path, "metrics_all.png"),
                loss_color='darkred', acc_color='darkgreen'
            )
            plot_kept_ratio(
                self.history["kept_ratio"],
                f"{self.model_name} XAI-DBPD Kept Ratio",
                os.path.join(self.save_path, "kept_ratio.png"),
                color='darkred'
            )
            plot_metrics_test(
                self.history["val_acc"],
                f"{self.model_name}_XAI-DBPD_Validation",
                os.path.join(self.save_path, "val_metrics.png"),
                color='darkblue'
            )

        plot_accuracy_time_multi_test(
            self.model_name, self.history["val_acc"], self.history["time_per_epoch"],
            self.history["samples_used"], self.loss_threshold,
            os.path.join(self.save_path, "test_accuracy_time.png"),
            os.path.join(self.save_path, "model_data.json")
        )

        if os.path.exists(ckpt_path):
            self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device, weights_only=True))

        effective_epochs = total_samples_processed / len(self.train_loader.dataset)
        total_time = time.time() - start_time
        print(f"\nXAI-DBPD Training Complete.")
        print(f"Total Training Time: {total_time/60:.2f} minutes")
        print(f"Total samples processed: {total_samples_processed}")
        print(f"Effective Epochs: {effective_epochs:.2f}")
        print(f"Total CAM-filtered samples: {total_cam_filtered}")

        survival_log_path = os.path.join(self.save_path, "survival_log.json")
        with open(survival_log_path, "w") as f:
            json.dump({str(k): v for k, v in survival_log.items()}, f, indent=2)
        print(f"Survival log saved to {survival_log_path}")

        label_log_path = os.path.join(self.save_path, "label_log.json")
        with open(label_log_path, "w") as f:
            json.dump(dict(label_log), f, indent=2)
        print(f"Label log saved to {label_log_path}")

        with open(os.path.join(self.save_path, "xai_dbpd_history.json"), 'w') as f:
            json.dump(self.history, f, indent=2)

        self._final_confusion_matrix("xai_dbpd")

        return self.model, total_samples_processed

    def train_with_xai_smrd(self, start_revision, cls_num_list, batch_size,
                             cam_keep_frac=0.7, cam_start_epoch=3, warm_up_epochs=5):
        """
        XAI-SMRD: Like SMRD, but after randomly matching the DBPD sample count, we
        filter *those* using Grad-CAM. Mostly just an ablation study baseline.
        """
        self.model.to(self.device)
        criterion = self._get_criterion(cls_num_list)
        optimizer = optim.AdamW(self.model.parameters(), lr=3e-4, weight_decay=0.01)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.98)

        ckpt_path = os.path.join(self.save_path, "model_xai_smrd.pt")
        best_val_f1 = -np.inf
        total_samples_processed = 0
        total_cam_filtered = 0

        survival_log = defaultdict(list)
        label_log = defaultdict(int)  # type: ignore[assignment]

        target_layers = [get_target_layer(self.model)]

        start_time = time.time()

        for epoch in range(self.epochs):
            epoch_start = time.time()
            self.model.train()

            train_loss_sum_used = 0.0
            train_correct_used = 0
            train_total_used = 0

            train_correct_all = 0
            train_total_all = 0

            samples_kept_in_epoch = 0
            total_samples_in_epoch = 0
            cam_filtered_in_epoch = 0

            pbar = tqdm(self.train_loader, desc=f"Epoch [{epoch+1}/{self.epochs}] XAI-SMRD")

            for batch in pbar:
                if len(batch) == 4:
                    x_meta, y_ecg, z_true, idx = batch
                else:
                    x_meta, y_ecg, z_true = batch

                x_meta = x_meta.to(self.device)
                y_ecg = y_ecg.to(self.device)
                z_true = z_true.to(self.device).long()
                current_batch_size = z_true.size(0)

                with torch.no_grad():
                    outputs = self.model(x_meta, y_ecg)
                    preds = torch.argmax(outputs, dim=1)
                    prob = torch.softmax(outputs, dim=1)
                    correct_class = prob[torch.arange(z_true.size(0)), z_true]
                    train_correct_all += torch.sum(preds == z_true).item()
                    train_total_all += current_batch_size

                    if self.loss_threshold == 0:
                        mask = torch.ne(preds, z_true)
                    else:
                        mask = torch.lt(correct_class, self.loss_threshold)

                    num_to_select = mask.sum().item()

                if epoch < warm_up_epochs:
                    if idx is not None:
                        survival_log[epoch].extend(idx.cpu().tolist())
                    for label in z_true.tolist():
                        label_log[int(label)] += 1

                    optimizer.zero_grad()
                    outputs_full = self.model(x_meta, y_ecg)
                    losses = criterion(outputs_full, z_true)
                    final_loss = losses.mean()
                    final_loss.backward()
                    optimizer.step()

                    num_kept = outputs_full.size(0)
                    train_loss_sum_used += losses.sum().item()

                    with torch.no_grad():
                        preds = outputs_full.argmax(dim=1)
                        train_correct_used += torch.sum(preds == z_true).item()
                        train_total_used += num_kept

                elif epoch < start_revision:
                    if num_to_select == 0:
                        total_samples_in_epoch += current_batch_size
                        continue

                    indices = torch.randperm(current_batch_size)[:num_to_select]
                    x_meta_sampled = x_meta[indices]
                    y_ecg_sampled = y_ecg[indices]
                    z_true_sampled = z_true[indices]

                    if epoch >= warm_up_epochs + cam_start_epoch and len(z_true_sampled) > 1:
                        # combined = (1-conf) × cam_focus, all classes
                        cam_scores = compute_cam_scores(
                            self.model,
                            x_meta_sampled, y_ecg_sampled, z_true_sampled,
                            target_layers, self.device
                        )
                        conf_sampled = correct_class[indices]
                        combined = (1.0 - conf_sampled) * cam_scores
                        k = max(1, int(len(combined) * cam_keep_frac))
                        _, top_k = torch.topk(combined, k)
                        keep_mask = torch.zeros(len(z_true_sampled), dtype=torch.bool, device=self.device)
                        keep_mask[top_k] = True
                        # class-aware rescue
                        for cls in z_true_sampled.unique():
                            cls_pos = (z_true_sampled == cls).nonzero(as_tuple=True)[0]
                            if not keep_mask[cls_pos].any():
                                best = cls_pos[combined[cls_pos].argmax()]
                                keep_mask[best] = True
                        cam_filtered_in_epoch += (~keep_mask).sum().item()
                        x_meta_sampled = x_meta_sampled[keep_mask]
                        y_ecg_sampled = y_ecg_sampled[keep_mask]
                        z_true_sampled = z_true_sampled[keep_mask]

                    if idx is not None:
                        kept_idx = idx[indices]
                        survival_log[epoch].extend(kept_idx.cpu().tolist())
                    for label in z_true_sampled.tolist():
                        label_log[int(label)] += 1

                    optimizer.zero_grad()
                    outputs_sampled = self.model(x_meta_sampled, y_ecg_sampled)
                    losses = criterion(outputs_sampled, z_true_sampled)
                    final_loss = losses.mean()
                    final_loss.backward()
                    optimizer.step()

                    num_kept = outputs_sampled.size(0)
                    train_loss_sum_used += losses.sum().item()

                    with torch.no_grad():
                        preds_kept = outputs_sampled.argmax(dim=1)
                        train_correct_used += torch.sum(preds_kept == z_true_sampled).item()
                        train_total_used += num_kept

                else:
                    if idx is not None:
                        survival_log[epoch].extend(idx.cpu().tolist())
                    for label in z_true.tolist():
                        label_log[int(label)] += 1

                    optimizer.zero_grad()
                    outputs_full = self.model(x_meta, y_ecg)
                    losses = criterion(outputs_full, z_true)
                    final_loss = losses.mean()
                    final_loss.backward()
                    optimizer.step()

                    num_kept = outputs_full.size(0)
                    train_loss_sum_used += losses.sum().item()

                    with torch.no_grad():
                        preds = outputs_full.argmax(dim=1)
                        train_correct_used += torch.sum(preds == z_true).item()
                        train_total_used += num_kept

                samples_kept_in_epoch += num_kept
                total_samples_in_epoch += current_batch_size
                total_samples_processed += num_kept

                batch_ratio = num_kept / current_batch_size if current_batch_size > 0 else 0
                pbar.set_postfix({'loss': f"{final_loss.item():.4f}", 'ratio': f"{batch_ratio:.2f}"})

            epoch_loss_used = train_loss_sum_used / train_total_used if train_total_used > 0 else 0.0
            epoch_acc_used = train_correct_used / train_total_used if train_total_used > 0 else 0.0
            epoch_acc_all = train_correct_all / train_total_all if train_total_all > 0 else 0.0
            epoch_kept_ratio = samples_kept_in_epoch / total_samples_in_epoch if total_samples_in_epoch > 0 else 1.0

            val_loss, val_acc, val_f1 = self._validate(criterion)

            epoch_duration = time.time() - epoch_start
            scheduler.step()
            total_cam_filtered += cam_filtered_in_epoch

            print(f"Epoch [{epoch+1}/{self.epochs}] "
                  f"Loss(Used): {epoch_loss_used:.4f} Acc(Used): {epoch_acc_used:.4f} Acc(All): {epoch_acc_all:.4f} "
                  f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f} Val F1: {val_f1:.4f} "
                  f"Kept: {epoch_kept_ratio:.2%} CAM-filtered: {cam_filtered_in_epoch}")

            self.history["train_loss_used"].append(epoch_loss_used)
            self.history["train_acc_used"].append(epoch_acc_used)
            self.history["train_acc_all"].append(epoch_acc_all)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["val_f1"].append(val_f1)
            self.history["time_per_epoch"].append(epoch_duration)
            self.history["kept_ratio"].append(epoch_kept_ratio)
            self.history["samples_used"].append(samples_kept_in_epoch)
            self.history["train_size"].append(total_samples_in_epoch)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(self.model.state_dict(), ckpt_path)
                print(f"  [*] Best Val F1 reached! Saved to {ckpt_path}")

            plot_metrics(
                self.history["train_loss_used"], self.history["train_acc_used"],
                f"{self.model_name}_XAI-SMRD (Used)",
                os.path.join(self.save_path, "metrics_used.png"),
                loss_color='purple', acc_color='teal'
            )
            plot_metrics(
                self.history["train_loss_used"], self.history["train_acc_all"],
                f"{self.model_name}_XAI-SMRD (All)",
                os.path.join(self.save_path, "metrics_all.png"),
                loss_color='purple', acc_color='teal'
            )
            plot_kept_ratio(
                self.history["kept_ratio"],
                f"{self.model_name} XAI-SMRD Kept Ratio",
                os.path.join(self.save_path, "kept_ratio.png"),
                color='purple'
            )
            plot_metrics_test(
                self.history["val_acc"],
                f"{self.model_name}_XAI-SMRD_Validation",
                os.path.join(self.save_path, "val_metrics.png"),
                color='darkmagenta'
            )

        plot_accuracy_time_multi_test(
            self.model_name, self.history["val_acc"], self.history["time_per_epoch"],
            self.history["samples_used"], self.loss_threshold,
            os.path.join(self.save_path, "test_accuracy_time.png"),
            os.path.join(self.save_path, "model_data.json")
        )

        if os.path.exists(ckpt_path):
            self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device, weights_only=True))

        effective_epochs = total_samples_processed / len(self.train_loader.dataset)
        total_time = time.time() - start_time
        print(f"\nXAI-SMRD Training Complete.")
        print(f"Total Training Time: {total_time/60:.2f} minutes")
        print(f"Total samples processed: {total_samples_processed}")
        print(f"Effective Epochs: {effective_epochs:.2f}")
        print(f"Total CAM-filtered samples: {total_cam_filtered}")

        survival_log_path = os.path.join(self.save_path, "survival_log.json")
        with open(survival_log_path, "w") as f:
            json.dump({str(k): v for k, v in survival_log.items()}, f, indent=2)
        print(f"Survival log saved to {survival_log_path}")

        label_log_path = os.path.join(self.save_path, "label_log.json")
        with open(label_log_path, "w") as f:
            json.dump(dict(label_log), f, indent=2)
        print(f"Label log saved to {label_log_path}")

        with open(os.path.join(self.save_path, "xai_smrd_history.json"), 'w') as f:
            json.dump(self.history, f, indent=2)

        self._final_confusion_matrix("xai_smrd")

        return self.model, total_samples_processed

