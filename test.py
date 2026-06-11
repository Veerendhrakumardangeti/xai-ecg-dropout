import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn import metrics as sk_metrics
from sklearn.metrics import f1_score as sklearn_f1_score


def test_model(model, test_loader, device, tag="MODEL", dataset="ptbxl"):
    """runs a full evaluation pass on the test set and prints a classification report"""
    print(f"\n{tag} test results:")
    model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss().to(device)
    test_correct = 0
    test_total = 0
    test_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Testing ({tag})"):
            x_meta = batch[0].to(device)
            y_ecg = batch[1].to(device)
            z_true = batch[2].to(device).long()
            
            outputs = model(x_meta, y_ecg)
            batch_loss = criterion(outputs, z_true)
            test_loss += batch_loss.item()
            
            preds = torch.argmax(outputs, dim=1)
            test_correct += torch.sum(preds == z_true).item()
            test_total += z_true.size(0)
            
            all_preds.append(preds.cpu())
            all_labels.append(z_true.cpu())

    accuracy = test_correct / test_total if test_total > 0 else 0
    val_loss = test_loss / len(test_loader)

    Z_pred = torch.cat(all_preds, dim=0).numpy().astype(int)
    Z_test = torch.cat(all_labels, dim=0).numpy().astype(int)

    from data import get_class_info
    _, labels = get_class_info(dataset)
    
    print(sk_metrics.classification_report(Z_test, Z_pred,
                                           target_names=labels, zero_division=0))
                                           
    macro_f1 = sklearn_f1_score(Z_test, Z_pred, average="macro", zero_division=0)
    acc = float(np.mean(Z_pred == Z_test))
    
    print(f"{tag} Macro F1: {macro_f1:.4f}  Accuracy: {acc:.4f}")
    return Z_pred, macro_f1, acc
