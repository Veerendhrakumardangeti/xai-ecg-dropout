import argparse
import sys
import os
import json
import torch

# force immediate output for slurm logs
sys.stdout.reconfigure(line_buffering=True)

from model import build_model
from data import load_dataset, get_class_info
from baseline import BaselineTrainer
from selective_gradient import TrainRevision
from test import test_model



def seed_everything(seed):
    """make things as deterministic as possible for reproducible research"""
    import random
    import numpy as np
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(
        description="ECG classification (PTB-XL / CPSC 2018 / Chapman) with XAI-DBPD")
    parser.add_argument("--dataset", type=str, default="ptbxl",
                        choices=["ptbxl", "cpsc2018", "chapman", "chapman_v2", "georgia"],
                        help="Which dataset to use")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["baseline", "train_with_revision", "smrd", "srd", "xai_dbpd", "xai_smrd", "xai_srd"],
                        help="Training mode")
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--save_path", type=str, default="./output")
    parser.add_argument("--data_path", type=str, default="/Users/veerendhrakumar/Desktop/baseline/ptbxl")
    parser.add_argument("--sampling_rate", type=int, default=100, choices=[100, 500])
    parser.add_argument("--loss_threshold", type=float, default=0.3)
    parser.add_argument("--start_revision", type=int, default=50)
    parser.add_argument("--warm_up_epochs", type=int, default=5,
                        help="Epochs of full-dataset warm-up before any filtering begins")
    parser.add_argument("--model", type=str, default="efficientnet",
                        choices=["efficientnet", "resnet18", "mobilenetv2"],
                        help="Model architecture to use")
    parser.add_argument("--seed", type=int, default=42)
    # xai specific hyperparams
    parser.add_argument("--cam_keep_frac", type=float, default=0.7)
    parser.add_argument("--cam_start_epoch", type=int, default=3)
    parser.add_argument("--srd_decay", type=float, default=0.99,
                        help="Per-epoch decay factor for SRD/XAI-SRD (default: 0.99)")

    args = parser.parse_args()

    seed_everything(args.seed)

    num_classes, class_names = get_class_info(args.dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_path, exist_ok=True)

    print(f"ECG classification — {args.dataset.upper()} ({num_classes} classes)")
    print("-" * 50)
    print(f"Dataset:          {args.dataset}")
    print("-" * 50)
    print(f"Mode:             {args.mode}")
    print(f"Data path:        {args.data_path}")
    print(f"Save path:        {args.save_path}")
    print(f"Batch size:       {args.batch_size}")
    print(f"Epochs:           {args.epoch}")
    if args.mode in ("train_with_revision", "smrd", "srd", "xai_dbpd", "xai_smrd", "xai_srd"):
        print(f"Warm-up epochs:   {args.warm_up_epochs}")
        print(f"Revision start:   epoch {args.start_revision + 1}")
        if args.mode not in ("srd", "xai_srd"):
            print(f"Loss Threshold:   {args.loss_threshold}")
    print(f"Device:           {device}")
    print(f"GPUs:             {torch.cuda.device_count()}")
    print()

    print(f"Loading {args.dataset} from: {args.data_path}")
    (train_loader, val_loader, test_loader, cls_num_list) = load_dataset(
        args.dataset, args.data_path, args.sampling_rate, args.batch_size
    )

    model = build_model(args.model, num_classes=num_classes).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # fire off the appropriate training routine based on the mode argument
    if args.mode == "baseline":
        trainer = BaselineTrainer(
            args.model, model, train_loader, val_loader, device,
            args.epoch, args.save_path
        )
        trained_model = trainer.train(args.batch_size)

    elif args.mode == "train_with_revision":
        trainer = TrainRevision(
            args.model, model, train_loader, val_loader, device,
            args.epoch, args.save_path, args.loss_threshold
        )
        trained_model, num_step = trainer.train_with_revision(
            args.start_revision, cls_num_list, args.batch_size,
            warm_up_epochs=args.warm_up_epochs
        )

    elif args.mode == "smrd":
        trainer = TrainRevision(
            args.model, model, train_loader, val_loader, device,
            args.epoch, args.save_path, args.loss_threshold
        )
        trained_model, num_step = trainer.train_with_smrd_matched(
            args.start_revision, cls_num_list, args.batch_size,
            warm_up_epochs=args.warm_up_epochs
        )

    elif args.mode == "srd":
        trainer = TrainRevision(
            args.model, model, train_loader, val_loader, device,
            args.epoch, args.save_path, args.loss_threshold
        )
        trained_model, num_step = trainer.train_with_srd_decay(
            args.start_revision, cls_num_list, args.batch_size,
            srd_decay=args.srd_decay, warm_up_epochs=args.warm_up_epochs
        )

    elif args.mode == "xai_dbpd":
        trainer = TrainRevision(
            args.model, model, train_loader, val_loader, device,
            args.epoch, args.save_path, args.loss_threshold
        )
        trained_model, num_step = trainer.train_with_xai_dbpd(
            args.start_revision, cls_num_list, args.batch_size,
            cam_keep_frac=args.cam_keep_frac, cam_start_epoch=args.cam_start_epoch,
            warm_up_epochs=args.warm_up_epochs
        )

    elif args.mode == "xai_smrd":
        trainer = TrainRevision(
            args.model, model, train_loader, val_loader, device,
            args.epoch, args.save_path, args.loss_threshold
        )
        trained_model, num_step = trainer.train_with_xai_smrd(
            args.start_revision, cls_num_list, args.batch_size,
            cam_keep_frac=args.cam_keep_frac, cam_start_epoch=args.cam_start_epoch,
            warm_up_epochs=args.warm_up_epochs
        )

    elif args.mode == "xai_srd":
        trainer = TrainRevision(
            args.model, model, train_loader, val_loader, device,
            args.epoch, args.save_path, args.loss_threshold
        )
        trained_model, num_step = trainer.train_with_xai_srd(
            args.start_revision, cls_num_list, args.batch_size,
            cam_keep_frac=args.cam_keep_frac, cam_start_epoch=args.cam_start_epoch,
            srd_decay=args.srd_decay, warm_up_epochs=args.warm_up_epochs
        )

    # run the final test evaluation on the unseen fold 10 data
    _, macro_f1, acc = test_model(trained_model, test_loader, device,
                                  tag=args.mode.upper(), dataset=args.dataset)

    # append to a shared json file so we can compare across script runs
    results_path = os.path.join(args.save_path, "comparison_results.json")
    all_results = {}
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            all_results = json.load(f)

    # calculate "effective epochs" as a measure of training cost
    # e.g. if we drop 50% of data for 10 epochs, that's only 5 effective epochs
    effective_epochs = args.epoch
    if args.mode in ("train_with_revision", "smrd", "srd", "xai_dbpd", "xai_smrd", "xai_srd"):
        total_processed = float(num_step)
        dataset_size = len(train_loader.dataset)
        effective_epochs = total_processed / dataset_size
    
    mode_labels = {
        "train_with_revision": f"DBPD t={args.loss_threshold}",
        "smrd": f"SMRD t={args.loss_threshold}",
        "srd": "SRD (0.99^epoch)",
        "baseline": "Baseline",
        "xai_dbpd": f"XAI-DBPD t={args.loss_threshold}",
        "xai_smrd": f"XAI-SMRD t={args.loss_threshold}",
        "xai_srd": "XAI-SRD (0.99^epoch)"
    }
    result_key = mode_labels.get(args.mode, args.mode)
    all_results[result_key] = {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "effective_epochs": float(effective_epochs)
    }

    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # save final weights
    torch.save(trained_model.state_dict(), os.path.join(args.save_path, "trained_model.pt"))
    print(f"Done. Results in {args.save_path}")


if __name__ == "__main__":
    main()
