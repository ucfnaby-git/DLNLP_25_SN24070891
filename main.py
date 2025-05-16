import os
import csv
import torch
import pandas as pd
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as lr_scheduler

from torch.utils.data import DataLoader
from A.utils import create_dirs, seed_worker, set_seed
from A.data_process import BERTTextDataset, load_data, split_dataset
from A.model import SentimentClassifier
from A.training import train_one_epoch
from A.testing import test_model

# === Configuration ===
SEED = 42
EPOCHS = 10
BATCH_SIZE = 128
NUM_WORKERS = 20
CSV_PATH="Datasets/IMDB_Dataset.csv"
OUTPUT_DIR = "logs"

def main():
    # === Set seed and prepare environment ===
    set_seed(SEED)
    create_dirs(OUTPUT_DIR)

    # === Device setup ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Load and prepare data ===
    texts, labels, tokenizer, label2id, id2label = load_data(CSV_PATH)
    dataset = BERTTextDataset(texts, labels, tokenizer, label2id)
    train_loader, val_loader, test_loader= split_dataset(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    # === Initialize classifier ===
    print(f"num_labels:{len(label2id)}")
    classifier = SentimentClassifier(num_labels=len(label2id))
    if torch.cuda.device_count() > 1:
        classifier = torch.nn.DataParallel(classifier, device_ids=[0, 1, 2, 3, 4, 5, 6, 8])
    classifier = classifier.to(device)

    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-2)
    # scheduler = lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=EPOCHS, eta_min=1e-3
    # )
    loss_fn = torch.nn.CrossEntropyLoss()
    log_path = os.path.join(OUTPUT_DIR, "metrics_log.csv")

    # === Logging headers ===
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "TrainLoss", "TrainAcc", "TrainF1", "ValLoss", "ValAcc", "ValF1"])

    # === Training Loop ===
    for epoch in range(1, EPOCHS + 1):
        print(f"\n=== Epoch {epoch}/{EPOCHS} ===")
        train_loss, train_acc, train_f1 = train_one_epoch(classifier, train_loader, optimizer, loss_fn, device)

        val_loss, val_acc, val_f1 = train_one_epoch(classifier, train_loader, optimizer, loss_fn, device, training=False)

        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")

        # scheduler.step()

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, train_acc, train_f1, val_loss, val_acc, val_f1])
        if epoch % 5 == 0:
            torch.save(classifier.state_dict(), f"{OUTPUT_DIR}/checkpoints/classifier_epoch_{epoch}.pt")

    # === Plot curves ===
    metrics_df = pd.read_csv(log_path)

    plt.figure(figsize=(8, 5))
    plt.plot(metrics_df["Epoch"], metrics_df["TrainLoss"], label="Train Loss")
    plt.plot(metrics_df["Epoch"], metrics_df["ValLoss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/plots/loss_curve.png")

    plt.figure(figsize=(10, 6))
    plt.plot(metrics_df["Epoch"], metrics_df["TrainAcc"], label="Train Accuracy")
    plt.plot(metrics_df["Epoch"], metrics_df["ValAcc"], label="Val Accuracy")
    plt.plot(metrics_df["Epoch"], metrics_df["TrainF1"], label="Train F1")
    plt.plot(metrics_df["Epoch"], metrics_df["ValF1"], label="Val F1")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Training Metrics")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/plots/metric_curves.png")

    #=== Run test evaluation using the last saved model ===
    # latest_ckpt = f"{OUTPUT_DIR}/checkpoints/classifier_epoch_{EPOCHS}.pt"
    # test_model(model_path=latest_ckpt, batch_size=BATCH_SIZE, device=device)

if __name__ == "__main__":
    main()
    