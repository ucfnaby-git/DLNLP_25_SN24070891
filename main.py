import os
import csv
import torch
import pandas as pd
import matplotlib.pyplot as plt
from transformers import BertModel

from torch.utils.data import DataLoader
from A.utils import create_dirs, seed_worker, set_seed
from A.data_process import BERTTextDataset, load_data
from A.model import SentimentClassifier
from A.training import train_one_epoch
from A.evaluation import evaluate
from A.testing import test_model

# === Configuration ===
SEED = 42
EPOCHS = 10
BATCH_SIZE = 16
CSV_PATH = "Datasets/Sentiment_Analysis.csv"
OUTPUT_DIR = "logs"

def main():
    # === Set seed and prepare environment ===
    set_seed(SEED)
    create_dirs(OUTPUT_DIR)

    # === Device setup ===
    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

    # === Load and freeze BERT ===
    bert = BertModel.from_pretrained("bert-base-uncased")
    for param in bert.parameters():
        param.requires_grad = False
    bert.eval()
    bert.to(device)

    # === Load and prepare data ===
    texts, labels, tokenizer, label2id, id2label = load_data(CSV_PATH)
    dataset = BERTTextDataset(texts, labels, tokenizer, label2id)

    train_size = int(0.7 * len(dataset))
    temp_size = len(dataset) - train_size
    train_dataset, temp_dataset = torch.utils.data.random_split(dataset, [train_size, temp_size])
    val_size = int(0.5 * len(temp_dataset))
    test_size = len(temp_dataset) - val_size
    val_dataset, test_dataset = torch.utils.data.random_split(temp_dataset, [val_size, test_size])
    
    torch.save(test_dataset, os.path.join("Datasets", "test_dataset.pt"))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=16, worker_init_fn=seed_worker)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=16, worker_init_fn=seed_worker)

    # === Initialize classifier ===
    classifier = SentimentClassifier(num_labels=len(label2id))
    if torch.cuda.device_count() > 1:
        classifier = torch.nn.DataParallel(classifier, device_ids=[4, 5, 6, 8, 9])
    classifier = classifier.to(device)

    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()
    log_path = os.path.join(OUTPUT_DIR, "metrics_log.csv")

    # === Logging headers ===
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "TrainLoss", "TrainAcc", "TrainF1", "ValLoss", "ValAcc", "ValF1", "ValAUC"])

    # === Training Loop ===
    for epoch in range(1, EPOCHS + 1):
        print(f"\\n=== Epoch {epoch}/{EPOCHS} ===")
        train_loss, train_acc, train_f1 = train_one_epoch(bert, classifier, train_loader, optimizer, loss_fn, device)
        val_loss, val_acc, val_f1, val_auc = evaluate(bert, classifier, val_loader, loss_fn, device, len(label2id))

        auc_str = f"{val_auc:.4f}" if val_auc is not None else "0.0000"
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f} | AUC: {auc_str}")

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, train_acc, train_f1, val_loss, val_acc, val_f1, val_auc or 0.0])

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
    plt.plot(metrics_df["Epoch"], metrics_df["ValAUC"], label="Val AUC")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Training Metrics")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/plots/metric_curves.png")

    #=== Run test evaluation using the last saved model ===
    latest_ckpt = f"{OUTPUT_DIR}/checkpoints/classifier_epoch_{EPOCHS}.pt"
    test_model(model_path=latest_ckpt)

if __name__ == "__main__":
    main()
    