import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from transformers import BertModel

from A.model import SentimentClassifier
from A.utils import set_seed, seed_worker
from A.evaluation import evaluate

def test_model(model_path, batch_size=16, seed=42, device=None):
    """
    Evaluate a trained sentiment classifier on a saved test set and produce a confusion matrix.

    Args:
        model_path (str): Path to the trained classifier model (state_dict .pt).
        batch_size (int): Batch size for test DataLoader.
        seed (int): Random seed for reproducibility.
        device (torch.device or None): Device for evaluation; defaults to CUDA if available.
    """
    set_seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Load saved test dataset ===

    torch.serialization.add_safe_globals([torch.utils.data.dataset.Subset])
    test_dataset = torch.load("Datasets/test_dataset.pt", weights_only=False)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=16, worker_init_fn=seed_worker)

    # === Load classifier ===
    classifier = SentimentClassifier(num_labels=2)
    if torch.cuda.device_count() > 1:
        classifier = torch.nn.DataParallel(classifier, device_ids=[0, 1, 2, 3, 4, 5, 6, 8])

    checkpoint = torch.load(model_path, map_location=device)
    classifier.load_state_dict(checkpoint)
    classifier.to(device)
    classifier.eval()

    # === Collect predictions ===
    all_preds, all_labels = [], []
    with torch.no_grad():
        for input_ids, attention_mask, token_type_ids, labels in test_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            labels = labels.to(device)

            out = classifier(input_ids, attention_mask, token_type_ids)
            preds = torch.argmax(out, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    # === Classification report ===
    print("Running test evaluation...")
    label_names = [
        "positive","negative"
    ]
    report = classification_report(all_labels, all_preds, target_names=label_names, digits=4)
    print(report)

    os.makedirs("logs", exist_ok=True)
    with open("logs/test_classification_report.txt", "w") as f:
        f.write(report)

    # === Confusion matrix ===
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
    plt.title("Confusion Matrix on Test Set")
    plt.tight_layout()
    plt.savefig("logs/plots/test_confusion_matrix.png")
    plt.close()

    print("Evaluation complete. Results saved to logs/")

    return cm, report
