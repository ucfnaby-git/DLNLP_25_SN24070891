import pandas as pd
import torch
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer

from A.utils import set_seed, seed_worker


class BERTTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, label2id, max_length=500):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.label2id[self.labels[idx]]

        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return (
            encoded["input_ids"].squeeze(0),
            encoded["attention_mask"].squeeze(0),
            encoded["token_type_ids"].squeeze(0),
            torch.tensor(label, dtype=torch.long),
        )

    def __len__(self):
        return len(self.texts)


def load_data(csv_path, max_length=500):
    df = pd.read_csv(csv_path)
    texts = df["review"].tolist()
    labels = df["sentiment"].tolist()

    unique_labels = sorted(set(labels))
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for label, i in label2id.items()}
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    return texts, labels, tokenizer, label2id, id2label

def split_dataset(dataset, batch_size, num_workers=20):
    train_size = int(0.7 * len(dataset))
    temp_size = len(dataset) - train_size
    train_dataset, temp_dataset = torch.utils.data.random_split(
        dataset, [train_size, temp_size])
    val_size = int(0.5 * len(temp_dataset))
    test_size = len(temp_dataset) - val_size
    val_dataset, test_dataset = torch.utils.data.random_split(temp_dataset, 
    [val_size, test_size])

    torch.save(test_dataset, os.path.join("Datasets", "test_dataset.pt"))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
    shuffle=True, num_workers=num_workers, worker_init_fn=seed_worker)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
    shuffle=False, num_workers=num_workers, worker_init_fn=seed_worker)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
    shuffle=False, num_workers=num_workers, worker_init_fn=seed_worker)
    return train_loader, val_loader, test_loader