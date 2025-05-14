import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer


class BERTTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, label2id, max_length=128):
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


def load_data(csv_path, max_length=128):
    df = pd.read_csv(csv_path)
    texts = df["content"].tolist()
    labels = df["sentiment"].tolist()

    unique_labels = sorted(set(labels))
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for label, i in label2id.items()}
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    return texts, labels, tokenizer, label2id, id2label
