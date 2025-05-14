import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm


def train_one_epoch(bert, classifier, dataloader, optimizer, loss_fn, device):
    '''
    Perform one epoch of training using a frozen BERT model and a trainable classification head.

    Args:
        bert (nn.Module): Frozen BERT model in eval mode.
        classifier (nn.Module): Trainable classifier head.
        dataloader (DataLoader): Training data loader.
        optimizer (torch.optim.Optimizer): Optimizer for classifier parameters.
        loss_fn (callable): Loss function (e.g., CrossEntropyLoss).
        device (torch.device): Target device for model and tensors.

    Returns:
        tuple: average_loss, accuracy, F1 score
    '''
    classifier.train()
    total_loss = 0
    all_preds, all_labels = [], []

    loop = tqdm(dataloader, desc="Training", leave=False)
    for input_ids, attn_mask, token_type_ids, labels in loop:
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)
        token_type_ids = token_type_ids.to(device)
        labels = labels.to(device)

        # Get [CLS] token from frozen BERT
        with torch.no_grad():
            cls_output = bert(input_ids, attention_mask=attn_mask, token_type_ids=token_type_ids).last_hidden_state[:, 0]

        optimizer.zero_grad()
        logits = classifier(cls_output)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    return avg_loss, acc, f1