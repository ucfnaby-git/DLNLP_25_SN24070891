import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm


def train_one_epoch(classifier, dataloader, optimizer, loss_fn, device, training=True):
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
    if training == True:
        classifier.train()
        loop = tqdm(dataloader, desc="Training", leave=False)
    else:
        classifier.eval()
        loop = tqdm(dataloader, desc="Evaluating", leave=False)
    total_loss = 0
    all_preds, all_labels = [], []

    for input_ids, attention_mask, token_type_ids, labels in loop:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        token_type_ids = token_type_ids.to(device)
        labels = labels.to(device)

        out = classifier(input_ids, attention_mask)
        loss = loss_fn(out, labels)

        optimizer.zero_grad()
        # loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(torch.argmax(out, dim=1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    return avg_loss, acc, f1