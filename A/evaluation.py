import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm


def evaluate(classifier, dataloader, loss_fn, device, num_labels):
    '''
    Evaluate the classifier using a frozen BERT encoder.

    Args:
        classifier (nn.Module): Classifier head to evaluate.
        dataloader (DataLoader): Validation or test data loader.
        loss_fn (callable): Loss function to evaluate.
        device (torch.device): Device on which models and data reside.
        num_labels (int): Number of output classes.

    Returns:
        tuple: avg_loss, accuracy, F1 score, AUC score (or None if not computable)
    '''
    classifier.eval()
    total_loss = 0
    all_preds, all_probs, all_labels = [], [], []

    loop = tqdm(dataloader, desc="Evaluating", leave=False)
    for input_ids, attn_mask, token_type_ids, labels in loop:
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)
        token_type_ids = token_type_ids.to(device)
        labels = labels.to(device)
        
        out = classifier(input_ids, attention_mask, token_type_ids)
        loss = loss_fn(out, labels)
        total_loss += loss.item()

        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        all_probs.extend(probs.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    return avg_loss, acc, f1