import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification

class SentimentClassifier(nn.Module):
    """
    Sentiment classification model using a frozen BERT encoder.
    The [CLS] token output is passed through a dropout and linear layer.
    """
    def __init__(self, num_labels: int, dropout_prob: float = 0.3):
        super(SentimentClassifier, self).__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels) 
        # Freeze all layers
        for param in self.bert.parameters():
            param.requires_grad = False

        for param in self.bert.classifier.parameters(): 
            param.requires_grad = True 

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        logits = outputs.logits
        return logits


class SentimentMLPClassifier(nn.Module):
    """
    A multi-layer perceptron (MLP) classifier for sentiment prediction.
    Assumes input is a 768-dim [CLS] token from BERT.
    """

    def __init__(self, num_labels: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_labels)
        )

    def forward(self, cls_embedding: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MLP classifier.

        Args:
            cls_embedding (torch.Tensor): BERT [CLS] embedding (batch_size, 768)

        Returns:
            torch.Tensor: Logits over sentiment classes (batch_size, num_labels)
        """
        return self.mlp(cls_embedding)