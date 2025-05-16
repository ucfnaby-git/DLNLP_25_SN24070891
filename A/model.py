import torch
import torch.nn as nn


class SentimentClassifier(nn.Module):
    """
    A classification head for sentiment prediction using the [CLS] token embedding
    from a frozen BERT encoder.

    This module assumes that BERT outputs are precomputed externally and passed
    into this model during training or inference.
    """

    def __init__(self, num_labels: int):
        """
        Initialize the classifier.

        Args:
            num_labels (int): Number of sentiment categories.
        """
        super().__init__()
        self.fc = nn.Linear(768, num_labels)

    def forward(self, cls_embedding: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the classifier.

        Args:
            cls_embedding (torch.Tensor): BERT [CLS] token representation (batch_size, 768)

        Returns:
            torch.Tensor: Raw class logits (batch_size, num_labels)
        """
        return self.fc(cls_embedding)

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