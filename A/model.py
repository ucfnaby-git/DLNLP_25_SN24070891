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
