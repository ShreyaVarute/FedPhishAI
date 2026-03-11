import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertConfig


class PhishingDetector(nn.Module):
    def __init__(self, num_labels: int = 2, dropout_prob: float = 0.3):
        super(PhishingDetector, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        hidden_size = self.distilbert.config.hidden_size  # 768

        self.pre_classifier = nn.Linear(hidden_size, 256)
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(256, num_labels)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # Use [CLS] token representation
        hidden_state = outputs.last_hidden_state[:, 0, :]

        x = self.pre_classifier(hidden_state)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.classifier(x)
        return logits

    def get_attention_weights(self, input_ids, attention_mask):
        """Extract attention weights for explainability."""
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True
        )
        return outputs.attentions  # tuple of attention layers