import torch
import torch.nn as nn
from transformers import DistilBertModel

class PhishingDetector(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(
            'distilbert-base-uncased',
            output_attentions=True   # Required for suspicious token extraction
        )
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, 2)  # Binary: 0=legit, 1=phishing

    def forward(self, input_ids, attention_mask):
        outputs    = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden     = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        attentions = outputs.attentions                     # All layer attentions
        logits     = self.classifier(self.dropout(hidden))
        return logits, attentions