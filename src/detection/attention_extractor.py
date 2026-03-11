import torch
import numpy as np
from transformers import DistilBertTokenizer
from src.detection.model import PhishingDetector
from src.data.preprocessor import EmailPreprocessor


class AttentionExtractor:
    def __init__(self, model_path: str, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = PhishingDetector(num_labels=2).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.preprocessor = EmailPreprocessor()
        self.tokenizer = self.preprocessor.tokenizer

    def extract(self, text: str):
        """
        Extract attention weights and return highlighted tokens.
        Returns: dict with tokens and their attention scores.
        """
        cleaned = self.preprocessor.clean_email(text)
        encoding = self.tokenizer(
            cleaned,
            max_length=256,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        with torch.no_grad():
            attentions = self.model.get_attention_weights(input_ids, attention_mask)

        # Average attention across all heads in last layer
        # attentions shape: (num_layers, batch, heads, seq, seq)
        last_layer_attn = attentions[-1]  # (batch, heads, seq, seq)
        avg_attn = last_layer_attn.mean(dim=1)  # (batch, seq, seq)
        cls_attn = avg_attn[0, 0, :]  # CLS token attention over all tokens

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())

        # Get actual (non-padding) tokens
        mask = encoding['attention_mask'][0].tolist()
        result_tokens = []
        result_scores = []
        for i, (token, m) in enumerate(zip(tokens, mask)):
            if m == 1 and token not in ['[CLS]', '[SEP]', '[PAD]']:
                result_tokens.append(token)
                result_scores.append(float(cls_attn[i].cpu()))

        # Normalize scores
        if result_scores:
            max_score = max(result_scores)
            if max_score > 0:
                result_scores = [s / max_score for s in result_scores]

        return {
            'tokens': result_tokens,
            'attention_scores': result_scores,
            'top_tokens': sorted(
                zip(result_tokens, result_scores),
                key=lambda x: x[1], reverse=True
            )[:10]
        }