import torch
from transformers import DistilBertTokenizer

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def extract_suspicious_tokens(
    input_ids: torch.Tensor,
    attentions: tuple,
    top_k: int = 5
) -> list:
    """
    Average attention across all heads and layers.
    Return the top-k highest-attention tokens as 'suspicious tokens'.
    These feed directly into the T5 explanation module.
    """
    # Stack all layers → avg → shape (batch, heads, seq, seq)
    avg_attn         = torch.stack(attentions).mean(dim=0)
    avg_attn         = avg_attn.mean(dim=1)          # avg over heads
    token_importance = avg_attn[0].sum(dim=0)         # sum received attention

    top_indices = token_importance.argsort(descending=True)[:top_k]
    tokens      = tokenizer.convert_ids_to_tokens(input_ids[0])

    skip = {'[CLS]', '[SEP]', '[PAD]', '.', ',', '!', '?'}
    return [tokens[i] for i in top_indices if tokens[i] not in skip]