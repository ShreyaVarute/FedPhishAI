import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def evaluate_model(model, loader, device) -> dict:
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            ids   = batch['input_ids'].to(device)
            mask  = batch['attention_mask'].to(device)
            lbls  = batch['label']
            logits, _ = model(ids, mask)
            preds     = logits.argmax(dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(lbls.tolist())

    acc            = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary')
    fn  = sum(1 for t, p in zip(all_labels, all_preds) if t==1 and p==0)
    fnr = fn / max(sum(all_labels), 1)

    return {'accuracy': acc, 'precision': prec,
            'recall': rec, 'f1': f1, 'FNR': fnr}