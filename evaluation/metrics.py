from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
import textstat

def detection_metrics(y_true, y_pred) -> dict:
    acc              = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    fn               = sum(1 for t, p in zip(y_true, y_pred) if t==1 and p==0)
    fnr              = fn / max(sum(y_true), 1)
    return {'accuracy': acc, 'precision': prec,
            'recall': rec, 'f1': f1, 'FNR': fnr}

def explanation_metrics(generated: str, reference: str) -> dict:
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
    rouge  = scorer.score(reference, generated)
    bleu   = sentence_bleu([reference.split()], generated.split())
    read   = textstat.flesch_reading_ease(generated)
    return {
        'bleu':        bleu,
        'rouge1':      rouge['rouge1'].fmeasure,
        'rouge2':      rouge['rouge2'].fmeasure,
        'rougeL':      rouge['rougeL'].fmeasure,
        'readability': read
    }