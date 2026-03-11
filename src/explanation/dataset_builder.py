"""
Build explanation training dataset from phishing emails.
Uses rule-based templates to generate training data for T5.
"""
import pandas as pd
import re
import os
from tqdm import tqdm


PHISHING_TEMPLATES = [
    "This email was flagged as phishing because it {reason}.",
    "This is a phishing email. It contains {reason}, which is a classic indicator of phishing.",
    "Phishing detected: The email {reason}, a common tactic used by attackers.",
]

LEGITIMATE_TEMPLATES = [
    "This email is legitimate. It uses natural language without urgency or suspicious requests.",
    "This appears to be a legitimate email. It does not contain threats, urgency, or requests for credentials.",
    "Legitimate email: The message is conversational and does not ask for sensitive information.",
]


def extract_phishing_reason(text: str) -> str:
    """Extract the most prominent phishing feature from email text."""
    text_lower = text.lower()
    reasons = []

    if re.search(r'http[s]?://', text):
        reasons.append("contains suspicious links")
    if any(w in text_lower for w in ['urgent', 'immediately', 'act now', 'expires']):
        reasons.append("uses urgency language")
    if any(w in text_lower for w in ['verify', 'confirm', 'password', 'credential']):
        reasons.append("requests sensitive credentials")
    if any(w in text_lower for w in ['suspended', 'locked', 'disabled', 'closed']):
        reasons.append("threatens account suspension")
    if any(w in text_lower for w in ['prize', 'won', 'lottery', 'reward', 'million']):
        reasons.append("promises unrealistic rewards")
    if any(w in text_lower for w in ['bank', 'paypal', 'amazon', 'account']):
        reasons.append("impersonates a trusted institution")

    if not reasons:
        reasons.append("uses deceptive language patterns common in phishing attacks")

    return " and ".join(reasons[:2])


def build_explanation_dataset(
    labeled_csv: str,
    output_dir: str,
    max_samples: int = 10000
):
    """
    Build T5 training dataset with (email, prediction, explanation) triples.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(labeled_csv)
    df = df.dropna(subset=['text', 'label']).reset_index(drop=True)

    if max_samples:
        df = df.sample(n=min(max_samples, len(df)), random_state=42)

    records = []
    import random
    random.seed(42)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building explanations"):
        text = str(row['text'])
        label = int(row['label'])

        if label == 1:  # phishing
            reason = extract_phishing_reason(text)
            template = random.choice(PHISHING_TEMPLATES)
            explanation = template.format(reason=reason)
            prediction = "phishing"
        else:  # legitimate
            explanation = random.choice(LEGITIMATE_TEMPLATES)
            prediction = "legitimate"

        records.append({
            'email_text': text[:500],  # Truncate to 500 chars
            'prediction': prediction,
            'explanation': explanation
        })

    result_df = pd.DataFrame(records)

    # Split train/val
    val_size = int(0.1 * len(result_df))
    val_df = result_df.sample(n=val_size, random_state=42)
    train_df = result_df.drop(val_df.index)

    train_df.to_csv(os.path.join(output_dir, 'train_explanations.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val_explanations.csv'), index=False)

    print(f"Explanation dataset built:")
    print(f"  Train: {len(train_df)} | Val: {len(val_df)}")
    print(f"  Saved to: {output_dir}")

    return train_df, val_df