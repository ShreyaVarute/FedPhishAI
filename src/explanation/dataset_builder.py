import pandas as pd

def build_prompt(email_text, prediction, suspicious_tokens, confidence) -> str:
    """Build structured prompt that feeds attention signals into T5."""
    tokens_str = ', '.join([f"'{t}'" for t in suspicious_tokens])
    return (
        f"Email: {email_text[:300]}\n"
        f"Prediction: {prediction}\n"
        f"Suspicious Tokens: [{tokens_str}]\n"
        f"Confidence: {confidence:.2f}\n"
        f"Explain:"
    )

def build_target(suspicious_tokens, prediction) -> str:
    """Generate rule-guided explanation target for T5 training."""
    if prediction == 'Phishing' and suspicious_tokens:
        tl = ', '.join(suspicious_tokens[:3])
        return (
            f"This email is classified as phishing because it contains "
            f"suspicious language including {tl}, "
            f"which are common indicators of social engineering attacks."
        )
    return "This email appears legitimate based on its language patterns."

def create_explanation_dataset(analyzed_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in analyzed_df.iterrows():
        prompt = build_prompt(
            row['text'], row['pred_label'],
            row['suspicious_tokens'], row['confidence']
        )
        target = build_target(row['suspicious_tokens'], row['pred_label'])
        rows.append({'input': prompt, 'target': target})
    return pd.DataFrame(rows)