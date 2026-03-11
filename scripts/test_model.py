"""
Test the trained phishing detection model on custom emails.
Run: python scripts/test_model.py
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from src.detection.model import PhishingDetector
from src.data.preprocessor import EmailPreprocessor


def predict_email(model, preprocessor, text: str, device: str):
    """Predict whether a single email is phishing or legitimate."""
    model.eval()
    encoding = preprocessor.tokenize(text)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(logits, dim=1).item()

    label = "PHISHING" if pred == 1 else "LEGITIMATE"
    phish_prob = probs[0][1].item()
    legit_prob = probs[0][0].item()

    return label, phish_prob, legit_prob


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = 'models/distilbert_baseline/best_model.pt'

    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please train first using scripts/run_baseline.py")
        return

    # Load model
    model = PhishingDetector(num_labels=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    preprocessor = EmailPreprocessor()

    # Test emails - mix of legitimate and phishing
    test_emails = [
        # Should be PHISHING
        {
            "text": "URGENT: Your PayPal account has been suspended. Click here immediately to verify your identity and restore access: http://paypal-secure-login.xyz/verify",
            "expected": "PHISHING"
        },
        {
            "text": "Congratulations! You have won $1,000,000 in our lottery. Send your bank details to claim your prize. Act now before it expires!",
            "expected": "PHISHING"
        },
        {
            "text": "Dear customer, your bank account will be closed in 24 hours unless you confirm your credentials at this link. This is your final warning.",
            "expected": "PHISHING"
        },
        # Should be LEGITIMATE
        {
            "text": "Hi John, just a reminder about our team meeting tomorrow at 2pm in conference room B. Please bring your project updates.",
            "expected": "LEGITIMATE"
        },
        {
            "text": "Happy Birthday! Hope you have a wonderful day filled with joy and celebration. Looking forward to seeing you at the party!",
            "expected": "LEGITIMATE"
        },
        {
            "text": "Hey, are you free for lunch this week? There's a new restaurant downtown I've been wanting to try. Let me know what works for you.",
            "expected": "LEGITIMATE"
        },
        {
            "text": "The quarterly report is attached. Please review the revenue figures for Q3 before our board meeting on Friday.",
            "expected": "LEGITIMATE"
        },
        {
            "text": "Your Amazon order #112-3456789 has shipped and will arrive by Thursday. Track your package using the order details in your account.",
            "expected": "LEGITIMATE"
        },
    ]

    print("=" * 70)
    print("PHISHING EMAIL DETECTION - TEST RESULTS")
    print("=" * 70)

    correct = 0
    for i, email in enumerate(test_emails):
        label, phish_prob, legit_prob = predict_email(model, preprocessor, email["text"], device)
        is_correct = label == email["expected"]
        correct += int(is_correct)
        status = "✓" if is_correct else "✗"

        print(f"\n[{i+1}] {status} Expected: {email['expected']}")
        print(f"    Predicted: {label}")
        print(f"    Phishing probability: {phish_prob:.4f}")
        print(f"    Legitimate probability: {legit_prob:.4f}")
        print(f"    Email: {email['text'][:80]}...")

    print("\n" + "=" * 70)
    print(f"Accuracy on test samples: {correct}/{len(test_emails)} ({100*correct/len(test_emails):.1f}%)")
    print("=" * 70)


if __name__ == '__main__':
    main()