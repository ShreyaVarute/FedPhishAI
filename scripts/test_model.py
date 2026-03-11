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

    # Improved test emails
    test_emails = [

        # ---------------- PHISHING EMAILS ----------------
        {
            "text": "URGENT: Your PayPal account has been suspended due to suspicious activity. Please verify your account immediately using the secure link below.",
            "expected": "PHISHING"
        },
        {
            "text": "Security Alert: We detected unauthorized login attempts on your bank account. Confirm your credentials within 24 hours to prevent suspension.",
            "expected": "PHISHING"
        },
        {
            "text": "Congratulations! You have been selected as the winner of our international lottery. Send your bank details to receive your prize.",
            "expected": "PHISHING"
        },
        {
            "text": "Your Netflix payment failed. Update your billing information now to continue enjoying our services without interruption.",
            "expected": "PHISHING"
        },
        {
            "text": "Your Apple ID has been locked due to suspicious activity. Click the link below to restore access to your account.",
            "expected": "PHISHING"
        },
        {
            "text": "We noticed suspicious activity on your Amazon account. Please reset your password immediately to protect your account.",
            "expected": "PHISHING"
        },
        {
            "text": "Important Notice: Your mailbox has exceeded its storage limit. Verify your account now to avoid losing incoming emails.",
            "expected": "PHISHING"
        },
        {
            "text": "You have received a tax refund of $450. Please confirm your details to process the refund to your bank account.",
            "expected": "PHISHING"
        },
        {
            "text": "Your package delivery failed due to incorrect address details. Update your shipping information immediately.",
            "expected": "PHISHING"
        },
        {
            "text": "Final warning: Your online banking access will be disabled unless you verify your identity immediately.",
            "expected": "PHISHING"
        },

        # ---------------- LEGITIMATE EMAILS ----------------
        {
            "text": "Subject: Meeting Reminder\n\nHi team,\nJust a reminder that our weekly project meeting will take place tomorrow at 10 AM in the conference room.",
            "expected": "LEGITIMATE"
        },
        {
            "text": "Hi John,\n\nCan you please review the updated financial report before the board meeting on Friday?",
            "expected": "LEGITIMATE"
        },
        {
            "text": "Subject: Team Lunch\n\nHello everyone,\nWe are planning a team lunch this Friday at 1 PM. Please confirm your availability.",
            "expected": "LEGITIMATE"
        },
        {
            "text": "Hi Sarah,\n\nThank you for sending the updated presentation slides. I'll review them before tomorrow's meeting.",
            "expected": "LEGITIMATE"
        },
        {
            "text": "Subject: Assignment Deadline\n\nDear students,\nThe deadline for submitting the assignment has been extended to Monday.",
            "expected": "LEGITIMATE"
        },
        {
            "text": "Hello,\n\nYour order has been successfully processed and will be shipped within the next two business days.",
            "expected": "LEGITIMATE"
        },
        {
            "text": "Hi Mark,\n\nThanks again for your help with the project report yesterday. I really appreciate your support.",
            "expected": "LEGITIMATE"
        },
        {
            "text": "Subject: Travel Itinerary\n\nAttached is the travel itinerary for your upcoming business trip next week.",
            "expected": "LEGITIMATE"
        },
        {
            "text": "Hi team,\n\nPlease find the meeting notes attached from today's discussion.",
            "expected": "LEGITIMATE"
        },
        {
            "text": "Hello,\n\nYour Amazon order has shipped and will arrive by Thursday. Track the shipment using your order page.",
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