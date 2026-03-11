import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from src.data.preprocessor import EmailPreprocessor


class ExplanationGenerator:
    """
    Generate human-readable explanations for phishing predictions using T5.
    """
    def __init__(self, model_path: str = None, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')
        self.preprocessor = EmailPreprocessor()

        if model_path and torch.cuda.is_available():
            self.model = T5ForConditionalGeneration.from_pretrained('t5-small')
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            # Use pretrained T5 for zero-shot explanation generation
            self.model = T5ForConditionalGeneration.from_pretrained('t5-small')

        self.model = self.model.to(self.device)
        self.model.eval()

    def generate_explanation(self, email_text: str, prediction: str, confidence: float = None) -> str:
        """
        Generate explanation for a phishing prediction.

        Args:
            email_text: Original email content
            prediction: 'phishing' or 'legitimate'
            confidence: Model confidence score (optional)

        Returns:
            Human-readable explanation string
        """
        cleaned = self.preprocessor.clean_email(email_text)
        truncated = cleaned[:400]  # Truncate for T5 input

        if prediction.lower() == 'phishing':
            prompt = (
                f"explain why this email is phishing: {truncated}"
            )
        else:
            prompt = (
                f"explain why this email is legitimate: {truncated}"
            )

        inputs = self.tokenizer(
            prompt,
            max_length=512,
            truncation=True,
            return_tensors='pt'
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=100,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=2
            )

        explanation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return explanation

    def generate_rule_based_explanation(self, email_text: str, prediction: str, top_tokens: list = None) -> str:
        """
        Fallback rule-based explanation when T5 model isn't trained yet.
        Uses email features and attention tokens to construct explanation.
        """
        import re
        text_lower = email_text.lower()

        urgency_words = ['urgent', 'immediately', 'suspended', 'expired', 'final warning', 'act now']
        financial_words = ['bank', 'account', 'password', 'credit card', 'verify', 'payment', 'paypal']
        threat_words = ['suspended', 'locked', 'disabled', 'closed', 'compromised']
        has_url = bool(re.search(r'http[s]?://', email_text))
        has_email_addr = bool(re.search(r'[\w\.-]+@[\w\.-]+', email_text))

        found_urgency = [w for w in urgency_words if w in text_lower]
        found_financial = [w for w in financial_words if w in text_lower]
        found_threat = [w for w in threat_words if w in text_lower]

        if prediction.lower() == 'phishing':
            reasons = []
            if found_urgency:
                reasons.append(f"urgency language ('{found_urgency[0]}')")
            if found_financial:
                reasons.append(f"financial/credential requests ('{found_financial[0]}')")
            if found_threat:
                reasons.append(f"threat language ('{found_threat[0]}')")
            if has_url:
                reasons.append("suspicious links")
            if not reasons:
                reasons.append("suspicious patterns typical of phishing")

            explanation = f"This email was classified as phishing because it contains {', and '.join(reasons)}, which are common indicators of phishing attacks."
        else:
            explanation = "This email was classified as legitimate because it uses normal conversational language without urgency, threats, or requests for sensitive information."

        return explanation