from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

class ExplanationGenerator:
    def __init__(self, model_path='models/t5_explanations/'):
        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')
        self.model     = T5ForConditionalGeneration.from_pretrained(model_path)
        self.model.eval()

    def generate(self, prompt: str, max_length: int = 128) -> str:
        inputs = self.tokenizer(
            prompt, return_tensors='pt', max_length=512, truncation=True)
        with torch.no_grad():
            output = self.model.generate(
                **inputs, max_length=max_length,
                num_beams=4, early_stopping=True
            )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)