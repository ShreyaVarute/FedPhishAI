"""
Fine-tune T5 model to generate natural language explanations for phishing predictions.
"""
import os
import json
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm


class ExplanationDataset(Dataset):
    """Dataset for T5 explanation generation training."""

    def __init__(self, data_path: str, tokenizer, max_input_length=512, max_target_length=128):
        self.df = pd.read_csv(data_path)
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

        # Expects columns: 'email_text', 'prediction', 'explanation'
        required_cols = ['email_text', 'prediction', 'explanation']
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"Missing column: {col}. Required: {required_cols}")

        self.df = self.df.dropna(subset=required_cols).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        email_text = str(row['email_text'])[:400]  # Truncate
        prediction = str(row['prediction'])

        if prediction.lower() == 'phishing':
            input_text = f"explain why this email is phishing: {email_text}"
        else:
            input_text = f"explain why this email is legitimate: {email_text}"

        target_text = str(row['explanation'])

        # Tokenize input
        input_enc = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Tokenize target
        target_enc = self.tokenizer(
            target_text,
            max_length=self.max_target_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        labels = target_enc['input_ids'].squeeze()
        # Replace padding token id with -100 so it's ignored in loss computation
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': input_enc['input_ids'].squeeze(),
            'attention_mask': input_enc['attention_mask'].squeeze(),
            'labels': labels
        }


class T5ExplanationTrainer:
    def __init__(
        self,
        train_csv: str,
        val_csv: str,
        model_save_dir: str = 'models/t5_explanations',
        epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 3e-4,
        device: str = None
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        os.makedirs(model_save_dir, exist_ok=True)
        self.model_save_dir = model_save_dir
        self.epochs = epochs

        print(f"Loading T5-small model...")
        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')
        self.model = T5ForConditionalGeneration.from_pretrained('t5-small').to(self.device)

        train_dataset = ExplanationDataset(train_csv, self.tokenizer)
        val_dataset = ExplanationDataset(val_csv, self.tokenizer)

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(self.train_loader) * epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for batch in tqdm(self.train_loader, desc="T5 Training"):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="T5 Validation"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                total_loss += outputs.loss.item()
        return total_loss / len(self.val_loader)

    def train(self):
        best_val_loss = float('inf')
        history = {'train_loss': [], 'val_loss': []}

        for epoch in range(self.epochs):
            print(f"\n=== T5 Epoch {epoch + 1}/{self.epochs} ===")
            train_loss = self.train_epoch()
            val_loss = self.validate()
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.model.save_pretrained(os.path.join(self.model_save_dir, 'best_model'))
                self.tokenizer.save_pretrained(os.path.join(self.model_save_dir, 'best_model'))
                print(f"  ✓ Best T5 model saved")

        with open(os.path.join(self.model_save_dir, 'history.json'), 'w') as f:
            json.dump(history, f, indent=2)

        return history