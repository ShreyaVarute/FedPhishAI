import torch
from torch.utils.data import Dataset
import pandas as pd
from src.data.preprocessor import clean_email, tokenizer


class EmailDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

        # Clean text
        self.df['text'] = self.df['text'].apply(clean_email)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.iloc[idx]['text']
        label = self.df.iloc[idx]['label']

        # Tokenize on-the-fly (FAST + memory efficient)
        encoding = tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=256,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }


def train_one_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0

    for i, batch in enumerate(loader):
        ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        lbls = batch['label'].to(device)

        optimizer.zero_grad()

        logits, _ = model(ids, mask)
        loss = criterion(logits, lbls)

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        # Show progress (VERY IMPORTANT)
        if i % 50 == 0:
            print(f"Batch {i}/{len(loader)} | Loss: {loss.item():.4f}")

    return total_loss / len(loader)