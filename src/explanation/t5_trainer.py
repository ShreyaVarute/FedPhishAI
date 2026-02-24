from transformers import (T5ForConditionalGeneration, T5Tokenizer,
                             Seq2SeqTrainer, Seq2SeqTrainingArguments)
from datasets import Dataset
import pandas as pd

tokenizer = T5Tokenizer.from_pretrained('t5-small')
model     = T5ForConditionalGeneration.from_pretrained('t5-small')

def tokenize_fn(examples):
    inputs  = tokenizer(examples['input'],  max_length=512, truncation=True, padding='max_length')
    targets = tokenizer(examples['target'], max_length=128, truncation=True, padding='max_length')
    inputs['labels'] = targets['input_ids']
    return inputs

def finetune_t5(
    csv_path='data/explanations/dataset.csv',
    output_dir='models/t5_explanations/'
):
    df      = pd.read_csv(csv_path)
    dataset = Dataset.from_pandas(df).map(tokenize_fn, batched=True)
    split   = dataset.train_test_split(test_size=0.1)

    args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        predict_with_generate=True,
        logging_dir='./logs',
    )
    trainer = Seq2SeqTrainer(
        model=model, args=args,
        train_dataset=split['train'],
        eval_dataset=split['test'],
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(output_dir)
    print(f'T5 saved to {output_dir}')