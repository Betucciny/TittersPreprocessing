from transformers import TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import pandas as pd


def load_dataset(file_path, tokenizer, block_size=128):
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size
    )
    return dataset


def fine_tune_gpt2(model, tokenizer, train_file_path, output_dir):
    train_dataset = load_dataset(train_file_path, tokenizer)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    trainer.train()
    
def main():
    train_file_path = 'data/train.txt'
    output_dir = './gpt2-fine-tuned'
    model_name = 'gpt2'
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    fine_tune_gpt2(model, tokenizer, train_file_path, output_dir)
    
    
if __name__ == '__main__':
    main()