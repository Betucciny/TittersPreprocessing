from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import DatasetDict
import torch

def main():
    # Load the dataset
    dataset = load_dataset("json", data_files="data/data.json")
    dataset = dataset['train'].train_test_split(test_size=0.1)
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    
    # Define the tokenize function based on the actual structure of your dataset
    def tokenize_function(examples):
        # Combine 'prompt' and 'response' for input and labels
        inputs = examples['prompt']
        targets = examples['response']
        model_inputs = tokenizer(inputs, max_length=100, truncation=True, padding="max_length")
        labels = tokenizer(targets, max_length=100, truncation=True, padding="max_length").input_ids
        model_inputs["labels"] = labels
        return model_inputs
    
    # Tokenize the dataset
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Load the model
    model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model.gradient_checkpointing_enable()  # Enable gradient checkpointing

    # Define the training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=1,  # Reduce batch size
        per_device_eval_batch_size=1,   # Reduce batch size
        num_train_epochs=3,
        weight_decay=0.01,
        gradient_accumulation_steps=8,  # Accumulate gradients over 8 steps
        fp16=True,  # Enable mixed precision training
        use_cpu=True,  # Use CPU for training
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],  # Use the 'test' split for validation
    )
    
    # Train the model
    trainer.train()
    
    # Save the fine-tuned model and tokenizer
    model.save_pretrained("./fine-tuned-model")
    tokenizer.save_pretrained("./fine-tuned-model")

if __name__ == "__main__":
    main()
