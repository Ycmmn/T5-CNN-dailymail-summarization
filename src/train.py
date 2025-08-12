#=========== import libraries =======
from datasets import load_dataset
from transformers import (
    T5TokenizerFast,
    T5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
import evaluate
import numpy as np
import torch


#========= Config Variables ==========
MODEL_NAME = "t5-small"
DATASET_NAME = "cnn_dailymail"
DATASET_CONFIG = "3.0.0"
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 128
BATCH_SIZE = 8
EPOCHS = 1
OUTPUT_DIR = "./t5_cnn_dailymail_finetuned"
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



# ======  Load tokenizer and model ======
tokenizer = T5TokenizerFast.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)


# ======= Load dataset =========
dataset = load_dataset(DATASET_NAME, DATASET_CONFIG)


prefix = "summarize: "

def preprocess_function(examples):
    # Add task prefix
    inputs = [prefix + doc for doc in examples["article"]]
    model_inputs = tokenizer(
        inputs, max_length=MAX_INPUT_LENGTH, truncation=True, padding="max_length"
    )

    # Tokenize targets (labels)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["highlights"],
            max_length=MAX_TARGET_LENGTH,
            truncation=True,
            padding="max_length"
        )

    # Replace pad tokens in labels with -100 to ignore in loss
    model_inputs["labels"]=labels["input_ids"]
    model_inputs["labels"] = [
        [(label if label != tokenizer.pad_token_id else -100) for label in label_seq]
        for label_seq in model_inputs["labels"]
    ]

    return model_inputs
    return {
        "input_ids": [...],
        "attention_mask" : [...],
        "labels": [...]
    }

print("Tokenizing dataset...")

tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,  # Process multiple samples at once for faster mapping
    remove_columns=dataset["train"].column_names  # Remove original columns (article, highlights)
)


# Create a data collator that will dynamically pad the inputs and labels for Seq2Seq models
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
