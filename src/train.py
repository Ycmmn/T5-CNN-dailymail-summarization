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
    batched=True,  # Process batches instead of single samples
    remove_columns=dataset["train"].column_names  # Remove original columns (article, highlights)
)


# Create a data collator that pads inputs and labels for Seq2Seq models
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


# Load the ROUGE metric for summarization evaluation
rouge = evaluate.load("rouge")



def postprocess_text(preds_text, labels_text):
    # Remove espace from each prediction
    preds_text = [pred.strip() for pred in preds_text]
    # Remove espace from each label
    labels_text = [label.strip() for label in labels_text]
    return preds_text, labels_text




def compute_metrics(eval_pred):
    # eval_pred: output from the Trainer as a tuple (predictions, labels)
    # Unpack eval_pred into predictions and labels
    preds, labels = eval_pred
    if isinstance(preds, tuple):
        preds = preds[0]
     

# Decode token IDs back into text, skipping special tokens like <pad> 
decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

# Replace -100 with the pad token ID
# np.where(condition, x, y)
labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

# Decode label token IDs back into text, skipping special tokens
decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

# Clean up whitespace in predictions and labels
decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

# Compute ROUGE scores between predictions and references, using stemming for better matching
result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

# Get the F1 score for each ROUGE metric 
result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

# Calculate the average of ROUGE-1, ROUGE-2, and ROUGE-L scores
result["rouge_mean"] = np.mean(
    [result.get("rouge1", 0), result.get("rouge2", 0), result.get("rougeL", 0)]
)

# This is a Dictionary Comprehension
# Round all metric values to 4 decimal places
result = {k: round(v, 4) for k, v in result.items()}

return result



# ---------- Trainer ----------
trainer = Seq2SeqTrainer(
    model=model,                         # Model
    args=training_args,                   # Training settings
    train_dataset=tokenized_dataset["train"],      # Training data
    eval_dataset=tokenized_dataset["validation"],  # Validation data
    tokenizer=tokenizer,                  # Tokenizer
    data_collator=data_collator,          # Pads batches
    compute_metrics=compute_metrics,      # Eval metrics
)

# --------------------- Training Model ---------------------
trainer.train()

# ----------------------- Validation ----------------
vali_metrics = trainer.evaluate(tokenized_dataset["validation"])
print("Validation metrics:", vali_metrics)

# ------------------------ Test ---------------------
# after training model
test_metrics = trainer.evaluate(tokenized_dataset["test"])
print("Test metrics:", test_metrics)

