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

