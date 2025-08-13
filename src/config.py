# Set training arguments for Seq2Seq model
# Training settings
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,                   # Save model here
    evaluation_strategy="epoch",             # Evaluate each epoch
    learning_rate=5e-5,                      # Learning rate
    per_device_train_batch_size=BATCH_SIZE,  # Train batch size
    per_device_eval_batch_size=BATCH_SIZE,   # Eval batch size
    weight_decay=0.01,                       # Regularization
    save_total_limit=3,                      # Max saved models
    num_train_epochs=EPOCHS,                 # Training epochs
    predict_with_generate=True,              # Generate text for eval
    fp16=torch.cuda.is_available(),          # Use 16-bit if GPU
    logging_dir='./logs',                    # Log folder
    logging_strategy='steps',                # Log by steps
    logging_steps=200,                       # Log every 200 steps
    seed=SEED,                               # Random seed
)

