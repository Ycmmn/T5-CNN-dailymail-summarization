def generate_summary(text: str, max_length: int = 128, num_beams: int = 4):
    # Add prefix for summarization
    input_text = prefix + text
    
    # Tokenize input text
    inputs = tokenizer(
        input_text, max_length=MAX_INPUT_LENGTH, truncation=True, return_tensors="pt"
    ).to(DEVICE)
    
    # Set model to evaluation mode
    model.eval()
    
    # Generate summary without gradient calculation
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_length=max_length,        # Max length of summary
            num_beams=num_beams,          # Beam search size
            early_stopping=True,          # Stop when EOS token is reached
            no_repeat_ngram_size=3,       # Avoid repeating 3-word sequences
        )
    
    # Decode token IDs to text
    summary = tokenizer.decode(
        generated_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    
    return summary




# ---------------------- Sample usage --------------------

# Get the first article from the test dataset
article = dataset["test"][0]["article"]

print("---- ARTICLE ----")
print(article)

print("\n---- GENERATED SUMMARY ----")
print(generate_summary(article))
