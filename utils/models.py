from transformers import T5ForConditionalGeneration, T5Tokenizer

def load_t5(model_name: str = "google-t5/t5-base"):
    """
    Loads the T5 model and its tokenizer.

    Args:
        model_name (str): The model variant to load. Default is "t5-base".

    Returns:
        model: An instance of T5ForConditionalGeneration.
        tokenizer: An instance of T5Tokenizer.
    """
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return model, tokenizer

if __name__ == "__main__":
    model, tokenizer = load_t5()
    print("T5 model and tokenizer loaded successfully.")