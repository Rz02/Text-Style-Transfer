import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import torch
from utils.model_utils import load_model
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch.optim as optim

if __name__ == "__main__":
    
    run_name = "t5_finetune_toxic" 
    results_dir = f"results/{run_name}"
    save_dir = f"{results_dir}/model_checkpoint"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize an optimizer instance before passing it
    optimizer = optim.AdamW(params=[torch.tensor(0.0)], lr=1e-5)  # Dummy initialization
    
    model, tokenizer = load_model(T5ForConditionalGeneration, T5Tokenizer, optimizer, save_dir, device)
    model.eval()
    
    sentence = "This is a fucking test sentence."
    
    # Tokenize input
    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    
    # Run inference
    with torch.no_grad():
        outputs = model.generate(**inputs)
    
    # Decode output
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Output:", result)