import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import T5Tokenizer
from tqdm import tqdm
import random
import numpy as np
from utils.data_reader import create_dataloader
from utils.models import load_t5
from utils.plot_logger import setup_logger, plot_losses, plot_metric
from utils.evaluate_performance import compute_all_metrics
from utils.model_utils import load_modellll, save_model
from transformers import T5ForConditionalGeneration, T5Tokenizer

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

def evaluate_model(model, tokenizer, dataloader, device):
    model.eval()
    predictions, references, original_texts = [], [], []
    total_loss, count = 0.0, 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()
            count += 1
            
            gen_outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=128).to(device)
            batch_preds = tokenizer.batch_decode(gen_outputs, skip_special_tokens=True)
            
            labels_for_decode = labels.clone()
            labels_for_decode[labels_for_decode == -100] = tokenizer.pad_token_id
            batch_refs = tokenizer.batch_decode(labels_for_decode, skip_special_tokens=True)
            
            batch_orig = tokenizer.batch_decode(input_ids, skip_special_tokens=True) if "toxic_text" not in batch else list(batch["toxic_text"])
            
            predictions.extend(batch_preds)
            references.extend(batch_refs)
            original_texts.extend(batch_orig)
    
    avg_loss = total_loss / count if count > 0 else 0.0
    metrics = compute_all_metrics(predictions, references, original_texts=original_texts)
    return metrics, avg_loss

def main():
    run_name = "t5_sequential_finetune_toxic_lr_3e-5_eval"
    load_proj = "t5_sequential_finetune_toxic_lr_3e-5"
    num_epochs = 30
    batch_size = 8
    lr = 1e-5
    
    results_dir = f"results/{run_name}"
    save_dir = f"{results_dir}/model_checkpoint"
    plot_dir = f"{results_dir}/plots"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # load_proj = "t5_pretrain_contrastive_toxic"
    load_dir = f"results/{load_proj}/model_checkpoint"
    model, tokenizer = load_modellll(T5ForConditionalGeneration, T5Tokenizer, load_dir, device)

    model.to(device)
    
    train_dataloader, val_dataloader = create_dataloader("Data/paradetox.tsv", tokenizer, batch_size=batch_size, max_length=128, eval_size=batch_size)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    train_losses, eval_losses, eval_bleu_scores, eval_meteor_scores, eval_toxicity_scores, eval_fluency_scores, eval_content_preservation_scores  = [], [], [], [], [], [], []
    
    for epoch in range(1, num_epochs + 1):
        scheduler.step()
        eval_metrics, eval_loss = evaluate_model(model, tokenizer, val_dataloader, device)
        eval_losses.append(eval_loss)
        
        eval_bleu_scores.append(eval_metrics["bleu"].get("score", 0.0))
        eval_meteor_scores.append(eval_metrics["meteor"].get("meteor", 0.0))
        eval_toxicity_scores.append(eval_metrics["toxicity"])
        eval_fluency_scores.append(eval_metrics["fluency"])
        eval_content_preservation_scores.append(eval_metrics["content_preservation"])
    
    try:
        plot_metric(eval_bleu_scores, "Validation BLEU", save_dir=plot_dir)
        plot_metric(eval_meteor_scores, "Validation METEOR", save_dir=plot_dir)
        plot_metric(eval_toxicity_scores, "Validation Toxicity", save_dir=plot_dir)
        plot_metric(eval_fluency_scores, "Validation Fluency", save_dir=plot_dir)
        plot_metric(eval_content_preservation_scores, "Validation Content Preservation", save_dir=plot_dir)
        plot_losses(train_losses, eval_losses, save_dir=plot_dir)
    except ImportError:
        print("Plotting module not found. Skipping metric plots.")
        
    try:
        save_model(model, tokenizer, optimizer, epoch=10, loss=0.3, file_path=save_dir)
        print(f"Model and tokenizer saved to models/{run_name}")
    except Exception as e:
        print(f"Error saving model and tokenizer: {e}")

if __name__ == "__main__":
    main()
