import torch
import torch.nn as nn
import torch.optim as optim
from utils.data_reader import create_dataloader
from utils.models import load_t5
from utils.plot_logger import setup_logger, plot_losses, plot_metric
from transformers import T5Tokenizer
from utils.evaluate_performance import compute_all_metrics
import random
import numpy as np
from tqdm import tqdm

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

def train_epoch(model, tokenizer, dataloader, optimizer, device, epoch, logger, lambda_cycle=1.0):
    model.train()
    total_forward_loss = 0.0
    total_cycle_loss = 0.0

    # Use tqdm for batch progress with dynamic postfix
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} Training", leave=False)
    for batch in progress_bar:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass: detoxification (toxic -> detoxified)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        forward_loss = outputs.loss

        # --- Cycle Consistency Step ---
        detox_outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=512)
        detox_texts = tokenizer.batch_decode(detox_outputs, skip_special_tokens=True)

        toxic_texts = batch.get("toxic_text", None)
        if toxic_texts is None:
            toxic_texts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        else:
            toxic_texts = list(toxic_texts)

        # Prepare reverse pass input with a reverse prompt "toxic: "
        reverse_inputs = ["toxic: " + text for text in detox_texts]
        reverse_encodings = tokenizer(
            reverse_inputs,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(device)

        toxic_encodings = tokenizer(
            toxic_texts,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(device)
        cycle_outputs = model(
            input_ids=reverse_encodings["input_ids"],
            attention_mask=reverse_encodings["attention_mask"],
            labels=toxic_encodings["input_ids"]
        )
        cycle_loss = cycle_outputs.loss

        total_loss = forward_loss + lambda_cycle * cycle_loss

        total_loss.backward()
        optimizer.step()

        total_forward_loss += forward_loss.item()
        total_cycle_loss += cycle_loss.item()

        # Update tqdm progress bar with current loss values.
        progress_bar.set_postfix({
            "Fwd": f"{forward_loss.item():.4f}",
            "Cycle": f"{cycle_loss.item():.4f}",
            "Total": f"{total_loss.item():.4f}"
        })

    avg_forward_loss = total_forward_loss / len(dataloader)
    avg_cycle_loss = total_cycle_loss / len(dataloader)
    logger.info(f"Epoch {epoch}: Avg Forward Loss = {avg_forward_loss:.4f}, Avg Cycle Loss = {avg_cycle_loss:.4f}")
    print(f"Epoch {epoch}: Avg Forward Loss = {avg_forward_loss:.4f}, Avg Cycle Loss = {avg_cycle_loss:.4f}")
    return avg_forward_loss, avg_cycle_loss


def evaluate_model(model, tokenizer, dataloader, device):
    model.eval()
    predictions = []
    references = []
    original_texts = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=128)
            batch_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            batch_refs = tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            if "toxic_text" in batch:
                batch_orig = list(batch["toxic_text"])
            else:
                batch_orig = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            
            predictions.extend(batch_preds)
            references.extend(batch_refs)
            original_texts.extend(batch_orig)

    metrics = compute_all_metrics(predictions, references, original_texts=original_texts)
    return metrics

def main():
    run_name = "cycle_consistency_epoch100_lr1e-5"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logger(f"{run_name}.log")
    model, tokenizer = load_t5("google-t5/t5-base")
    model.to(device)
    train_dataloader, val_dataloader = create_dataloader("Data/paradetox.tsv", tokenizer, batch_size=8, max_length=128)
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    num_epochs = 30
    train_forward_losses = []
    train_cycle_losses = []
    
    eval_bleu_scores = []
    eval_bert_scores = []
    eval_meteor_scores = []
    eval_sim_scores = []
    eval_fluency_scores = []
    
    train_bleu_scores = []  # New list for training BLEU scores

    for epoch in range(1, num_epochs + 1):
        logger.info(f"Starting epoch {epoch}")
        avg_forward_loss, avg_cycle_loss = train_epoch(
            model, tokenizer, train_dataloader, optimizer, device, epoch, logger, lambda_cycle=1.0
        )
        train_forward_losses.append(avg_forward_loss)
        train_cycle_losses.append(avg_cycle_loss)

        scheduler.step()
        torch.save(model.state_dict(), f"checkpoints/t5_epoch_{epoch}.pt")
        logger.info(f"Checkpoint saved for epoch {epoch}")

        # Evaluate on validation set
        eval_metrics = evaluate_model(model, tokenizer, val_dataloader, device)
        logger.info(f"Epoch {epoch} Evaluation Metrics: {eval_metrics}")
        
        bleu_val = eval_metrics["bleu"].get("score", 0.0)
        bert_val = eval_metrics["bert_score"].get("f1", 0.0)
        meteor_val = eval_metrics["meteor"].get("meteor", 0.0)
        sim_val = eval_metrics["content_preservation"] if eval_metrics["content_preservation"] is not None else 0.0
        fluency_val = eval_metrics["fluency"]

        eval_bleu_scores.append(bleu_val)
        eval_bert_scores.append(bert_val)
        eval_meteor_scores.append(meteor_val)
        eval_sim_scores.append(sim_val)
        eval_fluency_scores.append(fluency_val)
        
        # Evaluate on training set (only BLEU is computed here, but you can add others similarly)
        train_metrics = evaluate_model(model, tokenizer, train_dataloader, device)
        train_bleu = train_metrics["bleu"].get("score", 0.0)
        train_bleu_scores.append(train_bleu)
        logger.info(f"Epoch {epoch} Training BLEU: {train_bleu:.4f}")
    plots_dir = f"results/cycle_consistency/{run_name}"
    try:
        plot_losses(train_forward_losses, train_cycle_losses, save_dir=plots_dir)
        plot_metric(eval_bleu_scores, "Validation BLEU", save_dir=plots_dir)
        plot_metric(eval_bert_scores, "Validation BERTScore", save_dir=plots_dir)
        plot_metric(eval_meteor_scores, "Validation METEOR", save_dir=plots_dir)
        plot_metric(eval_sim_scores, "Validation Content Preservation", save_dir=plots_dir)
        plot_metric(eval_fluency_scores, "Validation Fluency", save_dir=plots_dir)
        plot_metric(train_bleu_scores, "Training BLEU", save_dir=plots_dir)
    except ImportError:
        logger.info("Plotting module not found. Skipping metric plots.")

if __name__ == "__main__":
    main()