import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import T5Tokenizer, T5EncoderModel
from tqdm import tqdm
import random
import numpy as np
from datasets import load_dataset
from utils.plot_logger import setup_logger, plot_losses
from utils.model_utils import save_model
from utils.data_reader import ContrastiveDetoxDataset, create_dataloader
from utils.models import load_t5

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(55)

class TripletLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Compute Euclidean distance between the anchor, positive, and negative
        positive_distance = F.pairwise_distance(anchor, positive, p=2)
        negative_distance = F.pairwise_distance(anchor, negative, p=2)

        # Compute triplet loss with the margin
        loss = F.relu(positive_distance - negative_distance + self.margin)

        return loss.mean()

class ContrastiveLogSoftmaxLoss(nn.Module):
    def __init__(self):
        """
        Contrastive loss based on log-softmax similarity.
        Uses only one negative sample per anchor.
        """
        super(ContrastiveLogSoftmaxLoss, self).__init__()

    def forward(self, anchor, positive, negative):
        """
        Computes the contrastive log-softmax loss.

        Args:
            anchor (Tensor): Shape (batch_size, embedding_dim), anchor embeddings.
            positive (Tensor): Shape (batch_size, embedding_dim), positive embeddings.
            negative (Tensor): Shape (batch_size, embedding_dim), negative embeddings.

        Returns:
            Tensor: Scalar loss value.
        """
        # Compute similarity (dot product) between anchor and positive
        pos_sim = torch.sum(anchor * positive, dim=-1)  # Shape: (batch_size,)

        # Compute similarity (dot product) between anchor and negative
        neg_sim = torch.sum(anchor * negative, dim=-1)  # Shape: (batch_size,)

        # Compute softmax numerator and denominator
        numerator = torch.exp(pos_sim)  # Shape: (batch_size,)
        denominator = numerator + torch.exp(neg_sim)  # Shape: (batch_size,)

        # Compute log-softmax loss
        loss = -torch.log(numerator / denominator)  # Shape: (batch_size,)

        return loss.mean()  # Return mean loss over batch

def train_epoch(model, tokenizer, dataloader, optimizer, device, epoch, logger, loss_func):
    model.train()
    total_loss, count = 0.0, 0
    
    # Use tqdm for batch progress with dynamic postfix
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} Training", leave=False)
    
    for batch in progress_bar:
        optimizer.zero_grad()
    
        number_of_neutral_sentences = batch["neutral_ids_all"].shape[1]
        # Skip batches with fewer than two neutral sentences
        if number_of_neutral_sentences < 2:
            continue 
        
        # Move the data to the appropriate device
        toxic_ids = batch["toxic_ids"].to(device)
        toxic_attention_mask = batch["toxic_attention_mask"].to(device)
        neutral_ids_all = [batch["neutral_ids_all"][:, i, :].to(device) for i in range(2)]
        neutral_attention_mask_all = [batch["neutral_attention_mask_all"][:, i, :].to(device) for i in range(2)]
        # neutral_attention_mask_all = [embed.to(device) for embed in batch["neutral_attention_mask_all"]]
        # neutral_ids_all = [embed.to(device) for embed in batch["neutral_ids_all"]]
        
        # Compute encoder last hidden state for toxic text (Encoder only forward pass)
        toxic_encoder_output = model.encoder(input_ids=toxic_ids, attention_mask=toxic_attention_mask).last_hidden_state
        toxic_embedding = toxic_encoder_output[:, 0, :]  # Get [CLS] token representation (batch_size, embed_dim)

        # Compute encoder last hidden state for neutral texts (encode each neutral separately)
        neutral_embeddings = []
        for neutral_text, neutral_attention_mask in zip(neutral_ids_all, neutral_attention_mask_all):
            neutral_encoder_output = model.encoder(
                input_ids=neutral_text,
                attention_mask=neutral_attention_mask
            ).last_hidden_state
            neutral_embedding = neutral_encoder_output[:, 0, :]  # Get [CLS] token representation
            neutral_embeddings.append(neutral_embedding)

        # For simplicity, use the first neutral sentence as the anchor, and the second as the positive.
        anchor = neutral_embeddings[0]   # Anchor: neutral sentence 1
        positive = neutral_embeddings[1] # Positive:  neutral sentence 2
        negative = toxic_embedding  # Negative: Toxic sentence

        # Calculate the Loss
        loss = loss_func(anchor, positive, negative)

        # Backpropagate and optimize
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        count += 1
        
        progress_bar.set_postfix({"Loss": f"{loss.item():.12f}"})
    
    avg_loss = total_loss / count if count > 0 else 0.0
    print("total_loss", total_loss)
    print("count", count)
    return avg_loss

def evaluate_model(model, tokenizer, dataloader, device, loss_func):
    model.eval()
    total_loss, count = 0.0, 0

    with torch.no_grad():  # Disable gradient computation for evaluation
        for batch in dataloader:
            number_of_neutral_sentences = batch["neutral_ids_all"].shape[1]
            if number_of_neutral_sentences < 2:
                continue  

            # Move the data to the appropriate device
            toxic_ids = batch["toxic_ids"].to(device)
            toxic_attention_mask = batch["toxic_attention_mask"].to(device)
            neutral_ids_all = [batch["neutral_ids_all"][:, i, :].to(device) for i in range(2)]
            neutral_attention_mask_all = [batch["neutral_attention_mask_all"][:, i, :].to(device) for i in range(2)]

            # Compute encoder output for toxic text
            toxic_encoder_output = model.encoder(input_ids=toxic_ids, attention_mask=toxic_attention_mask).last_hidden_state
            toxic_embedding = toxic_encoder_output[:, 0, :]  # Get [CLS] token representation

            # Compute encoder outputs for neutral texts
            neutral_embeddings = []
            for neutral_text, neutral_attention_mask in zip(neutral_ids_all, neutral_attention_mask_all):
                neutral_encoder_output = model.encoder(
                    input_ids=neutral_text,
                    attention_mask=neutral_attention_mask
                ).last_hidden_state
                neutral_embedding = neutral_encoder_output[:, 0, :]
                neutral_embeddings.append(neutral_embedding)

            # Use the first neutral sentence as the anchor, second as positive, and toxic as negative
            anchor = neutral_embeddings[0]   # Neutral sentence 1 (Anchor)
            positive = neutral_embeddings[1] # Neutral sentence 2 (Positive)
            negative = toxic_embedding       # Toxic sentence (Negative)

            # Compute loss
            loss = loss_func(anchor, positive, negative)

            total_loss += loss.item()
            count += 1

    avg_loss = total_loss / count if count > 0 else 0.0
    print("total_loss", total_loss)
    print("count", count)
    return avg_loss

def main():
    run_name = "t5_pretrain_triplet_toxic"
    num_epochs = 10
    batch_size = 8
    lr = 3e-5
    max_length = 128
    loss_func = TripletLoss(margin=1.2)  
    # loss_func = ContrastiveLogSoftmaxLoss()
    
    results_dir = f"results/{run_name}"
    save_dir = f"{results_dir}/model_checkpoint"
    plot_dir = f"{results_dir}/plots"
    
    logger = setup_logger(f"{results_dir}/train.log")
    logger.info("Starting training...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    model, tokenizer = load_t5("google-t5/t5-base")
    model.to(device)
    logger.info("Model loaded and moved to device.")
    
    train_dataloader, val_dataloader = create_dataloader("Data/paradetox.tsv", tokenizer, batch_size=batch_size, max_length=max_length, dataset_classs=ContrastiveDetoxDataset)
    logger.info("DataLoaders created.")
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # loss_fn = ContrastiveLoss()
    
    # class ContrastiveLoss(nn.Module):
    #     def __init__(self, temperature=0.07):
    #         super().__init__()
    #         self.temperature = temperature
    #         self.cosine_similarity = nn.CosineSimilarity(dim=-1)
        
    #     def forward(self, toxic_embeddings, non_toxic_embeddings):
    #         similarity = self.cosine_similarity(toxic_embeddings, non_toxic_embeddings)
    #         loss = -torch.log(torch.exp(similarity / self.temperature).sum(dim=-1))
    #         return loss.mean()
    
    # Freeze decoder parameters
    for param in model.decoder.parameters():
        param.requires_grad = False
    
    train_losses, eval_losses = [], []
    
    for epoch in range(1, num_epochs + 1):
        logger.info(f"Starting epoch {epoch}")
        
        avg_loss = train_epoch(model, tokenizer, train_dataloader, optimizer, device, epoch, logger, loss_func)
        train_losses.append(avg_loss)
        logger.info(f"Epoch {epoch} Training Loss: {avg_loss:.12f}")
        
        scheduler.step()
        
        eval_loss = evaluate_model(model, tokenizer, val_dataloader, device, loss_func)
        eval_losses.append(eval_loss)
        logger.info(f"Epoch {epoch} Evaluation Loss: {eval_loss:.12f}")
    
    try:
        save_model(model, tokenizer, optimizer, epoch=num_epochs, loss=train_losses[-1], file_path=save_dir)
        import pickle

        # Save training and evaluation losses after each epoch
        with open(f"{results_dir}/losses.pkl", "wb") as f:
            pickle.dump({"train_losses": train_losses, "eval_losses": eval_losses}, f)
            
        logger.info(f"Model and tokenizer saved to {save_dir}")
    except Exception as e:
        logger.error(f"Error saving model and tokenizer: {e}")
        
    try:
        plot_losses(train_losses, eval_losses, save_dir=plot_dir)
    except ImportError:
        logger.info("Plotting module not found. Skipping loss plot.")

if __name__ == "__main__":
    main()
