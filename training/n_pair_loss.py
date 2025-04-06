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

set_seed(42)

def n_pair_loss(toxic_embeds, neutral_embeds_all, temperature=0.1):
    """
    N-pair loss function to ensure:
    1. Neutral sentences (neutral1, neutral2, neutral3) are similar to each other.
    2. Toxic sentences are dissimilar to all neutral sentences.

    Args:
        toxic_embeds (Tensor): (batch_size, embed_dim), toxic text embeddings.
        neutral_embeds_all (list of Tensors): List of (batch_size, embed_dim) tensors, one per neutral sentence.
        temperature (float): Scaling factor for cosine similarity.

    Returns:
        Tensor: Computed N-pair contrastive loss.
    """
    # Convert list to tensor (batch_size x num_neutrals x embed_dim)
    neutral_embeds_all = torch.stack(neutral_embeds_all)

    # Get the first neutral embedding (batch_size x embed_dim)
    neutral_embeds = neutral_embeds_all[:, 0, :]

    # Normalize embeddings
    toxic_norm = F.normalize(toxic_embeds, p=2, dim=-1)
    neutral_norm = F.normalize(neutral_embeds, p=2, dim=-1)
    neutral_all_norm = F.normalize(neutral_embeds_all, p=2, dim=-1)

    # Compute similarity
    toxic_neutral_sim = torch.matmul(toxic_norm, neutral_norm.mT)  # batch_size x batch_size
    neutral_pair_sim = torch.matmul(neutral_all_norm.view(-1, neutral_all_norm.shape[-1]), 
                                    neutral_all_norm.view(-1, neutral_all_norm.shape[-1]).mT)  # (batch_size*num_neutrals) x (batch_size*num_neutrals)

    # Mask out diagonal (self-similarity)
    mask = torch.eye(neutral_pair_sim.size(0), device=neutral_pair_sim.device).bool()
    neutral_pair_sim = neutral_pair_sim.masked_fill(mask, -float('inf'))

    # Apply temperature scaling
    toxic_neutral_sim /= temperature
    neutral_pair_sim /= temperature

    # Contrastive loss for toxic vs neutral
    # For cross_entropy, the target should be a 1D tensor of class indices, here we set it as zeros (indicating class 0 for all)
    toxic_loss = F.cross_entropy(toxic_neutral_sim, torch.zeros(toxic_neutral_sim.size(0), dtype=torch.long, device=toxic_neutral_sim.device))

    # Loss for similarity between neutral texts
    neutral_loss = torch.mean(F.relu(-neutral_pair_sim))

    return toxic_loss + neutral_loss


def train_epoch(model, tokenizer, dataloader, optimizer, device, epoch, logger):
    model.train()
    total_loss = 0.0
    
    # Use tqdm for batch progress with dynamic postfix
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} Training", leave=False)
    
    for batch in progress_bar:
        optimizer.zero_grad()
        toxic_ids = batch["toxic_ids"].to(device)
        toxic_attention_mask = batch["toxic_attention_mask"].to(device)
        neutral_ids_all = [embed.to(device) for embed in batch["neutral_ids_all"]]
        neutral_attention_mask_all = [embed.to(device) for embed in batch["neutral_attention_mask_all"]]
        
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

        # Calculate the N-pair loss
        loss = n_pair_loss(toxic_embedding, neutral_embeddings)

        # Backpropagate and optimize
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
    
    avg_loss = total_loss / len(dataloader)
    logger.info(f"Epoch {epoch}: Avg Loss = {avg_loss:.4f}")
    print(f"Epoch {epoch}: Avg Loss = {avg_loss:.4f}")
    return avg_loss

def evaluate_model(model, tokenizer, dataloader, device):
    model.eval()
    total_loss, count = 0.0, 0

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)

        for batch in progress_bar:
            toxic_ids = batch["toxic_ids"].to(device)
            toxic_attention_mask = batch["toxic_attention_mask"].to(device)
            neutral_ids_all = [embed.to(device) for embed in batch["neutral_ids_all"]]
            neutral_attention_mask_all = [embed.to(device) for embed in batch["neutral_attention_mask_all"]]

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

            # Calculate the N-pair loss
            loss = n_pair_loss(toxic_embedding, neutral_embeddings)
            
            total_loss += loss.item()
            count += 1
            
    avg_loss = total_loss / count if count > 0 else 0.0
    return avg_loss

def main():
    run_name = "t5_pretrain_contrastive_toxic"
    num_epochs = 10
    batch_size = 8
    lr = 3e-5
    max_length = 128
    
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
    
    train_dataloader, val_dataloader = create_dataloader("Data/paradetox.tsv", tokenizer, batch_size=batch_size, max_length=max_length, eval_size=batch_size, dataset_classs=ContrastiveDetoxDataset)
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
        
        avg_loss = train_epoch(model, tokenizer, train_dataloader, optimizer, device, epoch, logger)
        train_losses.append(avg_loss)
        logger.info(f"Epoch {epoch} Training Loss: {avg_loss:.4f}")
        
        scheduler.step()
        
        eval_loss = evaluate_model(model, tokenizer, val_dataloader, device)
        eval_losses.append(eval_loss)
        logger.info(f"Epoch {epoch} Evaluation Loss: {eval_loss:.4f}")
    
    try:
        plot_losses(train_losses, save_dir=plot_dir)
    except ImportError:
        logger.info("Plotting module not found. Skipping loss plot.")
    
    try:
        save_model(model, tokenizer, optimizer, epoch=num_epochs, loss=train_losses[-1], file_path=save_dir)
        logger.info(f"Model and tokenizer saved to {save_dir}")
    except Exception as e:
        logger.error(f"Error saving model and tokenizer: {e}")

if __name__ == "__main__":
    main()
