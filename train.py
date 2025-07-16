#!/usr/bin/env python3
"""
Ultra-compact FineWeb training for 20GB disk space
This version uses streaming and minimal disk usage
"""

import os
import torch
from math import exp
import click
from sentencepiece import SentencePieceProcessor
from model import *
import wandb
import json
import logging
from datasets import load_dataset, IterableDataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import gc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_ROOT = os.path.abspath(os.path.dirname(__file__))
MODEL_DIR = _ROOT + "/model"
BPE_MODEL_PATH = _ROOT + "/model/tokenizer.model"

os.makedirs(MODEL_DIR, exist_ok=True)

class StreamingFineWebDataset(IterableDataset):
    """
    Streaming dataset that processes FineWeb data on-the-fly
    Uses minimal memory and disk space
    """
    def __init__(self, 
                 config: str = "sample-10BT",
                 max_seq_len: int = 1024,
                 min_seq_len: int = 64,
                 tokenizer_path: str = BPE_MODEL_PATH,
                 max_samples: int = 10000):  # Reduced for disk space
        
        self.config = config
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len
        self.max_samples = max_samples
        
        # Load tokenizer
        self.tokenizer = SentencePieceProcessor(model_file=tokenizer_path)
        logger.info(f"Loaded tokenizer with vocab size: {self.tokenizer.get_piece_size()}")
        
        # Load dataset in streaming mode
        logger.info(f"Loading FineWeb dataset in streaming mode: {config}")
        self.dataset = load_dataset("HuggingFaceFW/fineweb", config, split="train", streaming=True)
        
    def __iter__(self):
        sample_count = 0
        
        for sample in self.dataset:
            if sample_count >= self.max_samples:
                break
                
            try:
                text = sample["text"]
                if not text or len(text.strip()) < 10:
                    continue
                
                # Tokenize with BOS and EOS tokens
                tokens = [self.tokenizer.bos_id()] + self.tokenizer.encode_as_ids(text) + [self.tokenizer.eos_id()]
                
                if len(tokens) < self.min_seq_len:
                    continue
                
                # Split into sequences
                for start_idx in range(0, len(tokens) - self.min_seq_len, self.max_seq_len // 2):
                    end_idx = start_idx + self.max_seq_len
                    sequence = tokens[start_idx:end_idx]
                    
                    if len(sequence) >= self.min_seq_len:
                        input_seq = sequence[:-1]
                        target_seq = sequence[1:]
                        
                        # Convert to tensors
                        input_tensor = torch.tensor(input_seq, dtype=torch.long)
                        target_tensor = torch.tensor(target_seq, dtype=torch.long)
                        
                        yield input_tensor, target_tensor
                        sample_count += 1
                        
                        if sample_count >= self.max_samples:
                            break
                
                # Clear memory
                del tokens
                gc.collect()
                
            except Exception as e:
                logger.warning(f"Error processing sample: {e}")
                continue

def create_compact_dataloader(batch_size: int = 4,
                             max_seq_len: int = 1024,
                             config: str = "sample-10BT",
                             max_samples: int = 10000) -> DataLoader:
    """
    Create a compact dataloader with minimal memory usage
    """
    
    def collate_fn(batch):
        batch_x, batch_y = zip(*batch)
        batch_x = pad_sequence(batch_x, batch_first=True, padding_value=0)
        batch_y = pad_sequence(batch_y, batch_first=True, padding_value=-1)
        return batch_x, batch_y
    
    dataset = StreamingFineWebDataset(
        config=config,
        max_seq_len=max_seq_len,
        max_samples=max_samples
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        collate_fn=collate_fn,
        num_workers=2,  # Reduced for memory
        pin_memory=False,  # Disabled to save memory
        persistent_workers=False  # Disabled to save memory
    )
    
    return dataloader

@click.command()
@click.option('--num-layers', type=int, default=6, help="No. of decoder layers")
@click.option('--hidden-size', type=int, default=768, help="hidden size")
@click.option('--num-heads', type=int, default=12, help="Number of heads")
@click.option('--max-seq-len', type=int, default=1024, help="Seq length")
@click.option('--vocab-size', type=int, default=32000, help="Vocab size")
@click.option('--batch-size', type=int, default=4, help="batch size")
@click.option('--learning-rate', type=float, default=0.0001, help="learning rate")
@click.option('--epochs', type=int, default=1, help="number of epochs")
@click.option('--fineweb-config', type=str, default="sample-10BT", help="FineWeb config")
@click.option('--max-samples', type=int, default=10000, help="Max samples (reduced for disk space)")
@click.option('--save-steps', type=int, default=200, help="Save model every N steps")
@click.option('--eval-steps', type=int, default=100, help="Evaluate every N steps")
@click.option('--warmup-steps', type=int, default=200, help="Warmup steps")
@click.option('--gradient-accumulation-steps', type=int, default=1, help="Gradient accumulation steps")
@click.option('--max-grad-norm', type=float, default=1.0, help="Max gradient norm for clipping")
@click.option('--wandb-project', type=str, default="minillama-fineweb-compact", help="WandB project name")
def train(num_layers, hidden_size, num_heads, max_seq_len, vocab_size,
          batch_size, learning_rate, epochs, fineweb_config, max_samples,
          save_steps, eval_steps, warmup_steps, gradient_accumulation_steps,
          max_grad_norm, wandb_project):
    
    # Initialize wandb
    wandb.init(project=wandb_project, config={
        "model": "miniLLAMA-2-compact",
        "dataset": f"FineWeb-{fineweb_config}",
        "max_samples": max_samples,
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "num_heads": num_heads,
        "max_seq_len": max_seq_len,
        "vocab_size": vocab_size,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epochs": epochs
    })
    
    # Load tokenizer
    tokenizer = SentencePieceProcessor(model_file=BPE_MODEL_PATH)
    logger.info(f"Loaded tokenizer with vocab size: {tokenizer.get_piece_size()}")
    
    # Create compact dataloader
    logger.info(f"Creating compact FineWeb dataloader with {max_samples} samples")
    dataloader = create_compact_dataloader(
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        config=fineweb_config,
        max_samples=max_samples
    )
    
    # Model configuration
    config = {
        "vocab_size": vocab_size,
        "n_head": num_heads,
        "hidden_size": hidden_size,
        "n_layer": num_layers,
        "n_embd": hidden_size,
        "n_local_heads": 23,
        "n_local_kv_heads": 12,
        "eps": 1e-6,
        "max_len": max_seq_len,
        "rope_theta": 1.0,
        "num_key_value_heads": 12,
        "attention_dropout": 0.25,
        "rms_norm_eps": 1.0,
        "weight_decay": 0.1,
        "block_size": max_seq_len
    }
    
    # Initialize model
    model = LLAMA(config)
    model._init_weights(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Model loaded on device: {device}")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate, 
        betas=(0.9, 0.999),
        weight_decay=config["weight_decay"]
    )
    
    # Learning rate scheduler with warmup
    def get_lr_scheduler(optimizer, num_training_steps):
        from torch.optim.lr_scheduler import LambdaLR
        
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - warmup_steps)))
        
        return LambdaLR(optimizer, lr_lambda)
    
    # Estimate total steps
    total_batches = max_samples // batch_size
    total_steps = total_batches * epochs
    scheduler = get_lr_scheduler(optimizer, total_steps)
    
    logger.info(f"Total batches per epoch: {total_batches}")
    logger.info(f"Total training steps: {total_steps}")
    
    # Training loop
    global_step = 0
    model.train()
    
    for epoch in range(epochs):
        logger.info(f"Starting epoch {epoch + 1}/{epochs}")
        
        for batch_idx, (batch_x, batch_y) in enumerate(dataloader):
            # Move to device
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            # Forward pass
            logits, loss = model(batch_x, batch_y)
            
            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                # Optimizer step
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                global_step += 1
                
                # Logging (reduced frequency to save disk space)
                if global_step % 20 == 0:
                    perplexity = exp(loss.item() * gradient_accumulation_steps)
                    current_lr = scheduler.get_last_lr()[0]
                    
                    wandb.log({
                        "train_loss": loss.item() * gradient_accumulation_steps,
                        "train_perplexity": perplexity,
                        "learning_rate": current_lr,
                        "epoch": epoch + 1,
                        "global_step": global_step
                    })
                    
                    logger.info(f"Step {global_step}: Loss={loss.item() * gradient_accumulation_steps:.4f}, "
                              f"Perplexity={perplexity:.2f}, LR={current_lr:.6f}")
                
                # Save checkpoint (reduced frequency)
                if global_step % save_steps == 0:
                    save_path = os.path.join(MODEL_DIR, f"llama_compact_step_{global_step}.bin")
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'config': config,
                        'global_step': global_step,
                        'epoch': epoch + 1
                    }, save_path)
                    
                    # Save config separately
                    config_path = os.path.join(MODEL_DIR, f"config_compact_step_{global_step}.json")
                    with open(config_path, 'w') as f:
                        json.dump(config, f, indent=2)
                    
                    logger.info(f"Saved checkpoint at step {global_step}")
                
                # Evaluation (reduced frequency)
                if global_step % eval_steps == 0:
                    model.eval()
                    eval_loss = 0.0
                    eval_steps_count = 0
                    
                    with torch.no_grad():
                        # Quick evaluation on a few batches
                        for eval_batch_idx, (eval_batch_x, eval_batch_y) in enumerate(dataloader):
                            if eval_batch_idx >= 5:  # Reduced evaluation batches
                                break
                            
                            eval_batch_x = eval_batch_x.to(device)
                            eval_batch_y = eval_batch_y.to(device)
                            
                            eval_logits, eval_batch_loss = model(eval_batch_x, eval_batch_y)
                            eval_loss += eval_batch_loss.item()
                            eval_steps_count += 1
                    
                    avg_eval_loss = eval_loss / eval_steps_count
                    eval_perplexity = exp(avg_eval_loss)
                    
                    wandb.log({
                        "eval_loss": avg_eval_loss,
                        "eval_perplexity": eval_perplexity,
                        "global_step": global_step
                    })
                    
                    logger.info(f"Evaluation at step {global_step}: "
                              f"Loss={avg_eval_loss:.4f}, Perplexity={eval_perplexity:.2f}")
                    
                    model.train()
            
            # Aggressive memory cleanup
            del logits, loss
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()
    
    # Save final model
    final_save_path = os.path.join(MODEL_DIR, "llama_compact_final.bin")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'config': config,
        'global_step': global_step,
        'epoch': epochs
    }, final_save_path)
    
    # Save final config
    final_config_path = os.path.join(MODEL_DIR, "config_compact_final.json")
    with open(final_config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Training completed! Final model saved to {final_save_path}")
    wandb.finish()

if __name__ == "__main__":
    train() 