"""
VLM-AD Training Script with AUX Head and Distillation Loss.

Pipeline:
1. model(images) -> (output, f_ego, f_ego_proj)
2. loss = mse_loss(output, targets) + aux_loss(f_ego_proj, clip_embeddings)
3. where aux_loss = alignment_loss + 0.1 * action_loss
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader

# Add project and AUX_Head paths
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
aux_root = project_root / "AUX_Head"
sys.path.insert(0, str(aux_root))
sys.path.insert(0, str(aux_root / "AUX_head"))
sys.path.insert(0, str(aux_root / "AUX_loss"))

from AutoDriveModels.Dummy.model import NetworkNvidiaParallel_VLM_AD
from dataset_vlm_ad import VLMADDataset, collate_fn_vlm_ad

# Import AUX components
from AUX_head.aux_head import AUXHead
from AUX_loss.AUX_loss import AUX_loss


class VLMADTrainer:
    """Trainer for VLM-AD with AUX loss."""
    
    def __init__(
        self,
        model,
        aux_head,
        device,
        optimizer,
        scheduler,
        mse_criterion,
        epochs,
        train_loader,
        val_loader,
        output_dir,
        model_name="vlm_ad",
    ):
        self.model = model
        self.aux_head = aux_head
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.mse_criterion = mse_criterion
        self.epochs = epochs
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = output_dir
        self.model_name = model_name
        self.best_val_loss = float("inf")
        self.start_epoch = 0
        
        os.makedirs(output_dir, exist_ok=True)
    
    def _compute_loss(self, outputs, targets, f_ego_proj, clip_embeddings, action_labels):
        """
        Compute combined loss:
        total_loss = mse_loss(regression) + aux_loss(alignment + action)
        
        Args:
            outputs: [B, 3] regression predictions (steering, throttle, brake)
            targets: [B, 3] regression targets
            f_ego_proj: [B, 512] projected ego features
            clip_embeddings: dict with 'current', 'next', 'reasoning' [B, 512]
            action_labels: dict with 'control', 'turn', 'lane' labels
        """
        # MSE loss for regression
        mse_loss = self.mse_criterion(outputs, targets)
        
        # AUX loss from TextHead and ActionHead
        aux_head_outputs = self.aux_head(f_ego_proj)
        
        # Unpack aux head outputs
        fc_hat = aux_head_outputs["fc_hat"]  # [B, 512]
        ff_hat = aux_head_outputs["ff_hat"]  # [B, 512]
        fr_hat = aux_head_outputs["fr_hat"]  # [B, 512]
        
        control_hat = aux_head_outputs["control_hat"]  # [B, 4]
        turn_hat = aux_head_outputs["turn_hat"]  # [B, 4]
        lane_hat = aux_head_outputs["lane_hat"]  # [B, 5]
        
        # Compute AUX loss (alignment + action)
        aux_loss_val = AUX_loss(
            c_hat=fc_hat, yc=clip_embeddings["current"],
            f_hat=ff_hat, yf=clip_embeddings["next"],
            r_hat=fr_hat, yr=clip_embeddings["reasoning"],
            control_hat=control_hat, ycontrol=action_labels["control"],
            turn_hat=turn_hat, yturn=action_labels["turn"],
            lane_hat=lane_hat, ylane=action_labels["lane"],
        )
        
        # Combined loss
        total_loss = mse_loss + aux_loss_val
        
        return total_loss, mse_loss, aux_loss_val
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        self.aux_head.train()
        
        total_loss = 0.0
        total_mse = 0.0
        total_aux = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            images = batch["images"].to(self.device)
            regression_targets = batch["regression_targets"].to(self.device)
            
            clip_embeddings = {
                "current": batch["clip_current"].to(self.device),
                "next": batch["clip_next"].to(self.device),
                "reasoning": batch["clip_reasoning"].to(self.device),
            }
            
            action_labels = {
                "control": batch["control_labels"].to(self.device),
                "turn": batch["turn_labels"].to(self.device),
                "lane": batch["lane_labels"].to(self.device),
            }
            
            # Forward pass
            self.optimizer.zero_grad()
            
            outputs, f_ego, f_ego_proj = self.model(images, return_features=True)
            
            # Compute loss
            loss, mse_loss, aux_loss = self._compute_loss(
                outputs, regression_targets, f_ego_proj, clip_embeddings, action_labels
            )
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            total_mse += mse_loss.item()
            total_aux += aux_loss.item()
            num_batches += 1
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx + 1}/{len(self.train_loader)}: "
                      f"loss={loss.item():.6f}, mse={mse_loss.item():.6f}, aux={aux_loss.item():.6f}")
        
        avg_loss = total_loss / num_batches
        avg_mse = total_mse / num_batches
        avg_aux = total_aux / num_batches
        
        return avg_loss, avg_mse, avg_aux
    
    def val_epoch(self):
        """Validate for one epoch."""
        self.model.eval()
        self.aux_head.eval()
        
        total_loss = 0.0
        total_mse = 0.0
        total_aux = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch["images"].to(self.device)
                regression_targets = batch["regression_targets"].to(self.device)
                
                clip_embeddings = {
                    "current": batch["clip_current"].to(self.device),
                    "next": batch["clip_next"].to(self.device),
                    "reasoning": batch["clip_reasoning"].to(self.device),
                }
                
                action_labels = {
                    "control": batch["control_labels"].to(self.device),
                    "turn": batch["turn_labels"].to(self.device),
                    "lane": batch["lane_labels"].to(self.device),
                }
                
                outputs, f_ego, f_ego_proj = self.model(images, return_features=True)
                
                loss, mse_loss, aux_loss = self._compute_loss(
                    outputs, regression_targets, f_ego_proj, clip_embeddings, action_labels
                )
                
                total_loss += loss.item()
                total_mse += mse_loss.item()
                total_aux += aux_loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_mse = total_mse / num_batches
        avg_aux = total_aux / num_batches
        
        return avg_loss, avg_mse, avg_aux
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "aux_head_state_dict": self.aux_head.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
        }
        
        if is_best:
            path = os.path.join(self.output_dir, f"{self.model_name}_best.pth")
        else:
            path = os.path.join(self.output_dir, f"{self.model_name}_epoch{epoch:03d}.pth")
        
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")
    
    def train(self):
        """Full training loop."""
        print("=" * 80)
        print(f"Starting training for {self.epochs} epochs")
        print(f"Output directory: {self.output_dir}")
        print("=" * 80)
        
        for epoch in range(self.start_epoch, self.epochs):
            # Train
            train_loss, train_mse, train_aux = self.train_epoch()
            
            # Validate
            val_loss, val_mse, val_aux = self.val_epoch()
            
            # Scheduler step
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            lr = self.optimizer.param_groups[0]["lr"]
            print(f"Epoch [{epoch + 1}/{self.epochs}] "
                  f"train_loss={train_loss:.6f} (mse={train_mse:.6f}, aux={train_aux:.6f}) "
                  f"val_loss={val_loss:.6f} (mse={val_mse:.6f}, aux={val_aux:.6f}) "
                  f"lr={lr:.6e}")
            
            # Save best checkpoint
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch + 1, is_best=True)
            
            # Save periodic checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch + 1, is_best=False)
        
        print("Training finished!")


def main():
    parser = argparse.ArgumentParser(description="VLM-AD Training with AUX Loss")
    
    # Dataset
    parser.add_argument("--data-dir", type=str, 
                        default="d:\\NYU_Files\\2026_Spring\\summer_research\\VLM\\WEEK01_BASICSETUP\\AUX_Head\\MERGED_RESULT\\output_dataset\\neo_hf_dataset",
                        help="Path to HF dataset root")
    parser.add_argument("--max-train-samples", type=int, default=0, help="Max train samples (0=all)")
    parser.add_argument("--max-val-samples", type=int, default=0, help="Max val samples (0=all)")
    
    # Model
    parser.add_argument("--output-dim", type=int, default=3, help="Regression output dimension")
    parser.add_argument("--proj-dim", type=int, default=512, help="Projection dimension (CLIP embedding)")
    parser.add_argument("--aux-head-dim", type=int, default=512, help="AUX head input dimension")
    
    # Training
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw", "sgd"])
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["none", "step", "cosine", "plateau"])
    
    # Device
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    # Output
    parser.add_argument("--output-dir", type=str, default="./out", help="Output directory for checkpoints")
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = VLMADDataset(args.data_dir, split="train", max_samples=args.max_train_samples)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn_vlm_ad,
        num_workers=0,
    )
    print(f"Train dataset: {len(train_dataset)} samples")
    
    # Create model
    print("Creating model...")
    model = NetworkNvidiaParallel_VLM_AD(output_dim=args.output_dim).to(device)
    
    # Create AUX head
    print("Creating AUX head...")
    aux_head = AUXHead(args.aux_head_dim).to(device)
    
    # Create optimizer
    all_params = list(model.parameters()) + list(aux_head.parameters())
    if args.optimizer.lower() == "adam":
        optimizer = Adam(all_params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == "adamw":
        optimizer = AdamW(all_params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == "sgd":
        optimizer = SGD(all_params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    # Create scheduler
    if args.scheduler.lower() == "none":
        scheduler = None
    elif args.scheduler.lower() == "step":
        scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    elif args.scheduler.lower() == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    elif args.scheduler.lower() == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    
    # MSE criterion for regression
    mse_criterion = nn.MSELoss()
    
    # Create trainer
    trainer = VLMADTrainer(
        model=model,
        aux_head=aux_head,
        device=device,
        optimizer=optimizer,
        scheduler=scheduler,
        mse_criterion=mse_criterion,
        epochs=args.epochs,
        train_loader=train_loader,
        val_loader=train_loader,  # Use train as val for now (split in real training)
        output_dir=args.output_dir,
    )
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()
