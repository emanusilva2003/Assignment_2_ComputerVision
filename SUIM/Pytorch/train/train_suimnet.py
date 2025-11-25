"""
# Training pipeline for SUIM-Net in PyTorch
# Paper: https://arxiv.org/pdf/2004.01241.pdf  
"""
# Silenciar warnings do TensorFlow ANTES de qualquer import
# (TensorBoard importa TensorFlow automaticamente)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Silencia logs TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Desativa oneDNN

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from os.path import join, exists, dirname, abspath
from tqdm import tqdm

# Local imports
import sys
# Add parent directory to path to find models and utils
script_dir = dirname(abspath(__file__))
#print(f"Script directory: {script_dir}")
parent_dir = dirname(script_dir)  # Go up one level to SUIM/Pytorch/
#print(f"Parent directory: {parent_dir}")
sys.path.append(parent_dir)

from models.suim_net import SUIM_Net
from utils.data_utils import get_suim_dataloader

def train_suimnet(base='VGG', batch_size=8, num_epochs=50, learning_rate=1e-4, device='cuda'):
    """
    Train SUIM-Net model
    
    Args:
        base: 'RSB' or 'VGG'
        batch_size: Training batch size
        num_epochs: Number of training epochs
        learning_rate: Initial learning rate
        device: 'cuda' or 'cpu'
    """

    # Setup directories
    script_dir = dirname(abspath(__file__))
    train_dir = join(script_dir, "..","..", "train_val")
    ckpt_dir = join(script_dir, "..", "ckpt")
    log_dir = join(script_dir, "..", "logs")
    
    if not exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    if not exists(log_dir):
        os.makedirs(log_dir)
    
    # Model configuration
    if base == 'RSB':
        im_res = (320, 256)  # (width, height) #Rever height
        ckpt_name = "suimnet_rsb.pth"
    else:  # VGG
        im_res = (320, 256)
        ckpt_name = "suimnet_vgg.pth"
    
    model_ckpt_path = join(ckpt_dir, ckpt_name)
    
    # Initialize model
    print(f"Initializing SUIM-Net with {base} encoder...")
    model = SUIM_Net(base=base, n_classes=5, pretrained=True)
    model = model.to(device)
    
    total_params, trainable_params = model.count_parameters()
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Data augmentation parameters (matching Keras config)
    aug_params = {
        'rotation_range': 0.2,
        'width_shift_range': 0.05,
        'height_shift_range': 0.05,
        'shear_range': 0.05,
        'zoom_range': 0.05,
        'horizontal_flip': True
    }
    
    # Create data loader
    print(f"Loading training data from {train_dir}...")
    train_loader = get_suim_dataloader(
        train_dir=train_dir,
        batch_size=batch_size,
        image_folder="images",
        mask_folder="masks",
        target_size=im_res,
        augmentation=True,
        augmentation_params=aug_params,
        num_workers=4,
        shuffle=True
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Batches per epoch: {len(train_loader)}")
    
    # Loss function and optimizer
    criterion = nn.BCELoss()  # Binary cross-entropy for multi-label
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler (reduce on plateau)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # TensorBoard logger
    writer = SummaryWriter(log_dir=log_dir)
    
    # Training loop
    print(f"\nStarting training for {num_epochs} epochs...")
    best_loss = float('inf')
    global_step = 0
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, (images, masks) in enumerate(pbar):
            # Move to device
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            
            # Compute loss
            loss = criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            global_step += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
            
            # Log to TensorBoard
            if batch_idx % 10 == 0:
                writer.add_scalar('Loss/train_step', loss.item(), global_step)
        
        # Epoch statistics
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")
        
        # Log epoch metrics
        writer.add_scalar('Loss/train_epoch', avg_loss, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Update learning rate
        scheduler.step(avg_loss)
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            print(f"Saving best model (loss: {best_loss:.4f})...")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, model_ckpt_path)
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = join(ckpt_dir, f"{base.lower()}_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
    
    writer.close()
    print(f"\nTraining completed! Best loss: {best_loss:.4f}")
    print(f"Model saved to: {model_ckpt_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train SUIM-Net')
    parser.add_argument('--base', type=str, default='VGG', choices=['RSB', 'VGG'],
                        help='Encoder architecture: RSB or VGG (default: VGG)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size (default: 8)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs (default: 50)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device: cuda or cpu (default: cuda)')
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    print(f"Using device: {args.device}")
    if args.device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Train model
    train_suimnet(
        base="RSB",
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        device=args.device
    )
    