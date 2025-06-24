"""
Stage 1 Training Script for NetEase Anti-Spoofing Approach
ConvNext model training with binary cross-entropy loss
Generates soft labels for Stage 2
Uses ModelManager for model loading and caching
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import argparse
import logging
from datetime import datetime
import json

# Import your dataset and model
from wfas_dataset import create_dataloaders_with_debug
from model_manager import ModelManager


class Stage1Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup logging
        self.setup_logging()
        
        # Initialize model manager
        self.model_manager = ModelManager(
            cache_base_dir=self.config['model_cache_dir'],
            logger=self.logger
        )
        
        # Create model
        self.model = self.create_model()
        
        # Create dataloaders
        self.train_loader, self.val_loader, self.test_loader = self.create_dataloaders()
        
        # Setup optimizer and scheduler
        self.optimizer = self.create_optimizer()
        self.scheduler = self.create_scheduler()
        self.criterion = nn.CrossEntropyLoss()
        
        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.best_val_acer = float('inf')
        
        self.logger.info(f"Stage 1 Trainer initialized")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model: {self.config['model_name']}")
        self.logger.info(f"Train samples: {len(self.train_loader.dataset)}")
        self.logger.info(f"Val samples: {len(self.val_loader.dataset)}")
    
    def setup_logging(self):
        """Setup logging"""
        os.makedirs(self.config['log_dir'], exist_ok=True)
        log_file = os.path.join(self.config['log_dir'], f"stage1_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def create_model(self):
        """Create ConvNext model using ModelManager"""
        model = self.model_manager.load_or_download_model(
            stage='stage1',
            model_name=self.config['model_name'],
            num_classes=2,  # Binary classification: spoof vs live
            pretrained=self.config['pretrained'],
            device=self.device
        )
        
        return model
    
    def create_dataloaders(self):
        """Create dataloaders for Stage 1"""
        train_loader, val_loader, test_loader = create_dataloaders_with_debug(
            data_root=self.config['data_root'],
            stage='stage1',
            batch_size=self.config['batch_size'],
            num_workers=self.config['num_workers'],
            image_size=self.config['image_size'],
            debug=self.config['debug']
        )
        return train_loader, val_loader, test_loader
    
    def create_optimizer(self):
        """Create optimizer"""
        if self.config['optimizer'] == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            )
        elif self.config['optimizer'] == 'sgd':
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                momentum=0.9,
                weight_decay=self.config['weight_decay']
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config['optimizer']}")
        
        return optimizer
    
    def create_scheduler(self):
        """Create learning rate scheduler"""
        if self.config['scheduler'] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=self.config['num_epochs']
            )
        elif self.config['scheduler'] == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config['num_epochs'] // 3,
                gamma=0.1
            )
        else:
            scheduler = None
        
        return scheduler
    
    def train_epoch(self):
        """Train one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}/{self.config['num_epochs']}")
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = total_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # For ACER calculation
        all_predictions = []
        all_labels = []
        all_scores = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Store for ACER calculation
                probs = F.softmax(outputs, dim=1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_scores.extend(probs[:, 1].cpu().numpy())  # Live probability
        
        val_loss = total_loss / (len(self.val_loader) + 0.000001)
        val_acc = 100. * correct / (total + 0.0000001)
        
        # Calculate ACER
        acer = self.calculate_acer(all_labels, all_predictions, all_scores)
        
        return val_loss, val_acc, acer
    
    def calculate_acer(self, labels, predictions, scores):
        """Calculate ACER (Average Classification Error Rate)"""
        labels = np.array(labels)
        predictions = np.array(predictions)
        scores = np.array(scores)
        
        # Find optimal threshold using EER
        thresholds = np.linspace(0, 1, 1000)
        best_threshold = 0.5
        best_eer = 1.0
        
        for threshold in thresholds:
            pred_binary = (scores >= threshold).astype(int)
            
            # Calculate APCER and BPCER
            live_mask = (labels == 1)
            spoof_mask = (labels == 0)
            
            if live_mask.sum() > 0 and spoof_mask.sum() > 0:
                # APCER: Attack Presentation Classification Error Rate
                apcer = ((pred_binary[spoof_mask] == 1).sum()) / spoof_mask.sum()
                # BPCER: Bonafide Presentation Classification Error Rate  
                bpcer = ((pred_binary[live_mask] == 0).sum()) / live_mask.sum()
                
                eer = abs(apcer - bpcer)
                if eer < best_eer:
                    best_eer = eer
                    best_threshold = threshold
        
        # Calculate final ACER with best threshold
        pred_binary = (scores >= best_threshold).astype(int)
        live_mask = (labels == 1)
        spoof_mask = (labels == 0)
        
        if live_mask.sum() > 0 and spoof_mask.sum() > 0:
            apcer = ((pred_binary[spoof_mask] == 1).sum()) / spoof_mask.sum()
            bpcer = ((pred_binary[live_mask] == 0).sum()) / live_mask.sum()
            acer = (apcer + bpcer) / 2
        else:
            acer = 1.0
        
        return acer
    
    def save_checkpoint(self, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_acc': self.best_val_acc,
            'best_val_acer': self.best_val_acer,
            'config': self.config
        }
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(self.config['checkpoint_dir'], 'stage1_latest.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.config['checkpoint_dir'], 'stage1_best.pth')
            torch.save(checkpoint, best_path)
            self.logger.info(f"Best model saved with Val Acc: {self.best_val_acc:.2f}%, ACER: {self.best_val_acer:.4f}")
            
            # Also save using ModelManager for easy reuse
            metrics = {
                'val_acc': self.best_val_acc,
                'val_acer': self.best_val_acer,
                'num_classes': 2
            }
            self.model_manager.save_trained_model(
                model=self.model,
                stage='stage1',
                model_name=self.config['model_name'],
                epoch=self.current_epoch,
                metrics=metrics
            )
    
    def generate_soft_labels(self):
        """Generate soft labels for Stage 2"""
        self.logger.info("Generating soft labels for Stage 2...")
        
        # Load best model
        best_model_path = os.path.join(self.config['checkpoint_dir'], 'stage1_best.pth')
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info("Loaded best model for soft label generation")
        
        self.model.eval()
        soft_labels = {}
        
        with torch.no_grad():
            for batch in tqdm(self.train_loader, desc="Generating soft labels"):
                images = batch['image'].to(self.device)
                image_paths = batch['image_path']
                
                # Get model predictions
                outputs = self.model(images)
                probs = F.softmax(outputs, dim=1)
                
                # Store soft labels
                for i, img_path in enumerate(image_paths):
                    soft_labels[img_path] = probs[i].cpu().numpy()
        
        # Save soft labels with compatibility settings
        soft_labels_path = os.path.join(self.config['output_dir'], 'stage1_soft_labels.pth')
        os.makedirs(os.path.dirname(soft_labels_path), exist_ok=True)
        torch.save(soft_labels, soft_labels_path, _use_new_zipfile_serialization=False)
        
        self.logger.info(f"Soft labels saved to: {soft_labels_path}")
        self.logger.info(f"Generated soft labels for {len(soft_labels)} samples")
        
        return soft_labels_path
    
    def train(self):
        """Main training loop"""
        self.logger.info("Starting Stage 1 training...")
        
        for epoch in range(self.config['num_epochs']):
            self.current_epoch = epoch
            
            # Training
            train_loss, train_acc = self.train_epoch()
            
            # Validation
            val_loss, val_acc, val_acer = self.validate()
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Logging
            self.logger.info(f"Epoch {epoch+1}/{self.config['num_epochs']}:")
            self.logger.info(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            self.logger.info(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val ACER: {val_acer:.4f}")
            self.logger.info(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_val_acer = val_acer
            
            self.save_checkpoint(is_best)
        
        # Generate soft labels after training
        soft_labels_path = self.generate_soft_labels()
        
        self.logger.info("Stage 1 training completed!")
        self.logger.info(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        self.logger.info(f"Best validation ACER: {self.best_val_acer:.4f}")
        self.logger.info(f"Soft labels saved to: {soft_labels_path}")
        
        return soft_labels_path


def get_default_config():
    """Get default configuration for Stage 1"""
    return {
        # Data settings
        'data_root': '/path/to/WFAS/dataset',
        'image_size': 224,
        'batch_size': 32,
        'num_workers': 4,
        
        # Model settings
        'model_name': 'convnextv2_base',  # or convnext_small, convnext_large
        'pretrained': True,
        'model_cache_dir': './model',  # Base model cache directory
        
        # Training settings
        'num_epochs': 50,
        'learning_rate': 2e-4,
        'weight_decay': 0.01,
        'optimizer': 'adamw',  # adamw or sgd
        'scheduler': 'cosine',  # cosine, step, or none
        
        # Paths
        'checkpoint_dir': './checkpoints/stage1',
        'log_dir': './logs/stage1',
        'output_dir': './outputs',
        
        # Debug
        'debug': True
    }


def main():
    parser = argparse.ArgumentParser(description='Stage 1 Training - ConvNext')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--data_root', type=str, help='Path to dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--model', type=str, default='convnext_base', help='Model name')
    parser.add_argument('--model_cache_dir', type=str, default='./model', help='Model cache directory')
    
    args = parser.parse_args()
    
    # Get configuration
    config = get_default_config()
    
    # Override with command line arguments
    if args.data_root:
        config['data_root'] = args.data_root
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.epochs:
        config['num_epochs'] = args.epochs
    if args.lr:
        config['learning_rate'] = args.lr
    if args.model:
        config['model_name'] = args.model
    if args.model_cache_dir:
        config['model_cache_dir'] = args.model_cache_dir
    
    # Load config file if provided
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            file_config = json.load(f)
        config.update(file_config)
    
    # Create directories
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)
    os.makedirs(config['output_dir'], exist_ok=True)
    os.makedirs(config['model_cache_dir'], exist_ok=True)
    
    # Save config
    config_path = os.path.join(config['output_dir'], 'stage1_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Start training
    trainer = Stage1Trainer(config)
    soft_labels_path = trainer.train()
    
    print(f"\n=== Stage 1 Training Completed ===")
    print(f"Best validation accuracy: {trainer.best_val_acc:.2f}%")
    print(f"Best validation ACER: {trainer.best_val_acer:.4f}")
    print(f"Soft labels saved to: {soft_labels_path}")
    print(f"Now you can use this for Stage 2 training!")


if __name__ == "__main__":
    main()