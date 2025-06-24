"""
Jupyter-Friendly NetEase Anti-Spoofing Trainer with ModelManager Integration
Simple interface for running NetEase two-stage training in Jupyter notebooks
"""

import os
import torch
import json
from datetime import datetime
from typing import Dict, Optional, Tuple, Any
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display, HTML, clear_output
import warnings
warnings.filterwarnings('ignore')

# Import the updated training modules with ModelManager
from model_manager import ModelManager


class JupyterNeteaseTrainer:
    """
    Jupyter-friendly interface for NetEase two-stage anti-spoofing training
    Now uses ModelManager for efficient model caching
    """
    
    def __init__(self, data_root: str, output_dir: str = "./netease_experiment", 
                 model_cache_dir: str = "./model"):
        """
        Initialize the trainer
        
        Args:
            data_root: Path to WFAS dataset
            output_dir: Directory for saving outputs
            model_cache_dir: Directory for model caching
        """
        self.data_root = data_root
        self.output_dir = output_dir
        self.model_cache_dir = model_cache_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(model_cache_dir, exist_ok=True)
        
        # Initialize model manager
        self.model_manager = ModelManager(cache_base_dir=model_cache_dir)
        
        # Training history
        self.stage1_history = []
        self.stage2_history = []
        self.stage1_trainer = None
        self.stage2_trainer = None
        
        print(f"🚀 NetEase Trainer initialized")
        print(f"📁 Data root: {data_root}")
        print(f"💾 Output dir: {output_dir}")
        print(f"🗄️  Model cache: {model_cache_dir}")
        print(f"🔧 Device: {self.device}")
        
        # Check dataset structure
        self._check_dataset()
        
        # Show cached models
        self._show_cached_models()
    
    def _check_dataset(self):
        """Quick dataset structure check"""
        if not os.path.exists(self.data_root):
            print(f"❌ ERROR: Dataset root not found: {self.data_root}")
            return
        
        splits_found = []
        for split in ['Train', 'train', 'Dev', 'dev', 'Test', 'test']:
            if os.path.exists(os.path.join(self.data_root, split)):
                splits_found.append(split)
        
        if splits_found:
            print(f"✅ Dataset splits found: {splits_found}")
        else:
            print(f"⚠️  No standard splits found. Available: {os.listdir(self.data_root)}")
    
    def _show_cached_models(self):
        """Show currently cached models"""
        cached_models = self.model_manager.list_cached_models()
        if cached_models:
            print(f"📦 Cached models:")
            for stage, models in cached_models.items():
                if models:
                    print(f"  {stage}: {', '.join(models)}")
        else:
            print(f"📦 No cached models (models will be downloaded on first use)")
    
    def download_models(self, stage1_model: str = 'convnext_base', 
                       stage2_model: str = 'maxvit_base_tf_224'):
        """
        Pre-download models for offline use
        
        Args:
            stage1_model: Stage 1 model name
            stage2_model: Stage 2 model name
        """
        print("📥 Pre-downloading models...")
        
        # Download Stage 1 model
        print(f"  Downloading {stage1_model} for Stage 1...")
        try:
            self.model_manager.load_or_download_model(
                stage='stage1',
                model_name=stage1_model,
                num_classes=2,
                device='cpu'
            )
            print(f"  ✅ {stage1_model} cached")
        except Exception as e:
            print(f"  ❌ Failed to cache {stage1_model}: {e}")
        
        # Download Stage 2 model
        print(f"  Downloading {stage2_model} for Stage 2...")
        try:
            self.model_manager.load_or_download_model(
                stage='stage2',
                model_name=stage2_model,
                num_classes=2,
                device='cpu'
            )
            print(f"  ✅ {stage2_model} cached")
        except Exception as e:
            print(f"  ❌ Failed to cache {stage2_model}: {e}")
        
        print("📦 Model pre-download completed!")
    
    def create_stage1_config(self, **kwargs) -> Dict:
        """
        Create Stage 1 configuration with sensible defaults
        
        Args:
            **kwargs: Override any default parameters
        """
        default_config = {
            # Data settings
            'data_root': self.data_root,
            'image_size': 224,
            'batch_size': 32,
            'num_workers': 4,
            
            # Model settings
            'model_name': 'convnext_base',
            'pretrained': True,
            'model_cache_dir': self.model_cache_dir,
            
            # Training settings
            'num_epochs': 50,
            'learning_rate': 2e-4,
            'weight_decay': 0.01,
            'optimizer': 'adamw',
            'scheduler': 'cosine',
            
            # Paths
            'checkpoint_dir': os.path.join(self.output_dir, 'checkpoints', 'stage1'),
            'log_dir': os.path.join(self.output_dir, 'logs', 'stage1'),
            'output_dir': self.output_dir,
            
            # Debug
            'debug': False  # Less verbose for notebooks
        }
        
        # Update with user parameters
        default_config.update(kwargs)
        
        # Create directories
        os.makedirs(default_config['checkpoint_dir'], exist_ok=True)
        os.makedirs(default_config['log_dir'], exist_ok=True)
        os.makedirs(default_config['output_dir'], exist_ok=True)
        
        return default_config
    
    def create_stage2_config(self, soft_labels_path: str, **kwargs) -> Dict:
        """
        Create Stage 2 configuration with sensible defaults
        
        Args:
            soft_labels_path: Path to soft labels from Stage 1
            **kwargs: Override any default parameters
        """
        default_config = {
            # Data settings
            'data_root': self.data_root,
            'soft_labels_path': soft_labels_path,
            'image_size': 224,
            'batch_size': 32,
            'num_workers': 4,
            
            # Model settings
            'model_name': 'maxvit_base_tf_224',
            'feature_dim': 512,
            'pretrained': True,
            'model_cache_dir': self.model_cache_dir,
            
            # Training settings
            'num_epochs': 30,
            'learning_rate': 1e-4,
            'weight_decay': 0.01,
            'optimizer': 'adamw',
            'scheduler': 'cosine',
            
            # Loss settings
            'focal_alpha': 1.0,
            'focal_gamma': 2.0,
            'triplet_margin': 0.3,
            'kd_temperature': 3.0,
            'kd_alpha': 0.7,
            'focal_weight': 1.0,
            'triplet_weight': 0.5,
            'kd_weight': 1.0,
            
            # Mixup/Cutmix settings
            'use_mixup_cutmix': True,
            'mixup_alpha': 0.2,
            'cutmix_alpha': 1.0,
            'mixup_cutmix_prob': 0.5,
            
            # Paths
            'checkpoint_dir': os.path.join(self.output_dir, 'checkpoints', 'stage2'),
            'log_dir': os.path.join(self.output_dir, 'logs', 'stage2'),
            'output_dir': self.output_dir,
            
            # Debug
            'debug': False
        }
        
        # Update with user parameters
        default_config.update(kwargs)
        
        # Create directories
        os.makedirs(default_config['checkpoint_dir'], exist_ok=True)
        os.makedirs(default_config['log_dir'], exist_ok=True)
        os.makedirs(default_config['output_dir'], exist_ok=True)
        
        return default_config
    
    def train_stage1(self, 
                     epochs: int = 50,
                     batch_size: int = 32,
                     learning_rate: float = 2e-4,
                     model_name: str = 'convnext_base',
                     image_size: int = 224,
                     **kwargs) -> str:
        """
        Train Stage 1 (ConvNext) model
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            model_name: ConvNext model variant
            image_size: Input image size
            **kwargs: Additional config parameters
        
        Returns:
            Path to generated soft labels
        """
        print("🔥 Starting Stage 1 Training (ConvNext)")
        print("=" * 50)
        
        # Import here to avoid issues if modules aren't available
        try:
            from stage1_training import Stage1Trainer
        except ImportError:
            print("❌ Error: stage1_training module not found!")
            print("Make sure stage1_training.py is in the same directory")
            return None
        
        # Create config
        config = self.create_stage1_config(
            num_epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            model_name=model_name,
            image_size=image_size,
            **kwargs
        )
        
        # Save config
        with open(os.path.join(self.output_dir, 'stage1_config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        # Initialize trainer
        self.stage1_trainer = Stage1Trainer(config)
        
        # Custom training loop with notebook-friendly output
        self.stage1_trainer.logger.info("Starting Stage 1 training...")
        
        for epoch in range(config['num_epochs']):
            self.stage1_trainer.current_epoch = epoch
            
            # Training
            train_loss, train_acc = self.stage1_trainer.train_epoch()
            
            # Validation
            val_loss, val_acc, val_acer = self.stage1_trainer.validate()
            
            # Update scheduler
            if self.stage1_trainer.scheduler:
                self.stage1_trainer.scheduler.step()
            
            # Store history
            epoch_data = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_acer': val_acer,
                'lr': self.stage1_trainer.optimizer.param_groups[0]['lr']
            }
            self.stage1_history.append(epoch_data)
            
            # Save best model
            is_best = val_acc > self.stage1_trainer.best_val_acc
            if is_best:
                self.stage1_trainer.best_val_acc = val_acc
                self.stage1_trainer.best_val_acer = val_acer
            
            self.stage1_trainer.save_checkpoint(is_best)
            
            # Notebook-friendly progress update
            if (epoch + 1) % 5 == 0 or epoch == 0:
                clear_output(wait=True)
                print(f"📊 Stage 1 Progress: {epoch+1}/{config['num_epochs']} epochs")
                print(f"🎯 Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | ACER: {val_acer:.4f}")
                print(f"🏆 Best Val Acc: {self.stage1_trainer.best_val_acc:.2f}%")
                
                # Plot progress
                if len(self.stage1_history) > 1:
                    self._plot_stage1_progress()
        
        # Generate soft labels
        print("\n🔮 Generating soft labels for Stage 2...")
        soft_labels_path = self.stage1_trainer.generate_soft_labels()
        
        print(f"✅ Stage 1 completed!")
        print(f"🏆 Best validation accuracy: {self.stage1_trainer.best_val_acc:.2f}%")
        print(f"📊 Best validation ACER: {self.stage1_trainer.best_val_acer:.4f}")
        print(f"💾 Soft labels saved to: {soft_labels_path}")
        
        return soft_labels_path
    
    def train_stage2(self,
                     soft_labels_path: str,
                     epochs: int = 30,
                     batch_size: int = 32,
                     learning_rate: float = 1e-4,
                     model_name: str = 'maxvit_base_tf_224',
                     image_size: int = 224,
                     **kwargs) -> Tuple[float, float]:
        """
        Train Stage 2 (MaxViT) model
        
        Args:
            soft_labels_path: Path to soft labels from Stage 1
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            model_name: MaxViT model variant
            image_size: Input image size
            **kwargs: Additional config parameters
        
        Returns:
            Tuple of (test_accuracy, test_acer)
        """
        print("🔥 Starting Stage 2 Training (MaxViT)")
        print("=" * 50)
        
        if not os.path.exists(soft_labels_path):
            print(f"❌ Error: Soft labels not found at {soft_labels_path}")
            print("Please run Stage 1 first!")
            return None, None
        
        # Import here to avoid issues if modules aren't available
        try:
            from stage2_training import Stage2Trainer
        except ImportError:
            print("❌ Error: stage2_training module not found!")
            print("Make sure stage2_training.py is in the same directory")
            return None, None
        
        # Create config
        config = self.create_stage2_config(
            soft_labels_path=soft_labels_path,
            num_epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            model_name=model_name,
            image_size=image_size,
            **kwargs
        )
        
        # Save config
        with open(os.path.join(self.output_dir, 'stage2_config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        # Initialize trainer
        self.stage2_trainer = Stage2Trainer(config)
        
        # Custom training loop with notebook-friendly output
        self.stage2_trainer.logger.info("Starting Stage 2 training...")
        
        for epoch in range(config['num_epochs']):
            self.stage2_trainer.current_epoch = epoch
            
            # Training
            train_loss, train_acc, focal_loss, triplet_loss, kd_loss = self.stage2_trainer.train_epoch()
            
            # Validation
            val_loss, val_acc, val_acer = self.stage2_trainer.validate()
            
            # Update scheduler
            if self.stage2_trainer.scheduler:
                self.stage2_trainer.scheduler.step()
            
            # Store history
            epoch_data = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_acer': val_acer,
                'focal_loss': focal_loss,
                'triplet_loss': triplet_loss,
                'kd_loss': kd_loss,
                'lr': self.stage2_trainer.optimizer.param_groups[0]['lr']
            }
            self.stage2_history.append(epoch_data)
            
            # Save best model
            is_best = val_acc > self.stage2_trainer.best_val_acc
            if is_best:
                self.stage2_trainer.best_val_acc = val_acc
                self.stage2_trainer.best_val_acer = val_acer
            
            self.stage2_trainer.save_checkpoint(is_best)
            
            # Notebook-friendly progress update
            if (epoch + 1) % 3 == 0 or epoch == 0:
                clear_output(wait=True)
                print(f"📊 Stage 2 Progress: {epoch+1}/{config['num_epochs']} epochs")
                print(f"🎯 Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | ACER: {val_acer:.4f}")
                print(f"💥 Focal: {focal_loss:.4f} | Triplet: {triplet_loss:.4f} | KD: {kd_loss:.4f}")
                print(f"🏆 Best Val Acc: {self.stage2_trainer.best_val_acc:.2f}%")
                
                # Plot progress
                if len(self.stage2_history) > 1:
                    self._plot_stage2_progress()
        
        # Test on test set
        print("\n🧪 Testing on test set...")
        test_acc, test_acer, attack_results = self.stage2_trainer.test_model()
        
        print(f"✅ Stage 2 completed!")
        print(f"🏆 Best validation accuracy: {self.stage2_trainer.best_val_acc:.2f}%")
        print(f"📊 Best validation ACER: {self.stage2_trainer.best_val_acer:.4f}")
        print(f"🎯 Final test accuracy: {test_acc:.2f}%")
        print(f"📈 Final test ACER: {test_acer:.4f}")
        
        # Show attack-type specific results
        print(f"\n📋 Results by Attack Type:")
        for attack_type, results in attack_results.items():
            acc = 100. * results['correct'] / results['total']
            print(f"  {attack_type}: {acc:.2f}% ({results['correct']}/{results['total']})")
        
        return test_acc, test_acer
    
    def train_complete_pipeline(self,
                               stage1_epochs: int = 50,
                               stage2_epochs: int = 30,
                               batch_size: int = 32,
                               stage1_lr: float = 2e-4,
                               stage2_lr: float = 1e-4,
                               **kwargs) -> Dict[str, Any]:
        """
        Train complete two-stage pipeline
        
        Args:
            stage1_epochs: Stage 1 epochs
            stage2_epochs: Stage 2 epochs
            batch_size: Batch size for both stages
            stage1_lr: Stage 1 learning rate
            stage2_lr: Stage 2 learning rate
            **kwargs: Additional parameters
        
        Returns:
            Dictionary with all results
        """
        print("🚀 Starting Complete NetEase Two-Stage Pipeline")
        print("=" * 60)
        
        # Stage 1
        soft_labels_path = self.train_stage1(
            epochs=stage1_epochs,
            batch_size=batch_size,
            learning_rate=stage1_lr,
            **kwargs
        )
        
        if soft_labels_path is None:
            print("❌ Stage 1 failed, aborting pipeline")
            return None
        
        print("\n" + "="*60)
        
        # Stage 2
        test_acc, test_acer = self.train_stage2(
            soft_labels_path=soft_labels_path,
            epochs=stage2_epochs,
            batch_size=batch_size,
            learning_rate=stage2_lr,
            **kwargs
        )
        
        # Compile results
        results = {
            'stage1': {
                'best_val_acc': self.stage1_trainer.best_val_acc,
                'best_val_acer': self.stage1_trainer.best_val_acer,
                'history': self.stage1_history
            },
            'stage2': {
                'best_val_acc': self.stage2_trainer.best_val_acc,
                'best_val_acer': self.stage2_trainer.best_val_acer,
                'test_acc': test_acc,
                'test_acer': test_acer,
                'history': self.stage2_history
            },
            'soft_labels_path': soft_labels_path
        }
        
        # Save results
        with open(os.path.join(self.output_dir, 'final_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n🎉 Complete pipeline finished!")
        print(f"📊 Final test accuracy: {test_acc:.2f}%")
        print(f"📈 Final test ACER: {test_acer:.4f}")
        
        return results
    
    def _plot_stage1_progress(self):
        """Plot Stage 1 training progress"""
        if not self.stage1_history:
            return
        
        df = pd.DataFrame(self.stage1_history)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Loss
        ax1.plot(df['epoch'], df['train_loss'], label='Train Loss', color='blue')
        ax1.plot(df['epoch'], df['val_loss'], label='Val Loss', color='red')
        ax1.set_title('Stage 1: Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy
        ax2.plot(df['epoch'], df['train_acc'], label='Train Acc', color='blue')
        ax2.plot(df['epoch'], df['val_acc'], label='Val Acc', color='red')
        ax2.set_title('Stage 1: Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        # ACER
        ax3.plot(df['epoch'], df['val_acer'], label='Val ACER', color='green')
        ax3.set_title('Stage 1: ACER')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('ACER')
        ax3.legend()
        ax3.grid(True)
        
        # Learning Rate
        ax4.plot(df['epoch'], df['lr'], label='Learning Rate', color='orange')
        ax4.set_title('Stage 1: Learning Rate')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('LR')
        ax4.legend()
        ax4.grid(True)
        ax4.set_yscale('log')
        
        plt.tight_layout()
        plt.show()
    
    def _plot_stage2_progress(self):
        """Plot Stage 2 training progress"""
        if not self.stage2_history:
            return
        
        df = pd.DataFrame(self.stage2_history)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Loss components
        ax1.plot(df['epoch'], df['train_loss'], label='Total Loss', color='black', linewidth=2)
        ax1.plot(df['epoch'], df['focal_loss'], label='Focal Loss', color='red')
        ax1.plot(df['epoch'], df['triplet_loss'], label='Triplet Loss', color='blue')
        ax1.plot(df['epoch'], df['kd_loss'], label='KD Loss', color='green')
        ax1.set_title('Stage 2: Loss Components')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy
        ax2.plot(df['epoch'], df['train_acc'], label='Train Acc', color='blue')
        ax2.plot(df['epoch'], df['val_acc'], label='Val Acc', color='red')
        ax2.set_title('Stage 2: Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        # ACER
        ax3.plot(df['epoch'], df['val_acer'], label='Val ACER', color='green')
        ax3.set_title('Stage 2: ACER')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('ACER')
        ax3.legend()
        ax3.grid(True)
        
        # Learning Rate
        ax4.plot(df['epoch'], df['lr'], label='Learning Rate', color='orange')
        ax4.set_title('Stage 2: Learning Rate')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('LR')
        ax4.legend()
        ax4.grid(True)
        ax4.set_yscale('log')
        
        plt.tight_layout()
        plt.show()
    
    def load_checkpoint(self, stage: int, checkpoint_type: str = 'best'):
        """
        Load a saved checkpoint
        
        Args:
            stage: 1 or 2
            checkpoint_type: 'best' or 'latest'
        """
        checkpoint_path = os.path.join(
            self.output_dir, 'checkpoints', f'stage{stage}', 
            f'stage{stage}_{checkpoint_type}.pth'
        )
        
        if not os.path.exists(checkpoint_path):
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            return None
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        print(f"✅ Loaded Stage {stage} {checkpoint_type} checkpoint")
        print(f"   Epoch: {checkpoint['epoch']}")
        print(f"   Best Val Acc: {checkpoint['best_val_acc']:.2f}%")
        print(f"   Best Val ACER: {checkpoint['best_val_acer']:.4f}")
        
        return checkpoint
    
    def plot_complete_history(self):
        """Plot training history for both stages"""
        if self.stage1_history and self.stage2_history:
            print("📊 Complete Training History")
            print("=" * 40)
            self._plot_stage1_progress()
            self._plot_stage2_progress()
        elif self.stage1_history:
            print("📊 Stage 1 Training History")
            self._plot_stage1_progress()
        elif self.stage2_history:
            print("📊 Stage 2 Training History")
            self._plot_stage2_progress()
        else:
            print("❌ No training history available")
    
    def clear_model_cache(self, stage: str = None):
        """Clear model cache"""
        try:
            self.model_manager.clear_cache(stage=stage)
            print(f"✅ Model cache cleared for stage: {stage if stage else 'all'}")
            self._show_cached_models()
        except Exception as e:
            print(f"❌ Failed to clear cache: {e}")
    
    def show_model_info(self, stage: str, model_name: str):
        """Show information about a cached model"""
        model_dir = self.model_manager.get_model_cache_dir(stage, model_name)
        
        if self.model_manager.check_local_model_exists(model_dir):
            try:
                config = self.model_manager.load_model_config(model_dir)
                print(f"📋 Model Info: {stage}/{model_name}")
                print("-" * 40)
                for key, value in config.items():
                    print(f"  {key}: {value}")
            except Exception as e:
                print(f"❌ Could not load model config: {e}")
        else:
            print(f"❌ Model {stage}/{model_name} not found in cache")


# Convenience functions for quick training
def quick_train_stage1(data_root: str, 
                      output_dir: str = "./netease_experiment",
                      model_cache_dir: str = "./model",
                      epochs: int = 20,
                      batch_size: int = 16) -> str:
    """Quick Stage 1 training with minimal parameters"""
    trainer = JupyterNeteaseTrainer(data_root, output_dir, model_cache_dir)
    return trainer.train_stage1(epochs=epochs, batch_size=batch_size)


def quick_train_stage2(data_root: str,
                      soft_labels_path: str,
                      output_dir: str = "./netease_experiment",
                      model_cache_dir: str = "./model", 
                      epochs: int = 15,
                      batch_size: int = 16) -> Tuple[float, float]:
    """Quick Stage 2 training with minimal parameters"""
    trainer = JupyterNeteaseTrainer(data_root, output_dir, model_cache_dir)
    return trainer.train_stage2(soft_labels_path, epochs=epochs, batch_size=batch_size)


def quick_train_complete(data_root: str,
                        output_dir: str = "./netease_experiment",
                        model_cache_dir: str = "./model",
                        stage1_epochs: int = 20,
                        stage2_epochs: int = 15,
                        batch_size: int = 16) -> Dict[str, Any]:
    """Quick complete pipeline training"""
    trainer = JupyterNeteaseTrainer(data_root, output_dir, model_cache_dir)
    return trainer.train_complete_pipeline(
        stage1_epochs=stage1_epochs,
        stage2_epochs=stage2_epochs,
        batch_size=batch_size
    )


def download_models_for_offline_use(model_cache_dir: str = "./model"):
    """Download and cache all models for offline use"""
    manager = ModelManager(cache_base_dir=model_cache_dir)
    
    models_to_download = [
        ('stage1', 'convnext_base'),
        ('stage1', 'convnext_small'),
        ('stage1', 'convnext_large'),
        ('stage2', 'maxvit_base_tf_224'),
        ('stage2', 'maxvit_small_tf_224'),
        ('stage2', 'maxvit_large_tf_224'),
    ]
    
    print("📥 Downloading models for offline use...")
    for stage, model_name in models_to_download:
        print(f"  Downloading {model_name} for {stage}...")
        try:
            manager.load_or_download_model(
                stage=stage,
                model_name=model_name,
                num_classes=2,
                device='cpu'
            )
            print(f"  ✅ {model_name} cached")
        except Exception as e:
            print(f"  ❌ Failed: {e}")
    
    print("📦 Offline model download completed!")