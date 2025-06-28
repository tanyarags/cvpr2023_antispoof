"""
Stage 2 Training Script for NetEase Anti-Spoofing Approach
MaxViT model training with soft labels from Stage 1
Uses compound loss (focal + triplet) and knowledge distillation
Uses ModelManager for model loading and caching
FIXED: Device mismatch error
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
from wfas_dataset import create_dataloaders_with_debug, MixupCutmixCollator
from model_manager import ModelManager


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""

    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Handle both hard labels (Long) and soft labels (Float) from mixup/cutmix
        if targets.dtype == torch.float32:
            # Check if targets are 1D mixed values or 2D one-hot
            if targets.dim() == 1:
                # 1D mixed values from mixup - convert to one-hot-like format
                # Create one-hot encoding based on continuous values
                batch_size = targets.size(0)
                num_classes = inputs.size(1)

                # Create soft targets where the mixed value represents the weight for class 1
                soft_targets = torch.zeros(batch_size, num_classes, device=inputs.device)
                soft_targets[:, 0] = 1.0 - targets  # Weight for class 0 (spoof)
                soft_targets[:, 1] = targets        # Weight for class 1 (live)

                # Use KL divergence
                log_probs = F.log_softmax(inputs, dim=1)
                loss = -torch.sum(soft_targets * log_probs, dim=1)
            else:
                # 2D one-hot labels from cutmix
                log_probs = F.log_softmax(inputs, dim=1)
                loss = -torch.sum(targets * log_probs, dim=1)

            if self.reduction == 'mean':
                return loss.mean()
            elif self.reduction == 'sum':
                return loss.sum()
            else:
                return loss
        else:
            # Hard labels - use standard focal loss
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
            pt = torch.exp(-ce_loss)
            focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss

            if self.reduction == 'mean':
                return focal_loss.mean()
            elif self.reduction == 'sum':
                return focal_loss.sum()
            else:
                return focal_loss


class TripletLoss(nn.Module):
    """Triplet Loss for learning discriminative features"""

    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, features, labels):
        """
        Args:
            features: Feature embeddings [batch_size, feature_dim]
            labels: Class labels [batch_size]
        """
        # Handle mixed labels from mixup/cutmix
        if labels.dtype == torch.float32:
            # For mixed labels (1D continuous values), use the dominant label
            labels = (labels > 0.5).long()

        batch_size = features.size(0)

        # Skip triplet loss if batch is too small or all same class
        if batch_size < 2 or len(torch.unique(labels)) < 2:
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        # Compute pairwise distances
        distances = torch.cdist(features, features, p=2)

        # Create masks for positive and negative pairs
        labels_expand = labels.view(-1, 1)
        mask_positive = (labels_expand == labels_expand.t()).float()
        mask_negative = (labels_expand != labels_expand.t()).float()

        # Remove diagonal (self-similarity)
        mask_positive.fill_diagonal_(0)

        # Find hardest positive and negative for each anchor
        losses = []
        for i in range(batch_size):
            # Hardest positive (farthest positive sample)
            pos_distances = distances[i] * mask_positive[i]
            if pos_distances.sum() > 0:
                hardest_positive_dist = pos_distances.max()

                # Hardest negative (closest negative sample)
                neg_distances = distances[i] * mask_negative[i]
                neg_distances = neg_distances + (1 - mask_negative[i]) * 1e6  # Add large value to ignore positives
                hardest_negative_dist = neg_distances.min()

                # Triplet loss
                loss = F.relu(hardest_positive_dist - hardest_negative_dist + self.margin)
                losses.append(loss)

        if losses:
            return torch.stack(losses).mean()
        else:
            return torch.tensor(0.0, device=features.device, requires_grad=True)


class CompoundLoss(nn.Module):
    """Compound loss combining focal loss, triplet loss, and knowledge distillation"""

    def __init__(self, focal_alpha=1, focal_gamma=2, triplet_margin=0.3,
                 kd_temperature=3.0, kd_alpha=0.7, focal_weight=1.0,
                 triplet_weight=0.5, kd_weight=1.0):
        super(CompoundLoss, self).__init__()

        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.triplet_loss = TripletLoss(margin=triplet_margin)
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

        self.kd_temperature = kd_temperature
        self.kd_alpha = kd_alpha
        self.focal_weight = focal_weight
        self.triplet_weight = triplet_weight
        self.kd_weight = kd_weight

    def forward(self, logits, features, hard_labels, soft_labels=None):
        """
        Args:
            logits: Model output logits [batch_size, num_classes]
            features: Feature embeddings [batch_size, feature_dim]
            hard_labels: Original binary labels (may be mixed from augmentation)
            soft_labels: Soft labels from Stage 1 (optional)
        """
        # Focal loss (handles both hard and mixed labels)
        focal_loss = self.focal_loss(logits, hard_labels)

        # Triplet loss (handles mixed labels internally)
        triplet_loss = self.triplet_loss(features, hard_labels)

        # Knowledge distillation loss
        kd_loss = torch.tensor(0.0, device=logits.device)
        if soft_labels is not None:
            student_soft = F.log_softmax(logits / self.kd_temperature, dim=1)
            teacher_soft = F.softmax(torch.log(soft_labels + 1e-8) / self.kd_temperature, dim=1)
            kd_loss = self.kl_div(student_soft, teacher_soft) * (self.kd_temperature ** 2)

        # Combined loss
        total_loss = (self.focal_weight * focal_loss +
                     self.triplet_weight * triplet_loss +
                     self.kd_weight * kd_loss)

        return total_loss, focal_loss, triplet_loss, kd_loss


class MaxViTWithFeatures(nn.Module):
    """MaxViT model that outputs both logits and features for triplet loss"""

    def __init__(self, model_name='maxvit_base_tf_224', num_classes=2, feature_dim=512):
        super(MaxViTWithFeatures, self).__init__()

        # Load MaxViT backbone using ModelManager
        # Note: We'll get the backbone from the model_manager in the trainer
        self.model_name = model_name
        self.num_classes = num_classes
        self.feature_dim = feature_dim

        # These will be set during initialization
        self.backbone = None
        self.feature_projection = None
        self.classifier = None

    def initialize_with_backbone(self, backbone_model):
        """Initialize with a backbone model from ModelManager"""
        # Store the original model for feature extraction
        self.backbone = backbone_model

        # We'll use the model's forward_features method if available
        # or create a custom feature extractor

        # Get feature dimension by doing a forward pass with the actual model
        self.backbone.eval()
        with torch.no_grad():
            # Make sure dummy input is on the same device as the model
            device = next(self.backbone.parameters()).device
            dummy_input = torch.randn(2, 3, 224, 224).to(device)  # Use batch size 2 to be safe

            # Try different methods to extract features
            backbone_output = None

            # # Method 1: Try forward_features if available
            # if hasattr(self.backbone, 'forward_features'):
            #     try:
            #         backbone_output = self.backbone.forward_features(dummy_input)
            #         print(f"Used forward_features, output shape: {backbone_output.shape}")
            #     except Exception as e:
            #         print(f"forward_features failed: {e}")
            #         backbone_output = None

            # Method 2: Try the backbone directly
            if backbone_output is None:
                try:
                    backbone_output = self.backbone(dummy_input)
                    if isinstance(backbone_output, tuple):
                        backbone_output = backbone_output[0]
                    print(f"Used backbone forward, output shape: {backbone_output.shape}")
                except Exception as e:
                    print(f"Backbone forward failed: {e}")

            # Method 3: Try to use the original model's feature extraction
            if backbone_output is None and hasattr(backbone_model, 'forward_features'):
                try:
                    backbone_output = backbone_model.forward_features(dummy_input)
                    print(f"Used original model forward_features, output shape: {backbone_output.shape}")
                except Exception as e:
                    print(f"Original model forward_features failed: {e}")

            # Method 4: Fallback - remove the last layer and try again
            if backbone_output is None:
                try:
                    # Remove the last layer (classifier) from the original model
                    modules = list(backbone_model.children())[:-1]
                    temp_backbone = nn.Sequential(*modules)
                    backbone_output = temp_backbone(dummy_input)
                    if isinstance(backbone_output, tuple):
                        backbone_output = backbone_output[0]
                    print(f"Used modified backbone, output shape: {backbone_output.shape}")
                    # Update our backbone to this working version
                    self.backbone = temp_backbone
                except Exception as e:
                    print(f"Modified backbone failed: {e}")

            # If still None, raise an error
            if backbone_output is None:
                raise RuntimeError("Could not extract features from the backbone model")

            # Handle different output shapes
            if backbone_output.dim() > 2:
                # Apply global average pooling if needed
                if backbone_output.dim() == 4:  # [B, C, H, W]
                    backbone_output = F.adaptive_avg_pool2d(backbone_output, (1, 1))
                    backbone_output = torch.flatten(backbone_output, 1)
                elif backbone_output.dim() == 3:  # [B, T, C] or [B, C, T]
                    backbone_output = torch.mean(backbone_output, dim=1)
                else:
                    backbone_output = torch.flatten(backbone_output, 1)

            backbone_dim = backbone_output.shape[1]
            print(f"Final backbone dimension: {backbone_dim}")

        # Feature projection
        self.feature_projection = nn.Sequential(
            nn.Linear(backbone_dim, self.feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Classifier
        self.classifier = nn.Linear(self.feature_dim, self.num_classes)

        # Set backbone back to train mode
        self.backbone.train()

    def forward(self, x):
        # Extract features using forward_features if available
        if hasattr(self.backbone, 'forward_features'):
            try:
                backbone_features = self.backbone.forward_features(x)
            except Exception as e:
                print(f"forward_features failed in forward pass: {e}")
                # Fallback to other methods
                backbone_features = self._extract_features_fallback(x)
        else:
            backbone_features = self._extract_features_fallback(x)

        # Handle different output shapes - same as in initialization
        if backbone_features.dim() > 2:
            if backbone_features.dim() == 4:  # [B, C, H, W]
                backbone_features = F.adaptive_avg_pool2d(backbone_features, (1, 1))
                backbone_features = torch.flatten(backbone_features, 1)
            elif backbone_features.dim() == 3:  # [B, T, C] or [B, C, T]
                backbone_features = torch.mean(backbone_features, dim=1)
            else:
                backbone_features = torch.flatten(backbone_features, 1)

        features = self.feature_projection(backbone_features)
        logits = self.classifier(features)

        return logits, features

    def _extract_features_fallback(self, x):
        """Fallback method for feature extraction"""
        # Ensure input is on the same device as model
        device = next(self.backbone.parameters()).device
        x = x.to(device)

        # Try the same methods as in initialization
        if hasattr(self.backbone, 'head'):
            # Temporarily replace head with identity
            original_head = self.backbone.head
            self.backbone.head = nn.Identity()
            features = self.backbone(x)
            self.backbone.head = original_head
            return features
        elif hasattr(self.backbone, 'classifier'):
            # Temporarily replace classifier with identity
            original_classifier = self.backbone.classifier
            self.backbone.classifier = nn.Identity()
            features = self.backbone(x)
            self.backbone.classifier = original_classifier
            return features
        else:
            # Last resort - use full forward pass
            features = self.backbone(x)
            if isinstance(features, tuple):
                features = features[0]
            return features

class Stage2Trainer:
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
        self.criterion = self.create_criterion()

        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.best_val_acer = float('inf')

        self.logger.info(f"Stage 2 Trainer initialized")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model: {self.config['model_name']}")
        self.logger.info(f"Train samples: {len(self.train_loader.dataset)}")
        self.logger.info(f"Val samples: {len(self.val_loader.dataset)}")
        self.logger.info(f"Soft labels path: {self.config['soft_labels_path']}")

    def setup_logging(self):
        """Setup logging"""
        os.makedirs(self.config['log_dir'], exist_ok=True)
        log_file = os.path.join(self.config['log_dir'], f"stage2_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def verify_random_initialization(model: nn.Module, sample_size: int = 10) -> bool:
        """
        Verify that a model has random initialization (not pretrained weights)

        Args:
            model: Model to check
            sample_size: Number of parameters to sample

        Returns:
            True if weights appear to be randomly initialized
        """
        import numpy as np

        # Collect some weight statistics
        weight_means = []
        weight_stds = []

        for name, param in model.named_parameters():
            if 'weight' in name and param.dim() >= 2:  # Check weight matrices
                # Get statistics
                weight_mean = param.data.mean().item()
                weight_std = param.data.std().item()

                weight_means.append(weight_mean)
                weight_stds.append(weight_std)

                if len(weight_means) >= sample_size:
                    break

        # Check if weights look like random initialization
        # Random init typically has mean close to 0 and std following certain patterns
        mean_of_means = np.mean(weight_means)
        std_of_means = np.std(weight_means)
        mean_of_stds = np.mean(weight_stds)

        # Random initialization heuristics:
        # 1. Mean should be very close to 0
        # 2. Std should be consistent with initialization schemes (typically 0.01-0.1)
        # 3. Different layers should have different patterns (high std of means)

        is_random = (
            abs(mean_of_means) < 0.01 and  # Mean close to 0
            0.01 < mean_of_stds < 0.2 and   # Reasonable std range
            std_of_means < 0.01             # Means are clustered around 0
        )

        return is_random


    # Add this to Stage2Trainer.create_model() for verification:
    def create_model(self):
        """Create MaxViT model using ModelManager"""
        # Get the backbone model
        backbone_model = self.model_manager.load_or_download_model(
            stage='stage2',
            model_name=self.config['model_name'],
            num_classes=0,  # Keep original classes for feature extraction
            pretrained=self.config['pretrained'],
            device=self.device
        )

        # Verify initialization if pretrained=False
        if not self.config['pretrained']:
            is_random = verify_random_initialization(backbone_model)
            if is_random:
                self.logger.info("✓ Verified: Model has random initialization")
            else:
                self.logger.warning("⚠ Warning: Model weights don't appear to be randomly initialized!")
                # Optionally force re-initialization
                if self.config.get('force_random_init', True):
                    self.logger.info("Force re-initializing model weights...")
                    for name, param in backbone_model.named_parameters():
                        if 'weight' in name:
                            if param.dim() >= 2:
                                nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
                            elif param.dim() == 1:
                                nn.init.constant_(param, 0)
                        elif 'bias' in name:
                            nn.init.constant_(param, 0)
                    self.logger.info("Model weights re-initialized")

        # Create our custom model with features
        model = MaxViTWithFeatures(
            model_name=self.config['model_name'],
            num_classes=2,
            feature_dim=self.config['feature_dim']
        )

        # Initialize with the backbone
        model.initialize_with_backbone(backbone_model)
        model = model.to(self.device)

        # Log model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")

        return model

    def create_dataloaders(self):
        """Create dataloaders for Stage 2"""
        train_loader, val_loader, test_loader = create_dataloaders_with_debug(
            data_root=self.config['data_root'],
            stage='stage2',
            soft_labels_path=self.config['soft_labels_path'],
            batch_size=self.config['batch_size'],
            num_workers=self.config['num_workers'],
            image_size=self.config['image_size'],
            debug=self.config['debug']
        )

        # Replace train_loader with mixup/cutmix version
        if self.config['use_mixup_cutmix']:
            collate_fn = MixupCutmixCollator(
                mixup_alpha=self.config['mixup_alpha'],
                cutmix_alpha=self.config['cutmix_alpha'],
                prob=self.config['mixup_cutmix_prob']
            )

            train_loader = DataLoader(
                train_loader.dataset,
                batch_size=self.config['batch_size'],
                shuffle=True,
                num_workers=self.config['num_workers'],
                pin_memory=True,
                collate_fn=collate_fn
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

    def create_criterion(self):
        """Create compound loss function"""
        criterion = CompoundLoss(
            focal_alpha=self.config['focal_alpha'],
            focal_gamma=self.config['focal_gamma'],
            triplet_margin=self.config['triplet_margin'],
            kd_temperature=self.config['kd_temperature'],
            kd_alpha=self.config['kd_alpha'],
            focal_weight=self.config['focal_weight'],
            triplet_weight=self.config['triplet_weight'],
            kd_weight=self.config['kd_weight']
        )
        return criterion

    def train_epoch(self):
        """Train one epoch"""
        self.model.train()
        total_loss = 0.0
        focal_loss_total = 0.0
        triplet_loss_total = 0.0
        kd_loss_total = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}/{self.config['num_epochs']}")

        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)

            # Get original labels for accuracy calculation if available
            original_labels = batch.get('original_label', labels).to(self.device)

            # Get soft labels if available
            soft_labels = None
            if 'soft_label' in batch:
                soft_labels = batch['soft_label'].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            logits, features = self.model(images)

            # Calculate compound loss
            loss, focal_loss, triplet_loss, kd_loss = self.criterion(
                logits, features, labels, soft_labels
            )

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Statistics
            total_loss += loss.item()
            focal_loss_total += focal_loss.item()
            triplet_loss_total += triplet_loss.item()
            kd_loss_total += kd_loss.item()

            _, predicted = logits.max(1)

            # Use original labels for accuracy calculation
            total += original_labels.size(0)
            correct += predicted.eq(original_labels).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Focal': f'{focal_loss.item():.4f}',
                'Triplet': f'{triplet_loss.item():.4f}',
                'KD': f'{kd_loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })

        epoch_loss = total_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        focal_avg = focal_loss_total / len(self.train_loader)
        triplet_avg = triplet_loss_total / len(self.train_loader)
        kd_avg = kd_loss_total / len(self.train_loader)

        return epoch_loss, epoch_acc, focal_avg, triplet_avg, kd_avg

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

                logits, features = self.model(images)

                # Use simple cross-entropy for validation
                loss = F.cross_entropy(logits, labels)

                total_loss += loss.item()
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                # Store for ACER calculation
                probs = F.softmax(logits, dim=1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_scores.extend(probs[:, 1].cpu().numpy())  # Live probability

        val_loss = total_loss / (len(self.val_loader) + 0.0000001)
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
        checkpoint_path = os.path.join(self.config['checkpoint_dir'], 'stage2_latest.pth')
        torch.save(checkpoint, checkpoint_path)

        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.config['checkpoint_dir'], 'stage2_best.pth')
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
                stage='stage2',
                model_name=self.config['model_name'],
                epoch=self.current_epoch,
                metrics=metrics
            )

    def test_model(self):
        """Test the best model on test set"""
        self.logger.info("Testing best model on test set...")

        # Load best model
        best_model_path = os.path.join(self.config['checkpoint_dir'], 'stage2_best.pth')
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info("Loaded best model for testing")

        self.model.eval()
        correct = 0
        total = 0

        # For detailed metrics
        all_predictions = []
        all_labels = []
        all_scores = []
        attack_type_results = {}

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                attack_types = batch['attack_type']

                logits, features = self.model(images)
                probs = F.softmax(logits, dim=1)
                _, predicted = logits.max(1)

                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                # Store results
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_scores.extend(probs[:, 1].cpu().numpy())

                # Store by attack type
                for i, attack_type in enumerate(attack_types):
                    if attack_type not in attack_type_results:
                        attack_type_results[attack_type] = {'correct': 0, 'total': 0}
                    attack_type_results[attack_type]['total'] += 1
                    if predicted[i] == labels[i]:
                        attack_type_results[attack_type]['correct'] += 1

        test_acc = 100. * correct / total
        test_acer = self.calculate_acer(all_labels, all_predictions, all_scores)

        # Log results
        self.logger.info(f"Test Results:")
        self.logger.info(f"  Overall Accuracy: {test_acc:.2f}%")
        self.logger.info(f"  Overall ACER: {test_acer:.4f}")

        self.logger.info(f"  Results by Attack Type:")
        for attack_type, results in attack_type_results.items():
            acc = 100. * results['correct'] / results['total']
            self.logger.info(f"    {attack_type}: {acc:.2f}% ({results['correct']}/{results['total']})")

        return test_acc, test_acer, attack_type_results

    def train(self):
        """Main training loop"""
        self.logger.info("Starting Stage 2 training...")

        for epoch in range(self.config['num_epochs']):
            self.current_epoch = epoch

            # Training
            train_loss, train_acc, focal_loss, triplet_loss, kd_loss = self.train_epoch()

            # Validation
            val_loss, val_acc, val_acer = self.validate()

            # Update scheduler
            if self.scheduler:
                self.scheduler.step()

            # Logging
            self.logger.info(f"Epoch {epoch+1}/{self.config['num_epochs']}:")
            self.logger.info(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            self.logger.info(f"    Focal: {focal_loss:.4f}, Triplet: {triplet_loss:.4f}, KD: {kd_loss:.4f}")
            self.logger.info(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val ACER: {val_acer:.4f}")
            self.logger.info(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")

            # Save best model
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_val_acer = val_acer

            self.save_checkpoint(is_best)

        # Test on test set
        test_acc, test_acer, attack_results = self.test_model()

        self.logger.info("Stage 2 training completed!")
        self.logger.info(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        self.logger.info(f"Best validation ACER: {self.best_val_acer:.4f}")
        self.logger.info(f"Test accuracy: {test_acc:.2f}%")
        self.logger.info(f"Test ACER: {test_acer:.4f}")

        return test_acc, test_acer

        # In Stage2Trainer.__init__, add clear logging about initialization:
        def setup_logging(self):
            """Setup logging"""
            os.makedirs(self.config['log_dir'], exist_ok=True)
            log_file = os.path.join(self.config['log_dir'], f"stage2_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(log_file),
                    logging.StreamHandler()
                ]
            )
            self.logger = logging.getLogger(__name__)

            # Log initialization strategy clearly
            self.logger.info("="*60)
            self.logger.info("STAGE 2 TRAINING INITIALIZATION")
            self.logger.info("="*60)

            if self.config['pretrained']:
                self.logger.info("► Model Initialization: Using PRETRAINED weights (ImageNet)")
            else:
                self.logger.info("► Model Initialization: Using RANDOM weights")
                self.logger.info("  Note: Model will be initialized from scratch")
                self.logger.info("  This is typically used for: ")
                self.logger.info("  - Training from scratch on your dataset")
                self.logger.info("  - Avoiding ImageNet bias")
                self.logger.info("  - Research experiments requiring random init")

            self.logger.info("="*60)

def get_default_config():
    """Get default configuration for Stage 2"""
    return {
        # Data settings
        'data_root': '/path/to/WFAS/dataset',
        'soft_labels_path': './outputs/stage1_soft_labels.pth',  # From Stage 1
        'image_size': 224,
        'batch_size': 32,
        'num_workers': 4,

        # Model settings
        'model_name': 'maxvit_base_tf_224',  # or maxvit_small_tf_224, maxvit_large_tf_224
        'feature_dim': 512,
        'pretrained': True,
        'model_cache_dir': './model',  # Base model cache directory

        # Training settings
        'num_epochs': 30,
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'optimizer': 'adamw',  # adamw or sgd
        'scheduler': 'cosine',  # cosine, step, or none

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
        'checkpoint_dir': './checkpoints/stage2',
        'log_dir': './logs/stage2',
        'output_dir': './outputs',

        # Debug
        'debug': True
    }


def main():
    parser = argparse.ArgumentParser(description='Stage 2 Training - MaxViT')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--data_root', type=str, help='Path to dataset')
    parser.add_argument('--soft_labels_path', type=str, help='Path to soft labels from Stage 1')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--model', type=str, default='maxvit_base_tf_224', help='Model name')
    parser.add_argument('--model_cache_dir', type=str, default='./model', help='Model cache directory')

    args = parser.parse_args()

    # Get configuration
    config = get_default_config()

    # Override with command line arguments
    if args.data_root:
        config['data_root'] = args.data_root
    if args.soft_labels_path:
        config['soft_labels_path'] = args.soft_labels_path
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

    # Validate soft labels path
    if not os.path.exists(config['soft_labels_path']):
        print(f"ERROR: Soft labels file not found: {config['soft_labels_path']}")
        print("Please run Stage 1 training first to generate soft labels!")
        return

    # Create directories
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)
    os.makedirs(config['output_dir'], exist_ok=True)
    os.makedirs(config['model_cache_dir'], exist_ok=True)

    # Save config
    config_path = os.path.join(config['output_dir'], 'stage2_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    # Start training
    trainer = Stage2Trainer(config)
    test_acc, test_acer = trainer.train()

    print(f"\n=== Stage 2 Training Completed ===")
    print(f"Best validation accuracy: {trainer.best_val_acc:.2f}%")
    print(f"Best validation ACER: {trainer.best_val_acer:.4f}")
    print(f"Test accuracy: {test_acc:.2f}%")
    print(f"Test ACER: {test_acer:.4f}")


if __name__ == "__main__":
    main()