"""
Model Manager - Handles model loading, caching, and downloading
Supports both Stage 1 (ConvNext) and Stage 2 (MaxViT) models
"""

import os
import torch
import torch.nn as nn
import json
import logging
import timm
from typing import Optional, Dict, Any, Union


class ModelManager:
    """
    Centralized model management for loading, caching, and downloading models
    Supports HuggingFace timm models with local caching
    """
    
    def __init__(self, cache_base_dir: str = "./model", logger: Optional[logging.Logger] = None):
        """
        Initialize ModelManager
        
        Args:
            cache_base_dir: Base directory for model caching
            logger: Logger instance (optional)
        """
        self.cache_base_dir = cache_base_dir
        self.logger = logger or self._setup_default_logger()
        
        # Ensure cache directory exists
        os.makedirs(cache_base_dir, exist_ok=True)
    
    def _setup_default_logger(self) -> logging.Logger:
        """Setup default logger if none provided"""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def get_model_cache_dir(self, stage: str, model_name: str) -> str:
        """
        Get cache directory for a specific model
        
        Args:
            stage: Stage identifier (e.g., 'stage1', 'stage2')
            model_name: Name of the model
            
        Returns:
            Path to model cache directory
        """
        return os.path.join(self.cache_base_dir, stage, model_name)
    
    def check_local_model_exists(self, model_dir: str) -> bool:
        """
        Check if model files exist in local directory
        
        Args:
            model_dir: Directory to check
            
        Returns:
            True if model files exist, False otherwise
        """
        # Required files for a complete model
        required_files = ['config.json']
        
        # Check for different weight file formats
        weight_files = ['model.safetensors', 'pytorch_model.bin', 'model.pth']
        
        # Check if config exists
        has_config = all(os.path.exists(os.path.join(model_dir, f)) for f in required_files)
        
        # Check if at least one weight file exists
        has_weights = any(os.path.exists(os.path.join(model_dir, f)) for f in weight_files)
        
        return has_config and has_weights
    
    def save_model_config(self, model_dir: str, model_name: str, num_classes: int, 
                         pretrained: bool, additional_info: Optional[Dict] = None) -> None:
        """
        Save model configuration
        
        Args:
            model_dir: Directory to save config
            model_name: Name of the model
            num_classes: Number of classes
            pretrained: Whether model was pretrained
            additional_info: Additional configuration info
        """
        model_config = {
            'model_name': model_name,
            'num_classes': num_classes,
            'pretrained': pretrained,
            'downloaded_from': 'timm',
            'version': '1.0'
        }
        
        if additional_info:
            model_config.update(additional_info)
        
        config_path = os.path.join(model_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(model_config, f, indent=2)
        
        self.logger.info(f"Model config saved to: {config_path}")
    
    def load_model_config(self, model_dir: str) -> Dict[str, Any]:
        """
        Load model configuration
        
        Args:
            model_dir: Directory containing config
            
        Returns:
            Model configuration dictionary
        """
        config_path = os.path.join(model_dir, 'config.json')
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Model config not found: {config_path}")
        
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def download_and_save_model(self, model_name: str, model_dir: str, 
                               pretrained: bool = True, num_classes: int = 1000) -> bool:
        """
        Download model from HuggingFace and save to local directory
        
        Args:
            model_name: Name of the model to download
            model_dir: Directory to save the model
            pretrained: Whether to use pretrained weights
            num_classes: Number of classes for the model
            
        Returns:
            True if successful, False otherwise
        """
        self.logger.info(f"Downloading model '{model_name}' from HuggingFace...")
        
        try:
            # Create model directory
            os.makedirs(model_dir, exist_ok=True)
            
            # Create model from timm (this will download if not cached)
            model = timm.create_model(
                model_name, 
                pretrained=pretrained,
                num_classes=num_classes
            )
            
            # Save the model state dict
            model_path = os.path.join(model_dir, 'pytorch_model.bin')
            torch.save(model.state_dict(), model_path)
            
            # Save model configuration
            self.save_model_config(
                model_dir=model_dir,
                model_name=model_name,
                num_classes=num_classes,
                pretrained=pretrained
            )
            
            self.logger.info(f"Model saved to: {model_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to download and save model: {str(e)}")
            return False
    
    def load_local_model(self, model_name: str, model_dir: str, 
                        target_num_classes: Optional[int] = None) -> Optional[nn.Module]:
        """
        Load model from local directory
        
        Args:
            model_name: Name of the model
            model_dir: Directory containing the model
            target_num_classes: Target number of classes (if different from saved)
            
        Returns:
            Loaded model or None if failed
        """
        try:
            # Load model configuration
            model_config = self.load_model_config(model_dir)
            
            # Create model with original number of classes
            original_num_classes = model_config.get('num_classes', 1000)
            model = timm.create_model(
                model_name,
                pretrained=False,  # Don't download pretrained weights
                num_classes=original_num_classes
            )
            
            # Find and load weights
            weight_files = ['pytorch_model.bin', 'model.safetensors', 'model.pth']
            model_path = None
            
            for weight_file in weight_files:
                potential_path = os.path.join(model_dir, weight_file)
                if os.path.exists(potential_path):
                    model_path = potential_path
                    break
            
            if model_path is None:
                self.logger.warning(f"No model weights found in {model_dir}")
                return None
            
            # Load weights based on file type
            if model_path.endswith('.bin') or model_path.endswith('.pth'):
                state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
            elif model_path.endswith('.safetensors'):
                try:
                    from safetensors.torch import load_file
                    state_dict = load_file(model_path)
                except ImportError:
                    self.logger.warning("safetensors not installed, cannot load .safetensors files")
                    return None
            else:
                self.logger.error(f"Unsupported weight file format: {model_path}")
                return None
            
            # Load state dict
            model.load_state_dict(state_dict)
            
            # Modify classifier layer if needed
            if target_num_classes is not None and target_num_classes != original_num_classes:
                model = self._modify_classifier(model, target_num_classes)
            
            self.logger.info(f"Loaded model weights from: {model_path}")
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load local model: {str(e)}")
            return None
    
    def _modify_classifier(self, model: nn.Module, num_classes: int) -> nn.Module:
        """
        Modify the classifier layer of a model
        
        Args:
            model: Model to modify
            num_classes: New number of classes
            
        Returns:
            Modified model
        """
        if hasattr(model, 'head'):
            # ConvNext and some other models
            if hasattr(model.head, 'fc'):
                in_features = model.head.fc.in_features
                model.head.fc = nn.Linear(in_features, num_classes)
            else:
                in_features = model.head.in_features
                model.head = nn.Linear(in_features, num_classes)
        elif hasattr(model, 'classifier'):
            # ResNet and some other models
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes)
        elif hasattr(model, 'fc'):
            # Some models have direct fc layer
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
        else:
            self.logger.warning("Could not find classifier layer to modify")
        
        return model
    
    def load_or_download_model(self, stage: str, model_name: str, 
                              num_classes: int = 2, pretrained: bool = True,
                              device: str = 'cpu') -> nn.Module:
        """
        Load model from cache or download if not available
        
        Args:
            stage: Stage identifier ('stage1', 'stage2', etc.)
            model_name: Name of the model
            num_classes: Number of classes for final layer
            pretrained: Whether to use pretrained weights
            device: Device to load model on
            
        Returns:
            Loaded model
        """
        model_dir = self.get_model_cache_dir(stage, model_name)
        
        # Try to load from local cache first
        if self.check_local_model_exists(model_dir):
            self.logger.info(f"Loading model from local directory: {model_dir}")
            model = self.load_local_model(model_name, model_dir, target_num_classes=num_classes)
            
            if model is not None:
                model = model.to(device)
                self.logger.info("Successfully loaded model from local cache")
                return model
            else:
                self.logger.warning("Failed to load local model, falling back to download")
        else:
            self.logger.info(f"Local model not found in: {model_dir}")
        
        # Download and save model
        if self.download_and_save_model(model_name, model_dir, pretrained, num_classes=1000):
            # Try loading the downloaded model
            model = self.load_local_model(model_name, model_dir, target_num_classes=num_classes)
        else:
            model = None
        
        # Fallback to direct timm download
        if model is None:
            self.logger.info("Using timm directly as fallback...")
            model = timm.create_model(
                model_name, 
                pretrained=pretrained,
                num_classes=num_classes
            )
        
        model = model.to(device)
        
        # Log model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Model loaded: {model_name}")
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
        
        return model
    
    def save_trained_model(self, model: nn.Module, stage: str, model_name: str, 
                          epoch: int, metrics: Dict[str, float], 
                          additional_data: Optional[Dict] = None) -> str:
        """
        Save a trained model with metadata
        
        Args:
            model: Trained model to save
            stage: Stage identifier
            model_name: Base model name
            epoch: Training epoch
            metrics: Training metrics
            additional_data: Additional data to save
            
        Returns:
            Path to saved model
        """
        model_dir = self.get_model_cache_dir(stage, f"{model_name}_trained")
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model state dict
        model_path = os.path.join(model_dir, 'trained_model.pth')
        
        save_data = {
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'metrics': metrics,
            'model_name': model_name,
            'stage': stage
        }
        
        if additional_data:
            save_data.update(additional_data)
        
        torch.save(save_data, model_path)
        
        # Save configuration
        self.save_model_config(
            model_dir=model_dir,
            model_name=model_name,
            num_classes=metrics.get('num_classes', 2),
            pretrained=True,
            additional_info={
                'trained': True,
                'epoch': epoch,
                'metrics': metrics,
                'stage': stage
            }
        )
        
        self.logger.info(f"Trained model saved to: {model_path}")
        return model_path
    
    def list_cached_models(self, stage: Optional[str] = None) -> Dict[str, list]:
        """
        List all cached models
        
        Args:
            stage: Optional stage filter
            
        Returns:
            Dictionary of stage -> list of model names
        """
        cached_models = {}
        
        if stage:
            stage_dirs = [stage]
        else:
            stage_dirs = [d for d in os.listdir(self.cache_base_dir) 
                         if os.path.isdir(os.path.join(self.cache_base_dir, d))]
        
        for stage_dir in stage_dirs:
            stage_path = os.path.join(self.cache_base_dir, stage_dir)
            if os.path.exists(stage_path):
                models = [d for d in os.listdir(stage_path) 
                         if os.path.isdir(os.path.join(stage_path, d)) 
                         and self.check_local_model_exists(os.path.join(stage_path, d))]
                cached_models[stage_dir] = models
        
        return cached_models
    
    def clear_cache(self, stage: Optional[str] = None, model_name: Optional[str] = None) -> None:
        """
        Clear model cache
        
        Args:
            stage: Optional stage to clear (if None, clears all)
            model_name: Optional specific model to clear
        """
        import shutil
        
        if stage and model_name:
            # Clear specific model
            model_dir = self.get_model_cache_dir(stage, model_name)
            if os.path.exists(model_dir):
                shutil.rmtree(model_dir)
                self.logger.info(f"Cleared cache for {stage}/{model_name}")
        elif stage:
            # Clear entire stage
            stage_dir = os.path.join(self.cache_base_dir, stage)
            if os.path.exists(stage_dir):
                shutil.rmtree(stage_dir)
                self.logger.info(f"Cleared cache for stage: {stage}")
        else:
            # Clear all cache
            if os.path.exists(self.cache_base_dir):
                shutil.rmtree(self.cache_base_dir)
                os.makedirs(self.cache_base_dir)
                self.logger.info("Cleared all model cache")


# Convenience functions for backward compatibility
def create_model_with_cache(stage: str, model_name: str, num_classes: int = 2,
                           pretrained: bool = True, device: str = 'cpu',
                           cache_dir: str = "./model", logger: Optional[logging.Logger] = None) -> nn.Module:
    """
    Convenience function to create a model with caching
    
    Args:
        stage: Stage identifier
        model_name: Model name
        num_classes: Number of output classes
        pretrained: Use pretrained weights
        device: Device to load on
        cache_dir: Cache directory
        logger: Logger instance
        
    Returns:
        Loaded model
    """
    manager = ModelManager(cache_base_dir=cache_dir, logger=logger)
    return manager.load_or_download_model(
        stage=stage,
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        device=device
    )


def list_available_models(cache_dir: str = "./model") -> Dict[str, list]:
    """
    List all available cached models
    
    Args:
        cache_dir: Cache directory
        
    Returns:
        Dictionary of available models by stage
    """
    manager = ModelManager(cache_base_dir=cache_dir)
    return manager.list_cached_models()
