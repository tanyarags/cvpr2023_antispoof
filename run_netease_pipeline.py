"""
Complete NetEase Two-Stage Training Pipeline with ModelManager Integration
Runs Stage 1 (ConvNext) followed by Stage 2 (MaxViT)
Now includes model caching and management features
"""

import os
import json
import argparse
from datetime import datetime
from model_manager import ModelManager


def create_stage1_config(data_root, output_dir, model_cache_dir):
    """Create configuration for Stage 1"""
    config = {
        # Data settings
        'data_root': data_root,
        'image_size': 224,
        'batch_size': 32,
        'num_workers': 4,
        
        # Model settings
        'model_name': 'convnext_base',
        'pretrained': True,
        'model_cache_dir': model_cache_dir,
        
        # Training settings
        'num_epochs': 50,
        'learning_rate': 2e-4,
        'weight_decay': 0.01,
        'optimizer': 'adamw',
        'scheduler': 'cosine',
        
        # Paths
        'checkpoint_dir': os.path.join(output_dir, 'checkpoints', 'stage1'),
        'log_dir': os.path.join(output_dir, 'logs', 'stage1'),
        'output_dir': output_dir,
        
        # Debug
        'debug': True
    }
    return config


def create_stage2_config(data_root, output_dir, model_cache_dir, soft_labels_path):
    """Create configuration for Stage 2"""
    config = {
        # Data settings
        'data_root': data_root,
        'soft_labels_path': soft_labels_path,
        'image_size': 224,
        'batch_size': 32,
        'num_workers': 4,
        
        # Model settings
        'model_name': 'maxvit_base_tf_224',
        'feature_dim': 512,
        'pretrained': True,
        'model_cache_dir': model_cache_dir,
        
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
        'checkpoint_dir': os.path.join(output_dir, 'checkpoints', 'stage2'),
        'log_dir': os.path.join(output_dir, 'logs', 'stage2'),
        'output_dir': output_dir,
        
        # Debug
        'debug': True
    }
    return config


def check_models_availability(model_cache_dir, stage1_model, stage2_model):
    """Check if models are available in cache or need to be downloaded"""
    manager = ModelManager(cache_base_dir=model_cache_dir)
    
    print("🔍 Checking model availability...")
    
    # Check Stage 1 model
    stage1_dir = manager.get_model_cache_dir('stage1', stage1_model)
    stage1_available = manager.check_local_model_exists(stage1_dir)
    
    # Check Stage 2 model
    stage2_dir = manager.get_model_cache_dir('stage2', stage2_model)
    stage2_available = manager.check_local_model_exists(stage2_dir)
    
    print(f"  Stage 1 ({stage1_model}): {'✅ Cached' if stage1_available else '📥 Will download'}")
    print(f"  Stage 2 ({stage2_model}): {'✅ Cached' if stage2_available else '📥 Will download'}")
    
    # Show all cached models
    cached_models = manager.list_cached_models()
    if cached_models:
        print("📦 Currently cached models:")
        for stage, models in cached_models.items():
            if models:
                print(f"  {stage}: {', '.join(models)}")
    else:
        print("📦 No models currently cached")
    
    return stage1_available, stage2_available


def run_stage1(config):
    """Run Stage 1 training"""
    print("="*60)
    print("STARTING STAGE 1: ConvNext Training")
    print("="*60)
    
    try:
        from stage1_training import Stage1Trainer
    except ImportError:
        print("❌ Error: stage1_training module not found!")
        print("Make sure stage1_training.py is in the same directory")
        return None
    
    trainer = Stage1Trainer(config)
    soft_labels_path = trainer.train()
    
    print(f"\nStage 1 completed!")
    print(f"Best validation accuracy: {trainer.best_val_acc:.2f}%")
    print(f"Best validation ACER: {trainer.best_val_acer:.4f}")
    print(f"Soft labels saved to: {soft_labels_path}")
    
    return soft_labels_path


def run_stage2(config):
    """Run Stage 2 training"""
    print("\n" + "="*60)
    print("STARTING STAGE 2: MaxViT Training with Soft Labels")
    print("="*60)
    
    try:
        from stage2_training import Stage2Trainer
    except ImportError:
        print("❌ Error: stage2_training module not found!")
        print("Make sure stage2_training.py is in the same directory")
        return None
    
    trainer = Stage2Trainer(config)
    test_acc, test_acer = trainer.train()
    
    print(f"\nStage 2 completed!")
    print(f"Best validation accuracy: {trainer.best_val_acc:.2f}%")
    print(f"Best validation ACER: {trainer.best_val_acer:.4f}")
    print(f"Final test accuracy: {test_acc:.2f}%")
    print(f"Final test ACER: {test_acer:.4f}")
    
    return trainer.best_val_acc, trainer.best_val_acer, test_acc, test_acer


def download_models_only(model_cache_dir, stage1_model, stage2_model):
    """Download models without training"""
    print("📥 Downloading models...")
    manager = ModelManager(cache_base_dir=model_cache_dir)
    
    try:
        print(f"  Downloading {stage1_model} for Stage 1...")
        manager.load_or_download_model(
            stage='stage1',
            model_name=stage1_model,
            num_classes=2,
            device='cpu'
        )
        print(f"  ✅ {stage1_model} downloaded and cached")
    except Exception as e:
        print(f"  ❌ Failed to download {stage1_model}: {e}")
    
    try:
        print(f"  Downloading {stage2_model} for Stage 2...")
        manager.load_or_download_model(
            stage='stage2',
            model_name=stage2_model,
            num_classes=2,
            device='cpu'
        )
        print(f"  ✅ {stage2_model} downloaded and cached")
    except Exception as e:
        print(f"  ❌ Failed to download {stage2_model}: {e}")
    
    print("📦 Model download completed!")


def clear_model_cache(model_cache_dir, stage=None, model_name=None):
    """Clear model cache"""
    manager = ModelManager(cache_base_dir=model_cache_dir)
    
    try:
        manager.clear_cache(stage=stage, model_name=model_name)
        if stage and model_name:
            print(f"✅ Cleared cache for {stage}/{model_name}")
        elif stage:
            print(f"✅ Cleared cache for stage: {stage}")
        else:
            print("✅ Cleared all model cache")
    except Exception as e:
        print(f"❌ Failed to clear cache: {e}")


def list_cached_models(model_cache_dir):
    """List all cached models"""
    manager = ModelManager(cache_base_dir=model_cache_dir)
    cached_models = manager.list_cached_models()
    
    if cached_models:
        print("📦 Cached Models:")
        print("=" * 30)
        for stage, models in cached_models.items():
            print(f"\n{stage.upper()}:")
            if models:
                for model in models:
                    model_dir = manager.get_model_cache_dir(stage, model)
                    try:
                        config = manager.load_model_config(model_dir)
                        print(f"  • {model} (pretrained: {config.get('pretrained', 'unknown')})")
                    except:
                        print(f"  • {model}")
            else:
                print("  No models cached")
    else:
        print("📦 No cached models found")


def main():
    parser = argparse.ArgumentParser(description='NetEase Two-Stage Anti-Spoofing Pipeline')
    parser.add_argument('--data_root', type=str, help='Path to WFAS dataset')
    parser.add_argument('--output_dir', type=str, default='./netease_experiment', help='Output directory')
    parser.add_argument('--model_cache_dir', type=str, default='./model', help='Model cache directory')
    
    # Training modes
    parser.add_argument('--stage1_only', action='store_true', help='Run only Stage 1')
    parser.add_argument('--stage2_only', action='store_true', help='Run only Stage 2')
    parser.add_argument('--soft_labels_path', type=str, help='Path to soft labels (for stage2_only)')
    
    # Model management modes
    parser.add_argument('--download_only', action='store_true', help='Download models without training')
    parser.add_argument('--list_models', action='store_true', help='List cached models')
    parser.add_argument('--clear_cache', action='store_true', help='Clear model cache')
    parser.add_argument('--clear_stage', type=str, help='Clear cache for specific stage')
    parser.add_argument('--clear_model', type=str, help='Clear cache for specific model (use with --clear_stage)')
    
    # Stage 1 specific arguments
    parser.add_argument('--stage1_epochs', type=int, default=50, help='Stage 1 epochs')
    parser.add_argument('--stage1_lr', type=float, default=2e-4, help='Stage 1 learning rate')
    parser.add_argument('--stage1_model', type=str, default='convnext_base', help='Stage 1 model')
    
    # Stage 2 specific arguments
    parser.add_argument('--stage2_epochs', type=int, default=30, help='Stage 2 epochs')
    parser.add_argument('--stage2_lr', type=float, default=1e-4, help='Stage 2 learning rate')
    parser.add_argument('--stage2_model', type=str, default='maxvit_base_tf_224', help='Stage 2 model')
    
    # Common arguments
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--image_size', type=int, default=224, help='Image size')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.model_cache_dir, exist_ok=True)
    
    # Handle model management commands
    if args.list_models:
        list_cached_models(args.model_cache_dir)
        return
    
    if args.clear_cache:
        clear_model_cache(args.model_cache_dir)
        return
    
    if args.clear_stage:
        clear_model_cache(args.model_cache_dir, stage=args.clear_stage, model_name=args.clear_model)
        return
    
    if args.download_only:
        if not args.data_root:
            print("❌ --data_root is required for model download")
            return
        download_models_only(args.model_cache_dir, args.stage1_model, args.stage2_model)
        return
    
    # Validate data_root for training
    if not args.data_root:
        print("❌ --data_root is required for training")
        return
    
    if not os.path.exists(args.data_root):
        print(f"❌ Data root not found: {args.data_root}")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save experiment info
    experiment_info = {
        'timestamp': datetime.now().isoformat(),
        'args': vars(args),
        'description': 'NetEase Two-Stage Anti-Spoofing Training with ModelManager'
    }
    
    with open(os.path.join(args.output_dir, 'experiment_info.json'), 'w') as f:
        json.dump(experiment_info, f, indent=2)
    
    print("NetEase Two-Stage Anti-Spoofing Training Pipeline")
    print(f"Data root: {args.data_root}")
    print(f"Output directory: {args.output_dir}")
    print(f"Model cache: {args.model_cache_dir}")
    print(f"Timestamp: {experiment_info['timestamp']}")
    
    # Check model availability
    check_models_availability(args.model_cache_dir, args.stage1_model, args.stage2_model)
    
    soft_labels_path = None
    stage1_results = None
    stage2_results = None
    
    # Run Stage 1
    if not args.stage2_only:
        stage1_config = create_stage1_config(args.data_root, args.output_dir, args.model_cache_dir)
        
        # Override with command line arguments
        stage1_config['num_epochs'] = args.stage1_epochs
        stage1_config['learning_rate'] = args.stage1_lr
        stage1_config['model_name'] = args.stage1_model
        stage1_config['batch_size'] = args.batch_size
        stage1_config['image_size'] = args.image_size
        
        # Save Stage 1 config
        with open(os.path.join(args.output_dir, 'stage1_config.json'), 'w') as f:
            json.dump(stage1_config, f, indent=2)
        
        try:
            soft_labels_path = run_stage1(stage1_config)
            stage1_results = "completed"
        except Exception as e:
            print(f"❌ Stage 1 failed: {e}")
            if args.stage1_only:
                return
            else:
                print("⚠️  Continuing to Stage 2 with existing soft labels...")
    
    # Run Stage 2
    if not args.stage1_only:
        # Determine soft labels path
        if args.stage2_only and args.soft_labels_path:
            soft_labels_path = args.soft_labels_path
        elif soft_labels_path is None:
            soft_labels_path = os.path.join(args.output_dir, 'stage1_soft_labels.pth')
        
        if not os.path.exists(soft_labels_path):
            print(f"❌ ERROR: Soft labels file not found: {soft_labels_path}")
            print("Please run Stage 1 first or provide correct --soft_labels_path")
            return
        
        stage2_config = create_stage2_config(args.data_root, args.output_dir, args.model_cache_dir, soft_labels_path)
        
        # Override with command line arguments
        stage2_config['num_epochs'] = args.stage2_epochs
        stage2_config['learning_rate'] = args.stage2_lr
        stage2_config['model_name'] = args.stage2_model
        stage2_config['batch_size'] = args.batch_size
        stage2_config['image_size'] = args.image_size
        
        # Save Stage 2 config
        with open(os.path.join(args.output_dir, 'stage2_config.json'), 'w') as f:
            json.dump(stage2_config, f, indent=2)
        
        try:
            stage2_results = run_stage2(stage2_config)
        except Exception as e:
            print(f"❌ Stage 2 failed: {e}")
            return
    
    # Final summary
    print("\n" + "="*60)
    print("TRAINING PIPELINE COMPLETED")
    print("="*60)
    
    if stage1_results:
        print("✅ Stage 1 (ConvNext): Completed")
        print(f"  Soft labels saved to: {soft_labels_path}")
    
    if stage2_results:
        val_acc, val_acer, test_acc, test_acer = stage2_results
        print("✅ Stage 2 (MaxViT): Completed")
        print(f"  Best validation accuracy: {val_acc:.2f}%")
        print(f"  Best validation ACER: {val_acer:.4f}")
        print(f"  Final test accuracy: {test_acc:.2f}%")
        print(f"  Final test ACER: {test_acer:.4f}")
    
    print(f"\n📁 All outputs saved to: {args.output_dir}")
    print("📋 Checkpoints, logs, and configs are available in subdirectories.")
    print(f"🗄️  Models cached in: {args.model_cache_dir}")


if __name__ == "__main__":
    main()
