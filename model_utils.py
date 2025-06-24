"""
Model Management Utilities
Command-line interface for managing cached models
"""

import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_manager import ModelManager
import logging


def setup_logging():
    """Setup basic logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def list_models(args):
    """List cached models"""
    logger = setup_logging()
    manager = ModelManager(cache_base_dir=args.cache_dir, logger=logger)
    
    cached_models = manager.list_cached_models(stage=args.stage)
    
    if not cached_models:
        print("No cached models found.")
        return
    
    print("Cached Models:")
    print("=" * 50)
    
    for stage, models in cached_models.items():
        print(f"\nStage: {stage}")
        print("-" * 20)
        if models:
            for model in models:
                print(f"  • {model}")
        else:
            print("  No models cached")


def download_model(args):
    """Download and cache a model"""
    logger = setup_logging()
    manager = ModelManager(cache_base_dir=args.cache_dir, logger=logger)
    
    print(f"Downloading model: {args.model_name} for {args.stage}")
    
    try:
        model = manager.load_or_download_model(
            stage=args.stage,
            model_name=args.model_name,
            num_classes=args.num_classes,
            pretrained=args.pretrained,
            device='cpu'  # Just download, don't load to GPU
        )
        print(f"✅ Successfully downloaded and cached: {args.model_name}")
        
    except Exception as e:
        print(f"❌ Failed to download model: {str(e)}")


def clear_cache(args):
    """Clear model cache"""
    logger = setup_logging()
    manager = ModelManager(cache_base_dir=args.cache_dir, logger=logger)
    
    if args.confirm or input(f"Are you sure you want to clear cache? (y/N): ").lower() == 'y':
        try:
            manager.clear_cache(stage=args.stage, model_name=args.model_name)
            print("✅ Cache cleared successfully")
        except Exception as e:
            print(f"❌ Failed to clear cache: {str(e)}")
    else:
        print("Cache clear cancelled")


def check_model(args):
    """Check if a model exists in cache"""
    logger = setup_logging()
    manager = ModelManager(cache_base_dir=args.cache_dir, logger=logger)
    
    model_dir = manager.get_model_cache_dir(args.stage, args.model_name)
    exists = manager.check_local_model_exists(model_dir)
    
    if exists:
        print(f"✅ Model {args.model_name} exists in {args.stage} cache")
        
        # Try to load config
        try:
            config = manager.load_model_config(model_dir)
            print("\nModel Info:")
            print("-" * 20)
            for key, value in config.items():
                print(f"  {key}: {value}")
        except Exception as e:
            print(f"⚠️  Could not load model config: {str(e)}")
    else:
        print(f"❌ Model {args.model_name} not found in {args.stage} cache")
        print(f"Cache directory: {model_dir}")


def main():
    parser = argparse.ArgumentParser(description='Model Management Utilities')
    parser.add_argument('--cache_dir', type=str, default='./model', 
                       help='Model cache directory (default: ./model)')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List cached models')
    list_parser.add_argument('--stage', type=str, help='Filter by stage (optional)')
    list_parser.set_defaults(func=list_models)
    
    # Download command
    download_parser = subparsers.add_parser('download', help='Download and cache a model')
    download_parser.add_argument('stage', type=str, help='Stage identifier (e.g., stage1, stage2)')
    download_parser.add_argument('model_name', type=str, help='Model name (e.g., convnext_base)')
    download_parser.add_argument('--num_classes', type=int, default=2, help='Number of classes')
    download_parser.add_argument('--no-pretrained', dest='pretrained', action='store_false',
                               help='Don\'t use pretrained weights')
    download_parser.set_defaults(func=download_model, pretrained=True)
    
    # Clear command
    clear_parser = subparsers.add_parser('clear', help='Clear model cache')
    clear_parser.add_argument('--stage', type=str, help='Stage to clear (optional)')
    clear_parser.add_argument('--model_name', type=str, help='Specific model to clear (optional)')
    clear_parser.add_argument('--confirm', action='store_true', help='Skip confirmation prompt')
    clear_parser.set_defaults(func=clear_cache)
    
    # Check command
    check_parser = subparsers.add_parser('check', help='Check if a model exists in cache')
    check_parser.add_argument('stage', type=str, help='Stage identifier')
    check_parser.add_argument('model_name', type=str, help='Model name')
    check_parser.set_defaults(func=check_model)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    args.func(args)


if __name__ == "__main__":
    main()
