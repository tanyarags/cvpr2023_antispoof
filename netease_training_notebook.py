# NetEase Anti-Spoofing Training Notebook
# Two-Stage Training: ConvNext (Stage 1) → MaxViT (Stage 2)

## Cell 1: Setup and Imports
```python
import os
import sys
import torch
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up plotting
plt.style.use('default')
sns.set_palette("husl")
%matplotlib inline

# Check if we're in the right directory
print("📁 Current working directory:", os.getcwd())
print("🔧 PyTorch version:", torch.__version__)
print("🖥️  CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("🚀 GPU:", torch.cuda.get_device_name(0))
    print("💾 GPU Memory:", f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Import our training modules
try:
    from jupyter_trainer import JupyterNeteaseTrainer
    from model_manager import ModelManager
    print("✅ All modules imported successfully!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure all training files are in the current directory")
```

## Cell 2: Configuration and Setup
```python
# ==========================================
# CONFIGURATION - MODIFY THESE PATHS
# ==========================================

# Dataset configuration
data_root = "/path/to/your/WFAS/dataset"  # 🔴 CHANGE THIS TO YOUR DATASET PATH
output_dir = "./netease_experiment_notebook"
model_cache_dir = "./model"

# Training configuration
STAGE1_EPOCHS = 10  # Stage 1 training epochs
STAGE2_EPOCHS = 10  # Stage 2 training epochs
BATCH_SIZE = 16     # Adjust based on your GPU memory
IMAGE_SIZE = 224    # Input image size

# Model configuration
STAGE1_MODEL = "convnext_base"      # convnext_base, convnext_small, convnext_large
STAGE2_MODEL = "maxvit_base_tf_224" # maxvit_base_tf_224, maxvit_small_tf_224

print("🔧 Training Configuration:")
print(f"  📁 Data root: {data_root}")
print(f"  💾 Output dir: {output_dir}")
print(f"  🗄️  Model cache: {model_cache_dir}")
print(f"  🏋️  Stage 1 epochs: {STAGE1_EPOCHS}")
print(f"  🏋️  Stage 2 epochs: {STAGE2_EPOCHS}")
print(f"  📦 Batch size: {BATCH_SIZE}")
print(f"  🖼️  Image size: {IMAGE_SIZE}")
print(f"  🤖 Stage 1 model: {STAGE1_MODEL}")
print(f"  🤖 Stage 2 model: {STAGE2_MODEL}")
```

## Cell 3: Initialize Trainer and Check Dataset
```python
# ==========================================
# INITIALIZE TRAINER
# ==========================================

# Create trainer instance
trainer = JupyterNeteaseTrainer(
    data_root=data_root,
    output_dir=output_dir,
    model_cache_dir=model_cache_dir
)

print("\n" + "="*60)
print("🔍 DATASET VERIFICATION")
print("="*60)

# Check if dataset exists
if not os.path.exists(data_root):
    print(f"❌ ERROR: Dataset not found at {data_root}")
    print("Please update the data_root path in the configuration cell above")
else:
    print(f"✅ Dataset found at {data_root}")
    
    # List dataset contents
    print(f"\n📂 Dataset structure:")
    try:
        for item in sorted(os.listdir(data_root)):
            item_path = os.path.join(data_root, item)
            if os.path.isdir(item_path):
                file_count = len([f for f in os.listdir(item_path) 
                                if os.path.isfile(os.path.join(item_path, f))])
                print(f"  📁 {item}/ ({file_count} items)")
            else:
                print(f"  📄 {item}")
    except Exception as e:
        print(f"  ⚠️  Could not list contents: {e}")
```

## Cell 4: Pre-download Models (Optional)
```python
# ==========================================
# PRE-DOWNLOAD MODELS (OPTIONAL)
# ==========================================

print("🔍 Checking model availability...")

# Check current cache status
model_manager = ModelManager(cache_base_dir=model_cache_dir)
cached_models = model_manager.list_cached_models()

if cached_models:
    print("📦 Currently cached models:")
    for stage, models in cached_models.items():
        if models:
            print(f"  {stage}: {', '.join(models)}")
else:
    print("📦 No models currently cached")

# Option to pre-download models
download_models = input(f"\n🤔 Pre-download models? This ensures offline training. (y/N): ").lower() == 'y'

if download_models:
    print(f"\n📥 Pre-downloading models...")
    trainer.download_models(stage1_model=STAGE1_MODEL, stage2_model=STAGE2_MODEL)
else:
    print("⏭️  Skipping pre-download. Models will be downloaded during training if needed.")
```

## Cell 5: Stage 1 Training (ConvNext)
```python
# ==========================================
# STAGE 1 TRAINING: ConvNext
# ==========================================

print("🚀 Starting Stage 1 Training")
print("="*60)
print(f"🤖 Model: {STAGE1_MODEL}")
print(f"🏋️  Epochs: {STAGE1_EPOCHS}")
print(f"📦 Batch size: {BATCH_SIZE}")
print("="*60)

# Record start time
stage1_start_time = datetime.now()

# Train Stage 1
try:
    soft_labels_path = trainer.train_stage1(
        epochs=STAGE1_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=2e-4,
        model_name=STAGE1_MODEL,
        image_size=IMAGE_SIZE,
        # Additional parameters can be added here
        weight_decay=0.01,
        optimizer='adamw',
        scheduler='cosine'
    )
    
    # Record completion
    stage1_end_time = datetime.now()
    stage1_duration = stage1_end_time - stage1_start_time
    
    print(f"\n🎉 Stage 1 Training Completed!")
    print(f"⏱️  Duration: {stage1_duration}")
    print(f"🏆 Best validation accuracy: {trainer.stage1_trainer.best_val_acc:.2f}%")
    print(f"📊 Best validation ACER: {trainer.stage1_trainer.best_val_acer:.4f}")
    print(f"💾 Soft labels saved to: {soft_labels_path}")
    
except Exception as e:
    print(f"❌ Stage 1 training failed: {e}")
    import traceback
    traceback.print_exc()
    soft_labels_path = None
```

## Cell 6: Stage 1 Results Analysis
```python
# ==========================================
# STAGE 1 RESULTS ANALYSIS
# ==========================================

if trainer.stage1_history and trainer.stage1_trainer:
    print("📊 Stage 1 Training Analysis")
    print("="*40)
    
    # Create DataFrame from training history
    df_stage1 = pd.DataFrame(trainer.stage1_history)
    
    # Display summary statistics
    print(f"📈 Training Summary:")
    print(f"  • Total epochs: {len(df_stage1)}")
    print(f"  • Best validation accuracy: {trainer.stage1_trainer.best_val_acc:.2f}%")
    print(f"  • Best validation ACER: {trainer.stage1_trainer.best_val_acer:.4f}")
    print(f"  • Final training accuracy: {df_stage1.iloc[-1]['train_acc']:.2f}%")
    print(f"  • Final validation accuracy: {df_stage1.iloc[-1]['val_acc']:.2f}%")
    
    # Plot training progress
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curves
    ax1.plot(df_stage1['epoch'], df_stage1['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(df_stage1['epoch'], df_stage1['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_title('Stage 1: Loss Curves', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curves
    ax2.plot(df_stage1['epoch'], df_stage1['train_acc'], 'b-', label='Train Acc', linewidth=2)
    ax2.plot(df_stage1['epoch'], df_stage1['val_acc'], 'r-', label='Val Acc', linewidth=2)
    ax2.set_title('Stage 1: Accuracy Curves', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # ACER curve
    ax3.plot(df_stage1['epoch'], df_stage1['val_acer'], 'g-', label='Val ACER', linewidth=2)
    ax3.set_title('Stage 1: ACER (Lower is Better)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('ACER')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Learning rate
    ax4.plot(df_stage1['epoch'], df_stage1['lr'], 'orange', label='Learning Rate', linewidth=2)
    ax4.set_title('Stage 1: Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Learning Rate')
    ax4.set_yscale('log')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Display training history table
    print(f"\n📋 Detailed Training History:")
    display(df_stage1.round(4))
    
else:
    print("❌ No Stage 1 training history available")
```

## Cell 7: Stage 1 Model Information
```python
# ==========================================
# STAGE 1 MODEL INFORMATION
# ==========================================

if trainer.stage1_trainer:
    print("🔍 Stage 1 Model Information")
    print("="*40)
    
    # Model architecture info
    model = trainer.stage1_trainer.model
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"🤖 Model: {STAGE1_MODEL}")
    print(f"📊 Total parameters: {total_params:,}")
    print(f"🎛️  Trainable parameters: {trainable_params:,}")
    print(f"💾 Model size: ~{total_params * 4 / 1e6:.1f} MB (float32)")
    
    # Training configuration
    print(f"\n⚙️  Training Configuration:")
    print(f"  • Optimizer: {trainer.stage1_trainer.config['optimizer']}")
    print(f"  • Learning rate: {trainer.stage1_trainer.config['learning_rate']}")
    print(f"  • Weight decay: {trainer.stage1_trainer.config['weight_decay']}")
    print(f"  • Scheduler: {trainer.stage1_trainer.config['scheduler']}")
    print(f"  • Batch size: {trainer.stage1_trainer.config['batch_size']}")
    
    # Checkpoint info
    checkpoint_dir = trainer.stage1_trainer.config['checkpoint_dir']
    best_checkpoint = os.path.join(checkpoint_dir, 'stage1_best.pth')
    latest_checkpoint = os.path.join(checkpoint_dir, 'stage1_latest.pth')
    
    print(f"\n💾 Saved Checkpoints:")
    if os.path.exists(best_checkpoint):
        size = os.path.getsize(best_checkpoint) / 1e6
        print(f"  ✅ Best checkpoint: {best_checkpoint} ({size:.1f} MB)")
    if os.path.exists(latest_checkpoint):
        size = os.path.getsize(latest_checkpoint) / 1e6
        print(f"  ✅ Latest checkpoint: {latest_checkpoint} ({size:.1f} MB)")
```

## Cell 8: Prepare for Stage 2
```python
# ==========================================
# PREPARE FOR STAGE 2
# ==========================================

print("🔄 Preparing for Stage 2 Training")
print("="*40)

# Check if soft labels exist
if soft_labels_path and os.path.exists(soft_labels_path):
    print(f"✅ Soft labels found: {soft_labels_path}")
    
    # Load and inspect soft labels
    try:
        soft_labels = torch.load(soft_labels_path, map_location='cpu')
        print(f"📊 Soft labels statistics:")
        print(f"  • Number of samples: {len(soft_labels)}")
        print(f"  • Sample paths: {list(soft_labels.keys())[:3]}...")
        
        # Analyze soft label distribution
        import numpy as np
        probs = np.array(list(soft_labels.values()))
        live_probs = probs[:, 1]  # Probability of live class
        
        print(f"  • Live probability stats:")
        print(f"    - Mean: {live_probs.mean():.3f}")
        print(f"    - Std: {live_probs.std():.3f}")
        print(f"    - Min: {live_probs.min():.3f}")
        print(f"    - Max: {live_probs.max():.3f}")
        
        # Plot distribution
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        plt.hist(live_probs, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Distribution of Live Probabilities')
        plt.xlabel('Live Probability')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.hist(probs[:, 0], bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
        plt.title('Distribution of Spoof Probabilities')
        plt.xlabel('Spoof Probability')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"⚠️  Could not analyze soft labels: {e}")
        
else:
    print(f"❌ Soft labels not found!")
    if not soft_labels_path:
        print("Stage 1 training may have failed.")
    else:
        print(f"Expected path: {soft_labels_path}")
    
    # Ask user what to do
    action = input("\n🤔 What would you like to do?\n"
                  "1. Continue with Stage 2 anyway (will fail)\n"
                  "2. Stop here and fix Stage 1\n"
                  "3. Provide custom soft labels path\n"
                  "Enter choice (1/2/3): ")
    
    if action == "2":
        print("🛑 Stopping execution. Please fix Stage 1 training first.")
        # You can stop execution here if in a script
    elif action == "3":
        custom_path = input("Enter path to soft labels file: ")
        if os.path.exists(custom_path):
            soft_labels_path = custom_path
            print(f"✅ Using custom soft labels: {soft_labels_path}")
        else:
            print(f"❌ File not found: {custom_path}")
```

## Cell 9: Stage 2 Training (MaxViT)
```python
# ==========================================
# STAGE 2 TRAINING: MaxViT
# ==========================================

if soft_labels_path and os.path.exists(soft_labels_path):
    print("🚀 Starting Stage 2 Training")
    print("="*60)
    print(f"🤖 Model: {STAGE2_MODEL}")
    print(f"🏋️  Epochs: {STAGE2_EPOCHS}")
    print(f"📦 Batch size: {BATCH_SIZE}")
    print(f"🔮 Soft labels: {soft_labels_path}")
    print("="*60)
    
    # Record start time
    stage2_start_time = datetime.now()
    
    # Train Stage 2
    try:
        test_acc, test_acer = trainer.train_stage2(
            soft_labels_path=soft_labels_path,
            epochs=STAGE2_EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=1e-4,
            model_name=STAGE2_MODEL,
            image_size=IMAGE_SIZE,
            # Stage 2 specific parameters
            feature_dim=512,
            focal_alpha=1.0,
            focal_gamma=2.0,
            triplet_margin=0.3,
            kd_temperature=3.0,
            focal_weight=1.0,
            triplet_weight=0.5,
            kd_weight=1.0,
            use_mixup_cutmix=True,
            mixup_alpha=0.2,
            cutmix_alpha=1.0
        )
        
        # Record completion
        stage2_end_time = datetime.now()
        stage2_duration = stage2_end_time - stage2_start_time
        total_duration = stage2_end_time - stage1_start_time
        
        print(f"\n🎉 Stage 2 Training Completed!")
        print(f"⏱️  Stage 2 duration: {stage2_duration}")
        print(f"⏱️  Total pipeline duration: {total_duration}")
        print(f"🏆 Best validation accuracy: {trainer.stage2_trainer.best_val_acc:.2f}%")
        print(f"📊 Best validation ACER: {trainer.stage2_trainer.best_val_acer:.4f}")
        print(f"🎯 Final test accuracy: {test_acc:.2f}%")
        print(f"📈 Final test ACER: {test_acer:.4f}")
        
    except Exception as e:
        print(f"❌ Stage 2 training failed: {e}")
        import traceback
        traceback.print_exc()
        test_acc, test_acer = None, None
        
else:
    print("❌ Cannot start Stage 2: No valid soft labels found")
    test_acc, test_acer = None, None
```

## Cell 10: Stage 2 Results Analysis
```python
# ==========================================
# STAGE 2 RESULTS ANALYSIS
# ==========================================

if trainer.stage2_history and trainer.stage2_trainer:
    print("📊 Stage 2 Training Analysis")
    print("="*40)
    
    # Create DataFrame from training history
    df_stage2 = pd.DataFrame(trainer.stage2_history)
    
    # Display summary statistics
    print(f"📈 Training Summary:")
    print(f"  • Total epochs: {len(df_stage2)}")
    print(f"  • Best validation accuracy: {trainer.stage2_trainer.best_val_acc:.2f}%")
    print(f"  • Best validation ACER: {trainer.stage2_trainer.best_val_acer:.4f}")
    print(f"  • Final test accuracy: {test_acc:.2f}%" if test_acc else "  • Test accuracy: N/A")
    print(f"  • Final test ACER: {test_acer:.4f}" if test_acer else "  • Test ACER: N/A")
    
    # Plot training progress with loss components
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss components
    ax1.plot(df_stage2['epoch'], df_stage2['train_loss'], 'black', label='Total Loss', linewidth=3)
    ax1.plot(df_stage2['epoch'], df_stage2['focal_loss'], 'red', label='Focal Loss', linewidth=2)
    ax1.plot(df_stage2['epoch'], df_stage2['triplet_loss'], 'blue', label='Triplet Loss', linewidth=2)
    ax1.plot(df_stage2['epoch'], df_stage2['kd_loss'], 'green', label='KD Loss', linewidth=2)
    ax1.set_title('Stage 2: Loss Components', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curves
    ax2.plot(df_stage2['epoch'], df_stage2['train_acc'], 'b-', label='Train Acc', linewidth=2)
    ax2.plot(df_stage2['epoch'], df_stage2['val_acc'], 'r-', label='Val Acc', linewidth=2)
    ax2.set_title('Stage 2: Accuracy Curves', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # ACER curve
    ax3.plot(df_stage2['epoch'], df_stage2['val_acer'], 'g-', label='Val ACER', linewidth=2)
    ax3.set_title('Stage 2: ACER (Lower is Better)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('ACER')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Learning rate
    ax4.plot(df_stage2['epoch'], df_stage2['lr'], 'orange', label='Learning Rate', linewidth=2)
    ax4.set_title('Stage 2: Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Learning Rate')
    ax4.set_yscale('log')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Loss component analysis
    print(f"\n🔍 Loss Component Analysis (Final Epoch):")
    final_epoch = df_stage2.iloc[-1]
    total_loss = final_epoch['train_loss']
    focal_loss = final_epoch['focal_loss']
    triplet_loss = final_epoch['triplet_loss']
    kd_loss = final_epoch['kd_loss']
    
    print(f"  • Total Loss: {total_loss:.4f}")
    print(f"  • Focal Loss: {focal_loss:.4f} ({focal_loss/total_loss*100:.1f}%)")
    print(f"  • Triplet Loss: {triplet_loss:.4f} ({triplet_loss/total_loss*100:.1f}%)")
    print(f"  • KD Loss: {kd_loss:.4f} ({kd_loss/total_loss*100:.1f}%)")
    
    # Display training history table
    print(f"\n📋 Detailed Training History:")
    display(df_stage2.round(4))
    
else:
    print("❌ No Stage 2 training history available")
```

## Cell 11: Complete Pipeline Summary
```python
# ==========================================
# COMPLETE PIPELINE SUMMARY
# ==========================================

print("🎯 NetEase Two-Stage Training Summary")
print("="*60)

# Overall results
if trainer.stage1_trainer and trainer.stage2_trainer:
    print("✅ TRAINING COMPLETED SUCCESSFULLY")
    print("\n📊 Final Results:")
    print(f"  🥇 Stage 1 (ConvNext):")
    print(f"     • Best Val Accuracy: {trainer.stage1_trainer.best_val_acc:.2f}%")
    print(f"     • Best Val ACER: {trainer.stage1_trainer.best_val_acer:.4f}")
    
    print(f"  🥇 Stage 2 (MaxViT):")
    print(f"     • Best Val Accuracy: {trainer.stage2_trainer.best_val_acc:.2f}%")
    print(f"     • Best Val ACER: {trainer.stage2_trainer.best_val_acer:.4f}")
    if test_acc and test_acer:
        print(f"     • Test Accuracy: {test_acc:.2f}%")
        print(f"     • Test ACER: {test_acer:.4f}")
    
    # Performance improvement
    stage1_acc = trainer.stage1_trainer.best_val_acc
    stage2_acc = trainer.stage2_trainer.best_val_acc
    improvement = stage2_acc - stage1_acc
    print(f"\n📈 Performance Improvement:")
    print(f"  • Accuracy improvement: {improvement:+.2f}%")
    print(f"  • Relative improvement: {improvement/stage1_acc*100:+.1f}%")
    
else:
    print("❌ Training incomplete or failed")

# Training configuration summary
print(f"\n⚙️  Training Configuration:")
print(f"  • Stage 1 Model: {STAGE1_MODEL}")
print(f"  • Stage 2 Model: {STAGE2_MODEL}")
print(f"  • Stage 1 Epochs: {STAGE1_EPOCHS}")
print(f"  • Stage 2 Epochs: {STAGE2_EPOCHS}")
print(f"  • Batch Size: {BATCH_SIZE}")
print(f"  • Image Size: {IMAGE_SIZE}")

# File locations
print(f"\n📁 Output Files:")
print(f"  • Output directory: {output_dir}")
print(f"  • Model cache: {model_cache_dir}")
if soft_labels_path:
    print(f"  • Soft labels: {soft_labels_path}")

# Timing information
if 'stage1_duration' in locals() and 'stage2_duration' in locals():
    print(f"\n⏱️  Timing:")
    print(f"  • Stage 1 duration: {stage1_duration}")
    print(f"  • Stage 2 duration: {stage2_duration}")
    print(f"  • Total duration: {stage1_duration + stage2_duration}")

print(f"\n🎉 Notebook execution completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
```

## Cell 12: Save Results and Cleanup
```python
# ==========================================
# SAVE RESULTS AND CLEANUP
# ==========================================

print("💾 Saving Results and Cleanup")
print("="*40)

# Compile all results
results_summary = {
    'experiment_info': {
        'timestamp': datetime.now().isoformat(),
        'stage1_model': STAGE1_MODEL,
        'stage2_model': STAGE2_MODEL,
        'stage1_epochs': STAGE1_EPOCHS,
        'stage2_epochs': STAGE2_EPOCHS,
        'batch_size': BATCH_SIZE,
        'image_size': IMAGE_SIZE
    }
}

# Add Stage 1 results
if trainer.stage1_trainer:
    results_summary['stage1'] = {
        'best_val_acc': trainer.stage1_trainer.best_val_acc,
        'best_val_acer': trainer.stage1_trainer.best_val_acer,
        'training_history': trainer.stage1_history
    }

# Add Stage 2 results
if trainer.stage2_trainer:
    results_summary['stage2'] = {
        'best_val_acc': trainer.stage2_trainer.best_val_acc,
        'best_val_acer': trainer.stage2_trainer.best_val_acer,
        'test_acc': test_acc,
        'test_acer': test_acer,
        'training_history': trainer.stage2_history
    }

# Save results
results_file = os.path.join(output_dir, 'notebook_results.json')
try:
    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"✅ Results saved to: {results_file}")
except Exception as e:
    print(f"⚠️  Could not save results: {e}")

# List all generated files
print(f"\n📋 Generated Files:")
if os.path.exists(output_dir):
    for root, dirs, files in os.walk(output_dir):
        level = root.replace(output_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}📁 {os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            file_path = os.path.join(root, file)
            size = os.path.getsize(file_path) / 1024  # KB
            print(f"{subindent}📄 {file} ({size:.1f} KB)")

# Memory cleanup
print(f"\n🧹 Cleanup:")
if 'trainer' in locals():
    # Clear GPU memory if used
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("✅ GPU memory cleared")
    print("✅ Training objects available for further analysis")

print(f"\n🎯 Training Complete!")
print("You can now:")
print("  • Analyze the results using trainer.plot_complete_history()")
print("  • Load checkpoints with trainer.load_checkpoint(stage, 'best')")
print("  • Access training history via trainer.stage1_history and trainer.stage2_history")
print("  • Use the trained models for inference")
```

## Cell 13: Optional - Additional Analysis and Visualization
```python
# ==========================================
# OPTIONAL: ADDITIONAL ANALYSIS
# ==========================================

print("📊 Additional Analysis and Visualizations")
print("="*50)

# Combined training curves comparison
if trainer.stage1_history and trainer.stage2_history:
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Prepare data
    df1 = pd.DataFrame(trainer.stage1_history)
    df2 = pd.DataFrame(trainer.stage2_history)
    
    # Adjust Stage 2 epochs to continue from Stage 1
    df2_adjusted = df2.copy()
    df2_adjusted['epoch'] = df2_adjusted['epoch'] + STAGE1_EPOCHS
    
    # Combined accuracy plot
    ax1.plot(df1['epoch'], df1['train_acc'], 'b-', label='Stage 1 Train', linewidth=2, alpha=0.8)
    ax1.plot(df1['epoch'], df1['val_acc'], 'b--', label='Stage 1 Val', linewidth=2, alpha=0.8)
    ax1.plot(df2_adjusted['epoch'], df2_adjusted['train_acc'], 'r-', label='Stage 2 Train', linewidth=2, alpha=0.8)
    ax1.plot(df2_adjusted['epoch'], df2_adjusted['val_acc'], 'r--', label='Stage 2 Val', linewidth=2, alpha=0.8)
    ax1.axvline(x=STAGE1_EPOCHS, color='gray', linestyle=':', alpha=0.7, label='Stage Transition')
    ax1.set_title('Complete Pipeline: Accuracy Progression', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Combined ACER plot
    ax2.plot(df1['epoch'], df1['val_acer'], 'b-', label='Stage 1 ACER', linewidth=2, alpha=0.8)
    ax2.plot(df2_adjusted['epoch'], df2_adjusted['val_acer'], 'r-', label='Stage 2 ACER', linewidth=2, alpha=0.8)
    ax2.axvline(x=STAGE1_EPOCHS, color='gray', linestyle=':', alpha=0.7, label='Stage Transition')
    ax2.set_title('Complete Pipeline: ACER Progression', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('ACER (Lower is Better)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Stage 2 loss components breakdown
    ax3.stackplot(df2['epoch'], df2['focal_loss'], df2['triplet_loss'], df2['kd_loss'], 
                  labels=['Focal Loss', 'Triplet Loss', 'KD Loss'], alpha=0.7)
    ax3.set_title('Stage 2: Loss Components Breakdown', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss Value')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # Performance comparison bar chart
    metrics = ['Val Accuracy', 'Val ACER']
    stage1_vals = [trainer.stage1_trainer.best_val_acc, trainer.stage1_trainer.best_val_acer * 100]  # Scale ACER for visibility
    stage2_vals = [trainer.stage2_trainer.best_val_acc, trainer.stage2_trainer.best_val_acer * 100]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax4.bar(x - width/2, stage1_vals, width, label='Stage 1', alpha=0.8, color='skyblue')
    ax4.bar(x + width/2, stage2_vals, width, label='Stage 2', alpha=0.8, color='lightcoral')
    ax4.set_title('Stage Comparison: Best Results', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Value (%)')
    ax4.set_xticks(x)
    ax4.set_xticklabels(['Accuracy (%)', 'ACER × 100'])
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (s1, s2) in enumerate(zip(stage1_vals, stage2_vals)):
        ax4.text(i - width/2, s1 + 0.5, f'{s1:.1f}', ha='center', va='bottom', fontweight='bold')
        ax4.text(i + width/2, s2 + 0.5, f'{s2:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Performance metrics table
    print("\n📋 Performance Comparison Table:")
    comparison_df = pd.DataFrame({
        'Metric': ['Best Validation Accuracy (%)', 'Best Validation ACER', 'Final Test Accuracy (%)', 'Final Test ACER'],
        'Stage 1': [trainer.stage1_trainer.best_val_acc, trainer.stage1_trainer.best_val_acer, 'N/A', 'N/A'],
        'Stage 2': [trainer.stage2_trainer.best_val_acc, trainer.stage2_trainer.best_val_acer, 
                   test_acc if test_acc else 'N/A', test_acer if test_acer else 'N/A'],
        'Improvement': [
            f"{trainer.stage2_trainer.best_val_acc - trainer.stage1_trainer.best_val_acc:+.2f}",
            f"{trainer.stage2_trainer.best_val_acer - trainer.stage1_trainer.best_val_acer:+.4f}",
            'N/A', 'N/A'
        ]
    })
    display(comparison_df)

else:
    print("⚠️  Cannot create comparison plots - missing training history")
```

## Cell 14: Model Inference Example (Optional)
```python
# ==========================================
# OPTIONAL: MODEL INFERENCE EXAMPLE
# ==========================================

print("🔮 Model Inference Example")
print("="*40)

if trainer.stage2_trainer:
    print("Setting up inference example...")
    
    # Load the best Stage 2 model
    best_checkpoint = trainer.load_checkpoint(2, 'best')
    
    if best_checkpoint:
        print("✅ Best Stage 2 model loaded successfully")
        
        # Example inference function
        def predict_single_image(model, image_path, device, transform=None):
            """
            Predict on a single image
            
            Args:
                model: Trained model
                image_path: Path to image
                device: Device to run inference on
                transform: Image transformations
            
            Returns:
                prediction, confidence
            """
            from PIL import Image
            import torchvision.transforms as transforms
            
            # Default transform if none provided
            if transform is None:
                transform = transforms.Compose([
                    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
                ])
            
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            # Inference
            model.eval()
            with torch.no_grad():
                logits, features = model(image_tensor)
                probs = torch.softmax(logits, dim=1)
                predicted_class = torch.argmax(probs, dim=1).item()
                confidence = probs[0, predicted_class].item()
            
            class_names = ['Spoof', 'Live']
            return class_names[predicted_class], confidence, probs[0].cpu().numpy()
        
        print("🎯 Inference function defined")
        print("Usage example:")
        print("  prediction, confidence, probs = predict_single_image(")
        print("      trainer.stage2_trainer.model, '/path/to/image.jpg', trainer.device)")
        print("  print(f'Prediction: {prediction} (confidence: {confidence:.3f})')")
        
        # Show model summary
        print(f"\n🤖 Model Summary:")
        print(f"  • Architecture: {STAGE2_MODEL}")
        print(f"  • Input size: {IMAGE_SIZE}x{IMAGE_SIZE}")
        print(f"  • Output classes: 2 (Spoof, Live)")
        print(f"  • Feature dimension: {trainer.stage2_trainer.config['feature_dim']}")
        
    else:
        print("❌ Could not load best checkpoint for inference")
else:
    print("❌ No trained Stage 2 model available for inference")
```

## Cell 15: Export and Save Final Models
```python
# ==========================================
# EXPORT AND SAVE FINAL MODELS
# ==========================================

print("📦 Exporting Final Models")
print("="*40)

export_dir = os.path.join(output_dir, 'exported_models')
os.makedirs(export_dir, exist_ok=True)

# Export Stage 1 model
if trainer.stage1_trainer:
    try:
        stage1_export_path = os.path.join(export_dir, f'stage1_{STAGE1_MODEL}_final.pth')
        
        # Create export package
        export_package = {
            'model_state_dict': trainer.stage1_trainer.model.state_dict(),
            'config': trainer.stage1_trainer.config,
            'best_val_acc': trainer.stage1_trainer.best_val_acc,
            'best_val_acer': trainer.stage1_trainer.best_val_acer,
            'model_name': STAGE1_MODEL,
            'stage': 1,
            'training_epochs': STAGE1_EPOCHS,
            'export_timestamp': datetime.now().isoformat()
        }
        
        torch.save(export_package, stage1_export_path)
        size = os.path.getsize(stage1_export_path) / 1e6
        print(f"✅ Stage 1 model exported: {stage1_export_path} ({size:.1f} MB)")
        
    except Exception as e:
        print(f"❌ Failed to export Stage 1 model: {e}")

# Export Stage 2 model
if trainer.stage2_trainer:
    try:
        stage2_export_path = os.path.join(export_dir, f'stage2_{STAGE2_MODEL}_final.pth')
        
        # Create export package
        export_package = {
            'model_state_dict': trainer.stage2_trainer.model.state_dict(),
            'config': trainer.stage2_trainer.config,
            'best_val_acc': trainer.stage2_trainer.best_val_acc,
            'best_val_acer': trainer.stage2_trainer.best_val_acer,
            'test_acc': test_acc,
            'test_acer': test_acer,
            'model_name': STAGE2_MODEL,
            'stage': 2,
            'training_epochs': STAGE2_EPOCHS,
            'soft_labels_path': soft_labels_path,
            'export_timestamp': datetime.now().isoformat()
        }
        
        torch.save(export_package, stage2_export_path)
        size = os.path.getsize(stage2_export_path) / 1e6
        print(f"✅ Stage 2 model exported: {stage2_export_path} ({size:.1f} MB)")
        
    except Exception as e:
        print(f"❌ Failed to export Stage 2 model: {e}")

# Create model loading script
loading_script = f'''
# Model Loading Script
# Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

import torch
from model_manager import ModelManager

def load_trained_model(export_path, device='cpu'):
    """Load exported model"""
    checkpoint = torch.load(export_path, map_location=device)
    
    # Recreate model using ModelManager
    manager = ModelManager()
    model = manager.load_or_download_model(
        stage=f"stage{{checkpoint['stage']}}",
        model_name=checkpoint['model_name'],
        num_classes=2,
        device=device
    )
    
    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded {{checkpoint['model_name']}} (Stage {{checkpoint['stage']}})")
    print(f"Best Val Acc: {{checkpoint['best_val_acc']:.2f}}%")
    print(f"Best Val ACER: {{checkpoint['best_val_acer']:.4f}}")
    
    return model, checkpoint

# Usage examples:
# stage1_model, stage1_info = load_trained_model('{stage1_export_path if 'stage1_export_path' in locals() else 'stage1_model.pth'}')
# stage2_model, stage2_info = load_trained_model('{stage2_export_path if 'stage2_export_path' in locals() else 'stage2_model.pth'}')
'''

script_path = os.path.join(export_dir, 'load_models.py')
with open(script_path, 'w') as f:
    f.write(loading_script)

print(f"✅ Model loading script saved: {script_path}")

# Summary of exports
print(f"\n📋 Export Summary:")
if os.path.exists(export_dir):
    for file in os.listdir(export_dir):
        file_path = os.path.join(export_dir, file)
        if os.path.isfile(file_path):
            size = os.path.getsize(file_path) / 1e6
            print(f"  📄 {file} ({size:.1f} MB)")

print(f"\n🎉 All exports completed successfully!")
print(f"Models and scripts saved in: {export_dir}")
```

## Cell 16: Final Cleanup and Summary
```python
# ==========================================
# FINAL CLEANUP AND SUMMARY
# ==========================================

print("🏁 Final Summary and Cleanup")
print("="*60)

# Training summary
total_epochs = STAGE1_EPOCHS + STAGE2_EPOCHS
print(f"✅ NetEase Two-Stage Training Completed Successfully!")
print(f"\n📊 Training Summary:")
print(f"  • Total epochs trained: {total_epochs}")
print(f"  • Stage 1 ({STAGE1_MODEL}): {STAGE1_EPOCHS} epochs")
print(f"  • Stage 2 ({STAGE2_MODEL}): {STAGE2_EPOCHS} epochs")
print(f"  • Batch size used: {BATCH_SIZE}")

if trainer.stage1_trainer and trainer.stage2_trainer:
    print(f"\n🏆 Best Results:")
    print(f"  • Stage 1 Best Val Acc: {trainer.stage1_trainer.best_val_acc:.2f}%")
    print(f"  • Stage 2 Best Val Acc: {trainer.stage2_trainer.best_val_acc:.2f}%")
    print(f"  • Final Test Accuracy: {test_acc:.2f}%" if test_acc else "  • Test Accuracy: Not available")
    print(f"  • Final Test ACER: {test_acer:.4f}" if test_acer else "  • Test ACER: Not available")

# File system summary
print(f"\n📁 Generated Files and Directories:")
print(f"  • Main output: {output_dir}")
print(f"  • Model cache: {model_cache_dir}")
print(f"  • Exported models: {os.path.join(output_dir, 'exported_models')}")

# Memory cleanup
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"\n🧹 GPU memory cleared")

print(f"\n💡 Next Steps:")
print(f"  1. Analyze results using the generated plots and metrics")
print(f"  2. Use exported models for inference on new data")
print(f"  3. Experiment with different hyperparameters if needed")
print(f"  4. Consider longer training for better performance")

print(f"\n🎯 Notebook completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Happy experimenting! 🚀")
```