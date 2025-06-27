"""
Debug script to compare Script 1 and Script 2 models
Helps identify why Script 2 is not converging while Script 1 is training properly
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
import timm

# Import your model classes
# from stage2_training_v1 import MaxViTWithFeatures as MaxViTWithFeatures_V1
# from stage2_training_v2 import MaxViTWithFeatures as MaxViTWithFeatures_V2, ModelManager

# For standalone testing, we'll define minimal versions here
class MaxViTWithFeatures_V1(nn.Module):
    """Script 1 version - Simple"""
    def __init__(self, model_name='maxvit_base_tf_224', num_classes=2, feature_dim=512):
        super(MaxViTWithFeatures_V1, self).__init__()
        
        # Load MaxViT backbone
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            backbone_output = self.backbone(dummy_input)
            backbone_dim = backbone_output.shape[1]
        
        # Feature projection
        self.feature_projection = nn.Sequential(
            nn.Linear(backbone_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Classifier
        self.classifier = nn.Linear(feature_dim, num_classes)
    
    def forward(self, x):
        # Extract features
        backbone_features = self.backbone(x)
        features = self.feature_projection(backbone_features)
        logits = self.classifier(features)
        
        return logits, features


def debug_models(model1_path=None, model2_path=None, device='cuda'):
    """
    Compare two models comprehensively
    
    Args:
        model1_path: Path to Script 1 checkpoint (optional)
        model2_path: Path to Script 2 checkpoint (optional)
        device: Device to run tests on
    """
    
    print("="*80)
    print("MODEL COMPARISON DEBUG SCRIPT")
    print("="*80)
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Create or load models
    print("\n1. CREATING/LOADING MODELS")
    print("-"*40)
    
    # For testing, create new models - replace with your actual model loading
    model1 = MaxViTWithFeatures_V1(model_name='maxvit_small_tf_224', num_classes=2, feature_dim=512)
    model1 = model1.to(device)
    
    # For model2, you would load your Script 2 version
    # model2 = MaxViTWithFeatures_V2(...)
    # For now, create another instance for comparison
    model2 = MaxViTWithFeatures_V1(model_name='maxvit_small_tf_224', num_classes=2, feature_dim=512)
    model2 = model2.to(device)
    
    # Load checkpoints if provided
    if model1_path:
        checkpoint1 = torch.load(model1_path, map_location=device)
        model1.load_state_dict(checkpoint1['model_state_dict'])
        print(f"Loaded Script 1 model from: {model1_path}")
    
    if model2_path:
        checkpoint2 = torch.load(model2_path, map_location=device)
        model2.load_state_dict(checkpoint2['model_state_dict'])
        print(f"Loaded Script 2 model from: {model2_path}")
    
    # 2. MODEL ARCHITECTURE COMPARISON
    print("\n2. MODEL ARCHITECTURE COMPARISON")
    print("-"*40)
    
    # Count parameters
    def count_parameters(model):
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = total - trainable
        return total, trainable, frozen
    
    total1, trainable1, frozen1 = count_parameters(model1)
    total2, trainable2, frozen2 = count_parameters(model2)
    
    print(f"Script 1 Model:")
    print(f"  Total params: {total1:,}")
    print(f"  Trainable params: {trainable1:,}")
    print(f"  Frozen params: {frozen1:,}")
    
    print(f"\nScript 2 Model:")
    print(f"  Total params: {total2:,}")
    print(f"  Trainable params: {trainable2:,}")
    print(f"  Frozen params: {frozen2:,}")
    
    print(f"\nDifference:")
    print(f"  Total: {total2 - total1:,}")
    print(f"  Trainable: {trainable2 - trainable1:,}")
    print(f"  Frozen: {frozen2 - frozen1:,}")
    
    # 3. LAYER-BY-LAYER COMPARISON
    print("\n3. LAYER-BY-LAYER PARAMETER COMPARISON")
    print("-"*40)
    
    # Get all named parameters
    params1 = dict(model1.named_parameters())
    params2 = dict(model2.named_parameters())
    
    # Compare layer names
    keys1 = set(params1.keys())
    keys2 = set(params2.keys())
    
    print(f"Layers only in Script 1: {keys1 - keys2}")
    print(f"Layers only in Script 2: {keys2 - keys1}")
    
    # Check requires_grad for each layer
    print("\nRequires_grad status for key layers:")
    for name in ['backbone', 'feature_projection', 'classifier']:
        print(f"\n{name}:")
        for key in params1.keys():
            if name in key:
                grad1 = params1[key].requires_grad
                grad2 = params2.get(key, torch.tensor([0])).requires_grad if key in params2 else None
                if grad2 is not None and grad1 != grad2:
                    print(f"  {key}: Script1={grad1}, Script2={grad2} ⚠️")
                break
    
    # 4. FORWARD PASS COMPARISON
    print("\n4. FORWARD PASS COMPARISON")
    print("-"*40)
    
    # Set both models to eval mode for consistent comparison
    model1.eval()
    model2.eval()
    
    # Create identical input
    torch.manual_seed(42)
    test_input = torch.randn(4, 3, 224, 224).to(device)
    
    # Forward pass
    with torch.no_grad():
        logits1, features1 = model1(test_input)
        logits2, features2 = model2(test_input)
    
    print(f"Input shape: {test_input.shape}")
    print(f"\nScript 1 outputs:")
    print(f"  Logits shape: {logits1.shape}")
    print(f"  Features shape: {features1.shape}")
    print(f"  Logits mean: {logits1.mean().item():.6f}, std: {logits1.std().item():.6f}")
    print(f"  Features mean: {features1.mean().item():.6f}, std: {features1.std().item():.6f}")
    
    print(f"\nScript 2 outputs:")
    print(f"  Logits shape: {logits2.shape}")
    print(f"  Features shape: {features2.shape}")
    print(f"  Logits mean: {logits2.mean().item():.6f}, std: {logits2.std().item():.6f}")
    print(f"  Features mean: {features2.mean().item():.6f}, std: {features2.std().item():.6f}")
    
    # Compare outputs
    if logits1.shape == logits2.shape:
        logit_diff = torch.abs(logits1 - logits2).mean().item()
        feature_diff = torch.abs(features1 - features2).mean().item()
        print(f"\nOutput differences:")
        print(f"  Mean absolute logit difference: {logit_diff:.6f}")
        print(f"  Mean absolute feature difference: {feature_diff:.6f}")
    
    # 5. GRADIENT FLOW TEST
    print("\n5. GRADIENT FLOW TEST")
    print("-"*40)
    
    # Set models to train mode
    model1.train()
    model2.train()
    
    # Create dummy labels
    labels = torch.randint(0, 2, (4,)).to(device)
    
    # Test gradient flow for both models
    for model_name, model in [("Script 1", model1), ("Script 2", model2)]:
        print(f"\n{model_name} Gradient Flow:")
        
        # Zero gradients
        model.zero_grad()
        
        # Forward pass
        logits, features = model(test_input)
        
        # Simple loss
        loss = F.cross_entropy(logits, labels)
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        grad_stats = OrderedDict()
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_mean = param.grad.mean().item()
                grad_std = param.grad.std().item()
                grad_max = param.grad.abs().max().item()
                
                # Group by component
                component = name.split('.')[0]
                if component not in grad_stats:
                    grad_stats[component] = {
                        'layers': 0,
                        'with_grad': 0,
                        'mean_grad': [],
                        'max_grad': []
                    }
                
                grad_stats[component]['layers'] += 1
                grad_stats[component]['with_grad'] += 1
                grad_stats[component]['mean_grad'].append(abs(grad_mean))
                grad_stats[component]['max_grad'].append(grad_max)
            else:
                component = name.split('.')[0]
                if component not in grad_stats:
                    grad_stats[component] = {
                        'layers': 0,
                        'with_grad': 0,
                        'mean_grad': [],
                        'max_grad': []
                    }
                grad_stats[component]['layers'] += 1
        
        # Print summary
        for component, stats in grad_stats.items():
            avg_mean_grad = np.mean(stats['mean_grad']) if stats['mean_grad'] else 0
            avg_max_grad = np.mean(stats['max_grad']) if stats['max_grad'] else 0
            print(f"  {component}: {stats['with_grad']}/{stats['layers']} layers have gradients")
            if stats['with_grad'] > 0:
                print(f"    Avg mean grad: {avg_mean_grad:.6f}, Avg max grad: {avg_max_grad:.6f}")
    
    # 6. INTERMEDIATE ACTIVATIONS
    print("\n6. INTERMEDIATE ACTIVATIONS CHECK")
    print("-"*40)
    
    # Hook to capture intermediate activations
    activations = {'model1': {}, 'model2': {}}
    
    def create_hook(model_name, layer_name):
        def hook(module, input, output):
            activations[model_name][layer_name] = {
                'shape': output.shape if hasattr(output, 'shape') else str(type(output)),
                'mean': output.mean().item() if hasattr(output, 'mean') else 'N/A',
                'std': output.std().item() if hasattr(output, 'std') else 'N/A'
            }
        return hook
    
    # Register hooks
    hooks = []
    
    # For model1
    if hasattr(model1, 'backbone'):
        hooks.append(model1.backbone.register_forward_hook(create_hook('model1', 'backbone')))
    if hasattr(model1, 'feature_projection'):
        hooks.append(model1.feature_projection.register_forward_hook(create_hook('model1', 'feature_projection')))
    
    # For model2
    if hasattr(model2, 'backbone'):
        hooks.append(model2.backbone.register_forward_hook(create_hook('model2', 'backbone')))
    if hasattr(model2, 'feature_projection'):
        hooks.append(model2.feature_projection.register_forward_hook(create_hook('model2', 'feature_projection')))
    
    # Forward pass to trigger hooks
    with torch.no_grad():
        _ = model1(test_input)
        _ = model2(test_input)
    
    # Print activation statistics
    for model_name in ['model1', 'model2']:
        print(f"\n{model_name} activations:")
        for layer_name, stats in activations[model_name].items():
            print(f"  {layer_name}:")
            print(f"    Shape: {stats['shape']}")
            print(f"    Mean: {stats['mean']}")
            print(f"    Std: {stats['std']}")
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # 7. TRAINING BEHAVIOR TEST
    print("\n7. MINI TRAINING TEST")
    print("-"*40)
    
    # Quick training test
    optimizer1 = torch.optim.Adam(model1.parameters(), lr=1e-4)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-4)
    
    model1.train()
    model2.train()
    
    initial_losses = {'model1': [], 'model2': []}
    
    print("Running 10 mini-batches...")
    for i in range(10):
        # Create random batch
        batch_input = torch.randn(8, 3, 224, 224).to(device)
        batch_labels = torch.randint(0, 2, (8,)).to(device)
        
        # Train model1
        optimizer1.zero_grad()
        logits1, features1 = model1(batch_input)
        loss1 = F.cross_entropy(logits1, batch_labels)
        loss1.backward()
        optimizer1.step()
        initial_losses['model1'].append(loss1.item())
        
        # Train model2
        optimizer2.zero_grad()
        logits2, features2 = model2(batch_input)
        loss2 = F.cross_entropy(logits2, batch_labels)
        loss2.backward()
        optimizer2.step()
        initial_losses['model2'].append(loss2.item())
    
    print(f"\nModel 1 losses: {initial_losses['model1']}")
    print(f"Model 2 losses: {initial_losses['model2']}")
    
    print(f"\nModel 1: First loss: {initial_losses['model1'][0]:.4f}, Last loss: {initial_losses['model1'][-1]:.4f}")
    print(f"Model 2: First loss: {initial_losses['model2'][0]:.4f}, Last loss: {initial_losses['model2'][-1]:.4f}")
    
    # Check if losses are decreasing
    model1_decreasing = initial_losses['model1'][-1] < initial_losses['model1'][0]
    model2_decreasing = initial_losses['model2'][-1] < initial_losses['model2'][0]
    
    print(f"\nModel 1 loss decreasing: {model1_decreasing} {'✓' if model1_decreasing else '✗'}")
    print(f"Model 2 loss decreasing: {model2_decreasing} {'✓' if model2_decreasing else '✗'}")
    
    print("\n" + "="*80)
    print("DEBUG COMPLETE")
    print("="*80)


def check_backbone_differences(model1, model2):
    """Additional check specifically for backbone differences"""
    print("\nBACKBONE DETAILED COMPARISON")
    print("-"*40)
    
    if hasattr(model1, 'backbone') and hasattr(model2, 'backbone'):
        print("Backbone 1 type:", type(model1.backbone))
        print("Backbone 2 type:", type(model2.backbone))
        
        # Check if backbone is in eval vs train mode
        print(f"\nBackbone 1 training mode: {model1.backbone.training}")
        print(f"Backbone 2 training mode: {model2.backbone.training}")
        
        # Check if backbone parameters require grad
        backbone1_requires_grad = any(p.requires_grad for p in model1.backbone.parameters())
        backbone2_requires_grad = any(p.requires_grad for p in model2.backbone.parameters())
        
        print(f"\nBackbone 1 requires_grad: {backbone1_requires_grad}")
        print(f"Backbone 2 requires_grad: {backbone2_requires_grad}")


if __name__ == "__main__":
    # Run the debug script
    # You can pass checkpoint paths if you have saved models
    debug_models(
        model1_path=None,  # Replace with path to Script 1 checkpoint
        model2_path=None,  # Replace with path to Script 2 checkpoint
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )