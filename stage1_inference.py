"""
Stage 1 Inference Script for NetEase Anti-Spoofing Approach
Loads trained ConvNext model from .pth file and runs inference
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import argparse
import json
from tqdm import tqdm
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import timm

# Import your dataset if needed for batch inference
try:
    from wfas_dataset import create_dataloaders_with_debug
except ImportError:
    print("Warning: wfas_dataset not found. Single image inference will still work.")


class Stage1Inference:
    def __init__(self, model_path: str, device: str = 'auto'):
        """
        Initialize Stage 1 inference
        
        Args:
            model_path: Path to the trained .pth file
            device: Device to run inference on ('cuda', 'cpu', or 'auto')
        """
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load model and config
        self.model, self.config = self.load_model(model_path)
        self.model.eval()
        
        # Image preprocessing
        self.image_size = self.config.get('image_size', 224)
        self.setup_preprocessing()
        
    def load_model(self, model_path: str) -> Tuple[nn.Module, Dict]:
        """
        Load model from .pth file
        
        Args:
            model_path: Path to the .pth checkpoint file
            
        Returns:
            Tuple of (model, config)
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        print(f"Loading model from: {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Extract config
        config = checkpoint.get('config', {})
        model_name = config.get('model_name', 'convnextv2_base')
        
        # Create model architecture
        print(f"Creating model: {model_name}")
        model = timm.create_model(
            model_name,
            pretrained=False,  # We'll load our own weights
            num_classes=2  # Binary classification
        )
        
        # Load model weights
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Sometimes the checkpoint might be just the state dict
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        
        # Print model info
        print(f"Model loaded successfully!")
        print(f"Model type: {model_name}")
        print(f"Best validation accuracy: {checkpoint.get('best_val_acc', 'N/A'):.2f}%")
        print(f"Best validation ACER: {checkpoint.get('best_val_acer', 'N/A'):.4f}")
        
        return model, config
    
    def setup_preprocessing(self):
        """Setup image preprocessing transforms"""
        from torchvision import transforms
        
        # Standard preprocessing for inference
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Preprocess a single image
        
        Args:
            image_path: Path to the image
            
        Returns:
            Preprocessed image tensor
        """
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image)
        return image_tensor.unsqueeze(0)  # Add batch dimension
    
    def predict_single_image(self, image_path: str, return_probs: bool = True) -> Dict:
        """
        Run inference on a single image
        
        Args:
            image_path: Path to the image
            return_probs: Whether to return probabilities or just the prediction
            
        Returns:
            Dictionary with prediction results
        """
        # Preprocess image
        image_tensor = self.preprocess_image(image_path).to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probs = F.softmax(outputs, dim=1)
            
            # Get prediction
            _, predicted = outputs.max(1)
            predicted_class = predicted.item()
            
            # Get probabilities
            spoof_prob = probs[0, 0].item()
            live_prob = probs[0, 1].item()
        
        result = {
            'image_path': image_path,
            'prediction': 'live' if predicted_class == 1 else 'spoof',
            'predicted_class': predicted_class,
            'confidence': max(spoof_prob, live_prob)
        }
        
        if return_probs:
            result['spoof_probability'] = spoof_prob
            result['live_probability'] = live_prob
            result['probabilities'] = probs[0].cpu().numpy()
        
        return result
    
    def predict_batch(self, image_paths: List[str], batch_size: int = 32) -> List[Dict]:
        """
        Run inference on multiple images
        
        Args:
            image_paths: List of image paths
            batch_size: Batch size for inference
            
        Returns:
            List of prediction results
        """
        results = []
        
        # Process in batches
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing batches"):
            batch_paths = image_paths[i:i + batch_size]
            batch_tensors = []
            
            # Preprocess batch
            for path in batch_paths:
                try:
                    tensor = self.preprocess_image(path)
                    batch_tensors.append(tensor)
                except Exception as e:
                    print(f"Error processing {path}: {e}")
                    continue
            
            if not batch_tensors:
                continue
            
            # Stack tensors
            batch = torch.cat(batch_tensors, dim=0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(batch)
                probs = F.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
            
            # Process results
            for j, path in enumerate(batch_paths[:len(batch_tensors)]):
                result = {
                    'image_path': path,
                    'prediction': 'live' if predicted[j].item() == 1 else 'spoof',
                    'predicted_class': predicted[j].item(),
                    'spoof_probability': probs[j, 0].item(),
                    'live_probability': probs[j, 1].item(),
                    'confidence': max(probs[j, 0].item(), probs[j, 1].item())
                }
                results.append(result)
        
        return results
    
    def predict_directory(self, directory: str, extensions: List[str] = None, 
                         recursive: bool = True) -> List[Dict]:
        """
        Run inference on all images in a directory
        
        Args:
            directory: Directory containing images
            extensions: List of image extensions to process
            recursive: Whether to search subdirectories
            
        Returns:
            List of prediction results
        """
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        image_paths = []
        
        if recursive:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in extensions):
                        image_paths.append(os.path.join(root, file))
        else:
            for file in os.listdir(directory):
                if any(file.lower().endswith(ext) for ext in extensions):
                    image_paths.append(os.path.join(directory, file))
        
        print(f"Found {len(image_paths)} images in {directory}")
        
        return self.predict_batch(image_paths)
    
    def predict_with_dataloader(self, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run inference using a DataLoader
        
        Args:
            dataloader: PyTorch DataLoader
            
        Returns:
            Tuple of (predictions, labels, probabilities)
        """
        all_predictions = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Running inference"):
                images = batch['image'].to(self.device)
                labels = batch['label'].cpu().numpy()
                
                outputs = self.model(images)
                probs = F.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels)
                all_probs.extend(probs.cpu().numpy())
        
        return np.array(all_predictions), np.array(all_labels), np.array(all_probs)
    
    def calculate_metrics(self, predictions: np.ndarray, labels: np.ndarray, 
                         probs: np.ndarray) -> Dict[str, float]:
        """
        Calculate evaluation metrics
        
        Args:
            predictions: Predicted classes
            labels: True labels
            probs: Prediction probabilities
            
        Returns:
            Dictionary of metrics
        """
        # Accuracy
        accuracy = (predictions == labels).mean() * 100
        
        # Per-class accuracy
        live_mask = (labels == 1)
        spoof_mask = (labels == 0)
        
        live_acc = (predictions[live_mask] == labels[live_mask]).mean() * 100 if live_mask.sum() > 0 else 0
        spoof_acc = (predictions[spoof_mask] == labels[spoof_mask]).mean() * 100 if spoof_mask.sum() > 0 else 0
        
        # ACER calculation
        live_scores = probs[:, 1]  # Live probability
        
        # Find optimal threshold
        thresholds = np.linspace(0, 1, 1000)
        best_threshold = 0.5
        best_eer = 1.0
        
        for threshold in thresholds:
            pred_binary = (live_scores >= threshold).astype(int)
            
            if live_mask.sum() > 0 and spoof_mask.sum() > 0:
                apcer = ((pred_binary[spoof_mask] == 1).sum()) / spoof_mask.sum()
                bpcer = ((pred_binary[live_mask] == 0).sum()) / live_mask.sum()
                
                eer = abs(apcer - bpcer)
                if eer < best_eer:
                    best_eer = eer
                    best_threshold = threshold
        
        # Calculate final ACER
        pred_binary = (live_scores >= best_threshold).astype(int)
        if live_mask.sum() > 0 and spoof_mask.sum() > 0:
            apcer = ((pred_binary[spoof_mask] == 1).sum()) / spoof_mask.sum()
            bpcer = ((pred_binary[live_mask] == 0).sum()) / live_mask.sum()
            acer = (apcer + bpcer) / 2
        else:
            acer = 1.0
        
        metrics = {
            'accuracy': accuracy,
            'live_accuracy': live_acc,
            'spoof_accuracy': spoof_acc,
            'acer': acer,
            'best_threshold': best_threshold,
            'apcer': apcer if 'apcer' in locals() else None,
            'bpcer': bpcer if 'bpcer' in locals() else None
        }
        
        return metrics
    
    def save_results(self, results: List[Dict], output_path: str, format: str = 'csv'):
        """
        Save inference results to file
        
        Args:
            results: List of prediction results
            output_path: Path to save results
            format: Output format ('csv' or 'json')
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if format == 'csv':
            df = pd.DataFrame(results)
            df.to_csv(output_path, index=False)
        elif format == 'json':
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Stage 1 Inference - ConvNext Anti-Spoofing')
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to trained .pth model file')
    parser.add_argument('--input', type=str, required=True,
                       help='Input image path, directory, or "dataloader" for test set')
    parser.add_argument('--output', type=str, default='./inference_results.csv',
                       help='Output file path for results')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for inference')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to run inference on')
    parser.add_argument('--format', type=str, default='csv',
                       choices=['csv', 'json'],
                       help='Output format')
    parser.add_argument('--data_root', type=str,
                       help='Data root for dataloader mode')
    
    args = parser.parse_args()
    
    # Initialize inference
    inferencer = Stage1Inference(args.model_path, device=args.device)
    
    # Run inference based on input type
    if args.input == 'dataloader':
        # Use test dataloader
        if not args.data_root:
            raise ValueError("--data_root required for dataloader mode")
        
        _, _, test_loader = create_dataloaders_with_debug(
            data_root=args.data_root,
            stage='stage1',
            batch_size=args.batch_size,
            num_workers=4,
            image_size=inferencer.image_size,
            debug=False
        )
        
        # Run inference
        predictions, labels, probs = inferencer.predict_with_dataloader(test_loader)
        
        # Calculate metrics
        metrics = inferencer.calculate_metrics(predictions, labels, probs)
        
        print("\n=== Test Set Metrics ===")
        print(f"Accuracy: {metrics['accuracy']:.2f}%")
        print(f"Live Accuracy: {metrics['live_accuracy']:.2f}%")
        print(f"Spoof Accuracy: {metrics['spoof_accuracy']:.2f}%")
        print(f"ACER: {metrics['acer']:.4f}")
        print(f"Best Threshold: {metrics['best_threshold']:.3f}")
        
        # Save metrics
        metrics_path = args.output.replace('.csv', '_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to: {metrics_path}")
        
    elif os.path.isdir(args.input):
        # Directory mode
        results = inferencer.predict_directory(args.input)
        inferencer.save_results(results, args.output, format=args.format)
        
        # Print summary
        total = len(results)
        live_count = sum(1 for r in results if r['prediction'] == 'live')
        spoof_count = total - live_count
        
        print(f"\n=== Inference Summary ===")
        print(f"Total images: {total}")
        print(f"Live predictions: {live_count} ({live_count/total*100:.1f}%)")
        print(f"Spoof predictions: {spoof_count} ({spoof_count/total*100:.1f}%)")
        
    else:
        # Single image mode
        result = inferencer.predict_single_image(args.input)
        
        print(f"\n=== Prediction Result ===")
        print(f"Image: {result['image_path']}")
        print(f"Prediction: {result['prediction'].upper()}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Live probability: {result['live_probability']:.4f}")
        print(f"Spoof probability: {result['spoof_probability']:.4f}")
        
        # Save single result
        inferencer.save_results([result], args.output, format=args.format)


if __name__ == "__main__":
    main()
