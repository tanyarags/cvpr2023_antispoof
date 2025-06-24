import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import pandas as pd
from PIL import Image
import torch

# Import your dataset class (assuming it's in the same directory or installed)
from wfas_dataset import WFASDataset, debug_dataset_structure

def comprehensive_dataset_analysis(data_root, save_plots=True, output_dir="dataset_analysis"):
    """
    Comprehensive analysis of WFAS dataset
    """
    print("="*60)
    print("COMPREHENSIVE WFAS DATASET ANALYSIS")
    print("="*60)
    
    # Create output directory for plots
    if save_plots:
        os.makedirs(output_dir, exist_ok=True)
    
    # First, run the basic structure debug
    print("\n1. BASIC STRUCTURE ANALYSIS")
    print("-"*40)
    debug_dataset_structure(data_root)
    
    # Detailed analysis for each split
    splits = ['train', 'dev', 'test']
    analysis_results = {}
    
    for split in splits:
        print(f"\n2. DETAILED ANALYSIS FOR {split.upper()} SPLIT")
        print("-"*40)
        
        try:
            # Create dataset instance with debugging
            dataset = WFASDataset(
                data_root=data_root,
                split=split,
                stage='stage1',
                image_size=224,
                augment=False,
                debug=True
            )
            
            if len(dataset) == 0:
                print(f"No samples found in {split} split")
                continue
            
            # Analyze the dataset
            split_analysis = analyze_split(dataset, split, save_plots, output_dir)
            analysis_results[split] = split_analysis
            
        except Exception as e:
            print(f"Error analyzing {split} split: {e}")
            continue
    
    # Cross-split comparison
    if len(analysis_results) > 1:
        print(f"\n3. CROSS-SPLIT COMPARISON")
        print("-"*40)
        compare_splits(analysis_results, save_plots, output_dir)
    
    # Generate summary report
    print(f"\n4. SUMMARY REPORT")
    print("-"*40)
    generate_summary_report(analysis_results, save_plots, output_dir)
    
    return analysis_results

def analyze_split(dataset, split_name, save_plots=True, output_dir="dataset_analysis"):
    """
    Detailed analysis of a single split
    """
    samples = dataset.samples
    split_analysis = {
        'total_samples': len(samples),
        'live_samples': 0,
        'spoof_samples': 0,
        'attack_types': Counter(),
        'sample_paths': [],
        'image_sizes': [],
        'bbox_sizes': []
    }
    
    print(f"Analyzing {len(samples)} samples in {split_name} split...")
    
    # Count samples and collect statistics
    for i, sample in enumerate(samples):
        if sample['label'] == 1:
            split_analysis['live_samples'] += 1
        else:
            split_analysis['spoof_samples'] += 1
        
        split_analysis['attack_types'][sample['attack_type']] += 1
        split_analysis['sample_paths'].append(sample['image_path'])
        
        # Get image dimensions if possible
        try:
            img = cv2.imread(sample['image_path'])
            if img is not None:
                h, w = img.shape[:2]
                split_analysis['image_sizes'].append((w, h))
                
                # Try to get bbox size
                bbox, _ = dataset._parse_annotation(sample['annotation_path'])
                if bbox:
                    bbox_w = bbox[2] - bbox[0]
                    bbox_h = bbox[3] - bbox[1]
                    split_analysis['bbox_sizes'].append((bbox_w, bbox_h))
        except Exception as e:
            if i < 5:  # Only print first few errors
                print(f"Error reading image {sample['image_path']}: {e}")
    
    # Print statistics
    print(f"Total samples: {split_analysis['total_samples']}")
    print(f"Live samples: {split_analysis['live_samples']}")
    print(f"Spoof samples: {split_analysis['spoof_samples']}")
    print(f"Live/Spoof ratio: {split_analysis['live_samples']/max(split_analysis['spoof_samples'], 1):.2f}")
    print(f"Attack types: {dict(split_analysis['attack_types'])}")
    
    if split_analysis['image_sizes']:
        widths = [size[0] for size in split_analysis['image_sizes']]
        heights = [size[1] for size in split_analysis['image_sizes']]
        print(f"Image size stats - Width: {np.mean(widths):.1f}±{np.std(widths):.1f}, Height: {np.mean(heights):.1f}±{np.std(heights):.1f}")
    
    # Create visualizations
    if save_plots:
        create_split_visualizations(split_analysis, split_name, output_dir)
    
    return split_analysis

def create_split_visualizations(split_analysis, split_name, output_dir):
    """
    Create visualizations for a single split
    """
    # 1. Attack type distribution
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    attack_types = split_analysis['attack_types']
    plt.pie(attack_types.values(), labels=attack_types.keys(), autopct='%1.1f%%')
    plt.title(f'{split_name.title()} - Attack Type Distribution')
    
    # 2. Live vs Spoof
    plt.subplot(1, 3, 2)
    labels = ['Live', 'Spoof']
    sizes = [split_analysis['live_samples'], split_analysis['spoof_samples']]
    colors = ['lightgreen', 'lightcoral']
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors)
    plt.title(f'{split_name.title()} - Live vs Spoof')
    
    # 3. Sample count by attack type
    plt.subplot(1, 3, 3)
    attack_types = split_analysis['attack_types']
    plt.bar(range(len(attack_types)), list(attack_types.values()))
    plt.xticks(range(len(attack_types)), list(attack_types.keys()), rotation=45)
    plt.title(f'{split_name.title()} - Samples per Attack Type')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{split_name}_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Image size distribution (if available)
    if split_analysis['image_sizes']:
        plt.figure(figsize=(10, 4))
        
        widths = [size[0] for size in split_analysis['image_sizes']]
        heights = [size[1] for size in split_analysis['image_sizes']]
        
        plt.subplot(1, 2, 1)
        plt.hist(widths, bins=20, alpha=0.7, label='Width')
        plt.hist(heights, bins=20, alpha=0.7, label='Height')
        plt.xlabel('Pixels')
        plt.ylabel('Frequency')
        plt.title(f'{split_name.title()} - Image Size Distribution')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.scatter(widths, heights, alpha=0.5)
        plt.xlabel('Width')
        plt.ylabel('Height')
        plt.title(f'{split_name.title()} - Width vs Height')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{split_name}_image_sizes.png'), dpi=300, bbox_inches='tight')
        plt.close()

def compare_splits(analysis_results, save_plots=True, output_dir="dataset_analysis"):
    """
    Compare statistics across splits
    """
    splits = list(analysis_results.keys())
    
    # Create comparison DataFrame
    comparison_data = {
        'Split': splits,
        'Total Samples': [analysis_results[split]['total_samples'] for split in splits],
        'Live Samples': [analysis_results[split]['live_samples'] for split in splits],
        'Spoof Samples': [analysis_results[split]['spoof_samples'] for split in splits],
        'Live/Spoof Ratio': [analysis_results[split]['live_samples']/max(analysis_results[split]['spoof_samples'], 1) for split in splits]
    }
    
    df = pd.DataFrame(comparison_data)
    print(df.to_string(index=False))
    
    if save_plots:
        # Split comparison visualization
        plt.figure(figsize=(15, 5))
        
        # Total samples
        plt.subplot(1, 3, 1)
        plt.bar(splits, [analysis_results[split]['total_samples'] for split in splits])
        plt.title('Total Samples per Split')
        plt.ylabel('Count')
        
        # Live vs Spoof comparison
        plt.subplot(1, 3, 2)
        live_counts = [analysis_results[split]['live_samples'] for split in splits]
        spoof_counts = [analysis_results[split]['spoof_samples'] for split in splits]
        
        x = np.arange(len(splits))
        width = 0.35
        
        plt.bar(x - width/2, live_counts, width, label='Live', color='lightgreen')
        plt.bar(x + width/2, spoof_counts, width, label='Spoof', color='lightcoral')
        plt.xticks(x, splits)
        plt.title('Live vs Spoof Samples per Split')
        plt.ylabel('Count')
        plt.legend()
        
        # Attack type distribution across splits
        plt.subplot(1, 3, 3)
        all_attack_types = set()
        for split in splits:
            all_attack_types.update(analysis_results[split]['attack_types'].keys())
        
        attack_type_data = {attack_type: [analysis_results[split]['attack_types'].get(attack_type, 0) for split in splits] 
                           for attack_type in all_attack_types}
        
        bottom = np.zeros(len(splits))
        for attack_type in all_attack_types:
            plt.bar(splits, attack_type_data[attack_type], bottom=bottom, label=attack_type)
            bottom += attack_type_data[attack_type]
        
        plt.title('Attack Type Distribution per Split')
        plt.ylabel('Count')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'split_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

def generate_summary_report(analysis_results, save_plots=True, output_dir="dataset_analysis"):
    """
    Generate a comprehensive summary report
    """
    total_samples = sum(result['total_samples'] for result in analysis_results.values())
    total_live = sum(result['live_samples'] for result in analysis_results.values())
    total_spoof = sum(result['spoof_samples'] for result in analysis_results.values())
    
    # Collect all attack types
    all_attack_types = Counter()
    for result in analysis_results.values():
        all_attack_types.update(result['attack_types'])
    
    print(f"DATASET SUMMARY:")
    print(f"Total samples across all splits: {total_samples}")
    print(f"Total live samples: {total_live}")
    print(f"Total spoof samples: {total_spoof}")
    print(f"Overall live/spoof ratio: {total_live/max(total_spoof, 1):.2f}")
    print(f"Attack types across dataset: {dict(all_attack_types)}")
    
    # Save summary to file
    if save_plots:
        summary_path = os.path.join(output_dir, 'dataset_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("WFAS DATASET SUMMARY REPORT\n")
            f.write("="*50 + "\n\n")
            
            f.write(f"Total samples: {total_samples}\n")
            f.write(f"Total live samples: {total_live}\n")
            f.write(f"Total spoof samples: {total_spoof}\n")
            f.write(f"Overall live/spoof ratio: {total_live/max(total_spoof, 1):.2f}\n\n")
            
            f.write("Split-wise breakdown:\n")
            for split, result in analysis_results.items():
                f.write(f"\n{split.upper()}:\n")
                f.write(f"  Total: {result['total_samples']}\n")
                f.write(f"  Live: {result['live_samples']}\n")
                f.write(f"  Spoof: {result['spoof_samples']}\n")
                f.write(f"  Attack types: {dict(result['attack_types'])}\n")
            
            f.write(f"\nOverall attack type distribution: {dict(all_attack_types)}\n")
        
        print(f"\nAnalysis complete! Results saved to: {output_dir}")

def sample_visualization(dataset, num_samples=16, save_plots=True, output_dir="dataset_analysis"):
    """
    Visualize random samples from the dataset
    """
    if len(dataset) == 0:
        print("No samples to visualize")
        return
    
    # Get random samples
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    # Calculate grid size
    cols = 4
    rows = (len(indices) + cols - 1) // cols
    
    plt.figure(figsize=(16, 4 * rows))
    
    for i, idx in enumerate(indices):
        sample = dataset[idx]
        
        plt.subplot(rows, cols, i + 1)
        
        # Convert tensor to numpy and denormalize
        image = sample['image'].permute(1, 2, 0).numpy()
        
        # Denormalize (assuming ImageNet normalization)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = image * std + mean
        image = np.clip(image, 0, 1)
        
        plt.imshow(image)
        label = "Live" if sample['label'].item() == 1 else "Spoof"
        attack_type = sample['attack_type']
        plt.title(f"{label} - {attack_type}")
        plt.axis('off')
    
    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(output_dir, 'sample_visualization.png'), dpi=300, bbox_inches='tight')
    plt.show()

# Main execution function
def run_analysis(data_root, visualize_samples=True):
    """
    Run the complete dataset analysis
    """
    # Run comprehensive analysis
    results = comprehensive_dataset_analysis(data_root)
    
    # Visualize samples if requested
    if visualize_samples and results:
        for split in ['train', 'dev', 'test']:
            try:
                dataset = WFASDataset(
                    data_root=data_root,
                    split=split,
                    stage='stage1',
                    image_size=224,
                    augment=False,
                    debug=False
                )
                if len(dataset) > 0:
                    print(f"\nVisualizing samples from {split} split:")
                    sample_visualization(dataset, num_samples=8, save_plots=True, 
                                       output_dir=f"dataset_analysis")
            except Exception as e:
                print(f"Could not visualize {split} samples: {e}")
    
    return results

# Example usage
if __name__ == "__main__":
    # Replace with your actual dataset path
    data_root = "/path/to/your/WFAS/dataset"
    
    # Run the analysis
    analysis_results = run_analysis(data_root, visualize_samples=True)
