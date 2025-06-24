#!/usr/bin/env python3
"""
Dataset Splitter Script
Splits a dataset folder into train/val/test sets with 70:15:15 ratio.
Maintains folder structure and moves (not copies) folders for efficiency.
"""

import os
import shutil
import random
import json
from pathlib import Path
from typing import List, Dict, Tuple

def get_numbered_folders(path: Path) -> List[str]:
    """Get all numbered folders (like 000001, 000002, etc.) in a directory."""
    if not path.exists():
        return []
    
    folders = []
    for item in path.iterdir():
        if item.is_dir() and item.name.isdigit():
            folders.append(item.name)
    
    return sorted(folders)

def split_folders(folders: List[str], train_ratio: float = 0.7, val_ratio: float = 0.15) -> Tuple[List[str], List[str], List[str]]:
    """Split folders into train/val/test sets with given ratios."""
    random.shuffle(folders)
    
    total = len(folders)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    
    train_folders = folders[:train_size]
    val_folders = folders[train_size:train_size + val_size]
    test_folders = folders[train_size + val_size:]
    
    return train_folders, val_folders, test_folders

def create_directory_structure(base_path: Path, categories: List[str]) -> None:
    """Create the directory structure for train/val/test splits."""
    for split in ['train', 'val', 'test']:
        for category in categories:
            (base_path / split / category).mkdir(parents=True, exist_ok=True)

def move_folders(source_base: Path, target_base: Path, category: str, folders: List[str], split: str) -> None:
    """Move folders from source to target directory."""
    source_category_path = source_base / category
    target_category_path = target_base / split / category
    
    for folder in folders:
        source_folder = source_category_path / folder
        target_folder = target_category_path / folder
        
        if source_folder.exists():
            #print(f"Moving {source_folder} -> {target_folder}")
            shutil.move(str(source_folder), str(target_folder))
        else:
            print(f"Warning: {source_folder} does not exist, skipping...")

def save_split_info(base_path: Path, split_info: Dict) -> None:
    """Save split information to a JSON file for later reconstruction."""
    split_file = base_path / 'split_info.json'
    with open(split_file, 'w') as f:
        json.dump(split_info, f, indent=2)
    print(f"Split information saved to {split_file}")

def perform_split(original_train_path=Path("train"), output_base_path= Path('.')):
    #configuration
    original_train_path=Path(original_train_path)
    output_base_path= Path(output_base_path)
    
    # Set random seed for reproducibility (optional)
    random.seed(42)
    
    if not original_train_path.exists():
        print(f"Error: {original_train_path} does not exist!")
        return
    
    print("Starting dataset split...")
    print(f"Source: {original_train_path}")
    print(f"Target: {output_base_path}")
    
    # Discover all categories
    categories = []
    split_info = {}
    
    # Add 'living' category
    living_path = original_train_path / "living"
    if living_path.exists():
        categories.append("living")
        living_folders = get_numbered_folders(living_path)
        print(f"Found {len(living_folders)} folders in 'living' category")
        
        train_folders, val_folders, test_folders = split_folders(living_folders)
        split_info["living"] = {
            "train": train_folders,
            "val": val_folders,
            "test": test_folders
        }
    
    # Add spoof subcategories
    spoof_path = original_train_path / "spoof"
    if spoof_path.exists():
        for spoof_subdir in spoof_path.iterdir():
            if spoof_subdir.is_dir():
                category_name = f"spoof/{spoof_subdir.name}"
                categories.append(category_name)
                
                spoof_folders = get_numbered_folders(spoof_subdir)
                print(f"Found {len(spoof_folders)} folders in '{category_name}' category")
                
                train_folders, val_folders, test_folders = split_folders(spoof_folders)
                split_info[category_name] = {
                    "train": train_folders,
                    "val": val_folders,
                    "test": test_folders
                }
    
    if not categories:
        print("No categories found to split!")
        return
    
    print(f"\nCategories found: {categories}")
    
    # Create directory structure
    print("\nCreating directory structure...")
    create_directory_structure(output_base_path, categories)
    
    # Move folders for each category and split
    print("\nMoving folders...")
    for category in categories:
        print(f"\nProcessing category: {category}")
        
        # Move train folders
        train_folders = split_info[category]["train"]
        move_folders(original_train_path, output_base_path, category, train_folders, "train")
        print(f"  Train: {len(train_folders)} folders")
        
        # Move val folders
        val_folders = split_info[category]["val"]
        move_folders(original_train_path, output_base_path, category, val_folders, "val")
        print(f"  Val: {len(val_folders)} folders")
        
        # Move test folders
        test_folders = split_info[category]["test"]
        move_folders(original_train_path, output_base_path, category, test_folders, "test")
        print(f"  Test: {len(test_folders)} folders")
    
    # Save split information
    save_split_info(output_base_path, split_info)
    
    # Remove empty directories
    print("\nCleaning up empty directories...")
    try:
        # Remove empty category directories
        for category in categories:
            category_path = original_train_path / category.replace('spoof/', 'spoof/')
            if category_path.exists() and not any(category_path.iterdir()):
                category_path.rmdir()
                print(f"Removed empty directory: {category_path}")
        
        # Remove empty spoof directory if it exists and is empty
        spoof_path = original_train_path / "spoof"
        if spoof_path.exists() and not any(spoof_path.iterdir()):
            spoof_path.rmdir()
            print(f"Removed empty directory: {spoof_path}")
        
        # Remove empty train directory if it's completely empty
        if original_train_path.exists() and not any(original_train_path.iterdir()):
            original_train_path.rmdir()
            print(f"Removed empty directory: {original_train_path}")
    
    except OSError as e:
        print(f"Note: Could not remove some empty directories: {e}")
    
    print("\nDataset split completed successfully!")
    print(f"Structure created:")
    print(f"  train/ - {sum(len(info['train']) for info in split_info.values())} folders")
    print(f"  val/   - {sum(len(info['val']) for info in split_info.values())} folders")
    print(f"  test/  - {sum(len(info['test']) for info in split_info.values())} folders")

if __name__ == "__main__":
    perform_split(original_train_path=Path("train"), output_base_path= Path('.'))
