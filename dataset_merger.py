#!/usr/bin/env python3
"""
Dataset Merger Script
Merges train/val/test splits back into the original train folder structure.
Works without split_info.json by automatically discovering existing directories.
Robust - handles missing train/val/test directories gracefully.
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Set

def discover_categories_and_folders(base_path: Path) -> Dict[str, Dict[str, List[str]]]:
    """
    Automatically discover all categories and folders in train/val/test directories.
    Returns: {category: {split: [folders]}}
    """
    discovered = {}
    splits = ['train', 'val', 'test']
    
    for split in splits:
        split_path = base_path / split
        if not split_path.exists():
            print(f"Directory {split}/ not found, skipping...")
            continue
            
        print(f"Scanning {split}/ directory...")
        
        # Check for 'living' category
        living_path = split_path / "living"
        if living_path.exists():
            folders = [d.name for d in living_path.iterdir() if d.is_dir() and d.name.isdigit()]
            if folders:
                if "living" not in discovered:
                    discovered["living"] = {}
                discovered["living"][split] = sorted(folders)
                print(f"  Found {len(folders)} folders in living/")
        
        # Check for spoof categories
        spoof_path = split_path / "spoof"
        if spoof_path.exists():
            for spoof_subdir in spoof_path.iterdir():
                if spoof_subdir.is_dir():
                    category = f"spoof/{spoof_subdir.name}"
                    folders = [d.name for d in spoof_subdir.iterdir() if d.is_dir() and d.name.isdigit()]
                    if folders:
                        if category not in discovered:
                            discovered[category] = {}
                        discovered[category][split] = sorted(folders)
                        print(f"  Found {len(folders)} folders in {category}/")
    
    return discovered

def create_target_structure(base_path: Path, categories: Set[str]) -> None:
    """Create the target train directory structure."""
    train_path = base_path / "train"
    
    for category in categories:
        target_path = train_path / category
        target_path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {target_path}")

def move_folders_to_train(source_base: Path, target_base: Path, category: str, folders: List[str], split: str) -> int:
    """Move folders from split directories to train directory. Returns count of moved folders."""
    source_category_path = source_base / split / category
    target_category_path = target_base / category
    
    moved_count = 0
    for folder in folders:
        source_folder = source_category_path / folder
        target_folder = target_category_path / folder
        
        if source_folder.exists():
            if target_folder.exists():
                print(f"Warning: {target_folder} already exists, skipping...")
                continue
                
            print(f"Moving {source_folder} -> {target_folder}")
            shutil.move(str(source_folder), str(target_folder))
            moved_count += 1
        else:
            print(f"Warning: {source_folder} does not exist, skipping...")
    
    return moved_count

def remove_empty_directories(base_path: Path) -> None:
    """Remove empty directories left after moving folders."""
    splits = ['val', 'test', 'train']  # Don't remove train directory
    
    for split in splits:
        split_path = base_path / split
        if not split_path.exists():
            continue
        
        # Remove empty category directories (bottom-up approach)
        for root, dirs, files in os.walk(split_path, topdown=False):
            root_path = Path(root)
            # Only remove if directory is empty
            if not any(root_path.iterdir()):
                try:
                    root_path.rmdir()
                    print(f"Removed empty directory: {root_path}")
                except OSError as e:
                    print(f"Could not remove {root_path}: {e}")

def verify_merge(base_path: Path, discovered_data: Dict) -> None:
    """Verify that folders have been merged correctly."""
    train_path = base_path
    
    print("\nVerifying merge...")
    total_expected = 0
    total_found = 0
    
    for category, splits_data in discovered_data.items():
        category_path = train_path / category
        expected_count = sum(len(folders) for folders in splits_data.values())
        
        if category_path.exists():
            found_folders = [d.name for d in category_path.iterdir() if d.is_dir() and d.name.isdigit()]
            found_count = len(found_folders)
        else:
            found_count = 0
        
        print(f"  {category}: Expected {expected_count}, Found {found_count}")
        total_expected += expected_count
        total_found += found_count
    
    print(f"\nTotal: Expected {total_expected}, Found {total_found}")
    
    if total_expected == total_found:
        print("✓ All folders successfully merged back!")
    else:
        print("⚠ Warning: Some folders may be missing or duplicated!")

def get_train_folder_count(base_path: Path) -> int:
    """Count existing folders in train directory for comparison."""
    train_path = base_path
    if not train_path.exists():
        return 0
    
    count = 0
    # Count living folders
    living_path = train_path / "living"
    if living_path.exists():
        count += len([d for d in living_path.iterdir() if d.is_dir() and d.name.isdigit()])
    
    # Count spoof folders
    spoof_path = train_path / "spoof"
    if spoof_path.exists():
        for spoof_subdir in spoof_path.iterdir():
            if spoof_subdir.is_dir():
                count += len([d for d in spoof_subdir.iterdir() if d.is_dir() and d.name.isdigit()])
    
    return count

def perform_merge(base_path=Path('.')):
    # Configuration
    base_path = Path(base_path)  # Current directory
    
    print("Dataset Merger - Robust Version")
    print("Automatically discovers and merges train/val/test splits")
    print("Works without split_info.json file\n")
    
    # Check if any split directories exist
    available_splits = []
    for split in ['train', 'val', 'test']:
        if (base_path / split).exists():
            available_splits.append(split)
    
    if not available_splits:
        print("No train/, val/, or test/ directories found!")
        print("Nothing to merge.")
        return
    
    print(f"Found split directories: {', '.join(available_splits)}")
    
    # Get current train folder count for comparison
    initial_train_count = get_train_folder_count(base_path / "train")
    if initial_train_count > 0:
        print(f"Existing folders in train/: {initial_train_count}")
    
    # Discover all categories and folders automatically
    print("\nDiscovering dataset structure...")
    discovered_data = discover_categories_and_folders(base_path)
    
    if not discovered_data:
        print("No valid dataset folders found in any split directories!")
        print("Looking for numbered folders (e.g., 000001, 000002, etc.)")
        return
    
    # Show what was discovered
    all_categories = set(discovered_data.keys())
    print(f"\nDiscovered categories: {sorted(list(all_categories))}")
    
    total_folders_to_move = 0
    for category, splits_data in discovered_data.items():
        category_total = sum(len(folders) for folders in splits_data.values())
        total_folders_to_move += category_total
        splits_list = list(splits_data.keys())
        print(f"  {category}: {category_total} folders across {splits_list}")
    
    if total_folders_to_move == 0:
        print("No folders to move!")
        return
    
    # Ask for confirmation
    print(f"\nReady to merge {total_folders_to_move} folders into train/ directory.")
    response = input("Proceed? [y/N]: ")
    if response.lower() not in ['y', 'yes']:
        print("Merge cancelled.")
        return
    
    # Create target directory structure
    print("\nCreating train directory structure...")
    create_target_structure(base_path, all_categories)
    
    # Move folders from each split back to train
    print("\nMoving folders to train directory...")
    total_moved = 0
    
    for category in sorted(all_categories):
        splits_data = discovered_data[category]
        print(f"\nProcessing category: {category}")
        
        category_moved = 0
        for split in ['train', 'val', 'test']:  # Process in order
            if split in splits_data:
                folders = splits_data[split]
                # if split == 'train':
                #     # Skip if folders are already in train (avoid moving to themselves)
                #     print(f"  {len(folders)} folders already in train/, skipping move")
                #     category_moved += len(folders)
                # else:
                moved = move_folders_to_train(base_path, base_path, category, folders, split)
                print(f"  Moved {moved}/{len(folders)} folders from {split}/")
                category_moved += moved
        
        total_moved += category_moved
        print(f"  Total for {category}: {category_moved} folders")
    
    # Clean up empty directories
    print("\nCleaning up empty directories...")
    remove_empty_directories(base_path)
    
    # Verify the merge
    verify_merge(base_path, discovered_data)
    
    # Final summary
    final_train_count = get_train_folder_count(base_path)
    print(f"\nMerge Summary:")
    print(f"  Initial train/ folders: {initial_train_count}")
    print(f"  Folders moved: {total_moved}")
    print(f"  Final train/ folders: {final_train_count}")
    print(f"  Expected final count: {initial_train_count + total_moved}")
    
    # Check for leftover split directories
    remaining_splits = [split for split in ['val', 'test'] if (base_path / split).exists()]
    if remaining_splits:
        print(f"\nNote: The following directories still exist: {remaining_splits}")
        print("You can manually remove them if they are empty.")
    
    print("\nDataset merge completed!")    

if __name__ == "__main__":
    perform_merge()
