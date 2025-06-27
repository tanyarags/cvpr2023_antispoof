import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2


class WFASDataset(Dataset):
    """
    Wild Face Anti-Spoofing Dataset reader for NetEase approach
    Supports two-stage training with different field of view strategies
    Updated to work with simplified flat folder structure:
    train/
    ├── living/
    │   ├── 1.jpg
    │   ├── 2.jpg
    │   └── ...
    └── spoof/
        ├── 3.jpg
        ├── 4.jpg
        └── ...
    """
    
    def __init__(self, 
                 data_root, 
                 split='train',  # 'train', 'dev', 'test'
                 stage='stage1',  # 'stage1' (ConvNext), 'stage2' (MaxViT)
                 soft_labels_path=None,  # Path to soft labels from stage1 for stage2
                 image_size=224,
                 augment=True,
                 debug=True):  # Add debug flag
        
        self.data_root = data_root
        self.split = split
        self.stage = stage
        self.image_size = image_size
        self.augment = augment and split == 'train'
        self.debug = debug
        
        # Load soft labels for stage2
        self.soft_labels = {}
        if stage == 'stage2' and soft_labels_path and os.path.exists(soft_labels_path):
            try:
                # Try with weights_only=False for compatibility with numpy arrays
                self.soft_labels = torch.load(soft_labels_path, weights_only=False)
                if self.debug:
                    print(f"Loaded {len(self.soft_labels)} soft labels from {soft_labels_path}")
            except Exception as e:
                if self.debug:
                    print(f"Warning: Could not load soft labels from {soft_labels_path}: {e}")
                    print("Continuing without soft labels...")
                self.soft_labels = {}
        
        # Debug: Check data root
        if self.debug:
            print(f"Dataset root: {data_root}")
            print(f"Split: {split}")
            print(f"Data root exists: {os.path.exists(data_root)}")
            if os.path.exists(data_root):
                print(f"Contents of data root: {os.listdir(data_root)}")
        
        # Load dataset
        self.samples = self._load_samples()
        
        if self.debug:
            print(f"Total samples loaded: {len(self.samples)}")
            if len(self.samples) == 0:
                print("WARNING: No samples found! Check your dataset structure.")
            else:
                # Show sample distribution
                live_count = sum(1 for s in self.samples if s['label'] == 1)
                spoof_count = sum(1 for s in self.samples if s['label'] == 0)
                print(f"Live samples: {live_count}, Spoof samples: {spoof_count}")
        
        # Setup transforms based on stage
        self._setup_transforms()
    
    def _load_samples(self):
        """Load all samples from the simplified dataset structure"""
        samples = []
        
        # Try different split directory patterns
        possible_split_dirs = [
            os.path.join(self.data_root, self.split.capitalize()),  # Train, Dev, Test
            os.path.join(self.data_root, self.split.lower()),      # train, dev, test
            os.path.join(self.data_root, self.split.upper()),      # TRAIN, DEV, TEST
            os.path.join(self.data_root, self.split),              # whatever user provides
        ]
        
        split_dir = None
        for possible_dir in possible_split_dirs:
            if os.path.exists(possible_dir):
                split_dir = possible_dir
                break
        
        if split_dir is None:
            if self.debug:
                print(f"ERROR: Could not find split directory for '{self.split}'")
                print(f"Tried: {possible_split_dirs}")
                print(f"Available directories in {self.data_root}: {os.listdir(self.data_root) if os.path.exists(self.data_root) else 'ROOT NOT FOUND'}")
            return samples
        
        if self.debug:
            print(f"Using split directory: {split_dir}")
            print(f"Contents: {os.listdir(split_dir)}")
        
        # Load living samples
        living_dir = os.path.join(split_dir, 'living')
        if os.path.exists(living_dir):
            living_samples = self._load_samples_from_dir(living_dir, label=1, attack_type='living')
            samples.extend(living_samples)
            if self.debug:
                print(f"Loaded {len(living_samples)} living samples")
        else:
            if self.debug:
                print(f"Living directory not found: {living_dir}")
        
        # Load spoof samples
        spoof_dir = os.path.join(split_dir, 'spoof')
        if os.path.exists(spoof_dir):
            spoof_samples = self._load_samples_from_dir(spoof_dir, label=0, attack_type='spoof')
            samples.extend(spoof_samples)
            if self.debug:
                print(f"Loaded {len(spoof_samples)} spoof samples")
        else:
            if self.debug:
                print(f"Spoof directory not found: {spoof_dir}")
        
        return samples
    
    def _load_samples_from_dir(self, directory, label, attack_type):
        """Load samples from a flat directory"""
        samples = []
        
        if not os.path.exists(directory):
            return samples
        
        for img_file in os.listdir(directory):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                img_path = os.path.join(directory, img_file)
                
                # Create sample entry
                samples.append({
                    'image_path': img_path,
                    'label': label,
                    'attack_type': attack_type,
                    'filename': img_file
                })
        
        return samples
    
    def _setup_transforms(self):
        """Setup transforms based on training stage"""
        
        # Common normalization
        normalize = A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        if self.stage == 'stage1':  # ConvNext stage
            if self.augment:
                # NetEase augmentations for stage1
                self.transform = A.Compose([
                    A.Resize(self.image_size, self.image_size),
                    A.HorizontalFlip(p=0.5),
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                    A.CLAHE(p=0.3),
                    A.GaussianBlur(blur_limit=(3, 7), p=0.3),
                    A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.2),
                    normalize,
                    ToTensorV2()
                ])
            else:
                self.transform = A.Compose([
                    A.Resize(self.image_size, self.image_size),
                    normalize,
                    ToTensorV2()
                ])
                
        else:  # stage2 - MaxViT stage
            if self.augment:
                # Enhanced augmentations for stage2
                self.transform = A.Compose([
                    A.Resize(self.image_size, self.image_size),
                    A.HorizontalFlip(p=0.5),
                    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15, p=0.6),
                    A.CLAHE(clip_limit=2.0, p=0.4),
                    A.GaussianBlur(blur_limit=(3, 9), p=0.4),
                    A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.4, p=0.3),
                    A.OneOf([
                        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
                    ], p=0.5),
                    normalize,
                    ToTensorV2()
                ])
            else:
                self.transform = A.Compose([
                    A.Resize(self.image_size, self.image_size),
                    normalize,
                    ToTensorV2()
                ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        try:
            # Load image
            image = cv2.imread(sample['image_path'])
            if image is None:
                raise ValueError(f"Could not load image: {sample['image_path']}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Apply transforms
            transformed = self.transform(image=image)
            image_tensor = transformed['image']
            
            # Prepare output (maintaining API compatibility with original)
            output = {
                'image': image_tensor,
                'label': torch.tensor(sample['label'], dtype=torch.long),
                'attack_type': sample['attack_type'],
                'image_path': sample['image_path'],
                'bbox': torch.tensor([0, 0, image.shape[1], image.shape[0]], dtype=torch.float32),  # Dummy bbox (full image)
                'landmarks': torch.tensor([[25, 25], [75, 25], [50, 50], [35, 75], [65, 75]], dtype=torch.float32),  # Dummy landmarks
                'has_annotation': False  # No annotation files in this structure
            }
            
            # Add soft labels for stage2
            if self.stage == 'stage2' and sample['image_path'] in self.soft_labels:
                output['soft_label'] = torch.tensor(self.soft_labels[sample['image_path']], dtype=torch.float32)
            
            return output
            
        except Exception as e:
            if self.debug:
                print(f"Error loading sample {idx}: {e}")
                print(f"Sample info: {sample}")
            
            # Return a dummy sample to avoid crashing
            dummy_image = torch.zeros(3, self.image_size, self.image_size)
            return {
                'image': dummy_image,
                'label': torch.tensor(0, dtype=torch.long),
                'attack_type': 'error',
                'image_path': sample['image_path'],
                'bbox': torch.tensor([0, 0, 100, 100], dtype=torch.float32),
                'landmarks': torch.tensor([[25, 25], [75, 25], [50, 50], [35, 75], [65, 75]], dtype=torch.float32),
                'has_annotation': False
            }


class MixupCutmixCollator:
    """
    Collator for implementing mixup and cutmix augmentations
    Used in stage2 training as mentioned by NetEase
    """
    
    def __init__(self, mixup_alpha=0.2, cutmix_alpha=1.0, prob=0.5):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
    
    def __call__(self, batch):
        # Standard batch collation
        images = torch.stack([item['image'] for item in batch])
        labels = torch.stack([item['label'] for item in batch])
        original_labels = labels.clone()  # Store original labels for accuracy calculation
        
        # Apply mixup or cutmix with probability
        if random.random() < self.prob:
            if random.random() < 0.5:
                # Mixup
                images, labels = self._mixup(images, labels)
            else:
                # Cutmix
                images, labels = self._cutmix(images, labels)
        
        # Prepare batch dict
        batch_dict = {
            'image': images,
            'label': labels,
            'original_label': original_labels,  # Keep original labels for accuracy
            'attack_type': [item['attack_type'] for item in batch],
            'image_path': [item['image_path'] for item in batch],
            'bbox': torch.stack([item['bbox'] for item in batch]),
            'landmarks': torch.stack([item['landmarks'] for item in batch]),
            'has_annotation': [item['has_annotation'] for item in batch]
        }
        
        # Add soft labels if available
        if 'soft_label' in batch[0]:
            batch_dict['soft_label'] = torch.stack([item['soft_label'] for item in batch])
        
        return batch_dict
    
    def _mixup(self, images, labels):
        """Apply mixup augmentation"""
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        batch_size = images.size(0)
        index = torch.randperm(batch_size)
        
        mixed_images = lam * images + (1 - lam) * images[index]
        
        # For binary classification, create mixed labels as continuous values
        # where the value represents the weight for the positive class (live=1)
        mixed_labels = lam * labels.float() + (1 - lam) * labels[index].float()
        
        return mixed_images, mixed_labels
    
    def _cutmix(self, images, labels):
        """Apply cutmix augmentation"""
        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        batch_size = images.size(0)
        index = torch.randperm(batch_size)
        
        _, _, H, W = images.shape
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        # Random center point
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        # Bounding box
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        # Apply cutmix
        mixed_images = images.clone()
        mixed_images[:, :, bby1:bby2, bbx1:bbx2] = images[index, :, bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        
        # For binary classification, create mixed labels as continuous values
        # where the value represents the weight for the positive class (live=1)
        mixed_labels = lam * labels.float() + (1 - lam) * labels[index].float()
        
        return mixed_images, mixed_labels


def debug_dataset_structure(data_root):
    """
    Debug function to understand your dataset structure
    """
    print("=== DATASET STRUCTURE ANALYSIS ===")
    print(f"Root directory: {data_root}")
    print(f"Root exists: {os.path.exists(data_root)}")
    
    if not os.path.exists(data_root):
        print("ERROR: Root directory does not exist!")
        return
    
    print(f"Root contents: {os.listdir(data_root)}")
    
    # Check for common split directory patterns
    splits = ['train', 'dev', 'test', 'Train', 'Dev', 'Test', 'TRAIN', 'DEV', 'TEST']
    
    for split in splits:
        split_path = os.path.join(data_root, split)
        if os.path.exists(split_path):
            print(f"\n--- {split} directory found ---")
            print(f"Contents: {os.listdir(split_path)}")
            
            # Check living/spoof structure
            living_path = os.path.join(split_path, 'living')
            spoof_path = os.path.join(split_path, 'spoof')
            
            if os.path.exists(living_path):
                living_files = [f for f in os.listdir(living_path) 
                               if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                print(f"Living images: {len(living_files)}")
                print(f"Living samples (first 5): {living_files[:5]}")
            else:
                print(f"Living directory not found: {living_path}")
            
            if os.path.exists(spoof_path):
                spoof_files = [f for f in os.listdir(spoof_path) 
                              if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                print(f"Spoof images: {len(spoof_files)}")
                print(f"Spoof samples (first 5): {spoof_files[:5]}")
            else:
                print(f"Spoof directory not found: {spoof_path}")


def create_dataloaders_with_debug(data_root, 
                                 stage='stage1',
                                 soft_labels_path=None,
                                 batch_size=32,
                                 num_workers=4,
                                 image_size=224,
                                 debug=True):
    """
    Create dataloaders with debugging information
    """
    
    if debug:
        debug_dataset_structure(data_root)
    
    # Create datasets with debugging enabled
    train_dataset = WFASDataset(
        data_root=data_root,
        split='train',
        stage=stage,
        soft_labels_path=soft_labels_path,
        image_size=image_size,
        augment=True,
        debug=debug
    )
    
    if len(train_dataset) == 0:
        print("ERROR: Training dataset is empty!")
        print("Please check your dataset structure and paths.")
        return None, None, None
    
    dev_dataset = WFASDataset(
        data_root=data_root,
        split='dev',
        stage=stage,
        soft_labels_path=soft_labels_path,
        image_size=image_size,
        augment=False,
        debug=debug
    )
    
    test_dataset = WFASDataset(
        data_root=data_root,
        split='test',
        stage=stage,
        image_size=image_size,
        augment=False,
        debug=debug
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, dev_loader, test_loader


def create_dataloaders(data_root, 
                      stage='stage1',
                      soft_labels_path=None,
                      batch_size=32,
                      num_workers=4,
                      image_size=224):
    """
    Backward compatibility function - calls create_dataloaders_with_debug
    """
    return create_dataloaders_with_debug(
        data_root=data_root,
        stage=stage,
        soft_labels_path=soft_labels_path,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
        debug=False
    )


# Example usage
if __name__ == "__main__":
    # First, debug your dataset structure
    data_root = "/path/to/your/dataset"  # Replace with your actual path
    debug_dataset_structure(data_root)
    
    # Then create dataloaders with debugging
    train_loader, dev_loader, test_loader = create_dataloaders_with_debug(
        data_root=data_root,
        stage='stage1',
        batch_size=4,  # Small batch for testing
        debug=True
    )
    
    if train_loader is not None:
        print(f"\nDataloaders created successfully!")
        print(f"Train samples: {len(train_loader.dataset)}")
        print(f"Dev samples: {len(dev_loader.dataset)}")
        print(f"Test samples: {len(test_loader.dataset)}")
        
        # Test loading a batch
        try:
            batch = next(iter(train_loader))
            print(f"Batch loaded successfully!")
            print(f"Image shape: {batch['image'].shape}")
            print(f"Label shape: {batch['label'].shape}")
            print(f"Attack types: {batch['attack_type']}")
            print(f"Has annotations: {batch['has_annotation']}")
            
        except Exception as e:
            print(f"Error loading batch: {e}")
