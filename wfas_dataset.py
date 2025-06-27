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
    Now includes images without annotation files using whole image as fallback
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
                
                # Show samples with/without annotations
                with_ann = sum(1 for s in self.samples if s['annotation_path'] is not None)
                without_ann = len(self.samples) - with_ann
                print(f"Samples with annotations: {with_ann}, without annotations: {without_ann}")
                
                # Show attack type distribution
                attack_types = {}
                for s in self.samples:
                    attack_type = s['attack_type']
                    attack_types[attack_type] = attack_types.get(attack_type, 0) + 1
                print(f"Attack type distribution: {attack_types}")
        
        # Setup transforms based on stage
        self._setup_transforms()
    
    def _load_samples(self):
        """Load all samples from the dataset with flexible path handling"""
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
        
        # Load live samples
        samples.extend(self._load_live_samples(split_dir))
        
        # Load spoof samples  
        samples.extend(self._load_spoof_samples(split_dir))
        
        return samples
    
    def _load_live_samples(self, split_dir):
        """Load live samples - now includes images without annotations"""
        samples = []
        live_dir = os.path.join(split_dir, 'living')
        
        if not os.path.exists(live_dir):
            if self.debug:
                print(f"Live directory not found: {live_dir}")
            return samples
        
        if self.debug:
            print(f"Loading live samples from: {live_dir}")
            print(f"Live directory contents: {os.listdir(live_dir)[:10]}...")  # Show first 10
        
        live_count = 0
        live_with_ann = 0
        live_without_ann = 0
        
        for subject_id in os.listdir(live_dir):
            subject_path = os.path.join(live_dir, subject_id)
            if os.path.isdir(subject_path):
                for img_file in os.listdir(subject_path):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(subject_path, img_file)
                        
                        # Look for annotation file
                        base_name = os.path.splitext(img_file)[0]
                        ann_path = os.path.join(subject_path, f"{base_name}.txt")
                        
                        if os.path.exists(ann_path):
                            # Sample with annotation
                            samples.append({
                                'image_path': img_path,
                                'annotation_path': ann_path,
                                'label': 1,  # Live
                                'attack_type': 'living',
                                'has_annotation': True
                            })
                            live_with_ann += 1
                        else:
                            # Sample without annotation - include it anyway
                            samples.append({
                                'image_path': img_path,
                                'annotation_path': None,  # No annotation file
                                'label': 1,  # Live
                                'attack_type': 'living',
                                'has_annotation': False
                            })
                            live_without_ann += 1
                            if self.debug and live_without_ann <= 5:  # Only show first few missing annotations
                                print(f"Including image without annotation: {img_path}")
                        
                        live_count += 1
        
        if self.debug:
            print(f"Loaded {live_count} live samples ({live_with_ann} with annotations, {live_without_ann} without)")
        
        return samples
    
    def _load_spoof_samples(self, split_dir):
        """Load spoof samples - now includes images without annotations"""
        samples = []
        spoof_dir = os.path.join(split_dir, 'spoof')
        
        if not os.path.exists(spoof_dir):
            if self.debug:
                print(f"Spoof directory not found: {spoof_dir}")
            return samples
        
        if self.debug:
            print(f"Loading spoof samples from: {spoof_dir}")
            print(f"Spoof directory contents: {os.listdir(spoof_dir)}")
        
        spoof_count = 0
        spoof_with_ann = 0
        spoof_without_ann = 0
        
        for attack_category in os.listdir(spoof_dir):
            category_path = os.path.join(spoof_dir, attack_category)
            if os.path.isdir(category_path):
                if self.debug:
                    print(f"  Processing attack category: {attack_category}")
                    category_contents = os.listdir(category_path)
                    print(f"    Category contents (first 5): {category_contents[:5]}")
                
                category_count = 0
                category_with_ann = 0
                category_without_ann = 0
                
                for subject_id in os.listdir(category_path):
                    subject_path = os.path.join(category_path, subject_id)
                    if os.path.isdir(subject_path):
                        for img_file in os.listdir(subject_path):
                            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                                img_path = os.path.join(subject_path, img_file)
                                
                                # Look for annotation file
                                base_name = os.path.splitext(img_file)[0]
                                ann_path = os.path.join(subject_path, f"{base_name}.txt")
                                
                                if os.path.exists(ann_path):
                                    # Sample with annotation
                                    samples.append({
                                        'image_path': img_path,
                                        'annotation_path': ann_path,
                                        'label': 0,  # Spoof
                                        'attack_type': attack_category,
                                        'has_annotation': True
                                    })
                                    category_with_ann += 1
                                    spoof_with_ann += 1
                                else:
                                    # Sample without annotation - include it anyway
                                    samples.append({
                                        'image_path': img_path,
                                        'annotation_path': None,  # No annotation file
                                        'label': 0,  # Spoof
                                        'attack_type': attack_category,
                                        'has_annotation': False
                                    })
                                    category_without_ann += 1
                                    spoof_without_ann += 1
                                    if self.debug and spoof_without_ann <= 5:  # Only show first few
                                        print(f"    Including image without annotation: {img_path}")
                                
                                category_count += 1
                                spoof_count += 1
                
                if self.debug:
                    print(f"    Loaded {category_count} samples from {attack_category} ({category_with_ann} with ann, {category_without_ann} without)")
        
        if self.debug:
            print(f"Loaded {spoof_count} total spoof samples ({spoof_with_ann} with annotations, {spoof_without_ann} without)")
        
        return samples
    
    def _parse_annotation(self, ann_path):
        """Parse annotation file to get bbox and landmarks - returns None if no file"""
        if ann_path is None:
            return None, None
            
        try:
            with open(ann_path, 'r') as f:
                lines = f.readlines()
            
            if len(lines) < 7:
                if self.debug:
                    print(f"Invalid annotation file (insufficient lines): {ann_path}")
                return None, None
            
            # Parse bbox (first two lines)
            left, top = map(int, lines[0].strip().split())
            right, bottom = map(int, lines[1].strip().split())
            bbox = [left, top, right, bottom]
            
            # Parse landmarks (next 5 lines)
            landmarks = []
            for i in range(2, 7):
                x, y = map(int, lines[i].strip().split())
                landmarks.append([x, y])
            
            return bbox, landmarks
            
        except Exception as e:
            if self.debug:
                print(f"Error parsing annotation {ann_path}: {e}")
            return None, None
    
    def _generate_fallback_bbox_landmarks(self, img_shape):
        """Generate fallback bbox and landmarks for whole image"""
        h, w = img_shape[:2]
        
        # Use entire image as bounding box
        bbox = [0, 0, w, h]
        
        # Generate reasonable default landmarks (approximate face positions)
        # Left eye, right eye, nose, left mouth corner, right mouth corner
        landmarks = [
            [w // 4, h // 3],          # Left eye
            [3 * w // 4, h // 3],      # Right eye  
            [w // 2, h // 2],          # Nose
            [w // 3, 2 * h // 3],      # Left mouth corner
            [2 * w // 3, 2 * h // 3]   # Right mouth corner
        ]
        
        return bbox, landmarks
    
    def _expand_bbox(self, bbox, expansion_factor, img_shape):
        """Expand bbox by given factor"""
        left, top, right, bottom = bbox
        h, w = img_shape[:2]
        
        # Calculate center and size
        center_x = (left + right) / 2
        center_y = (top + bottom) / 2
        box_w = right - left
        box_h = bottom - top
        
        # Expand
        new_w = box_w * expansion_factor
        new_h = box_h * expansion_factor
        
        # Calculate new coordinates
        new_left = max(0, int(center_x - new_w / 2))
        new_top = max(0, int(center_y - new_h / 2))
        new_right = min(w, int(center_x + new_w / 2))
        new_bottom = min(h, int(center_y + new_h / 2))
        
        return [new_left, new_top, new_right, new_bottom]
    
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
                # Enhanced augmentations for stage2 including mixup/cutmix preparations
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
            
            # Parse annotation or use fallback
            bbox, landmarks = self._parse_annotation(sample['annotation_path'])
            if bbox is None or landmarks is None:
                # Generate fallback bbox and landmarks for whole image
                bbox, landmarks = self._generate_fallback_bbox_landmarks(image.shape)
                if self.debug and idx < 5:  # Only print for first few samples
                    annotation_status = "no annotation file" if sample['annotation_path'] is None else "malformed annotation"
                    print(f"Using whole image fallback for {sample['image_path']} ({annotation_status})")
            
            # Determine expansion factor based on stage
            if self.stage == 'stage1':
                # ConvNext: fixed 1.2x expansion
                expansion_factor = 1.2
            else:
                # MaxViT: random expansion during training
                if self.augment:
                    expansion_factor = random.uniform(1.0, 1.4)
                else:
                    expansion_factor = 1.2
            
            # Expand bbox
            expanded_bbox = self._expand_bbox(bbox, expansion_factor, image.shape)
            
            # Crop face region
            left, top, right, bottom = expanded_bbox
            if right > left and bottom > top:
                face_img = image[top:bottom, left:right]
            else:
                face_img = image  # Fallback to full image
            
            # Apply transforms
            if len(face_img.shape) == 3 and face_img.shape[0] > 0 and face_img.shape[1] > 0:
                transformed = self.transform(image=face_img)
                face_tensor = transformed['image']
            else:
                # Fallback if crop failed
                transformed = self.transform(image=image)
                face_tensor = transformed['image']
            
            # Prepare output
            output = {
                'image': face_tensor,
                'label': torch.tensor(sample['label'], dtype=torch.long),
                'attack_type': sample['attack_type'],
                'image_path': sample['image_path'],
                'bbox': torch.tensor(bbox, dtype=torch.float32),
                'landmarks': torch.tensor(landmarks, dtype=torch.float32),
                'has_annotation': sample['has_annotation']
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
            
            # Check live/spoof structure
            live_path = os.path.join(split_path, 'living')
            spoof_path = os.path.join(split_path, 'spoof')
            
            if os.path.exists(live_path):
                live_subjects = os.listdir(live_path)
                print(f"Live subjects (first 5): {live_subjects[:5]}")
                
                # Check first live subject
                if live_subjects:
                    first_subject = os.path.join(live_path, live_subjects[0])
                    if os.path.isdir(first_subject):
                        files = os.listdir(first_subject)
                        img_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                        txt_files = [f for f in files if f.endswith('.txt')]
                        print(f"First live subject: {len(img_files)} images, {len(txt_files)} annotations")
                        print(f"Sample files: {files[:5]}")
            
            if os.path.exists(spoof_path):
                spoof_categories = os.listdir(spoof_path)
                print(f"Spoof categories: {spoof_categories}")
                
                # Check first spoof category
                if spoof_categories:
                    first_category = os.path.join(spoof_path, spoof_categories[0])
                    if os.path.isdir(first_category):
                        subjects = os.listdir(first_category)
                        print(f"First spoof category subjects (first 5): {subjects[:5]}")
                        
                        if subjects:
                            first_subject = os.path.join(first_category, subjects[0])
                            if os.path.isdir(first_subject):
                                files = os.listdir(first_subject)
                                img_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                                txt_files = [f for f in files if f.endswith('.txt')]
                                print(f"First spoof subject: {len(img_files)} images, {len(txt_files)} annotations")
                                print(f"Sample files: {files[:5]}")


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
    data_root = "/path/to/your/WFAS/dataset"  # Replace with your actual path
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
            
            # Count samples with/without annotations in this batch
            with_ann = sum(batch['has_annotation'])
            without_ann = len(batch['has_annotation']) - with_ann
            print(f"Batch annotation status: {with_ann} with annotations, {without_ann} without")
            
        except Exception as e:
            print(f"Error loading batch: {e}")