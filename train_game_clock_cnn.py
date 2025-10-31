"""
Game Clock CNN Training Script

Trains a CNN to recognize game clock time in M:SS format (0:00 to 7:59)
Architecture predicts:
- Minute digit (0-7): 8 classes
- First second digit (0-5): 6 classes
- Second second digit (0-9): 10 classes
- Blocked flag: binary (sigmoid)
- Blank/0:00 flag: binary (sigmoid)

Total outputs: 8 + 6 + 10 + 1 + 1 = 26 outputs

Usage:
    python train_game_clock_cnn.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import cv2
import json
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
from scipy import ndimage
from collections import Counter

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class ImageAugmentation:
    """Apply random augmentations to images for training"""
    
    def __init__(self, apply_prob=0.8):
        self.apply_prob = apply_prob
    
    def random_rotation(self, img, max_angle=15):
        if random.random() < self.apply_prob:
            angle = random.uniform(-max_angle, max_angle)
            img = ndimage.rotate(img, angle, reshape=False, mode='constant', cval=0)
        return img
    
    def random_brightness(self, img, factor_range=(0.7, 1.3)):
        if random.random() < self.apply_prob:
            factor = random.uniform(*factor_range)
            img = np.clip(img * factor, 0, 255)
        return img
    
    def random_contrast(self, img, factor_range=(0.8, 1.2)):
        if random.random() < self.apply_prob:
            factor = random.uniform(*factor_range)
            mean = img.mean()
            img = np.clip((img - mean) * factor + mean, 0, 255)
        return img
    
    def random_noise(self, img, noise_level=10):
        if random.random() < self.apply_prob:
            noise = np.random.normal(0, noise_level, img.shape)
            img = np.clip(img + noise, 0, 255)
        return img
    
    def random_blur(self, img, max_kernel=5):
        """Apply random Gaussian or motion blur"""
        if random.random() < self.apply_prob:
            blur_type = random.choice(['gaussian', 'motion', 'median'])
            
            if blur_type == 'gaussian':
                kernel_size = random.choice([1, 3, 5, 7])
                if kernel_size > 1 and kernel_size <= max_kernel:
                    img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
            
            elif blur_type == 'motion':
                # Motion blur in random direction
                kernel_size = random.choice([3, 5, 7])
                angle = random.uniform(0, 360)
                
                # Create motion blur kernel
                kernel = np.zeros((kernel_size, kernel_size))
                kernel[kernel_size // 2, :] = np.ones(kernel_size)
                kernel = kernel / kernel_size
                
                # Rotate kernel
                M = cv2.getRotationMatrix2D((kernel_size // 2, kernel_size // 2), angle, 1)
                kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
                
                img = cv2.filter2D(img, -1, kernel)
            
            elif blur_type == 'median':
                kernel_size = random.choice([3, 5])
                img = cv2.medianBlur(img.astype(np.uint8), kernel_size).astype(np.float32)
        
        return img
    
    def random_shift(self, img, max_shift=3):
        if random.random() < self.apply_prob:
            shift_x = random.randint(-max_shift, max_shift)
            shift_y = random.randint(-max_shift, max_shift)
            
            rows, cols = img.shape
            M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            img = cv2.warpAffine(img, M, (cols, rows), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return img
    
    def random_scale(self, img, scale_range=(0.9, 1.1)):
        if random.random() < self.apply_prob:
            scale = random.uniform(*scale_range)
            h, w = img.shape
            new_h, new_w = int(h * scale), int(w * scale)
            
            img = cv2.resize(img, (new_w, new_h))
            
            if scale > 1:
                # Crop center to original size
                start_y = (new_h - h) // 2
                start_x = (new_w - w) // 2
                img = img[start_y:start_y+h, start_x:start_x+w]
            else:
                # Pad to original size
                pad_y = (h - new_h) // 2
                pad_x = (w - new_w) // 2
                pad_bottom = h - new_h - pad_y
                pad_right = w - new_w - pad_x
                img = cv2.copyMakeBorder(img, pad_y, pad_bottom, pad_x, pad_right, 
                                        cv2.BORDER_CONSTANT, value=0)
        return img
    
    def random_invert(self, img):
        if random.random() < 0.3:
            img = 255 - img
        return img
    
    def random_color_shuffle(self, img):
        """Randomly shuffle color channels (for grayscale, this adds color tinting)"""
        if random.random() < self.apply_prob:
            # For grayscale images, we can simulate color shuffling by:
            # 1. Converting to a pseudo-RGB by duplicating channels
            # 2. Applying different brightness/contrast to each channel
            # 3. Converting back to grayscale with different weights
            
            # Create 3-channel version
            img_rgb = np.stack([img, img, img], axis=-1)
            
            # Apply different transformations to each channel
            for c in range(3):
                # Random brightness per channel
                brightness_factor = random.uniform(0.7, 1.3)
                img_rgb[:, :, c] = np.clip(img_rgb[:, :, c] * brightness_factor, 0, 255)
                
                # Random contrast per channel
                contrast_factor = random.uniform(0.8, 1.2)
                mean = img_rgb[:, :, c].mean()
                img_rgb[:, :, c] = np.clip((img_rgb[:, :, c] - mean) * contrast_factor + mean, 0, 255)
            
            # Convert back to grayscale with shuffled weights
            # Normal weights: [0.299, 0.587, 0.114]
            # Shuffle them
            weights = [0.299, 0.587, 0.114]
            random.shuffle(weights)
            
            img = (img_rgb[:, :, 0] * weights[0] + 
                   img_rgb[:, :, 1] * weights[1] + 
                   img_rgb[:, :, 2] * weights[2])
            
            img = np.clip(img, 0, 255)
        
        return img
    
    def photometric_distort(self, img):
        """Apply random photometric distortions including brightness, contrast, saturation, hue"""
        if random.random() < self.apply_prob:
            # Brightness distortion
            if random.random() < 0.5:
                delta = random.uniform(-32, 32)
                img = np.clip(img + delta, 0, 255)
            
            # Contrast distortion
            if random.random() < 0.5:
                alpha = random.uniform(0.5, 1.5)
                img = np.clip(img * alpha, 0, 255)
                
            # For grayscale, simulate saturation by adjusting dynamic range
            if random.random() < 0.5:
                # Increase or decrease the range of values
                min_val, max_val = img.min(), img.max()
                range_factor = random.uniform(0.7, 1.3)
                center = (min_val + max_val) / 2
                img = np.clip((img - center) * range_factor + center, 0, 255)
        
        return img
    
    def random_affine(self, img, rotation_range=15, translation_range=0.1, scale_range=(0.8, 1.2), shear_range=10):
        """Apply random affine transformations"""
        if random.random() < self.apply_prob:
            h, w = img.shape
            
            # Random rotation
            angle = random.uniform(-rotation_range, rotation_range)
            
            # Random translation
            tx = random.uniform(-translation_range, translation_range) * w
            ty = random.uniform(-translation_range, translation_range) * h
            
            # Random scaling
            scale = random.uniform(*scale_range)
            
            # Random shear
            shear_x = random.uniform(-shear_range, shear_range)
            shear_y = random.uniform(-shear_range, shear_range)
            
            # Create transformation matrix
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, scale)
            
            # Add translation
            M[0, 2] += tx
            M[1, 2] += ty
            
            # Add shear
            shear_matrix = np.array([[1, np.tan(np.radians(shear_x)), 0],
                                   [np.tan(np.radians(shear_y)), 1, 0]], dtype=np.float32)
            M = np.dot(shear_matrix, np.vstack([M, [0, 0, 1]]))[:2]
            
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
        return img
    
    def elastic_transform(self, img, alpha=50, sigma=5):
        """Apply elastic deformation"""
        if random.random() < self.apply_prob:
            h, w = img.shape
            
            # Create random displacement fields
            dx = cv2.GaussianBlur((np.random.rand(h, w) * 2 - 1), (0, 0), sigma) * alpha
            dy = cv2.GaussianBlur((np.random.rand(h, w) * 2 - 1), (0, 0), sigma) * alpha
            
            # Create coordinate grids
            x, y = np.meshgrid(np.arange(w), np.arange(h))
            x_new = np.clip(x + dx, 0, w - 1).astype(np.float32)
            y_new = np.clip(y + dy, 0, h - 1).astype(np.float32)
            
            # Apply transformation
            img = cv2.remap(img, x_new, y_new, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
        return img
    
    def color_jitter(self, img, brightness=0.2, contrast=0.2):
        """Apply color jittering (adapted for grayscale)"""
        if random.random() < self.apply_prob:
            # Brightness jitter
            if random.random() < 0.8:
                brightness_factor = random.uniform(1 - brightness, 1 + brightness)
                img = np.clip(img * brightness_factor, 0, 255)
            
            # Contrast jitter
            if random.random() < 0.8:
                contrast_factor = random.uniform(1 - contrast, 1 + contrast)
                mean = img.mean()
                img = np.clip((img - mean) * contrast_factor + mean, 0, 255)
                
            # Simulate hue jitter by shifting intensity histogram
            if random.random() < 0.8:
                hue_shift = random.uniform(-20, 20)
                img = np.clip(img + hue_shift, 0, 255)
        
        return img
    
    def random_posterize(self, img, bits_range=(4, 8)):
        """Apply random posterization (reduce number of bits per channel)"""
        if random.random() < self.apply_prob:
            bits = random.randint(*bits_range)
            shift = 8 - bits
            img = ((img.astype(np.uint8) >> shift) << shift).astype(np.float32)
        
        return img
    
    def __call__(self, img):
        img = img.astype(np.float32)
        
        # 10% chance of no augmentation to preserve original data quality
        if random.random() < 0.1:
            return np.clip(img, 0, 255).astype(np.uint8)
        
        # Define all available augmentations
        augmentations = [
            self.random_rotation,
            self.random_brightness,
            self.random_contrast,
            self.random_noise,
            self.random_blur,
            self.random_shift,
            self.random_scale,
            self.random_color_shuffle,
            self.random_invert,
            self.photometric_distort,
            self.random_affine,
            self.elastic_transform,
            self.color_jitter,
            self.random_posterize
        ]
        
        # Randomly select only 2-4 augmentations to apply (much more reasonable)
        num_augmentations = random.randint(2, 4)
        selected_augmentations = random.sample(augmentations, num_augmentations)
        
        # Apply selected augmentations in random order
        random.shuffle(selected_augmentations)
        
        for aug in selected_augmentations:
            img = aug(img)
        
        return np.clip(img, 0, 255).astype(np.uint8)


def create_balanced_sampler(image_files, labels_dict):
    """
    Create a WeightedRandomSampler that ensures equal probability for each unique time value.
    
    For example, if we have:
    - 3 examples of 1:03 
    - 6 examples of 1:04
    
    Each image of 1:03 gets weight = 1/3 of total weight for 1:03
    Each image of 1:04 gets weight = 1/6 of total weight for 1:04
    
    This ensures each time value has equal probability of being selected.
    """
    # Count occurrences of each unique label
    label_counts = Counter([labels_dict[img_file] for img_file in image_files])
    
    # Calculate weight for each sample
    # Weight = 1 / (count of that label)
    # This makes each unique label equally likely to be selected
    sample_weights = []
    for img_file in image_files:
        label = labels_dict[img_file]
        weight = 1.0 / label_counts[label]
        sample_weights.append(weight)
    
    print(f"Label distribution for balanced sampling:")
    for label, count in sorted(label_counts.items()):
        weight_per_sample = 1.0 / count
        total_weight_for_label = weight_per_sample * count
        print(f"  {label}: {count} images, weight per sample: {weight_per_sample:.4f}, total weight: {total_weight_for_label:.4f}")
    
    # Create weighted sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),  # Sample as many as we have images per epoch
        replacement=True  # Allow replacement since we want equal label probability
    )
    
    return sampler


class GameClockDataset(Dataset):
    """Dataset for game clock images with M:SS labels"""
    
    def __init__(self, image_files, labels_dict, dataset_dir, augment=False):
        self.image_files = image_files
        self.labels_dict = labels_dict
        self.dataset_dir = dataset_dir
        self.augment = augment
        
        if augment:
            self.augmentation = ImageAugmentation(apply_prob=0.8)
        
        print(f"Dataset: {len(image_files)} images, augmentation={'ON' if augment else 'OFF'}")
    
    def __len__(self):
        return len(self.image_files)
    
    def parse_label(self, label_str):
        """
        Parse label string into minute, seconds1, seconds2, blocked, blank
        Returns: (minute_class, sec1_class, sec2_class, is_blocked, is_blank)
        """
        if label_str == "BLOCKED":
            return 0, 0, 0, 1.0, 0.0
        elif label_str == "0:00" or label_str == "BLANK":
            return 0, 0, 0, 0.0, 1.0
        elif label_str == "INCONCLUSIVE":
            # Treat inconclusive as blocked for training
            return 0, 0, 0, 1.0, 0.0
        else:
            # Parse M:SS format
            try:
                parts = label_str.split(':')
                if len(parts) != 2:
                    print(f"Warning: Invalid label format: {label_str}")
                    return 0, 0, 0, 1.0, 0.0
                
                minute = int(parts[0])
                seconds = int(parts[1])
                
                # Validate ranges
                if not (0 <= minute <= 7):
                    print(f"Warning: Invalid minute {minute} in {label_str}")
                    minute = 0
                if not (0 <= seconds <= 59):
                    print(f"Warning: Invalid seconds {seconds} in {label_str}")
                    seconds = 0
                
                # Split seconds into two digits
                sec1 = seconds // 10  # Tens digit (0-5)
                sec2 = seconds % 10   # Ones digit (0-9)
                
                return minute, sec1, sec2, 0.0, 0.0
            except Exception as e:
                print(f"Error parsing label {label_str}: {e}")
                return 0, 0, 0, 1.0, 0.0
    
    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        label_str = self.labels_dict[img_file]
        
        # Load image
        img_path = os.path.join(self.dataset_dir, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            print(f"Error loading image: {img_path}")
            img = np.zeros((32, 64), dtype=np.uint8)
        
        # Apply augmentation if training
        if self.augment:
            img = self.augmentation(img)
        
        # Resize to 64x32 (width x height)
        img = cv2.resize(img, (64, 32))
        
        # Normalize
        img = img.astype(np.float32) / 255.0
        
        # Add channel dimension (1, H, W)
        img = np.expand_dims(img, axis=0)
        
        # Parse label
        minute, sec1, sec2, blocked, blank = self.parse_label(label_str)
        
        # Convert to tensors
        img_tensor = torch.from_numpy(img).float()
        minute_tensor = torch.tensor(minute, dtype=torch.long)
        sec1_tensor = torch.tensor(sec1, dtype=torch.long)
        sec2_tensor = torch.tensor(sec2, dtype=torch.long)
        blocked_tensor = torch.tensor(blocked, dtype=torch.float32)
        blank_tensor = torch.tensor(blank, dtype=torch.float32)
        
        return img_tensor, minute_tensor, sec1_tensor, sec2_tensor, blocked_tensor, blank_tensor


class GameClockCNN(nn.Module):
    """CNN for game clock recognition in M:SS format"""
    
    def __init__(self):
        super(GameClockCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout_conv = nn.Dropout2d(0.3)
        
        # Calculate flattened size after convolutions
        # Input: 64x32 -> after 3 pools: 8x4 -> 8*4*256 = 8192
        self.fc1 = nn.Linear(256 * 8 * 4, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.dropout_fc1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.dropout_fc2 = nn.Dropout(0.4)
        
        # Output heads with separate hidden layers
        self.fc_minute_hidden = nn.Linear(256, 128)
        self.fc_minute = nn.Linear(128, 8)  # Minute (0-7)
        
        self.fc_sec1_hidden = nn.Linear(256, 128)
        self.fc_sec1 = nn.Linear(128, 6)  # First second digit (0-5)
        
        # Larger hidden layer for sec2 (hardest digit - 10 classes, changes fastest)
        self.fc_sec2_hidden = nn.Linear(256, 192)
        self.dropout_sec2 = nn.Dropout(0.3)  # Extra regularization for hardest digit
        self.fc_sec2 = nn.Linear(192, 10)  # Second second digit (0-9)
        
        self.fc_blocked = nn.Linear(256, 1)  # Blocked (sigmoid)
        self.fc_blank = nn.Linear(256, 1)  # Blank/0:00 (sigmoid)
    
    def forward(self, x):
        # Conv layers
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.dropout_conv(x)
        
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.dropout_conv(x)
        
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = torch.relu(self.bn4(self.conv4(x)))
        x = self.dropout_conv(x)
        
        # Flatten
        x = x.view(-1, 256 * 8 * 4)
        
        # Fully connected layers
        x = torch.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout_fc1(x)
        
        x = torch.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout_fc2(x)
        
        # Output heads
        minute_features = torch.relu(self.fc_minute_hidden(x))
        minute = self.fc_minute(minute_features)
        
        sec1_features = torch.relu(self.fc_sec1_hidden(x))
        sec1 = self.fc_sec1(sec1_features)
        
        sec2_features = torch.relu(self.fc_sec2_hidden(x))
        sec2_features = self.dropout_sec2(sec2_features)  # Extra dropout for hardest digit
        sec2 = self.fc_sec2(sec2_features)
        
        blocked = torch.sigmoid(self.fc_blocked(x))
        blank = torch.sigmoid(self.fc_blank(x))
        
        return minute, sec1, sec2, blocked, blank


def train_epoch(model, train_loader, criterion_ce, criterion_ce_sec2, criterion_bce, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct_minute = 0
    correct_sec1 = 0
    correct_sec2 = 0
    correct_all = 0
    total = 0
    
    for batch_idx, (images, minutes, sec1s, sec2s, blockeds, blanks) in enumerate(train_loader):
        images = images.to(device)
        minutes = minutes.to(device)
        sec1s = sec1s.to(device)
        sec2s = sec2s.to(device)
        blockeds = blockeds.to(device).unsqueeze(1)
        blanks = blanks.to(device).unsqueeze(1)
        
        optimizer.zero_grad()
        
        # Forward pass
        minute_out, sec1_out, sec2_out, blocked_out, blank_out = model(images)
        
        # Calculate losses (use label smoothing for sec2)
        loss_minute = criterion_ce(minute_out, minutes)
        loss_sec1 = criterion_ce(sec1_out, sec1s)
        loss_sec2 = criterion_ce_sec2(sec2_out, sec2s)  # Label smoothing for hardest digit
        loss_blocked = criterion_bce(blocked_out, blockeds)
        loss_blank = criterion_bce(blank_out, blanks)
        
        # Combined loss (weight sec2 most heavily since it's hardest to predict)
        # sec2 has 10 classes and changes fastest, so needs extra emphasis
        loss = 2.0 * loss_minute + 2.5 * loss_sec1 + 4.0 * loss_sec2 + loss_blocked + loss_blank
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Calculate accuracy
        _, pred_minute = torch.max(minute_out, 1)
        _, pred_sec1 = torch.max(sec1_out, 1)
        _, pred_sec2 = torch.max(sec2_out, 1)
        
        correct_minute += (pred_minute == minutes).sum().item()
        correct_sec1 += (pred_sec1 == sec1s).sum().item()
        correct_sec2 += (pred_sec2 == sec2s).sum().item()
        
        # All digits correct
        all_correct = (pred_minute == minutes) & (pred_sec1 == sec1s) & (pred_sec2 == sec2s)
        correct_all += all_correct.sum().item()
        
        total += minutes.size(0)
    
    avg_loss = total_loss / len(train_loader)
    acc_minute = 100.0 * correct_minute / total
    acc_sec1 = 100.0 * correct_sec1 / total
    acc_sec2 = 100.0 * correct_sec2 / total
    acc_all = 100.0 * correct_all / total
    
    return avg_loss, acc_minute, acc_sec1, acc_sec2, acc_all


def validate(model, val_loader, criterion_ce, criterion_ce_sec2, criterion_bce, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    correct_minute = 0
    correct_sec1 = 0
    correct_sec2 = 0
    correct_all = 0
    total = 0
    
    with torch.no_grad():
        for images, minutes, sec1s, sec2s, blockeds, blanks in val_loader:
            images = images.to(device)
            minutes = minutes.to(device)
            sec1s = sec1s.to(device)
            sec2s = sec2s.to(device)
            blockeds = blockeds.to(device).unsqueeze(1)
            blanks = blanks.to(device).unsqueeze(1)
            
            # Forward pass
            minute_out, sec1_out, sec2_out, blocked_out, blank_out = model(images)
            
            # Calculate losses (use label smoothing for sec2)
            loss_minute = criterion_ce(minute_out, minutes)
            loss_sec1 = criterion_ce(sec1_out, sec1s)
            loss_sec2 = criterion_ce_sec2(sec2_out, sec2s)
            loss_blocked = criterion_bce(blocked_out, blockeds)
            loss_blank = criterion_bce(blank_out, blanks)
            
            # Combined loss (weight sec2 most heavily since it's hardest to predict)
            loss = 2.0 * loss_minute + 2.5 * loss_sec1 + 4.0 * loss_sec2 + loss_blocked + loss_blank
            
            total_loss += loss.item()
            
            # Calculate accuracy
            _, pred_minute = torch.max(minute_out, 1)
            _, pred_sec1 = torch.max(sec1_out, 1)
            _, pred_sec2 = torch.max(sec2_out, 1)
            
            correct_minute += (pred_minute == minutes).sum().item()
            correct_sec1 += (pred_sec1 == sec1s).sum().item()
            correct_sec2 += (pred_sec2 == sec2s).sum().item()
            
            all_correct = (pred_minute == minutes) & (pred_sec1 == sec1s) & (pred_sec2 == sec2s)
            correct_all += all_correct.sum().item()
            
            total += minutes.size(0)
    
    avg_loss = total_loss / len(val_loader)
    acc_minute = 100.0 * correct_minute / total
    acc_sec1 = 100.0 * correct_sec1 / total
    acc_sec2 = 100.0 * correct_sec2 / total
    acc_all = 100.0 * correct_all / total
    
    return avg_loss, acc_minute, acc_sec1, acc_sec2, acc_all


def main():
    print("=" * 60)
    print("GAME CLOCK CNN TRAINER")
    print("=" * 60)
    
    # Configuration
    dataset_dir = "game_clock_dataset"
    labels_file = "game_clock_labels.json"
    batch_size = 32
    num_epochs = 250
    learning_rate = 0.001
    
    # Load labels
    if not os.path.exists(labels_file):
        print(f"Error: Labels file not found: {labels_file}")
        print("Please run label_dataset_game_clock.py first")
        return
    
    with open(labels_file, 'r') as f:
        labels_dict = json.load(f)
    
    print(f"Loaded {len(labels_dict)} labels from {labels_file}")
    
    # Get labeled images
    labeled_files = list(labels_dict.keys())
    
    if len(labeled_files) < 10:
        print(f"Error: Not enough labeled images ({len(labeled_files)})")
        print("Please label more images first")
        return
    
    # Split into train/val (80/20)
    train_files, val_files = train_test_split(labeled_files, test_size=0.2, random_state=42)
    
    print(f"Train: {len(train_files)} images")
    print(f"Val: {len(val_files)} images")
    
    # Create datasets
    train_dataset = GameClockDataset(train_files, labels_dict, dataset_dir, augment=True)
    val_dataset = GameClockDataset(val_files, labels_dict, dataset_dir, augment=False)
    
    # Create balanced samplers for both training and validation
    # This ensures equal probability for each time value in both training and evaluation
    train_sampler = create_balanced_sampler(train_files, labels_dict)
    val_sampler = create_balanced_sampler(val_files, labels_dict)
    
    # Create dataloaders
    # Use balanced samplers for both training and validation
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=0)
    
    # Create model
    model = GameClockCNN().to(device)
    
    # Loss functions
    # Use label smoothing for sec2 (hardest digit) to reduce overconfidence
    criterion_ce = nn.CrossEntropyLoss()
    criterion_ce_sec2 = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing for sec2
    criterion_bce = nn.BCELoss()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, 
                                                     patience=10)
    
    # Training loop
    best_val_acc = 0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    print("\nStarting training...")
    print(f"{'Epoch':<6} {'Train Loss':<12} {'Train Acc':<12} {'Val Loss':<12} {'Val Acc':<12} {'Best':<8}")
    print("-" * 70)
    
    for epoch in range(num_epochs):
        # Train
        train_loss, train_acc_m, train_acc_s1, train_acc_s2, train_acc_all = train_epoch(
            model, train_loader, criterion_ce, criterion_ce_sec2, criterion_bce, optimizer, device
        )
        
        # Validate
        val_loss, val_acc_m, val_acc_s1, val_acc_s2, val_acc_all = validate(
            model, val_loader, criterion_ce, criterion_ce_sec2, criterion_bce, device
        )
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc_all)
        val_accs.append(val_acc_all)
        
        # Learning rate scheduling
        scheduler.step(val_acc_all)
        
        # Print progress
        best_marker = ""
        if val_acc_all > best_val_acc:
            best_val_acc = val_acc_all
            best_marker = "⭐ NEW"
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc_all': val_acc_all,
                'val_acc_minute': val_acc_m,
                'val_acc_sec1': val_acc_s1,
                'val_acc_sec2': val_acc_s2,
            }, 'game_clock_cnn_best.pth')
        
        print(f"{epoch+1:<6} {train_loss:<12.4f} {train_acc_all:<12.2f} {val_loss:<12.4f} {val_acc_all:<12.2f} {best_marker:<8}")
        
        # Detailed accuracy every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"  Detailed - Train: M={train_acc_m:.1f}% S1={train_acc_s1:.1f}% S2={train_acc_s2:.1f}%")
            print(f"  Detailed - Val:   M={val_acc_m:.1f}% S1={val_acc_s1:.1f}% S2={val_acc_s2:.1f}%")
    
    print("\n" + "=" * 70)
    print(f"Training complete!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: game_clock_cnn_best.pth")
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy (All Digits Correct)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('game_clock_training_curves.png', dpi=150)
    print(f"Training curves saved to: game_clock_training_curves.png")
    
    print("\nNext steps:")
    print("1. Check training curves in game_clock_training_curves.png")
    print("2. If accuracy is low, label more images and retrain")
    print("3. Use the trained model with annotate_video_game_clock.py")


if __name__ == "__main__":
    main()
