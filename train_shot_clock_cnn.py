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
        """
        Initialize augmentation
        apply_prob: probability of applying each augmentation
        """
        self.apply_prob = apply_prob
    
    def random_rotation(self, img, max_angle=15):
        """Randomly rotate image by small angle"""
        if random.random() < self.apply_prob:
            angle = random.uniform(-max_angle, max_angle)
            img = ndimage.rotate(img, angle, reshape=False, mode='constant', cval=0)
        return img
    
    def random_brightness(self, img, factor_range=(0.7, 1.3)):
        """Randomly adjust brightness"""
        if random.random() < self.apply_prob:
            factor = random.uniform(*factor_range)
            img = np.clip(img * factor, 0, 255)
        return img
    
    def random_contrast(self, img, factor_range=(0.8, 1.2)):
        """Randomly adjust contrast"""
        if random.random() < self.apply_prob:
            factor = random.uniform(*factor_range)
            mean = img.mean()
            img = np.clip((img - mean) * factor + mean, 0, 255)
        return img
    
    def random_noise(self, img, noise_level=10):
        """Add random Gaussian noise"""
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
        """Randomly shift image horizontally/vertically"""
        if random.random() < self.apply_prob:
            shift_x = random.randint(-max_shift, max_shift)
            shift_y = random.randint(-max_shift, max_shift)
            
            rows, cols = img.shape
            M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            img = cv2.warpAffine(img, M, (cols, rows), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return img
    
    def random_scale(self, img, scale_range=(0.9, 1.1)):
        """Randomly scale image slightly"""
        if random.random() < self.apply_prob:
            scale = random.uniform(*scale_range)
            h, w = img.shape
            new_h, new_w = int(h * scale), int(w * scale)
            
            # Resize
            img = cv2.resize(img, (new_w, new_h))
            
            # Crop or pad to original size
            if scale > 1:
                # Crop center
                start_y = (new_h - h) // 2
                start_x = (new_w - w) // 2
                img = img[start_y:start_y+h, start_x:start_x+w]
            else:
                # Pad
                pad_y = (h - new_h) // 2
                pad_x = (w - new_w) // 2
                img = cv2.copyMakeBorder(img, pad_y, h-new_h-pad_y, pad_x, w-new_w-pad_x, 
                                        cv2.BORDER_REPLICATE)
        return img
    
    def random_perspective(self, img, distortion=0.05):
        """Apply random perspective transformation"""
        if random.random() < self.apply_prob:
            h, w = img.shape
            
            # Define source points (corners)
            src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            
            # Add random distortion to destination points (much smaller distortion)
            dst_points = src_points.copy()
            for i in range(4):
                dst_points[i][0] += random.uniform(-w*distortion, w*distortion)
                dst_points[i][1] += random.uniform(-h*distortion, h*distortion)
            
            # Apply perspective transform with error handling
            try:
                M = cv2.getPerspectiveTransform(src_points, dst_points)
                img = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            except:
                # If transform fails, return original image
                pass
        return img
    
    def random_invert(self, img):
        """Randomly invert image colors"""
        if random.random() < 0.3:  # Lower probability for inversion
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
        """Apply a random subset of augmentations in random order"""
        # Store as float for transformations
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
            self.random_perspective,
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
        
        # Ensure values are in valid range
        img = np.clip(img, 0, 255).astype(np.uint8)
        
        return img

def create_balanced_sampler(image_paths, labels):
    """
    Create a WeightedRandomSampler that ensures equal probability for each unique shot clock value.
    
    For example, if we have:
    - 5 examples of "15" 
    - 10 examples of "20"
    
    Each image of "15" gets weight = 1/5 of total weight for "15"
    Each image of "20" gets weight = 1/10 of total weight for "20"
    
    This ensures each shot clock value has equal probability of being selected.
    """
    # Count occurrences of each unique label
    label_counts = Counter(labels)
    
    # Calculate weight for each sample
    # Weight = 1 / (count of that label)
    # This makes each unique label equally likely to be selected
    sample_weights = []
    for label in labels:
        weight = 1.0 / label_counts[label]
        sample_weights.append(weight)
    
    print(f"Shot clock label distribution for balanced sampling:")
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

class ShotClockDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, augment=False):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.augment = augment
        
        # Initialize augmentation if needed
        if self.augment:
            self.augmenter = ImageAugmentation(apply_prob=0.8)
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            raise ValueError(f"Could not load image: {img_path}")
        
        # Resize to standard size (e.g., 64x32)
        img = cv2.resize(img, (64, 32))
        
        # Apply augmentation if training
        if self.augment:
            # Add padding before augmentation to prevent cropping digits
            # Pad by 25% on each side (16 pixels horizontally, 8 pixels vertically)
            pad_h = 8
            pad_w = 16
            img_padded = cv2.copyMakeBorder(img, pad_h, pad_h, pad_w, pad_w, 
                                           cv2.BORDER_CONSTANT, value=0)
            
            # Apply augmentation to padded image
            img_padded = self.augmenter(img_padded)
            
            # Crop back to original size (center crop)
            h, w = img_padded.shape
            center_y, center_x = h // 2, w // 2
            img = img_padded[center_y-16:center_y+16, center_x-32:center_x+32]
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Add channel dimension (1, H, W)
        img = np.expand_dims(img, axis=0)
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img)
        
        # Get label
        label_tensor = torch.from_numpy(self.labels[idx])
        
        return img_tensor, label_tensor

class ShotClockCNN(nn.Module):
    def __init__(self):
        super(ShotClockCNN, self).__init__()
        
        # Convolutional layers with more depth and residual-like connections
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
        
        # Output heads with separate hidden layers for better specialization
        self.fc_digit1_hidden = nn.Linear(256, 128)
        self.fc_digit1 = nn.Linear(128, 10)  # First digit (0-9)
        
        self.fc_digit2_hidden = nn.Linear(256, 128)
        self.fc_digit2 = nn.Linear(128, 10)  # Second digit (0-9)
        
        self.fc_blocked = nn.Linear(256, 1)  # Blocked (sigmoid)
        self.fc_inconclusive = nn.Linear(256, 1)  # Inconclusive (sigmoid)
        
    def forward(self, x):
        # Conv layers with deeper architecture
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.dropout_conv(x)
        
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.dropout_conv(x)
        
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = torch.relu(self.bn4(self.conv4(x)))
        x = self.dropout_conv(x)
        
        # Flatten
        x = x.view(-1, 256 * 8 * 4)
        
        # Fully connected layers with batch norm
        x = torch.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout_fc1(x)
        
        x = torch.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout_fc2(x)
        
        # Output heads with separate hidden layers for better feature specialization
        digit1_features = torch.relu(self.fc_digit1_hidden(x))
        digit1 = self.fc_digit1(digit1_features)  # Logits for first digit
        
        digit2_features = torch.relu(self.fc_digit2_hidden(x))
        digit2 = self.fc_digit2(digit2_features)  # Logits for second digit
        
        blocked = torch.sigmoid(self.fc_blocked(x))  # Probability
        inconclusive = torch.sigmoid(self.fc_inconclusive(x))  # Probability
        
        return digit1, digit2, blocked, inconclusive

def load_dataset(dataset_dir):
    """Load dataset from directory with labels.json"""
    labels_path = os.path.join(dataset_dir, 'labels.json')
    
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"labels.json not found in {dataset_dir}")
    
    with open(labels_path, 'r') as f:
        labels_dict = json.load(f)
    
    image_paths = []
    labels = []
    
    for filename, label in labels_dict.items():
        img_path = os.path.join(dataset_dir, filename)
        
        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path}")
            continue
        
        # Convert label to 22-dim vector
        label_vector = parse_label(label)
        
        if label_vector is not None:
            image_paths.append(img_path)
            labels.append(label_vector)
    
    print(f"Loaded {len(image_paths)} images from {dataset_dir}")
    
    return image_paths, np.array(labels, dtype=np.float32)

def parse_label(label):
    """
    Parse label into 22-dimensional vector:
    - 10 dims for first digit (one-hot)
    - 10 dims for second digit (one-hot)
    - 1 dim for blocked (binary)
    - 1 dim for inconclusive (binary)
    """
    vector = np.zeros(22, dtype=np.float32)
    
    if label == 'blocked':
        vector[20] = 1.0  # Blocked flag
        return vector
    elif label == 'inconclusive':
        vector[21] = 1.0  # Inconclusive flag
        return vector
    elif label == 'blank':
        # Skip blank labels - they don't have clear digit values
        return None
    else:
        # Numeric label (0-30)
        try:
            value = int(label)
            if not (0 <= value <= 30):
                print(f"Warning: Invalid value {value}, skipping")
                return None
            
            # Prepend 0 for single digits (e.g., 5 -> 05)
            value_str = str(value).zfill(2)
            
            digit1 = int(value_str[0])  # Tens place
            digit2 = int(value_str[1])  # Ones place
            
            # One-hot encode digits
            vector[digit1] = 1.0  # First digit (0-9)
            vector[10 + digit2] = 1.0  # Second digit (0-9)
            
            return vector
        except ValueError:
            print(f"Warning: Could not parse label '{label}', skipping")
            return None

def train_epoch(model, dataloader, criterion_ce, criterion_bce, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct_digit1 = 0
    correct_digit2 = 0
    total_samples = 0
    
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        
        # Extract label components
        digit1_labels = torch.argmax(labels[:, :10], dim=1)  # First digit
        digit2_labels = torch.argmax(labels[:, 10:20], dim=1)  # Second digit
        blocked_labels = labels[:, 20:21]  # Blocked flag
        inconclusive_labels = labels[:, 21:22]  # Inconclusive flag
        
        # Forward pass
        digit1_out, digit2_out, blocked_out, inconclusive_out = model(images)
        
        # Calculate losses
        loss_digit1 = criterion_ce(digit1_out, digit1_labels)
        loss_digit2 = criterion_ce(digit2_out, digit2_labels)
        loss_blocked = criterion_bce(blocked_out, blocked_labels)
        loss_inconclusive = criterion_bce(inconclusive_out, inconclusive_labels)
        
        # Combined loss
        loss = loss_digit1 + loss_digit2 + loss_blocked + loss_inconclusive
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, pred1 = torch.max(digit1_out, 1)
        _, pred2 = torch.max(digit2_out, 1)
        correct_digit1 += (pred1 == digit1_labels).sum().item()
        correct_digit2 += (pred2 == digit2_labels).sum().item()
        total_samples += labels.size(0)
    
    avg_loss = total_loss / len(dataloader)
    acc_digit1 = 100 * correct_digit1 / total_samples
    acc_digit2 = 100 * correct_digit2 / total_samples
    
    return avg_loss, acc_digit1, acc_digit2

def validate(model, dataloader, criterion_ce, criterion_bce, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    correct_digit1 = 0
    correct_digit2 = 0
    correct_both = 0
    total_samples = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Extract label components
            digit1_labels = torch.argmax(labels[:, :10], dim=1)
            digit2_labels = torch.argmax(labels[:, 10:20], dim=1)
            blocked_labels = labels[:, 20:21]
            inconclusive_labels = labels[:, 21:22]
            
            # Forward pass
            digit1_out, digit2_out, blocked_out, inconclusive_out = model(images)
            
            # Calculate losses
            loss_digit1 = criterion_ce(digit1_out, digit1_labels)
            loss_digit2 = criterion_ce(digit2_out, digit2_labels)
            loss_blocked = criterion_bce(blocked_out, blocked_labels)
            loss_inconclusive = criterion_bce(inconclusive_out, inconclusive_labels)
            
            loss = loss_digit1 + loss_digit2 + loss_blocked + loss_inconclusive
            
            # Statistics
            total_loss += loss.item()
            _, pred1 = torch.max(digit1_out, 1)
            _, pred2 = torch.max(digit2_out, 1)
            correct_digit1 += (pred1 == digit1_labels).sum().item()
            correct_digit2 += (pred2 == digit2_labels).sum().item()
            correct_both += ((pred1 == digit1_labels) & (pred2 == digit2_labels)).sum().item()
            total_samples += labels.size(0)
    
    avg_loss = total_loss / len(dataloader)
    acc_digit1 = 100 * correct_digit1 / total_samples
    acc_digit2 = 100 * correct_digit2 / total_samples
    acc_both = 100 * correct_both / total_samples
    
    return avg_loss, acc_digit1, acc_digit2, acc_both

def main():
    # Hyperparameters
    BATCH_SIZE = 16  # Smaller batch size for better generalization
    LEARNING_RATE = 0.0005  # Lower learning rate for more stable training
    NUM_EPOCHS = 500
    DATASET_DIR = r'shotclock dataset\dataset_MVI_7463 - Trim'
    WEIGHT_DECAY = 1e-4  # L2 regularization
    
    # Load dataset
    print("Loading dataset...")
    image_paths, labels = load_dataset(DATASET_DIR)
    
    if len(image_paths) == 0:
        print("Error: No valid images found!")
        return
    
    print(f"Total samples: {len(image_paths)}")
    print(f"Label shape: {labels.shape}")
    
    # Split into train and validation
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42
    )
    
    print(f"Training samples: {len(train_paths)}")
    print(f"Validation samples: {len(val_paths)}")
    
    # Create datasets (with augmentation for training only)
    train_dataset = ShotClockDataset(train_paths, train_labels, augment=True)
    val_dataset = ShotClockDataset(val_paths, val_labels, augment=False)
    
    # Create balanced samplers for both training and validation
    # This ensures equal probability for each shot clock value in both training and evaluation
    train_sampler = create_balanced_sampler(train_paths, train_labels)
    val_sampler = create_balanced_sampler(val_paths, val_labels)
    
    # Create dataloaders
    # Use balanced samplers for both training and validation
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler)
    
    # Create model
    model = ShotClockCNN().to(device)
    print(f"\nModel architecture:")
    print(model)
    
    # Loss functions with label smoothing for better generalization
    criterion_ce = nn.CrossEntropyLoss(label_smoothing=0.1)  # For digit classification
    criterion_bce = nn.BCELoss()  # For blocked/inconclusive flags
    
    # Optimizer with weight decay (L2 regularization)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Learning rate scheduler - cosine annealing with warm restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc_digit1': [],
        'train_acc_digit2': [],
        'val_acc_digit1': [],
        'val_acc_digit2': [],
        'val_acc_both': []
    }
    
    best_val_acc = 0.0
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(NUM_EPOCHS):
        # Train
        train_loss, train_acc1, train_acc2 = train_epoch(
            model, train_loader, criterion_ce, criterion_bce, optimizer, device
        )
        
        # Validate
        val_loss, val_acc1, val_acc2, val_acc_both = validate(
            model, val_loader, criterion_ce, criterion_bce, device
        )
        
        # Update scheduler (cosine annealing doesn't need loss)
        scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc_digit1'].append(train_acc1)
        history['train_acc_digit2'].append(train_acc2)
        history['val_acc_digit1'].append(val_acc1)
        history['val_acc_digit2'].append(val_acc2)
        history['val_acc_both'].append(val_acc_both)
        
        # Print progress with learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | LR: {current_lr:.6f}")
        print(f"  Train Loss: {train_loss:.4f} | Digit1 Acc: {train_acc1:.2f}% | Digit2 Acc: {train_acc2:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Digit1 Acc: {val_acc1:.2f}% | Digit2 Acc: {val_acc2:.2f}% | Both Acc: {val_acc_both:.2f}%")
        
        # Save best model
        if val_acc_both > best_val_acc:
            best_val_acc = val_acc_both
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc_both': val_acc_both,
            }, 'shot_clock_cnn_best.pth')
            print(f"  ✓ Saved best model (Both Acc: {val_acc_both:.2f}%)")
    
    # Save final model
    torch.save({
        'epoch': NUM_EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
    }, 'shot_clock_cnn_final.pth')
    
    print("\nTraining complete!")
    print(f"Best validation accuracy (both digits): {best_val_acc:.2f}%")
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    
    # Plot loss
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot digit1 accuracy
    plt.subplot(1, 3, 2)
    plt.plot(history['train_acc_digit1'], label='Train Digit1')
    plt.plot(history['val_acc_digit1'], label='Val Digit1')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('First Digit Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot digit2 accuracy
    plt.subplot(1, 3, 3)
    plt.plot(history['train_acc_digit2'], label='Train Digit2')
    plt.plot(history['val_acc_digit2'], label='Val Digit2')
    plt.plot(history['val_acc_both'], label='Val Both Digits', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Second Digit & Both Digits Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("\nTraining history saved to training_history.png")

if __name__ == "__main__":
    main()
