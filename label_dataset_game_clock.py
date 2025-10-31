"""
Game Clock Dataset Labeling Tool
Interactive GUI for labeling game clock images extracted from video.

Game clock format: M:SS (minutes:seconds, where minutes is 0-7)
Special labels:
- 'B' = BLOCKED (obscured by player/object)
- 'X' = BLANK / 0:00 (no clock visible or time expired)
- 'S' = Inconclusive (unclear/blurry)
- 'G' = Auto-guess using OCR (Tesseract)

Usage:
    python label_dataset_game_clock.py

Keyboard shortcuts:
    0-9: Enter digits for M:SS
    :: Add colon separator (or press Enter after 1 digit for minutes)
    B: Mark as BLOCKED
    X: Mark as BLANK (0:00)
    S: Mark as Inconclusive
    G: Auto-guess with OCR
    N: Next image (skip without labeling)
    P: Previous image
    D: Delete current label
    Q/ESC: Quit and save
"""

import cv2
import numpy as np
import json
import os
import sys
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk
import subprocess
import re
import random

class GameClockLabeler:
    def __init__(self, dataset_dir="game_clock_dataset", labels_file="game_clock_labels.json", shuffle=True):
        self.dataset_dir = dataset_dir
        self.labels_file = labels_file
        self.labels = {}
        
        # Load existing labels if they exist
        if os.path.exists(labels_file):
            with open(labels_file, 'r') as f:
                self.labels = json.load(f)
            print(f"Loaded {len(self.labels)} existing labels from {labels_file}")
        
        # Get list of images
        self.image_files = sorted([f for f in os.listdir(dataset_dir) 
                                   if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        if not self.image_files:
            print(f"Error: No images found in {dataset_dir}")
            sys.exit(1)
        
        # Shuffle images for more diverse labeling
        if shuffle:
            random.seed(42)  # Use seed for reproducibility
            random.shuffle(self.image_files)
            print(f"Shuffled image order for diverse labeling")
        
        print(f"Found {len(self.image_files)} images")
        
        # Count labeled vs unlabeled
        labeled_count = sum(1 for f in self.image_files if f in self.labels)
        print(f"Labeled: {labeled_count}, Unlabeled: {len(self.image_files) - labeled_count}")
        
        self.current_idx = 0
        self.current_input = ""
        
        # Find first unlabeled image
        for i, img_file in enumerate(self.image_files):
            if img_file not in self.labels:
                self.current_idx = i
                break
        
        # Setup GUI
        self.setup_gui()
    
    def setup_gui(self):
        """Create the GUI interface"""
        self.root = tk.Tk()
        self.root.title("Game Clock Dataset Labeler")
        self.root.geometry("1400x900")
        self.root.configure(bg='#1e1e1e')
        
        # Bind keyboard events
        self.root.bind('<Key>', self.on_key_press)
        self.root.bind('<Escape>', lambda e: self.quit_app())
        
        # Main container
        main_frame = tk.Frame(self.root, bg='#1e1e1e')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title section
        title_frame = tk.Frame(main_frame, bg='#2d2d2d', relief=tk.RAISED, borderwidth=2)
        title_frame.pack(fill=tk.X, pady=(0, 15))
        
        tk.Label(
            title_frame,
            text="🎮 GAME CLOCK LABELER",
            bg='#2d2d2d',
            fg='#4CAF50',
            font=('Arial', 20, 'bold'),
            pady=15
        ).pack()
        
        # Stats bar
        stats_frame = tk.Frame(main_frame, bg='#2d2d2d', relief=tk.RIDGE, borderwidth=2)
        stats_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.stats_label = tk.Label(
            stats_frame,
            text="",
            bg='#2d2d2d',
            fg='#FFD700',
            font=('Arial', 12),
            pady=10
        )
        self.stats_label.pack()
        
        # Image display area
        image_frame = tk.Frame(main_frame, bg='#000000', relief=tk.SUNKEN, borderwidth=3)
        image_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        self.image_label = tk.Label(image_frame, bg='#000000')
        self.image_label.pack(expand=True)
        
        # Current label input area
        input_frame = tk.Frame(main_frame, bg='#2d2d2d', relief=tk.RAISED, borderwidth=2)
        input_frame.pack(fill=tk.X, pady=(0, 15))
        
        tk.Label(
            input_frame,
            text="Current Input:",
            bg='#2d2d2d',
            fg='#FFFFFF',
            font=('Arial', 12, 'bold')
        ).pack(side=tk.LEFT, padx=10, pady=10)
        
        self.input_label = tk.Label(
            input_frame,
            text="",
            bg='#000000',
            fg='#00FF00',
            font=('Courier New', 24, 'bold'),
            width=10,
            relief=tk.SUNKEN,
            borderwidth=2
        )
        self.input_label.pack(side=tk.LEFT, padx=10, pady=10)
        
        # Existing label display
        tk.Label(
            input_frame,
            text="Existing Label:",
            bg='#2d2d2d',
            fg='#FFFFFF',
            font=('Arial', 12, 'bold')
        ).pack(side=tk.LEFT, padx=(30, 10), pady=10)
        
        self.existing_label = tk.Label(
            input_frame,
            text="None",
            bg='#1e1e1e',
            fg='#FFD700',
            font=('Courier New', 18, 'bold'),
            width=12,
            relief=tk.SUNKEN,
            borderwidth=2
        )
        self.existing_label.pack(side=tk.LEFT, padx=10, pady=10)
        
        # Instructions
        instructions_frame = tk.Frame(main_frame, bg='#2d2d2d', relief=tk.GROOVE, borderwidth=2)
        instructions_frame.pack(fill=tk.X)
        
        instructions_text = """
KEYBOARD SHORTCUTS:
• 0-9: Type game clock time (M:SS format, e.g. 7:30, 4:15)
• B: BLOCKED (obscured)  • X: BLANK (0:00 / no clock visible)  • S: Inconclusive  • G: OCR Auto-guess
• Enter: Submit label  • N: Next (skip)  • P: Previous  • D: Delete label  • Q/ESC: Quit & Save
        """
        
        tk.Label(
            instructions_frame,
            text=instructions_text,
            bg='#2d2d2d',
            fg='#CCCCCC',
            font=('Arial', 10),
            justify=tk.LEFT,
            pady=10
        ).pack()
        
        # Load first image
        self.display_current_image()
    
    def display_current_image(self):
        """Display the current image and update stats"""
        if self.current_idx >= len(self.image_files):
            messagebox.showinfo("Complete", "All images labeled!")
            self.quit_app()
            return
        
        # Load image
        img_filename = self.image_files[self.current_idx]
        img_path = os.path.join(self.dataset_dir, img_filename)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Error loading image: {img_path}")
            self.current_idx += 1
            self.display_current_image()
            return
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize for display (scale up for better visibility)
        scale_factor = 4
        display_img = cv2.resize(img_rgb, None, fx=scale_factor, fy=scale_factor, 
                                interpolation=cv2.INTER_NEAREST)
        
        # Convert to PIL Image
        pil_img = Image.fromarray(display_img)
        
        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(pil_img)
        
        # Update image label
        self.image_label.configure(image=photo)
        self.image_label.image = photo  # Keep reference
        
        # Update stats
        labeled_count = sum(1 for f in self.image_files if f in self.labels)
        unlabeled_count = len(self.image_files) - labeled_count
        progress_pct = (labeled_count / len(self.image_files)) * 100
        
        stats_text = (f"Image {self.current_idx + 1} / {len(self.image_files)}  |  "
                     f"Labeled: {labeled_count}  |  Unlabeled: {unlabeled_count}  |  "
                     f"Progress: {progress_pct:.1f}%  |  File: {img_filename}")
        self.stats_label.config(text=stats_text)
        
        # Update existing label display
        if img_filename in self.labels:
            label_text = self.labels[img_filename]
            # Color code special labels
            if label_text == "BLOCKED":
                self.existing_label.config(text="🚫 " + label_text, fg='#FF5555')  # Red
            elif label_text == "0:00":
                self.existing_label.config(text="⏱️ " + label_text, fg='#55FFFF')  # Cyan
            elif label_text == "INCONCLUSIVE":
                self.existing_label.config(text="❓ " + label_text, fg='#FFAA55')  # Orange
            else:
                self.existing_label.config(text="✓ " + label_text, fg='#55FF55')  # Green
        else:
            self.existing_label.config(text="None", fg='#888888')
        
        # Clear current input
        self.current_input = ""
        self.input_label.config(text=self.current_input)
    
    def on_key_press(self, event):
        """Handle keyboard input"""
        key = event.char.lower()
        keysym = event.keysym
        
        img_filename = self.image_files[self.current_idx]
        
        # Navigation keys
        if keysym == 'Escape' or key == 'q':
            self.quit_app()
            return
        elif key == 'n' or keysym == 'Right':
            self.next_image()
            return
        elif key == 'p' or keysym == 'Left':
            self.previous_image()
            return
        elif key == 'd':
            self.delete_label()
            return
        
        # Special labels
        if key == 'b':
            self.save_label("BLOCKED")
            return
        elif key == 'x':
            # X key for blank/0:00 (easier than typing 000)
            self.save_label("0:00")
            return
        elif key == 's':
            self.save_label("INCONCLUSIVE")
            return
        elif key == 'g':
            self.ocr_guess()
            return
        
        # Digit input
        if key.isdigit():
            self.current_input += key
            
            # Auto-format: add colon after 1 digit (M:)
            if len(self.current_input) == 1 and ':' not in self.current_input:
                self.current_input += ':'
            
            # Check for special case: 0:00 (blank)
            if self.current_input == "0:00":
                self.save_label("0:00")
                return
            
            self.input_label.config(text=self.current_input, fg='#FFFFFF')
        
        # Colon input
        elif key == ':' and len(self.current_input) >= 1:
            if ':' not in self.current_input:
                self.current_input += ':'
                self.input_label.config(text=self.current_input, fg='#FFFFFF')
        
        # Backspace
        elif keysym == 'BackSpace':
            self.current_input = self.current_input[:-1]
            self.input_label.config(text=self.current_input, fg='#FFFFFF')
        
        # Enter to submit
        elif keysym == 'Return' or keysym == 'KP_Enter':
            if self.validate_input():
                self.save_label(self.current_input)
    
    def validate_input(self):
        """Validate M:SS format"""
        if not self.current_input:
            return False
        
        # Check for special labels
        if self.current_input.upper() in ["BLOCKED", "INCONCLUSIVE", "0:00"]:
            return True
        
        # Validate M:SS format (single digit minutes)
        pattern = r'^\d:\d{2}$'
        if not re.match(pattern, self.current_input):
            messagebox.showwarning("Invalid Format", 
                                  "Please enter time in M:SS format (e.g., 7:30, 4:15, 0:45)")
            return False
        
        # Parse and validate ranges
        try:
            parts = self.current_input.split(':')
            minutes = int(parts[0])
            seconds = int(parts[1])
            
            if minutes < 0 or minutes > 7:
                messagebox.showwarning("Invalid Minutes", "Minutes must be 0-7 (max 7 minute periods)")
                return False
            
            if seconds < 0 or seconds > 59:
                messagebox.showwarning("Invalid Seconds", "Seconds must be 0-59")
                return False
            
            return True
        except ValueError:
            return False
    
    def save_label(self, label):
        """Save label for current image"""
        img_filename = self.image_files[self.current_idx]
        self.labels[img_filename] = label
        
        # Save to file
        with open(self.labels_file, 'w') as f:
            json.dump(self.labels, f, indent=2)
        
        # Print with visual indicator
        if label == "BLOCKED":
            print(f"🚫 Labeled {img_filename} as: {label}")
        elif label == "0:00":
            print(f"⏱️ Labeled {img_filename} as: {label} (BLANK)")
        elif label == "INCONCLUSIVE":
            print(f"❓ Labeled {img_filename} as: {label}")
        else:
            print(f"✓ Labeled {img_filename} as: {label}")
        
        # Move to next image
        self.next_image()
    
    def next_image(self):
        """Move to next image"""
        if self.current_idx < len(self.image_files) - 1:
            self.current_idx += 1
            self.display_current_image()
        else:
            messagebox.showinfo("End", "Reached last image!")
    
    def previous_image(self):
        """Move to previous image"""
        if self.current_idx > 0:
            self.current_idx -= 1
            self.display_current_image()
        else:
            messagebox.showinfo("Start", "Already at first image!")
    
    def delete_label(self):
        """Delete label for current image"""
        img_filename = self.image_files[self.current_idx]
        if img_filename in self.labels:
            del self.labels[img_filename]
            
            # Save to file
            with open(self.labels_file, 'w') as f:
                json.dump(self.labels, f, indent=2)
            
            print(f"Deleted label for {img_filename}")
            self.display_current_image()
        else:
            messagebox.showinfo("No Label", "Current image has no label to delete")
    
    def ocr_guess(self):
        """Use OCR to guess the game clock value"""
        img_filename = self.image_files[self.current_idx]
        img_path = os.path.join(self.dataset_dir, img_filename)
        
        print(f"Running OCR on {img_filename}...")
        guess = self.run_ocr_on_image(img_path)
        
        if guess:
            self.current_input = guess
            self.input_label.config(text=f"OCR: {self.current_input}", fg='#FFD700')
            print(f"✓ OCR guess: {guess}")
            
            # Show message box with guess
            response = messagebox.askyesno(
                "OCR Guess", 
                f"OCR guessed: {guess}\n\nAccept this label?",
                icon='question'
            )
            
            if response:
                # User accepted - save it
                self.save_label(guess)
            else:
                # User wants to edit - leave it in input field
                self.input_label.config(text=f"Edit: {self.current_input}")
        else:
            self.input_label.config(text="OCR failed", fg='#FF5555')
            print(f"✗ OCR failed for {img_filename}")
            messagebox.showwarning("OCR Failed", "Could not recognize text. Try manual entry.")
    
    def run_ocr_on_image(self, image_path):
        """Run Tesseract OCR on image with preprocessing for game clock format M:SS"""
        try:
            # Load and preprocess image for better OCR
            img = cv2.imread(image_path, cv2.GRAYSCALE)
            
            # Enhance contrast
            img = cv2.equalizeHist(img)
            
            # Threshold to get clear black/white image
            _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Resize to larger size for better OCR (3x scale)
            height, width = img.shape
            img = cv2.resize(img, (width * 3, height * 3), interpolation=cv2.INTER_CUBIC)
            
            # Save preprocessed image temporarily
            temp_path = image_path + "_temp_ocr.png"
            cv2.imwrite(temp_path, img)
            
            # Try with custom tessdata first
            tessdata_path = "./tessdata_ssd"
            results = []
            
            if os.path.exists(tessdata_path):
                # Try with custom trained model (SSD display)
                for psm in ['7', '8', '13']:  # Try different page segmentation modes
                    result = subprocess.run(
                        ['tesseract', temp_path, 'stdout', '--tessdata-dir', tessdata_path, 
                         '-l', 'ssd_alphanum_plus', '--psm', psm, 
                         '-c', 'tessedit_char_whitelist=0123456789:'],
                        capture_output=True,
                        text=True
                    )
                    if result.returncode == 0:
                        text = result.stdout.strip()
                        if text:
                            results.append(text)
            
            # Also try with default Tesseract
            for psm in ['7', '8', '13']:
                result = subprocess.run(
                    ['tesseract', temp_path, 'stdout', '--psm', psm, 
                     '-c', 'tessedit_char_whitelist=0123456789:'],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    text = result.stdout.strip()
                    if text:
                        results.append(text)
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            # Process all results and find best match for M:SS format
            best_result = None
            for text in results:
                # Clean up OCR output
                text = re.sub(r'[^0-9:]', '', text)
                
                # Try to parse as M:SS
                if ':' in text:
                    parts = text.split(':')
                    if len(parts) == 2 and parts[0] and parts[1]:
                        try:
                            minutes = int(parts[0])
                            seconds = int(parts[1])
                            if 0 <= minutes <= 7 and 0 <= seconds <= 59:
                                return f"{minutes}:{seconds:02d}"
                        except ValueError:
                            pass
                
                # If no colon but we have 3-4 digits, try to infer format
                # e.g., "730" -> "7:30", "045" -> "0:45"
                if not ':' in text and len(text) >= 3:
                    if len(text) == 3:
                        # Assume format MSS (e.g., "730" = 7:30)
                        minutes = int(text[0])
                        seconds = int(text[1:3])
                    elif len(text) == 4:
                        # Assume format M:SS (e.g., "0730" = 7:30)
                        minutes = int(text[0])
                        seconds = int(text[2:4])
                    else:
                        continue
                    
                    if 0 <= minutes <= 7 and 0 <= seconds <= 59:
                        best_result = f"{minutes}:{seconds:02d}"
                        break
                
                # Keep first non-empty result as fallback
                if text and not best_result:
                    best_result = text
            
            return best_result
            
        except FileNotFoundError:
            messagebox.showerror("OCR Error", 
                               "Tesseract not found. Please install Tesseract OCR.")
            return None
        except Exception as e:
            print(f"OCR error: {e}")
            return None
    
    def quit_app(self):
        """Save and quit"""
        # Final save
        with open(self.labels_file, 'w') as f:
            json.dump(self.labels, f, indent=2)
        
        labeled_count = len(self.labels)
        print(f"\n✓ Saved {labeled_count} labels to {self.labels_file}")
        print("Goodbye!")
        
        self.root.quit()
        self.root.destroy()
    
    def run(self):
        """Start the GUI"""
        self.root.mainloop()

def main():
    print("=" * 60)
    print("GAME CLOCK DATASET LABELER")
    print("=" * 60)
    
    dataset_dir = "game_clock_dataset"
    
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory not found: {dataset_dir}")
        print("\nPlease run image_cropper_game_clock.py first to create the dataset")
        sys.exit(1)
    
    labeler = GameClockLabeler(dataset_dir)
    labeler.run()

if __name__ == "__main__":
    main()
