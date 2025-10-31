import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import json
import random
import pytesseract
import numpy as np

# Set up Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Set TESSDATA_PREFIX for custom models
standard_tessdata = r'C:\Program Files\Tesseract-OCR\tessdata'
custom_tessdata = os.path.abspath('./tessdata_ssd')
tessdata_path = f"{custom_tessdata};{standard_tessdata}"
os.environ['TESSDATA_PREFIX'] = tessdata_path

class DatasetLabeler:
    def __init__(self, root):
        self.root = root
        self.root.title("Shot Clock Dataset Labeler")
        self.root.geometry("1000x800")
        
        # Variables
        self.dataset_dir = None
        self.image_files = []
        self.current_index = 0
        self.current_image = None
        self.labels = {}  # Store labels as {filename: label}
        self.labels_file = None
        self.ocr_guess = None  # Store OCR guess for current image
        
        # Setup UI
        self.setup_ui()
        
        # Bind keyboard shortcuts
        self.root.bind('<Return>', lambda e: self.save_and_next())
        self.root.bind('<space>', lambda e: self.save_and_next())
        self.root.bind('<Left>', lambda e: self.prev_image())
        self.root.bind('<Right>', lambda e: self.next_image())
        self.root.bind('b', lambda e: self.set_value('blocked'))
        self.root.bind('B', lambda e: self.set_value('blocked'))
        self.root.bind('s', lambda e: self.set_value('inconclusive'))
        self.root.bind('S', lambda e: self.set_value('inconclusive'))
        self.root.bind('g', lambda e: self.use_ocr_guess())
        self.root.bind('G', lambda e: self.use_ocr_guess())
        
    def setup_ui(self):
        # Top frame for controls
        top_frame = tk.Frame(self.root, bg='#2a2a2a', pady=10)
        top_frame.pack(fill=tk.X)
        
        # Dataset selection
        tk.Button(
            top_frame, 
            text="📁 Select Dataset Folder", 
            command=self.load_dataset,
            bg='#4CAF50',
            fg='white',
            font=('Arial', 12, 'bold'),
            padx=20,
            pady=10
        ).pack(side=tk.LEFT, padx=10)
        
        self.dataset_label = tk.Label(
            top_frame, 
            text="No dataset loaded", 
            bg='#2a2a2a',
            fg='white',
            font=('Arial', 10)
        )
        self.dataset_label.pack(side=tk.LEFT, padx=10)
        
        # Progress label
        self.progress_label = tk.Label(
            top_frame,
            text="",
            bg='#2a2a2a',
            fg='#FFD700',
            font=('Arial', 10, 'bold')
        )
        self.progress_label.pack(side=tk.LEFT, padx=10)
        
        # Canvas for image display
        canvas_frame = tk.Frame(self.root, bg='#1a1a1a')
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.canvas = tk.Canvas(
            canvas_frame, 
            bg='#000000'
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Middle frame for label input
        label_frame = tk.Frame(self.root, bg='#2a2a2a', pady=15)
        label_frame.pack(fill=tk.X)
        
        tk.Label(
            label_frame,
            text="Shot Clock Value:",
            bg='#2a2a2a',
            fg='white',
            font=('Arial', 14, 'bold')
        ).pack(side=tk.LEFT, padx=20)
        
        self.label_entry = tk.Entry(
            label_frame,
            font=('Arial', 24, 'bold'),
            width=10,
            justify='center',
            bg='#1a1a1a',
            fg='#4CAF50',
            insertbackground='white'
        )
        self.label_entry.pack(side=tk.LEFT, padx=10)
        self.label_entry.focus()
        
        tk.Label(
            label_frame,
            text="(0-30)",
            bg='#2a2a2a',
            fg='#888',
            font=('Arial', 10, 'italic')
        ).pack(side=tk.LEFT, padx=5)
        
        # OCR Guess label
        self.ocr_guess_label = tk.Label(
            label_frame,
            text="OCR: --",
            bg='#2a2a2a',
            fg='#00BCD4',
            font=('Arial', 12, 'bold')
        )
        self.ocr_guess_label.pack(side=tk.LEFT, padx=20)
        
        # Number pad buttons
        numpad_frame = tk.Frame(label_frame, bg='#2a2a2a')
        numpad_frame.pack(side=tk.LEFT, padx=20)
        
        # Create 3x4 numpad plus special buttons
        numbers = [
            ['7', '8', '9'],
            ['4', '5', '6'],
            ['1', '2', '3'],
            ['00', 'Blank', 'Blocked']
        ]
        
        for row_idx, row in enumerate(numbers):
            row_frame = tk.Frame(numpad_frame, bg='#2a2a2a')
            row_frame.pack()
            for num in row:
                if num == 'Blank':
                    btn = tk.Button(
                        row_frame,
                        text='Blank (00)',
                        command=lambda: self.set_value('blank'),
                        bg='#9C27B0',
                        fg='white',
                        font=('Arial', 9, 'bold'),
                        width=8,
                        height=2
                    )
                elif num == 'Blocked':
                    btn = tk.Button(
                        row_frame,
                        text='Blocked (B)',
                        command=lambda: self.set_value('blocked'),
                        bg='#FF5722',
                        fg='white',
                        font=('Arial', 9, 'bold'),
                        width=8,
                        height=2
                    )
                elif num == '00':
                    btn = tk.Button(
                        row_frame,
                        text=num,
                        command=lambda: self.set_value('blank'),
                        bg='#9C27B0',
                        fg='white',
                        font=('Arial', 10, 'bold'),
                        width=6,
                        height=2
                    )
                else:
                    btn = tk.Button(
                        row_frame,
                        text=num,
                        command=lambda n=num: self.append_digit(n),
                        bg='#424242',
                        fg='white',
                        font=('Arial', 12, 'bold'),
                        width=6,
                        height=2
                    )
                btn.pack(side=tk.LEFT, padx=2, pady=2)
        
        # Bottom frame for navigation
        bottom_frame = tk.Frame(self.root, bg='#2a2a2a', pady=10)
        bottom_frame.pack(fill=tk.X)
        
        # Instructions
        instructions = tk.Label(
            bottom_frame,
            text="Enter value (0-30), type 00 for Blank, B for Blocked, S for Inconclusive, G to use OCR guess, or ENTER/Space to save. Use ← → to navigate.",
            bg='#2a2a2a',
            fg='#FFD700',
            font=('Arial', 10, 'italic')
        )
        instructions.pack(pady=5)
        
        # Navigation buttons
        nav_frame = tk.Frame(bottom_frame, bg='#2a2a2a')
        nav_frame.pack()
        
        tk.Button(
            nav_frame,
            text="⬅️ Previous",
            command=self.prev_image,
            bg='#FF9800',
            fg='white',
            font=('Arial', 11, 'bold'),
            padx=15,
            pady=8
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            nav_frame,
            text="Skip ⏭️",
            command=self.next_image,
            bg='#9E9E9E',
            fg='white',
            font=('Arial', 11, 'bold'),
            padx=15,
            pady=8
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            nav_frame,
            text="Inconclusive (S)",
            command=lambda: self.save_label_and_next('inconclusive'),
            bg='#607D8B',
            fg='white',
            font=('Arial', 11, 'bold'),
            padx=15,
            pady=8
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            nav_frame,
            text="Use OCR (G)",
            command=self.use_ocr_guess,
            bg='#00BCD4',
            fg='white',
            font=('Arial', 11, 'bold'),
            padx=15,
            pady=8
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            nav_frame,
            text="✅ Save & Next",
            command=self.save_and_next,
            bg='#4CAF50',
            fg='white',
            font=('Arial', 11, 'bold'),
            padx=15,
            pady=8
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            nav_frame,
            text="💾 Save Labels",
            command=self.save_labels,
            bg='#2196F3',
            fg='white',
            font=('Arial', 11, 'bold'),
            padx=15,
            pady=8
        ).pack(side=tk.LEFT, padx=5)
        
        # Current image filename
        self.filename_label = tk.Label(
            bottom_frame,
            text="",
            bg='#2a2a2a',
            fg='#888',
            font=('Arial', 9)
        )
        self.filename_label.pack(pady=5)
        
    def append_digit(self, digit):
        """Append a digit to the entry"""
        current = self.label_entry.get()
        if len(current) < 2:  # Max 2 digits
            self.label_entry.delete(0, tk.END)
            self.label_entry.insert(0, current + digit)
    
    def clear_entry(self):
        """Clear the entry"""
        self.label_entry.delete(0, tk.END)
    
    def set_value(self, value):
        """Set a specific value"""
        self.label_entry.delete(0, tk.END)
        self.label_entry.insert(0, value)
    
    def run_ocr_on_image(self, image_path):
        """Run OCR on an image using Tesseract with SSD model"""
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                return None
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Scale up the image for better OCR
            scale_factor = 8
            h, w = gray.shape
            scaled = cv2.resize(gray, (w * scale_factor, h * scale_factor), interpolation=cv2.INTER_CUBIC)
            
            # Apply inverted threshold (white digits on black background)
            _, binary = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Light blur to reduce noise
            blurred = cv2.GaussianBlur(binary, (3, 3), 0)
            
            # Run Tesseract with SSD model
            custom_config = r'--psm 8 --oem 1 -l ssd -c tessedit_char_whitelist=0123456789'
            text = pytesseract.image_to_string(blurred, config=custom_config)
            
            # Clean the result
            text = text.strip().replace(' ', '')
            
            # Try to parse as integer
            if text.isdigit():
                value = int(text)
                if 0 <= value <= 30:
                    return str(value)
            
            return None
            
        except Exception as e:
            print(f"OCR error: {e}")
            return None
    
    def use_ocr_guess(self):
        """Use the OCR guess as the label value"""
        if self.ocr_guess and self.ocr_guess != "??":
            self.set_value(self.ocr_guess)
    
    def load_dataset(self):
        """Load dataset folder"""
        folder_path = filedialog.askdirectory(title="Select Dataset Folder")
        
        if not folder_path:
            return
        
        self.dataset_dir = folder_path
        
        # Find all image files
        self.image_files = []
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                self.image_files.append(filename)
        
        if not self.image_files:
            messagebox.showerror("Error", "No image files found in selected folder")
            return
        
        # Randomize order for better coverage
        random.shuffle(self.image_files)
        
        # Load existing labels if available
        self.labels_file = os.path.join(folder_path, 'labels.json')
        if os.path.exists(self.labels_file):
            try:
                with open(self.labels_file, 'r') as f:
                    self.labels = json.load(f)
                print(f"Loaded {len(self.labels)} existing labels")
            except Exception as e:
                print(f"Error loading labels: {e}")
                self.labels = {}
        else:
            self.labels = {}
        
        # Update UI
        self.dataset_label.config(text=f"Dataset: {os.path.basename(folder_path)} ({len(self.image_files)} images)")
        
        # Start with first image
        self.current_index = 0
        self.show_image()
        
    def show_image(self):
        """Display current image and run OCR"""
        if not self.image_files:
            return
        
        filename = self.image_files[self.current_index]
        filepath = os.path.join(self.dataset_dir, filename)
        
        # Run OCR on the image
        self.ocr_guess = self.run_ocr_on_image(filepath)
        if self.ocr_guess:
            self.ocr_guess_label.config(text=f"OCR: {self.ocr_guess}", fg='#4CAF50')
        else:
            self.ocr_guess = "??"
            self.ocr_guess_label.config(text="OCR: ??", fg='#FF5722')
        
        # Load image
        image = cv2.imread(filepath)
        if image is None:
            messagebox.showerror("Error", f"Could not load image: {filename}")
            return
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Scale to fit canvas
        self.canvas.update()
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        height, width = rgb.shape[:2]
        
        # Calculate scale
        width_scale = canvas_width / width
        height_scale = canvas_height / height
        scale = min(width_scale, height_scale, 3.0)  # Max 3x zoom
        
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize
        resized = cv2.resize(rgb, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        
        # Convert to PhotoImage
        pil_image = Image.fromarray(resized)
        self.current_image = ImageTk.PhotoImage(pil_image)
        
        # Display
        self.canvas.delete('all')
        self.canvas.config(width=new_width, height=new_height)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.current_image)
        
        # Update labels
        labeled_count = len(self.labels)
        total_count = len(self.image_files)
        percentage = (labeled_count / total_count * 100) if total_count > 0 else 0
        
        self.progress_label.config(
            text=f"Progress: {labeled_count}/{total_count} ({percentage:.1f}%)"
        )
        
        self.filename_label.config(text=f"File: {filename}")
        
        # Pre-fill entry if already labeled
        if filename in self.labels:
            self.label_entry.delete(0, tk.END)
            self.label_entry.insert(0, str(self.labels[filename]))
        else:
            self.label_entry.delete(0, tk.END)
        
        self.label_entry.focus()
        
    def prev_image(self):
        """Go to previous image"""
        if not self.image_files:
            return
        
        self.current_index = (self.current_index - 1) % len(self.image_files)
        self.show_image()
        
    def next_image(self):
        """Go to next image"""
        if not self.image_files:
            return
        
        self.current_index = (self.current_index + 1) % len(self.image_files)
        self.show_image()
        
    def save_and_next(self):
        """Save current label and move to next image"""
        if not self.image_files:
            return
        
        # Get label value
        label_value = self.label_entry.get().strip().lower()
        
        if label_value:
            # Handle 'blank', 'blocked', or 'inconclusive' labels, or '00' as blank
            if label_value in ['blank', 'blocked', 'inconclusive']:
                filename = self.image_files[self.current_index]
                self.labels[filename] = label_value
                
                # Auto-save every 10 labels
                if len(self.labels) % 10 == 0:
                    self.save_labels(silent=True)
                
                # Move to next
                self.next_image()
            elif label_value == '00':
                filename = self.image_files[self.current_index]
                self.labels[filename] = 'blank'
                
                # Auto-save every 10 labels
                if len(self.labels) % 10 == 0:
                    self.save_labels(silent=True)
                
                # Move to next
                self.next_image()
            else:
                try:
                    value = int(label_value)
                    if 0 <= value <= 30:
                        filename = self.image_files[self.current_index]
                        self.labels[filename] = value
                        
                        # Auto-save every 10 labels
                        if len(self.labels) % 10 == 0:
                            self.save_labels(silent=True)
                        
                        # Move to next
                        self.next_image()
                    else:
                        messagebox.showwarning("Invalid Value", "Please enter a value between 0 and 30, 'blank', 'blocked', or 'inconclusive'")
                except ValueError:
                    messagebox.showwarning("Invalid Value", "Please enter a valid number, 'blank', 'blocked', or 'inconclusive'")
        else:
            # Skip without labeling
            self.next_image()
    
    def save_label_and_next(self, label_value):
        """Save a specific label and move to next image"""
        if not self.image_files:
            return
        
        filename = self.image_files[self.current_index]
        self.labels[filename] = label_value
        
        # Auto-save every 10 labels
        if len(self.labels) % 10 == 0:
            self.save_labels(silent=True)
        
        # Move to next
        self.next_image()
    
    def save_labels(self, silent=False):
        """Save labels to JSON file"""
        if not self.labels_file:
            return
        
        try:
            with open(self.labels_file, 'w') as f:
                json.dump(self.labels, f, indent=2)
            
            if not silent:
                messagebox.showinfo("Success", f"Saved {len(self.labels)} labels to:\n{self.labels_file}")
            else:
                print(f"Auto-saved {len(self.labels)} labels")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save labels:\n{e}")
    
    def on_closing(self):
        """Save labels before closing"""
        if self.labels:
            response = messagebox.askyesno(
                "Save Labels?",
                f"You have {len(self.labels)} labeled images.\nSave before closing?"
            )
            if response:
                self.save_labels()
        
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = DatasetLabeler(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
