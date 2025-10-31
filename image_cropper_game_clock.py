"""
Game Clock Dataset Creator - Extract and save frames from video(s)
This script allows you to:
1. Select a region of interest (game clock) from one or more videos
2. Extract frames at specified intervals
3. Average multiple consecutive frames to reduce noise
4. Save cropped images for later labeling and training
5. Process multiple videos with separate crop regions for each

Usage:
    python image_cropper_game_clock.py video_file1.mov [video_file2.mov ...]
    
    Or run without arguments to enter interactive mode:
    python image_cropper_game_clock.py
"""

import cv2
import numpy as np
import os
import sys
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, filedialog

class GameClockDatasetCreator:
    def __init__(self, video_path, output_dir="game_clock_dataset", start_index=0):
        self.video_path = video_path
        self.output_dir = output_dir
        self.crop_rect = None
        self.frame_interval = 30  # Extract every 30 frames (1 second at 30fps)
        self.num_frames_to_average = 5  # Average 5 consecutive frames
        self.start_index = start_index  # Starting index for filename numbering
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load video
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            sys.exit(1)
        
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\n{'='*60}")
        print(f"Processing: {Path(video_path).name}")
        print(f"{'='*60}")
        print(f"Video: {self.frame_width}x{self.frame_height} @ {self.fps}fps")
        print(f"Total frames: {self.total_frames}")
        print(f"Duration: {self.total_frames/self.fps:.1f} seconds")
        
    def select_crop_region(self):
        """Interactive Tkinter GUI to select game clock region"""
        print("\n=== SELECT GAME CLOCK REGION ===")
        print("Opening GUI window...")
        
        # Read first frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = self.cap.read()
        if not ret:
            print("Error: Could not read first frame")
            return False
        
        # Use Tkinter GUI for selection
        root = tk.Tk()
        root.title("Select Game Clock Region")
        root.geometry("1200x800")
        root.configure(bg='#2a2a2a')
        
        # Title
        title_frame = tk.Frame(root, bg='#2a2a2a', pady=15)
        title_frame.pack(fill=tk.X)
        
        tk.Label(
            title_frame,
            text="🕐 Select Game Clock Region",
            bg='#2a2a2a',
            fg='white',
            font=('Arial', 16, 'bold')
        ).pack()
        
        tk.Label(
            title_frame,
            text="Click and drag to select the game clock area, then click 'Confirm Selection'",
            bg='#2a2a2a',
            fg='#FFD700',
            font=('Arial', 10, 'italic')
        ).pack()
        
        # Canvas for image
        canvas_frame = tk.Frame(root, bg='#1a1a1a')
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        canvas = tk.Canvas(canvas_frame, bg='#000000', cursor='crosshair')
        canvas.pack(fill=tk.BOTH, expand=True)
        
        # Variables for selection
        selection = {'start': None, 'end': None, 'rect_id': None, 'crop_rect': None}
        
        def on_mouse_down(event):
            selection['start'] = (event.x, event.y)
            selection['end'] = (event.x, event.y)
        
        def on_mouse_move(event):
            if selection['start']:
                selection['end'] = (event.x, event.y)
                # Clear previous rectangle
                if selection['rect_id']:
                    canvas.delete(selection['rect_id'])
                # Draw new rectangle
                x1, y1 = selection['start']
                x2, y2 = selection['end']
                selection['rect_id'] = canvas.create_rectangle(
                    x1, y1, x2, y2, outline='#00FF00', width=2
                )
                # Update info label
                width = abs(x2 - x1)
                height = abs(y2 - y1)
                # Convert to original frame coordinates
                orig_width = int(width / scale_factor)
                orig_height = int(height / scale_factor)
                info_label.config(text=f"Selection: {orig_width}x{orig_height} pixels")
        
        def on_mouse_up(event):
            selection['end'] = (event.x, event.y)
        
        canvas.bind('<ButtonPress-1>', on_mouse_down)
        canvas.bind('<B1-Motion>', on_mouse_move)
        canvas.bind('<ButtonRelease-1>', on_mouse_up)
        
        # Display frame
        canvas.update()
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        frame_h, frame_w = frame.shape[:2]
        scale_w = canvas_width / frame_w
        scale_h = canvas_height / frame_h
        scale_factor = min(scale_w, scale_h, 1.0)
        
        new_w = int(frame_w * scale_factor)
        new_h = int(frame_h * scale_factor)
        resized_frame = cv2.resize(frame, (new_w, new_h))
        
        # Convert to RGB and then to PIL
        frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        from PIL import Image, ImageTk
        pil_img = Image.fromarray(frame_rgb)
        photo = ImageTk.PhotoImage(pil_img)
        
        canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        canvas.image = photo  # Keep reference
        
        # Bottom controls
        bottom_frame = tk.Frame(root, bg='#2a2a2a', pady=15)
        bottom_frame.pack(fill=tk.X)
        
        info_label = tk.Label(
            bottom_frame,
            text="Selection: Not selected",
            bg='#2a2a2a',
            fg='#4CAF50',
            font=('Arial', 11)
        )
        info_label.pack(pady=5)
        
        button_frame = tk.Frame(bottom_frame, bg='#2a2a2a')
        button_frame.pack()
        
        def clear_selection():
            selection['start'] = None
            selection['end'] = None
            if selection['rect_id']:
                canvas.delete(selection['rect_id'])
                selection['rect_id'] = None
            info_label.config(text="Selection: Cleared")
        
        def confirm_selection():
            if selection['start'] and selection['end']:
                # Convert canvas coordinates to original frame coordinates
                x1 = int(min(selection['start'][0], selection['end'][0]) / scale_factor)
                y1 = int(min(selection['start'][1], selection['end'][1]) / scale_factor)
                x2 = int(max(selection['start'][0], selection['end'][0]) / scale_factor)
                y2 = int(max(selection['start'][1], selection['end'][1]) / scale_factor)
                
                width = x2 - x1
                height = y2 - y1
                
                if width > 0 and height > 0:
                    selection['crop_rect'] = (x1, y1, width, height)
                    root.quit()
                else:
                    messagebox.showwarning("Invalid Selection", "Please select a valid region")
            else:
                messagebox.showwarning("No Selection", "Please select a region first")
        
        def cancel():
            selection['crop_rect'] = None
            root.quit()
        
        tk.Button(
            button_frame,
            text="🔄 Clear Selection",
            command=clear_selection,
            bg='#FF9800',
            fg='white',
            font=('Arial', 11, 'bold'),
            padx=20,
            pady=10
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            button_frame,
            text="✅ Confirm Selection",
            command=confirm_selection,
            bg='#4CAF50',
            fg='white',
            font=('Arial', 11, 'bold'),
            padx=20,
            pady=10
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            button_frame,
            text="❌ Cancel",
            command=cancel,
            bg='#f44336',
            fg='white',
            font=('Arial', 11, 'bold'),
            padx=20,
            pady=10
        ).pack(side=tk.LEFT, padx=5)
        
        root.mainloop()
        root.destroy()
        
        self.crop_rect = selection['crop_rect']
        
        if self.crop_rect and self.crop_rect[2] > 0 and self.crop_rect[3] > 0:
            print(f"✓ Selected region: x={self.crop_rect[0]}, y={self.crop_rect[1]}, "
                  f"w={self.crop_rect[2]}, h={self.crop_rect[3]}")
            return True
        else:
            print("No valid region selected")
            return False
    
    def extract_and_save_frames(self):
        """Extract frames at intervals, average them, and save"""
        if self.crop_rect is None:
            print("Error: No crop region defined")
            return
        
        x, y, w, h = self.crop_rect
        
        print(f"\n=== EXTRACTING FRAMES ===")
        print(f"Frame interval: every {self.frame_interval} frames ({self.frame_interval/self.fps:.2f} seconds)")
        print(f"Averaging: {self.num_frames_to_average} consecutive frames")
        print(f"Output directory: {self.output_dir}/")
        
        # Check for existing images and determine target size if appending
        target_size = None
        if self.start_index > 0:  # We're appending to existing dataset
            existing_images = [f for f in os.listdir(self.output_dir) 
                              if f.startswith("game_clock_") and f.endswith(".png")]
            if existing_images:
                # Load the first existing image to get target size
                first_existing = os.path.join(self.output_dir, existing_images[0])
                try:
                    sample_img = cv2.imread(first_existing)
                    if sample_img is not None:
                        target_size = (sample_img.shape[1], sample_img.shape[0])  # (width, height)
                        print(f"Detected existing image size: {target_size[0]}x{target_size[1]}")
                        print("New images will be rescaled to match existing size")
                except Exception as e:
                    print(f"Warning: Could not read existing image for size detection: {e}")
        
        # Reset video to beginning
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        saved_count = 0
        frame_idx = 0
        
        # Create frame buffer for averaging
        frame_buffer = []
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame_idx += 1
            
            # Check if this is a frame we want to extract
            if (frame_idx - 1) % self.frame_interval == 0:
                # Collect frames for averaging (current frame + next few frames)
                frames_to_average = []
                
                # Add current frame
                frames_to_average.append(frame)
                
                # Read additional frames for averaging
                current_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                for i in range(1, self.num_frames_to_average):
                    ret_avg, frame_avg = self.cap.read()
                    if ret_avg:
                        frames_to_average.append(frame_avg)
                    else:
                        break
                
                # Reset position to continue from correct frame
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
                
                # Average the frames
                if len(frames_to_average) > 0:
                    averaged_frame = np.mean(frames_to_average, axis=0).astype(np.uint8)
                    
                    # Crop the game clock region
                    cropped = averaged_frame[y:y+h, x:x+w]
                    
                    # Rescale to match existing dataset size if appending
                    if target_size is not None:
                        current_size = (cropped.shape[1], cropped.shape[0])  # (width, height)
                        if current_size != target_size:
                            cropped = cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)
                    
                    # Save the cropped image (with offset for multiple videos)
                    filename = f"game_clock_{self.start_index + saved_count:05d}.png"
                    filepath = os.path.join(self.output_dir, filename)
                    cv2.imwrite(filepath, cropped)
                    
                    saved_count += 1
                    
                    # Print progress every 10 images
                    if saved_count % 10 == 0:
                        progress = (frame_idx / self.total_frames) * 100
                        print(f"Progress: {frame_idx}/{self.total_frames} frames ({progress:.1f}%) - "
                              f"Saved: {saved_count} images (total dataset: {self.start_index + saved_count})")
        
        self.cap.release()
        
        print(f"\n✓ Done! Saved {saved_count} images from this video")
        print(f"  Total images in dataset: {self.start_index + saved_count}")
        print(f"  Saved to: {self.output_dir}/")
        
        return self.start_index + saved_count  # Return next available index
        print(f"\nNext steps:")
        print(f"1. Run: python label_dataset_game_clock.py")
        print(f"2. Label the images with keyboard shortcuts")
        print(f"3. Train CNN with: python train_game_clock_cnn.py")
    
    def run(self):
        """Main execution flow"""
        # Step 1: Select crop region
        if not self.select_crop_region():
            print("Crop selection cancelled")
            return None
        
        # Step 2: Configure extraction
        print("\n=== CONFIGURE EXTRACTION ===")
        
        # Ask for frame interval
        default_interval = self.frame_interval
        response = input(f"Frame interval (extract every N frames) [{default_interval}]: ").strip()
        if response:
            try:
                self.frame_interval = int(response)
            except ValueError:
                print(f"Invalid input, using default: {default_interval}")
        
        # Ask for averaging count
        default_avg = self.num_frames_to_average
        response = input(f"Number of frames to average [{default_avg}]: ").strip()
        if response:
            try:
                self.num_frames_to_average = int(response)
            except ValueError:
                print(f"Invalid input, using default: {default_avg}")
        
        # Estimate output
        estimated_images = self.total_frames // self.frame_interval
        print(f"\nEstimated output: ~{estimated_images} images from this video")
        
        # Confirm
        response = input("\nProceed with extraction? [Y/n]: ").strip().lower()
        if response and response != 'y' and response != 'yes':
            print("Extraction cancelled")
            return None
        
        # Step 3: Extract and save frames
        next_index = self.extract_and_save_frames()
        return next_index

def process_multiple_videos(video_paths, output_dir="game_clock_dataset"):
    """Process multiple videos and combine into one dataset"""
    print("=" * 60)
    print("GAME CLOCK DATASET CREATOR - MULTI-VIDEO MODE")
    print("=" * 60)
    print(f"\nProcessing {len(video_paths)} video(s)")
    print(f"Output directory: {output_dir}\n")
    
    # Check for existing dataset and ask user what to do
    existing_images = []
    append_mode = False
    start_index = 0
    
    if os.path.exists(output_dir):
        existing_images = [f for f in os.listdir(output_dir) 
                          if f.startswith("game_clock_") and f.endswith(".png")]
        if existing_images:
            print(f"Found existing dataset with {len(existing_images)} images.")
            while True:
                choice = input("\nWhat would you like to do?\n"
                             "1. Append new frames to existing dataset\n"
                             "2. Create new dataset (overwrite existing)\n"
                             "3. Cancel\n"
                             "Enter choice (1-3): ").strip()
                
                if choice == "1":
                    append_mode = True
                    start_index = len(existing_images)
                    print(f"Will append new frames starting from index {start_index}")
                    break
                elif choice == "2":
                    append_mode = False
                    start_index = 0
                    # Remove existing dataset
                    import shutil
                    shutil.rmtree(output_dir)
                    existing_images = []
                    print("Existing dataset will be overwritten")
                    break
                elif choice == "3":
                    print("Operation cancelled")
                    return
                else:
                    print("Invalid choice. Please enter 1, 2, or 3.")
        else:
            print("Directory exists but is empty")
    else:
        print("Creating new dataset directory")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    next_index = start_index
    total_extracted = 0
    
    for i, video_path in enumerate(video_paths, 1):
        print(f"\n{'#'*60}")
        print(f"VIDEO {i}/{len(video_paths)}: {Path(video_path).name}")
        print(f"{'#'*60}")
        
        if not os.path.exists(video_path):
            print(f"Error: Video file not found: {video_path}")
            continue
        
        # Create dataset creator for this video
        creator = GameClockDatasetCreator(video_path, output_dir, start_index=next_index)
        result = creator.run()
        
        if result is not None:
            images_from_video = result - next_index
            total_extracted += images_from_video
            next_index = result
        else:
            print(f"Skipped video {i}")
    
    print(f"\n{'='*60}")
    print(f"ALL VIDEOS PROCESSED")
    print(f"{'='*60}")
    
    if append_mode:
        print(f"Appended {total_extracted} new images to existing dataset")
        print(f"Total images in combined dataset: {next_index}")
    else:
        print(f"Created new dataset with {total_extracted} images")
    
    print(f"Output directory: {output_dir}/")
    print(f"\nNext steps:")
    print(f"1. Run: python label_dataset_game_clock.py")
    print(f"2. Label the images with keyboard shortcuts")
    print(f"3. Train CNN with: python train_game_clock_cnn.py")

def main():
    video_paths = []
    
    # Check command line arguments
    if len(sys.argv) > 1:
        # Videos provided as arguments
        video_paths = sys.argv[1:]
    else:
        # Interactive mode - ask for videos
        print("=" * 60)
        print("GAME CLOCK DATASET CREATOR")
        print("=" * 60)
        print("\nNo videos specified on command line.")
        print("Enter video file paths (one per line, empty line to finish):\n")
        
        while True:
            path = input("Video path (or press Enter to finish): ").strip()
            if not path:
                break
            if os.path.exists(path):
                video_paths.append(path)
                print(f"  ✓ Added: {Path(path).name}")
            else:
                print(f"  ✗ File not found: {path}")
        
        # If no videos entered, try file dialog
        if not video_paths:
            print("\nOpening file dialog...")
            root = tk.Tk()
            root.withdraw()
            files = filedialog.askopenfilenames(
                title="Select Video File(s)",
                filetypes=[
                    ("Video files", "*.mp4 *.mov *.avi *.mkv *.MOV *.MP4"),
                    ("All files", "*.*")
                ]
            )
            root.destroy()
            
            if files:
                video_paths = list(files)
            else:
                print("No videos selected. Exiting.")
                sys.exit(0)
    
    if not video_paths:
        print("No videos to process. Exiting.")
        sys.exit(0)
    
    # Process all videos
    if len(video_paths) == 1:
        # Single video mode - check for existing dataset
        output_dir = "game_clock_dataset"
        start_index = 0
        
        if os.path.exists(output_dir):
            existing_images = [f for f in os.listdir(output_dir) 
                              if f.startswith("game_clock_") and f.endswith(".png")]
            if existing_images:
                print(f"\nFound existing dataset with {len(existing_images)} images.")
                while True:
                    choice = input("\nWhat would you like to do?\n"
                                 "1. Append new frames to existing dataset\n"
                                 "2. Create new dataset (overwrite existing)\n"
                                 "3. Cancel\n"
                                 "Enter choice (1-3): ").strip()
                    
                    if choice == "1":
                        start_index = len(existing_images)
                        print(f"Will append new frames starting from index {start_index}")
                        break
                    elif choice == "2":
                        start_index = 0
                        # Remove existing dataset
                        import shutil
                        shutil.rmtree(output_dir)
                        print("Existing dataset will be overwritten")
                        break
                    elif choice == "3":
                        print("Operation cancelled")
                        return
                    else:
                        print("Invalid choice. Please enter 1, 2, or 3.")
        
        creator = GameClockDatasetCreator(video_paths[0], output_dir, start_index)
        creator.run()
    else:
        # Multi-video mode
        process_multiple_videos(video_paths)

if __name__ == "__main__":
    main()
