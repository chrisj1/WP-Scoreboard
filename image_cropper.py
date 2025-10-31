import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import numpy as np

class ShotClockDatasetCreator:
    def __init__(self, root):
        self.root = root
        self.root.title("Shot Clock Dataset Creator")
        self.root.geometry("1200x800")
        
        # Variables
        self.video_path = None
        self.cap = None
        self.current_frame = None
        self.display_frame = None
        self.crop_start = None
        self.crop_end = None
        self.is_selecting = False
        self.crop_rect = None
        self.scale_factor = 1.0
        
        # Setup UI
        self.setup_ui()
        
    def setup_ui(self):
        # Top frame for controls
        top_frame = tk.Frame(self.root, bg='#2a2a2a', pady=10)
        top_frame.pack(fill=tk.X)
        
        # Video selection
        tk.Button(
            top_frame, 
            text="📁 Select Video File", 
            command=self.load_video,
            bg='#4CAF50',
            fg='white',
            font=('Arial', 12, 'bold'),
            padx=20,
            pady=10
        ).pack(side=tk.LEFT, padx=10)
        
        self.video_label = tk.Label(
            top_frame, 
            text="No video loaded", 
            bg='#2a2a2a',
            fg='white',
            font=('Arial', 10)
        )
        self.video_label.pack(side=tk.LEFT, padx=10)
        
        # Canvas for video display
        canvas_frame = tk.Frame(self.root, bg='#1a1a1a')
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.canvas = tk.Canvas(
            canvas_frame, 
            bg='#000000',
            cursor='crosshair'
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Bind mouse events for crop selection
        self.canvas.bind('<ButtonPress-1>', self.on_mouse_down)
        self.canvas.bind('<B1-Motion>', self.on_mouse_move)
        self.canvas.bind('<ButtonRelease-1>', self.on_mouse_up)
        
        # Bottom frame for controls
        bottom_frame = tk.Frame(self.root, bg='#2a2a2a', pady=10)
        bottom_frame.pack(fill=tk.X)
        
        # Instructions
        instructions = tk.Label(
            bottom_frame,
            text="Instructions: 1) Load a video file  2) Click and drag to select the shot clock area  3) Click 'Extract Frames'",
            bg='#2a2a2a',
            fg='#FFD700',
            font=('Arial', 10, 'italic')
        )
        instructions.pack(pady=5)
        
        # Buttons frame
        buttons_frame = tk.Frame(bottom_frame, bg='#2a2a2a')
        buttons_frame.pack()
        
        self.clear_btn = tk.Button(
            buttons_frame,
            text="🔄 Clear Selection",
            command=self.clear_selection,
            bg='#FF9800',
            fg='white',
            font=('Arial', 11, 'bold'),
            padx=15,
            pady=8,
            state=tk.DISABLED
        )
        self.clear_btn.pack(side=tk.LEFT, padx=5)
        
        self.extract_btn = tk.Button(
            buttons_frame,
            text="💾 Extract Frames to Dataset",
            command=self.extract_frames,
            bg='#2196F3',
            fg='white',
            font=('Arial', 11, 'bold'),
            padx=15,
            pady=8,
            state=tk.DISABLED
        )
        self.extract_btn.pack(side=tk.LEFT, padx=5)
        
        # Crop info label
        self.crop_info_label = tk.Label(
            bottom_frame,
            text="Crop area: Not selected",
            bg='#2a2a2a',
            fg='#4CAF50',
            font=('Arial', 10)
        )
        self.crop_info_label.pack(pady=5)
        
        # Progress label
        self.progress_label = tk.Label(
            bottom_frame,
            text="",
            bg='#2a2a2a',
            fg='white',
            font=('Arial', 10, 'bold')
        )
        self.progress_label.pack(pady=5)
        
    def load_video(self):
        """Load video file and display first frame"""
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            return
        
        # Release previous video if any
        if self.cap:
            self.cap.release()
        
        # Open new video
        self.cap = cv2.VideoCapture(file_path)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open video file")
            return
        
        self.video_path = file_path
        video_name = os.path.basename(file_path)
        self.video_label.config(text=f"Video: {video_name}")
        
        # Get video info
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        info_text = f"Video: {video_name} | Frames: {total_frames} | FPS: {fps:.1f} | Duration: {duration:.1f}s"
        self.video_label.config(text=info_text)
        
        # Read and display first frame
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame
            self.display_frame_on_canvas(frame)
        
        # Reset selection
        self.clear_selection()
        
    def display_frame_on_canvas(self, frame):
        """Display frame on canvas, scaled to fit"""
        # Get canvas size
        self.canvas.update()
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # Get frame size
        frame_height, frame_width = frame.shape[:2]
        
        # Calculate scale to fit canvas
        width_scale = canvas_width / frame_width
        height_scale = canvas_height / frame_height
        self.scale_factor = min(width_scale, height_scale, 1.0)  # Don't scale up
        
        # Resize frame
        new_width = int(frame_width * self.scale_factor)
        new_height = int(frame_height * self.scale_factor)
        
        resized = cv2.resize(frame, (new_width, new_height))
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Convert to PhotoImage
        image = Image.fromarray(rgb)
        self.display_frame = ImageTk.PhotoImage(image)
        
        # Display on canvas
        self.canvas.delete('all')
        self.canvas.config(width=new_width, height=new_height)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.display_frame)
        
    def on_mouse_down(self, event):
        """Start crop selection"""
        if not self.current_frame is not None:
            return
        
        self.is_selecting = True
        self.crop_start = (event.x, event.y)
        
    def on_mouse_move(self, event):
        """Update crop rectangle during drag"""
        if not self.is_selecting:
            return
        
        # Remove old rectangle
        if self.crop_rect:
            self.canvas.delete(self.crop_rect)
        
        # Draw new rectangle
        x1, y1 = self.crop_start
        x2, y2 = event.x, event.y
        
        self.crop_rect = self.canvas.create_rectangle(
            x1, y1, x2, y2,
            outline='#4CAF50',
            width=3
        )
        
    def on_mouse_up(self, event):
        """Finish crop selection"""
        if not self.is_selecting:
            return
        
        self.is_selecting = False
        self.crop_end = (event.x, event.y)
        
        # Calculate crop coordinates (in original frame space)
        x1, y1 = self.crop_start
        x2, y2 = self.crop_end
        
        # Ensure x1 < x2 and y1 < y2
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        
        # Convert to original frame coordinates
        orig_x1 = int(x1 / self.scale_factor)
        orig_y1 = int(y1 / self.scale_factor)
        orig_x2 = int(x2 / self.scale_factor)
        orig_y2 = int(y2 / self.scale_factor)
        
        # Store crop area
        self.crop_area = (orig_x1, orig_y1, orig_x2 - orig_x1, orig_y2 - orig_y1)
        
        # Update UI
        x, y, w, h = self.crop_area
        self.crop_info_label.config(
            text=f"Crop area: x={x}, y={y}, width={w}, height={h}"
        )
        
        self.clear_btn.config(state=tk.NORMAL)
        self.extract_btn.config(state=tk.NORMAL)
        
    def clear_selection(self):
        """Clear crop selection"""
        if self.crop_rect:
            self.canvas.delete(self.crop_rect)
            self.crop_rect = None
        
        self.crop_start = None
        self.crop_end = None
        self.crop_area = None
        self.is_selecting = False
        
        self.crop_info_label.config(text="Crop area: Not selected")
        self.clear_btn.config(state=tk.DISABLED)
        self.extract_btn.config(state=tk.DISABLED)
        
    def extract_frames(self):
        """Extract all frames from video with crop applied"""
        if not self.video_path or not self.crop_area:
            messagebox.showwarning("Warning", "Please select a video and crop area first")
            return
        
        # Ask for output directory
        output_dir = filedialog.askdirectory(title="Select Dataset Output Folder")
        if not output_dir:
            return
        
        # Create subdirectory with video name
        video_name = os.path.splitext(os.path.basename(self.video_path))[0]
        dataset_dir = os.path.join(output_dir, f"dataset_{video_name}")
        
        # Check if dataset already exists
        existing_files = []
        append_mode = False
        if os.path.exists(dataset_dir):
            existing_files = [f for f in os.listdir(dataset_dir) if f.endswith('.png')]
            if existing_files:
                result = messagebox.askyesnocancel(
                    "Existing Dataset Found",
                    f"Found existing dataset with {len(existing_files)} frames.\n\n"
                    f"Yes: Append new frames to existing dataset\n"
                    f"No: Create new dataset (overwrite)\n"
                    f"Cancel: Stop extraction"
                )
                if result is None:  # Cancel
                    return
                elif result:  # Yes - append
                    append_mode = True
                else:  # No - overwrite
                    import shutil
                    shutil.rmtree(dataset_dir)
                    existing_files = []
        
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Determine target size for rescaling if appending
        target_size = None
        if append_mode and existing_files:
            # Load the first existing image to get target size
            first_existing = os.path.join(dataset_dir, existing_files[0])
            try:
                sample_img = cv2.imread(first_existing)
                if sample_img is not None:
                    target_size = (sample_img.shape[1], sample_img.shape[0])  # (width, height)
                    self.progress_label.config(
                        text=f"Will rescale new images to match existing size: {target_size[0]}x{target_size[1]}"
                    )
                    self.root.update()
            except Exception as e:
                print(f"Warning: Could not read existing image for size detection: {e}")
        
        # Determine starting number for new files
        start_number = 0
        if append_mode and existing_files:
            # Find the highest existing frame number
            frame_numbers = []
            for filename in existing_files:
                try:
                    # Extract number from frame_XXXXXX.png format
                    if filename.startswith('frame_') and filename.endswith('.png'):
                        num_str = filename[6:12]  # Extract 6-digit number
                        frame_numbers.append(int(num_str))
                except:
                    continue
            if frame_numbers:
                start_number = max(frame_numbers) + 1
        
        # Reset video to start
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Get video info
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Extract frames
        frame_count = 0
        saved_count = 0
        
        x, y, w, h = self.crop_area
        
        mode_text = "Appending to existing dataset..." if append_mode else "Extracting frames..."
        self.progress_label.config(text=mode_text)
        self.root.update()
        
        frame_buffer = []
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Crop frame
            cropped = frame[y:y+h, x:x+w]
            frame_buffer.append(cropped)
            
            # When we have 5 frames, average them and save
            if len(frame_buffer) == 5:
                # Average the 5 frames
                averaged = np.mean(frame_buffer, axis=0).astype(np.uint8)
                
                # Rescale to match existing dataset size if appending
                if target_size is not None:
                    current_size = (averaged.shape[1], averaged.shape[0])  # (width, height)
                    if current_size != target_size:
                        averaged = cv2.resize(averaged, target_size, interpolation=cv2.INTER_AREA)
                
                # Save averaged frame with proper numbering
                frame_number = start_number + saved_count
                filename = f"frame_{frame_number:06d}.png"
                filepath = os.path.join(dataset_dir, filename)
                cv2.imwrite(filepath, averaged)
                saved_count += 1
                
                # Clear buffer
                frame_buffer = []
            
            # Update progress
            progress = (frame_count / total_frames) * 100
            total_existing = len(existing_files) if append_mode else 0
            total_frames_now = total_existing + saved_count
            self.progress_label.config(
                text=f"Extracting: {frame_count}/{total_frames} ({progress:.1f}%) | "
                f"Saved: {saved_count} | Total: {total_frames_now}"
            )
            self.root.update()
        
        total_existing = len(existing_files) if append_mode else 0
        total_frames_final = total_existing + saved_count
        
        self.progress_label.config(
            text=f"✅ Complete! Saved {saved_count} new frames. Total dataset: {total_frames_final} frames"
        )
        
        action_text = "Appended" if append_mode else "Extracted"
        messagebox.showinfo(
            "Success",
            f"{action_text} {saved_count} frames to dataset.\n"
            f"Total frames in dataset: {total_frames_final}\n"
            f"Location: {dataset_dir}"
        )
        
    def on_closing(self):
        """Clean up on window close"""
        if self.cap:
            self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = ShotClockDatasetCreator(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()