"""
Shot Clock Dataset Creator - Game-clock-style workflow

This mirrors the interaction model of image_cropper_game_clock.py:
- Accept one or more video paths from CLI or interactive picker
- Select ROI per video via GUI
- Extract averaged crops at configurable frame intervals
- Append or overwrite an existing dataset

Output format:
- Directory: shot_clock_dataset/
- Filenames: shot_clock_00000.png, shot_clock_00001.png, ...

Usage:
    python image_cropper_shot_clock.py [--start-frame N] video1.mov [video2.mov ...]

    Or interactive mode:
    python image_cropper_shot_clock.py [--start-frame N]
"""

import cv2
import numpy as np
import os
import sys
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, filedialog


class ShotClockDatasetCreator:
    def __init__(self, video_path, output_dir="shot_clock_dataset", start_index=0, start_frame=0):
        self.video_path = video_path
        self.output_dir = output_dir
        self.crop_rect = None
        self.frame_interval = 5
        self.num_frames_to_average = 5
        self.start_index = start_index
        self.start_frame = max(0, int(start_frame))
        self.rotation_degrees = 0

        os.makedirs(output_dir, exist_ok=True)

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
        print(f"Duration: {self.total_frames / max(self.fps, 1):.1f} seconds")
        if self.start_frame > 0:
            print(f"Resume start frame: {self.start_frame}")

        self.rotation_degrees = self._detect_rotation_degrees()
        if self.rotation_degrees:
            print(f"Detected orientation metadata: {self.rotation_degrees}°")

    def _detect_rotation_degrees(self):
        """Best-effort orientation metadata read from OpenCV backend."""
        meta_prop = getattr(cv2, "CAP_PROP_ORIENTATION_META", None)
        if meta_prop is None:
            return 0

        try:
            value = int(round(self.cap.get(meta_prop)))
        except Exception:
            return 0

        if value in {90, 180, 270}:
            return value
        return 0

    def _configure_rotation(self):
        """Allow user to accept metadata-based rotation or provide manual override."""
        if self.rotation_degrees in {90, 180, 270}:
            response = input(
                f"Apply detected rotation correction {self.rotation_degrees}°? [Y/n]: "
            ).strip().lower()
            if response in {"n", "no"}:
                self.rotation_degrees = 0

        manual = input(
            "Manual rotation override (0/90/180/270) [Enter to keep current]: "
        ).strip()
        if manual:
            try:
                manual_value = int(manual)
                if manual_value in {0, 90, 180, 270}:
                    self.rotation_degrees = manual_value
                else:
                    print("Invalid manual rotation; keeping current setting")
            except ValueError:
                print("Invalid manual rotation; keeping current setting")

        if self.rotation_degrees:
            print(f"Using rotation correction: {self.rotation_degrees}°")

    def _apply_rotation(self, frame):
        """Apply configured rotation to a frame."""
        if frame is None:
            return frame

        if self.rotation_degrees == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        if self.rotation_degrees == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        if self.rotation_degrees == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    def select_crop_region(self):
        print("\n=== SELECT SHOT CLOCK REGION ===")
        print("Opening GUI window...")

        self._configure_rotation()

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        ret, frame = self.cap.read()
        if not ret:
            print("Error: Could not read first frame")
            return False

        frame = self._apply_rotation(frame)

        root = tk.Tk()
        root.title("Select Shot Clock Region")
        root.geometry("1200x800")
        root.configure(bg="#2a2a2a")

        title_frame = tk.Frame(root, bg="#2a2a2a", pady=15)
        title_frame.pack(fill=tk.X)

        tk.Label(
            title_frame,
            text="⏱️ Select Shot Clock Region",
            bg="#2a2a2a",
            fg="white",
            font=("Arial", 16, "bold"),
        ).pack()

        tk.Label(
            title_frame,
            text="Click and drag to select the shot clock area, then click 'Confirm Selection'",
            bg="#2a2a2a",
            fg="#FFD700",
            font=("Arial", 10, "italic"),
        ).pack()

        canvas_frame = tk.Frame(root, bg="#1a1a1a")
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        canvas = tk.Canvas(canvas_frame, bg="#000000", cursor="crosshair")
        canvas.pack(fill=tk.BOTH, expand=True)

        selection = {"start": None, "end": None, "rect_id": None, "crop_rect": None}

        def on_mouse_down(event):
            selection["start"] = (event.x, event.y)
            selection["end"] = (event.x, event.y)

        def on_mouse_move(event):
            if selection["start"]:
                selection["end"] = (event.x, event.y)
                if selection["rect_id"]:
                    canvas.delete(selection["rect_id"])
                x1, y1 = selection["start"]
                x2, y2 = selection["end"]
                selection["rect_id"] = canvas.create_rectangle(
                    x1, y1, x2, y2, outline="#00FF00", width=2
                )
                width = abs(x2 - x1)
                height = abs(y2 - y1)
                orig_width = int(width / scale_factor)
                orig_height = int(height / scale_factor)
                info_label.config(text=f"Selection: {orig_width}x{orig_height} pixels")

        def on_mouse_up(event):
            selection["end"] = (event.x, event.y)

        canvas.bind("<ButtonPress-1>", on_mouse_down)
        canvas.bind("<B1-Motion>", on_mouse_move)
        canvas.bind("<ButtonRelease-1>", on_mouse_up)

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

        frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        from PIL import Image, ImageTk

        pil_img = Image.fromarray(frame_rgb)
        photo = ImageTk.PhotoImage(pil_img)

        canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        canvas.image = photo

        bottom_frame = tk.Frame(root, bg="#2a2a2a", pady=15)
        bottom_frame.pack(fill=tk.X)

        info_label = tk.Label(
            bottom_frame,
            text="Selection: Not selected",
            bg="#2a2a2a",
            fg="#4CAF50",
            font=("Arial", 11),
        )
        info_label.pack(pady=5)

        button_frame = tk.Frame(bottom_frame, bg="#2a2a2a")
        button_frame.pack()

        def clear_selection():
            selection["start"] = None
            selection["end"] = None
            if selection["rect_id"]:
                canvas.delete(selection["rect_id"])
                selection["rect_id"] = None
            info_label.config(text="Selection: Cleared")

        def confirm_selection():
            if selection["start"] and selection["end"]:
                x1 = int(min(selection["start"][0], selection["end"][0]) / scale_factor)
                y1 = int(min(selection["start"][1], selection["end"][1]) / scale_factor)
                x2 = int(max(selection["start"][0], selection["end"][0]) / scale_factor)
                y2 = int(max(selection["start"][1], selection["end"][1]) / scale_factor)

                width = x2 - x1
                height = y2 - y1

                if width > 0 and height > 0:
                    selection["crop_rect"] = (x1, y1, width, height)
                    root.quit()
                else:
                    messagebox.showwarning("Invalid Selection", "Please select a valid region")
            else:
                messagebox.showwarning("No Selection", "Please select a region first")

        def cancel():
            selection["crop_rect"] = None
            root.quit()

        tk.Button(
            button_frame,
            text="🔄 Clear Selection",
            command=clear_selection,
            bg="#FF9800",
            fg="white",
            font=("Arial", 11, "bold"),
            padx=20,
            pady=10,
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            button_frame,
            text="✅ Confirm Selection",
            command=confirm_selection,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 11, "bold"),
            padx=20,
            pady=10,
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            button_frame,
            text="❌ Cancel",
            command=cancel,
            bg="#f44336",
            fg="white",
            font=("Arial", 11, "bold"),
            padx=20,
            pady=10,
        ).pack(side=tk.LEFT, padx=5)

        root.mainloop()
        root.destroy()

        self.crop_rect = selection["crop_rect"]

        if self.crop_rect and self.crop_rect[2] > 0 and self.crop_rect[3] > 0:
            print(
                f"✓ Selected region: x={self.crop_rect[0]}, y={self.crop_rect[1]}, "
                f"w={self.crop_rect[2]}, h={self.crop_rect[3]}"
            )
            return True

        print("No valid region selected")
        return False

    def extract_and_save_frames(self):
        if self.crop_rect is None:
            print("Error: No crop region defined")
            return self.start_index

        x, y, w, h = self.crop_rect

        print("\n=== EXTRACTING FRAMES ===")
        print(
            f"Frame interval: every {self.frame_interval} frames "
            f"({self.frame_interval / max(self.fps, 1):.2f} seconds)"
        )
        print(f"Averaging: {self.num_frames_to_average} consecutive frames")
        print(f"Start frame: {self.start_frame}")
        print(f"Output directory: {self.output_dir}/")

        target_size = None
        if self.start_index > 0:
            existing_images = [
                f
                for f in os.listdir(self.output_dir)
                if f.startswith("shot_clock_") and f.endswith(".png")
            ]
            if existing_images:
                first_existing = os.path.join(self.output_dir, existing_images[0])
                sample_img = cv2.imread(first_existing)
                if sample_img is not None:
                    target_size = (sample_img.shape[1], sample_img.shape[0])
                    print(f"Detected existing image size: {target_size[0]}x{target_size[1]}")
                    print("New images will be rescaled to match existing size")

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

        saved_count = 0
        frame_idx = self.start_frame

        if self.num_frames_to_average == 1:
            print("Fast extraction mode enabled (no averaging): skipping decode on non-target frames")

            while True:
                # For non-target frames, grab() advances without full decode (faster).
                if frame_idx % self.frame_interval != 0:
                    grabbed = self.cap.grab()
                    if not grabbed:
                        break
                    frame_idx += 1
                    continue

                ret, frame = self.cap.read()
                if not ret:
                    break

                frame = self._apply_rotation(frame)
                frame_idx += 1

                cropped = frame[y : y + h, x : x + w]

                if target_size is not None:
                    current_size = (cropped.shape[1], cropped.shape[0])
                    if current_size != target_size:
                        cropped = cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)

                filename = f"shot_clock_{self.start_index + saved_count:05d}.png"
                filepath = os.path.join(self.output_dir, filename)
                cv2.imwrite(filepath, cropped)

                saved_count += 1

                if saved_count % 25 == 0:
                    progress = (frame_idx / max(self.total_frames, 1)) * 100
                    print(
                        f"Progress: {frame_idx}/{self.total_frames} frames ({progress:.1f}%) - "
                        f"Saved: {saved_count} images (total dataset: {self.start_index + saved_count})"
                    )
        else:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                frame = self._apply_rotation(frame)

                frame_idx += 1

                if (frame_idx - 1) % self.frame_interval == 0:
                    frames_to_average = [frame]

                    current_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                    for _ in range(1, self.num_frames_to_average):
                        ret_avg, frame_avg = self.cap.read()
                        if ret_avg:
                            frame_avg = self._apply_rotation(frame_avg)
                            frames_to_average.append(frame_avg)
                        else:
                            break

                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)

                    averaged_frame = np.mean(frames_to_average, axis=0).astype(np.uint8)
                    cropped = averaged_frame[y : y + h, x : x + w]

                    if target_size is not None:
                        current_size = (cropped.shape[1], cropped.shape[0])
                        if current_size != target_size:
                            cropped = cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)

                    filename = f"shot_clock_{self.start_index + saved_count:05d}.png"
                    filepath = os.path.join(self.output_dir, filename)
                    cv2.imwrite(filepath, cropped)

                    saved_count += 1

                    if saved_count % 25 == 0:
                        progress = (frame_idx / max(self.total_frames, 1)) * 100
                        print(
                            f"Progress: {frame_idx}/{self.total_frames} frames ({progress:.1f}%) - "
                            f"Saved: {saved_count} images (total dataset: {self.start_index + saved_count})"
                        )

        self.cap.release()

        print(f"\n✓ Done! Saved {saved_count} images from this video")
        print(f"  Total images in dataset: {self.start_index + saved_count}")
        print(f"  Saved to: {self.output_dir}/")

        return self.start_index + saved_count

    def run(self):
        if not self.select_crop_region():
            print("Crop selection cancelled")
            return None

        print("\n=== CONFIGURE EXTRACTION ===")

        response = input(f"Frame interval (extract every N frames) [{self.frame_interval}]: ").strip()
        if response:
            try:
                self.frame_interval = max(1, int(response))
            except ValueError:
                print("Invalid interval, using default")

        response = input(f"Number of frames to average [{self.num_frames_to_average}]: ").strip()
        if response:
            try:
                self.num_frames_to_average = max(1, int(response))
            except ValueError:
                print("Invalid averaging count, using default")

        remaining_frames = max(0, self.total_frames - self.start_frame)
        estimated_images = remaining_frames // max(self.frame_interval, 1)
        print(f"\nEstimated output: ~{estimated_images} images from this video")

        response = input("\nProceed with extraction? [Y/n]: ").strip().lower()
        if response and response not in {"y", "yes"}:
            print("Extraction cancelled")
            return None

        return self.extract_and_save_frames()


def process_multiple_videos(video_paths, output_dir="shot_clock_dataset", start_frame=0):
    print("=" * 60)
    print("SHOT CLOCK DATASET CREATOR - MULTI-VIDEO MODE")
    print("=" * 60)
    print(f"\nProcessing {len(video_paths)} video(s)")
    print(f"Output directory: {output_dir}\n")

    append_mode = False
    start_index = 0

    if os.path.exists(output_dir):
        existing_images = [
            f for f in os.listdir(output_dir) if f.startswith("shot_clock_") and f.endswith(".png")
        ]
        if existing_images:
            print(f"Found existing dataset with {len(existing_images)} images.")
            while True:
                choice = input(
                    "\nWhat would you like to do?\n"
                    "1. Append new frames to existing dataset\n"
                    "2. Create new dataset (overwrite existing)\n"
                    "3. Cancel\n"
                    "Enter choice (1-3): "
                ).strip()

                if choice == "1":
                    append_mode = True
                    start_index = len(existing_images)
                    print(f"Will append new frames starting from index {start_index}")
                    break
                if choice == "2":
                    import shutil

                    shutil.rmtree(output_dir)
                    start_index = 0
                    print("Existing dataset will be overwritten")
                    break
                if choice == "3":
                    print("Operation cancelled")
                    return

                print("Invalid choice. Please enter 1, 2, or 3.")

    os.makedirs(output_dir, exist_ok=True)

    next_index = start_index
    total_extracted = 0

    for i, video_path in enumerate(video_paths, 1):
        print(f"\n{'#' * 60}")
        print(f"VIDEO {i}/{len(video_paths)}: {Path(video_path).name}")
        print(f"{'#' * 60}")

        if not os.path.exists(video_path):
            print(f"Error: Video file not found: {video_path}")
            continue

        current_start_frame = start_frame if i == 1 else 0
        creator = ShotClockDatasetCreator(
            video_path,
            output_dir,
            start_index=next_index,
            start_frame=current_start_frame,
        )
        result = creator.run()

        if result is not None:
            images_from_video = result - next_index
            total_extracted += images_from_video
            next_index = result

    print(f"\n{'=' * 60}")
    print("ALL VIDEOS PROCESSED")
    print(f"{'=' * 60}")

    if append_mode:
        print(f"Appended {total_extracted} new images to existing dataset")
        print(f"Total images in combined dataset: {next_index}")
    else:
        print(f"Created/updated dataset with {next_index} images")

    print(f"Output directory: {output_dir}/")
    print("\nNext steps:")
    print("1. Run: python label_dataset.py")
    print("2. Label images (values 0-30, BLOCKED, BLANK, INCONCLUSIVE)")
    print("3. Train CNN with: python train_shot_clock_cnn.py")


def main():
    video_paths = []
    start_frame = 0

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--start-frame":
            if i + 1 >= len(args):
                print("Error: --start-frame requires an integer value")
                sys.exit(1)
            try:
                start_frame = max(0, int(args[i + 1]))
            except ValueError:
                print(f"Error: invalid --start-frame value: {args[i + 1]}")
                sys.exit(1)
            i += 2
            continue

        if arg.startswith("--start-frame="):
            raw_value = arg.split("=", 1)[1]
            try:
                start_frame = max(0, int(raw_value))
            except ValueError:
                print(f"Error: invalid --start-frame value: {raw_value}")
                sys.exit(1)
            i += 1
            continue

        video_paths.append(arg)
        i += 1

    if not video_paths:
        print("=" * 60)
        print("SHOT CLOCK DATASET CREATOR")
        print("=" * 60)
        print("\nNo videos specified on command line.")
        print("Enter video file paths (one per line, empty line to finish):\n")
        if start_frame > 0:
            print(f"Resume extraction requested from frame {start_frame}")

        while True:
            path = input("Video path (or press Enter to finish): ").strip()
            if not path:
                break
            if os.path.exists(path):
                video_paths.append(path)
                print(f"  ✓ Added: {Path(path).name}")
            else:
                print(f"  ✗ File not found: {path}")

        if not video_paths:
            print("\nOpening file dialog...")
            root = tk.Tk()
            root.withdraw()
            files = filedialog.askopenfilenames(
                title="Select Video File(s)",
                filetypes=[
                    ("Video files", "*.mp4 *.mov *.avi *.mkv *.MOV *.MP4"),
                    ("All files", "*.*"),
                ],
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

    if len(video_paths) == 1:
        output_dir = "shot_clock_dataset"
        start_index = 0

        if os.path.exists(output_dir):
            existing_images = [
                f for f in os.listdir(output_dir) if f.startswith("shot_clock_") and f.endswith(".png")
            ]
            if existing_images:
                print(f"\nFound existing dataset with {len(existing_images)} images.")
                while True:
                    choice = input(
                        "\nWhat would you like to do?\n"
                        "1. Append new frames to existing dataset\n"
                        "2. Create new dataset (overwrite existing)\n"
                        "3. Cancel\n"
                        "Enter choice (1-3): "
                    ).strip()

                    if choice == "1":
                        start_index = len(existing_images)
                        print(f"Will append new frames starting from index {start_index}")
                        break
                    if choice == "2":
                        import shutil

                        shutil.rmtree(output_dir)
                        print("Existing dataset will be overwritten")
                        break
                    if choice == "3":
                        print("Operation cancelled")
                        return

                    print("Invalid choice. Please enter 1, 2, or 3.")

        creator = ShotClockDatasetCreator(
            video_paths[0],
            output_dir,
            start_index=start_index,
            start_frame=start_frame,
        )
        creator.run()
    else:
        process_multiple_videos(video_paths, start_frame=start_frame)


if __name__ == "__main__":
    main()
