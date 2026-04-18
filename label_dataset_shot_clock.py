"""
Shot Clock Dataset Labeling Tool (game-clock-style workflow)

Interactive GUI for labeling shot-clock images extracted from video.

Valid labels:
- 0..30
- BLOCKED
- BLANK
- INCONCLUSIVE

Usage:
    python label_dataset_shot_clock.py

Keyboard shortcuts:
    0-9: enter digits
    Enter: save entered numeric label
    B: BLOCKED
    X: BLANK
    S: INCONCLUSIVE
    G: OCR guess
    N/Right: next image (skip)
    P/Left: previous image
    D: delete current label
    Q/ESC: quit and save
"""

import cv2
import json
import os
import random
import re
import subprocess
import sys
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk


def normalize_shot_label(value):
    """Normalize legacy and new label values to canonical shot-clock format."""
    if value is None:
        return None

    if isinstance(value, (int, float)):
        int_value = int(value)
        if 0 <= int_value <= 30:
            return str(int_value)
        return None

    label = str(value).strip()
    if not label:
        return None

    lower = label.lower()

    if lower in {"blocked", "block", "b"}:
        return "BLOCKED"
    if lower in {"inconclusive", "unknown", "unclear", "s"}:
        return "INCONCLUSIVE"
    if lower in {"blank", "x", "00", "0:00"}:
        return "BLANK"

    if re.fullmatch(r"\d{1,2}", lower):
        value_int = int(lower)
        if 0 <= value_int <= 30:
            return str(value_int)

    return None


class ShotClockLabeler:
    def __init__(self, dataset_dir="shot_clock_dataset", labels_file="shot_clock_labels.json", shuffle=True):
        self.dataset_dir = dataset_dir
        self.labels_file = labels_file
        self.labels = {}

        if os.path.exists(labels_file):
            with open(labels_file, "r", encoding="utf-8") as f:
                raw_labels = json.load(f)
            for key, value in raw_labels.items():
                normalized = normalize_shot_label(value)
                if normalized is not None:
                    self.labels[key] = normalized
            print(f"Loaded {len(self.labels)} existing labels from {labels_file}")

        self.image_files = sorted(
            [f for f in os.listdir(dataset_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]
        )

        if not self.image_files:
            print(f"Error: No images found in {dataset_dir}")
            sys.exit(1)

        if shuffle:
            random.seed(42)
            random.shuffle(self.image_files)
            print("Shuffled image order for diverse labeling")

        print(f"Found {len(self.image_files)} images")

        labeled_count = sum(1 for f in self.image_files if f in self.labels)
        print(f"Labeled: {labeled_count}, Unlabeled: {len(self.image_files) - labeled_count}")

        self.current_idx = 0
        self.current_input = ""

        for i, img_file in enumerate(self.image_files):
            if img_file not in self.labels:
                self.current_idx = i
                break

        self.setup_gui()

    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("Shot Clock Dataset Labeler")
        self.root.geometry("1400x900")
        self.root.configure(bg="#1e1e1e")

        self.root.bind("<Key>", self.on_key_press)
        self.root.bind("<Escape>", lambda e: self.quit_app())

        main_frame = tk.Frame(self.root, bg="#1e1e1e")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        title_frame = tk.Frame(main_frame, bg="#2d2d2d", relief=tk.RAISED, borderwidth=2)
        title_frame.pack(fill=tk.X, pady=(0, 15))

        tk.Label(
            title_frame,
            text="⏱️ SHOT CLOCK LABELER",
            bg="#2d2d2d",
            fg="#4CAF50",
            font=("Arial", 20, "bold"),
            pady=15,
        ).pack()

        stats_frame = tk.Frame(main_frame, bg="#2d2d2d", relief=tk.RIDGE, borderwidth=2)
        stats_frame.pack(fill=tk.X, pady=(0, 15))

        self.stats_label = tk.Label(
            stats_frame,
            text="",
            bg="#2d2d2d",
            fg="#FFD700",
            font=("Arial", 12),
            pady=10,
        )
        self.stats_label.pack()

        image_frame = tk.Frame(main_frame, bg="#000000", relief=tk.SUNKEN, borderwidth=3)
        image_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))

        self.image_label = tk.Label(image_frame, bg="#000000")
        self.image_label.pack(expand=True)

        input_frame = tk.Frame(main_frame, bg="#2d2d2d", relief=tk.RAISED, borderwidth=2)
        input_frame.pack(fill=tk.X, pady=(0, 15))

        tk.Label(
            input_frame,
            text="Current Input:",
            bg="#2d2d2d",
            fg="#FFFFFF",
            font=("Arial", 12, "bold"),
        ).pack(side=tk.LEFT, padx=10, pady=10)

        self.input_label = tk.Label(
            input_frame,
            text="",
            bg="#000000",
            fg="#00FF00",
            font=("Courier New", 24, "bold"),
            width=10,
            relief=tk.SUNKEN,
            borderwidth=2,
        )
        self.input_label.pack(side=tk.LEFT, padx=10, pady=10)

        tk.Label(
            input_frame,
            text="Existing Label:",
            bg="#2d2d2d",
            fg="#FFFFFF",
            font=("Arial", 12, "bold"),
        ).pack(side=tk.LEFT, padx=(30, 10), pady=10)

        self.existing_label = tk.Label(
            input_frame,
            text="None",
            bg="#1e1e1e",
            fg="#FFD700",
            font=("Courier New", 18, "bold"),
            width=14,
            relief=tk.SUNKEN,
            borderwidth=2,
        )
        self.existing_label.pack(side=tk.LEFT, padx=10, pady=10)

        instructions_frame = tk.Frame(main_frame, bg="#2d2d2d", relief=tk.GROOVE, borderwidth=2)
        instructions_frame.pack(fill=tk.X)

        instructions_text = """
KEYBOARD SHORTCUTS:
• 0-9: Type shot clock value (0-30) and press Enter
• B: BLOCKED  • X: BLANK  • S: INCONCLUSIVE  • G: OCR Auto-guess
• Enter: Submit label  • N: Next (skip)  • P: Previous  • D: Delete label  • Q/ESC: Quit & Save
        """

        tk.Label(
            instructions_frame,
            text=instructions_text,
            bg="#2d2d2d",
            fg="#CCCCCC",
            font=("Arial", 10),
            justify=tk.LEFT,
            pady=10,
        ).pack()

        self.display_current_image()

    def display_current_image(self):
        if self.current_idx >= len(self.image_files):
            messagebox.showinfo("Complete", "All images labeled!")
            self.quit_app()
            return

        img_filename = self.image_files[self.current_idx]
        img_path = os.path.join(self.dataset_dir, img_filename)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Error loading image: {img_path}")
            self.current_idx += 1
            self.display_current_image()
            return

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        display_img = cv2.resize(img_rgb, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
        pil_img = Image.fromarray(display_img)
        photo = ImageTk.PhotoImage(pil_img)

        self.image_label.configure(image=photo)
        self.image_label.image = photo

        labeled_count = sum(1 for f in self.image_files if f in self.labels)
        progress_pct = (labeled_count / len(self.image_files)) * 100

        stats_text = (
            f"Image {self.current_idx + 1} / {len(self.image_files)}  |  "
            f"Labeled: {labeled_count}  |  "
            f"Unlabeled: {len(self.image_files) - labeled_count}  |  "
            f"Progress: {progress_pct:.1f}%  |  File: {img_filename}"
        )
        self.stats_label.config(text=stats_text)

        if img_filename in self.labels:
            label_text = self.labels[img_filename]
            if label_text == "BLOCKED":
                self.existing_label.config(text="🚫 BLOCKED", fg="#FF5555")
            elif label_text == "BLANK":
                self.existing_label.config(text="⏱️ BLANK", fg="#55FFFF")
            elif label_text == "INCONCLUSIVE":
                self.existing_label.config(text="❓ INCONCLUSIVE", fg="#FFAA55")
            else:
                self.existing_label.config(text=f"✓ {label_text}", fg="#55FF55")
        else:
            self.existing_label.config(text="None", fg="#888888")

        self.current_input = ""
        self.input_label.config(text=self.current_input)

    def validate_numeric_input(self, value):
        if not re.fullmatch(r"\d{1,2}", value):
            return False
        int_value = int(value)
        return 0 <= int_value <= 30

    def on_key_press(self, event):
        key = event.char.lower()
        keysym = event.keysym

        if keysym == "Escape" or key == "q":
            self.quit_app()
            return
        if key == "n" or keysym == "Right":
            self.next_image()
            return
        if key == "p" or keysym == "Left":
            self.previous_image()
            return
        if key == "d":
            self.delete_label()
            return

        if key == "b":
            self.save_label("BLOCKED")
            return
        if key == "x":
            self.save_label("BLANK")
            return
        if key == "s":
            self.save_label("INCONCLUSIVE")
            return
        if key == "g":
            self.ocr_guess()
            return

        if key.isdigit():
            if len(self.current_input) < 2:
                self.current_input += key
                self.input_label.config(text=self.current_input, fg="#FFFFFF")
            return

        if keysym == "BackSpace":
            self.current_input = self.current_input[:-1]
            self.input_label.config(text=self.current_input, fg="#FFFFFF")
            return

        if keysym in {"Return", "KP_Enter"}:
            if not self.current_input:
                return
            if self.validate_numeric_input(self.current_input):
                self.save_label(str(int(self.current_input)))
            else:
                messagebox.showwarning("Invalid Value", "Please enter a value between 0 and 30")

    def save_label(self, label):
        img_filename = self.image_files[self.current_idx]
        normalized = normalize_shot_label(label)
        if normalized is None:
            messagebox.showwarning("Invalid Label", f"Could not normalize label: {label}")
            return

        self.labels[img_filename] = normalized

        with open(self.labels_file, "w", encoding="utf-8") as f:
            json.dump(self.labels, f, indent=2)

        if normalized == "BLOCKED":
            print(f"🚫 Labeled {img_filename} as BLOCKED")
        elif normalized == "BLANK":
            print(f"⏱️ Labeled {img_filename} as BLANK")
        elif normalized == "INCONCLUSIVE":
            print(f"❓ Labeled {img_filename} as INCONCLUSIVE")
        else:
            print(f"✓ Labeled {img_filename} as {normalized}")

        self.next_image()

    def next_image(self):
        if self.current_idx < len(self.image_files) - 1:
            self.current_idx += 1
            self.display_current_image()
        else:
            messagebox.showinfo("End", "Reached last image!")

    def previous_image(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.display_current_image()
        else:
            messagebox.showinfo("Start", "Already at first image!")

    def delete_label(self):
        img_filename = self.image_files[self.current_idx]
        if img_filename in self.labels:
            del self.labels[img_filename]
            with open(self.labels_file, "w", encoding="utf-8") as f:
                json.dump(self.labels, f, indent=2)
            print(f"Deleted label for {img_filename}")
            self.display_current_image()
        else:
            messagebox.showinfo("No Label", "Current image has no label to delete")

    def ocr_guess(self):
        img_filename = self.image_files[self.current_idx]
        img_path = os.path.join(self.dataset_dir, img_filename)

        print(f"Running OCR on {img_filename}...")
        guess = self.run_ocr_on_image(img_path)

        if guess is not None:
            self.current_input = str(guess)
            self.input_label.config(text=f"OCR: {self.current_input}", fg="#FFD700")
            print(f"✓ OCR guess: {guess}")

            response = messagebox.askyesno(
                "OCR Guess", f"OCR guessed: {guess}\n\nAccept this label?", icon="question"
            )

            if response:
                self.save_label(str(guess))
            else:
                self.input_label.config(text=f"Edit: {self.current_input}")
        else:
            self.input_label.config(text="OCR failed", fg="#FF5555")
            print(f"✗ OCR failed for {img_filename}")
            messagebox.showwarning("OCR Failed", "Could not recognize value. Try manual entry.")

    def run_ocr_on_image(self, image_path):
        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return None

            img = cv2.equalizeHist(img)
            _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            height, width = img.shape
            img = cv2.resize(img, (width * 4, height * 4), interpolation=cv2.INTER_CUBIC)

            temp_path = image_path + "_temp_ocr.png"
            cv2.imwrite(temp_path, img)

            results = []

            if os.path.exists("./tessdata_ssd"):
                for psm in ["7", "8", "13"]:
                    result = subprocess.run(
                        [
                            "tesseract",
                            temp_path,
                            "stdout",
                            "--tessdata-dir",
                            "./tessdata_ssd",
                            "-l",
                            "ssd_alphanum_plus",
                            "--psm",
                            psm,
                            "-c",
                            "tessedit_char_whitelist=0123456789",
                        ],
                        capture_output=True,
                        text=True,
                    )
                    if result.returncode == 0 and result.stdout.strip():
                        results.append(result.stdout.strip())

            for psm in ["7", "8", "13"]:
                result = subprocess.run(
                    [
                        "tesseract",
                        temp_path,
                        "stdout",
                        "--psm",
                        psm,
                        "-c",
                        "tessedit_char_whitelist=0123456789",
                    ],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0 and result.stdout.strip():
                    results.append(result.stdout.strip())

            if os.path.exists(temp_path):
                os.remove(temp_path)

            for text in results:
                cleaned = re.sub(r"\D", "", text)
                if cleaned:
                    value = int(cleaned)
                    if 0 <= value <= 30:
                        return value

            return None

        except FileNotFoundError:
            messagebox.showerror("OCR Error", "Tesseract not found. Please install Tesseract OCR.")
            return None
        except Exception as exc:
            print(f"OCR error: {exc}")
            return None

    def quit_app(self):
        with open(self.labels_file, "w", encoding="utf-8") as f:
            json.dump(self.labels, f, indent=2)

        print(f"\n✓ Saved {len(self.labels)} labels to {self.labels_file}")
        print("Goodbye!")

        self.root.quit()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main():
    print("=" * 60)
    print("SHOT CLOCK DATASET LABELER")
    print("=" * 60)

    dataset_dir = "shot_clock_dataset"
    labels_file = "shot_clock_labels.json"

    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory not found: {dataset_dir}")
        print("\nPlease run image_cropper.py first to create the dataset")
        sys.exit(1)

    labeler = ShotClockLabeler(dataset_dir=dataset_dir, labels_file=labels_file)
    labeler.run()


if __name__ == "__main__":
    main()
