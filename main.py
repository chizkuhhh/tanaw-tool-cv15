import tkinter as tk
from tkinter import ttk, filedialog, messagebox, Toplevel
import cv2
import os
import sys
import yaml
import json
import threading
import math
import zipfile
import subprocess
from datetime import datetime
from pathlib import Path
from PIL import Image, ImageTk

try:
    from logic.histogram_extractor import extract_histogram_based
    from logic.gps_extractor import extract_gps_based
    from logic.anonymizer import anonymize_images
except ImportError:
    print("Warning: 'logic' modules not found. Ensure logic/ folder exists.")

from ultralytics import YOLO

class TanawApp:
    def __init__(self, root):
        self.root = root
        self.root.title("TANAW - Traffic-Sign Annotation and Augmentation Workspace")
        self.root.geometry("1200x900")
        
        # --- State Variables ---
        self.extracted_frames = [] 
        self.anonymized_images = []
        self.annotation_images = []
        
        self.current_extract_idx = 0
        self.current_anon_idx = 0
        self.current_anno_idx = 0
        
        self.CLASSES = {}
        
        # Anonymization Editor State
        self.anon_scale = 1.0
        self.anon_offset_x = 0
        self.anon_offset_y = 0
        self.drawing_start = None
        self.current_rect_id = None
        self.selected_rect_id = None
        self.anon_boxes = [] 
        self.rect_map = {}   
        
        self.load_classes()

        # --- Styles ---
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TButton", padding=5, font=('Helvetica', 10))
        style.configure("TLabel", font=('Helvetica', 11))
        style.configure("Header.TLabel", font=('Helvetica', 14, 'bold'))

        # --- Tabs ---
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)

        tab_extraction_container = ttk.Frame(self.notebook)
        tab_anonymization_container = ttk.Frame(self.notebook)
        tab_annotation_container = ttk.Frame(self.notebook)
        tab_augmentation_container = ttk.Frame(self.notebook)
        tab_review_container = ttk.Frame(self.notebook)

        self.tab_extraction = self.create_scrollable_frame(tab_extraction_container)
        self.tab_anonymization = self.create_scrollable_frame(tab_anonymization_container)
        self.tab_annotation = self.create_scrollable_frame(tab_annotation_container)
        self.tab_augmentation = self.create_scrollable_frame(tab_augmentation_container)
        self.tab_review = self.create_scrollable_frame(tab_review_container)

        self.notebook.add(tab_extraction_container, text="1. Extraction")
        self.notebook.add(tab_anonymization_container, text="2. Anonymization")
        self.notebook.add(tab_annotation_container, text="3. Annotation")
        self.notebook.add(tab_augmentation_container, text="4. Augmentation")
        self.notebook.add(tab_review_container, text="5. Review & Export")

        # --- Setup UI ---
        self.setup_extraction_ui()
        self.setup_anonymization_ui()
        self.setup_annotation_ui()

        # --- Bind Keys ---
        self.root.bind('<Left>', self.handle_keypress)
        self.root.bind('<Right>', self.handle_keypress)
        self.root.bind('<Delete>', self.on_delete_key)
        self.root.bind('<BackSpace>', self.on_delete_key)

    def create_scrollable_frame(self, parent):
        """Create a scrollable frame for a tab"""
        canvas = tk.Canvas(parent, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        
        scrollable_frame = ttk.Frame(canvas)
        
        def configure_frame(event):
            # Make frame width match canvas width
            canvas.itemconfig(frame_id, width=event.width)
            
            # Update scroll region based on actual content
            scrollable_frame.update_idletasks()
            content_height = scrollable_frame.winfo_reqheight()
            
            # If content is shorter than canvas, make frame fill canvas height
            # Otherwise, let it be as tall as content
            if content_height < event.height:
                canvas.itemconfig(frame_id, height=event.height)
                canvas.configure(scrollregion=(0, 0, event.width, event.height))
            else:
                canvas.itemconfig(frame_id, height=content_height)
                canvas.configure(scrollregion=(0, 0, event.width, content_height))
        
        canvas.bind("<Configure>", configure_frame)
        
        frame_id = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        def _bind_to_mousewheel(event):
            canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        def _unbind_from_mousewheel(event):
            canvas.unbind_all("<MouseWheel>")
        
        canvas.bind('<Enter>', _bind_to_mousewheel)
        canvas.bind('<Leave>', _unbind_from_mousewheel)
        
        return scrollable_frame
    
    def start_drag_extract(self, event):
        """Start dragging the extraction image"""
        self.extract_canvas.scan_mark(event.x, event.y)

    def drag_extract(self, event):
        """Drag the extraction image"""
        self.extract_canvas.scan_dragto(event.x, event.y, gain=1)

    def load_classes(self):
        if os.path.exists("classes.yaml"):
            with open("classes.yaml", 'r') as f:
                self.CLASSES = yaml.safe_load(f)

    def handle_keypress(self, event):
        idx = self.notebook.index(self.notebook.select())
        if idx == 0:
            if event.keysym == 'Left': self.prev_extract_image()
            elif event.keysym == 'Right': self.next_extract_image()
        elif idx == 1:
            if event.keysym == 'Left': self.prev_anon_image()
            elif event.keysym == 'Right': self.next_anon_image()
        elif idx == 2:
            if event.keysym == 'Left': self.prev_anno_image()
            elif event.keysym == 'Right': self.next_anno_image()

    def on_method_change(self):
        """
        Called when extraction method radia button changes
        """

        method = self.extract_method.get()

        if method == "Histogram":
            # disable GPX widgets
            self.gpx_label.config(state='disabled')
            self.gpx_entry.config(state='disabled')
            self.gpx_button.config(state='disabled')

            # hide the note for matching gpx files
            self.gps_note.grid_remove()

        else:
            # enable gpx widgets
            self.gpx_label.config(state='normal')
            self.gpx_entry.config(state='normal')
            self.gpx_button.config(state='normal')

            # hide the note for matching gpx files
            self.gps_note.grid()

    def on_delete_key(self, event):
        """Handle Delete key press - delete selected box in current tab"""
        current_tab = self.notebook.index(self.notebook.select())
        
        if current_tab == 1:  # Anonymization tab
            self.delete_anon_rect(event)
        elif current_tab == 2:  # Annotation tab
            self.delete_anno_rect(event)

    # =========================================================================
    # TAB 1: EXTRACTION
    # =========================================================================
    def setup_extraction_ui(self):
        frame = self.tab_extraction
        
        top_frame = ttk.Frame(frame)
        top_frame.pack(side='top', fill='x')

        config_frame = ttk.LabelFrame(top_frame, text="Configuration", padding=10)
        config_frame.pack(fill='x', padx=10, pady=5)

        # Select extraction method here
        ttk.Label(config_frame, text="Extraction Method:").grid(row=0, column=0, sticky='w')
        self.extract_method = tk.StringVar(value="Histogram")
        ttk.Radiobutton(config_frame, text="Histogram", variable=self.extract_method, value="Histogram", 
                        command=self.on_method_change).grid(row=0, column=1, sticky='w')
        ttk.Radiobutton(config_frame, text="GPS", variable=self.extract_method, value="GPS",
                        command=self.on_method_change).grid(row=0, column=2, sticky='w')
        
        # Note the gps extraction requires matching gpx file
        self.gps_note = ttk.Label(config_frame, text="Note: GPS extraction method requires corresponding .gpx file for each video.")
        self.gps_note.grid(row=3, column=0, columnspan=5, sticky='w', pady=(5, 0))

        ttk.Label(config_frame, text="Input:").grid(row=1, column=0, sticky='w', pady=5)
        self.video_input_path = tk.StringVar()
        ttk.Entry(config_frame, textvariable=self.video_input_path, width=40).grid(row=1, column=1, columnspan=2)
        ttk.Button(config_frame, text="File", command=self.browse_video_file).grid(row=1, column=3, padx=2)
        ttk.Button(config_frame, text="Folder", command=self.browse_video_folder).grid(row=1, column=4, padx=2)

        self.gpx_label = ttk.Label(config_frame, text="GPX:")
        self.gpx_label.grid(row=2, column=0, sticky='w', pady=5 )

        self.gpx_input_path = tk.StringVar()
        self.gpx_entry = ttk.Entry(config_frame, textvariable=self.gpx_input_path, width=40)
        self.gpx_entry.grid(row=2, column=1, columnspan=2)

        self.gpx_button = ttk.Button(config_frame, text="Browse", command=self.browse_gpx)
        self.gpx_button.grid(row=2, column=3, padx=2)

        mid_frame = ttk.Frame(top_frame)
        mid_frame.pack(fill='x', padx=10)

        meta_frame = ttk.LabelFrame(mid_frame, text="Metadata", padding=5)
        meta_frame.pack(side='left', fill='both', expand=True, padx=(0,5))
        
        ttk.Label(meta_frame, text="Loc:").grid(row=0, column=0)
        self.meta_location = tk.StringVar()
        ttk.Entry(meta_frame, textvariable=self.meta_location, width=15).grid(row=0, column=1, sticky='ew')
        
        ttk.Label(meta_frame, text="Time:").grid(row=0, column=2)
        self.meta_time = tk.StringVar(value="Day")
        ttk.Combobox(meta_frame, textvariable=self.meta_time, values=["Day","Night","Dawn/Dusk"], width=8).grid(row=0, column=3)
        
        ttk.Label(meta_frame, text="Weather:").grid(row=0, column=4)
        self.meta_weather = tk.StringVar(value="Clear")
        ttk.Combobox(meta_frame, textvariable=self.meta_weather, values=["Clear", "Rainy", "Foggy", "Cloudy", "Other"], width=8).grid(row=0, column=5)

        ttk.Label(meta_frame, text="Lighting:").grid(row=0, column=6)
        self.meta_lighting = tk.StringVar(value="Normal")
        ttk.Combobox(meta_frame, textvariable=self.meta_lighting, values=["Normal", "Low-light (Tunnel)", "Low-light (Underpass)", "Low-light (Other)"], width=8).grid(row=0, column=7)

        ttk.Label(meta_frame, text="Road:").grid(row=1, column=0)
        self.meta_road_type = tk.StringVar(value="Highway")
        ttk.Combobox(meta_frame, textvariable=self.meta_road_type, 
                    values=["Highway", "Urban", "Rural", "Residential", "Other"], 
                    width=8).grid(row=1, column=1)

        param_frame = ttk.LabelFrame(mid_frame, text="Parameters", padding=5)
        param_frame.pack(side='left', fill='both', expand=True, padx=(5,0))
        
        ttk.Label(param_frame, text="Dist(m):").pack(side='left')
        self.target_dist = tk.DoubleVar(value=3.5)
        ttk.Entry(param_frame, textvariable=self.target_dist, width=5).pack(side='left', padx=2)
        
        ttk.Label(param_frame, text="Spd(km/h):").pack(side='left')
        self.speed_kph = tk.DoubleVar(value=40)
        ttk.Entry(param_frame, textvariable=self.speed_kph, width=5).pack(side='left', padx=2)
        
        ttk.Label(param_frame, text="Thresh:").pack(side='left')
        self.hist_thresh = tk.DoubleVar(value=0.15)
        ttk.Scale(param_frame, variable=self.hist_thresh, from_=0.05, to=0.5, 
                command=lambda v: self.lbl_hist.config(text=f"{float(v):.2f}")).pack(side='left')
        self.lbl_hist = ttk.Label(param_frame, text="0.15")
        self.lbl_hist.pack(side='left')

        action_frame = ttk.Frame(top_frame, padding=5)
        action_frame.pack(fill='x')
        self.btn_extract = ttk.Button(action_frame, text="START EXTRACTION", command=self.run_extraction_thread)
        self.btn_extract.pack(side='left', fill='x', expand=True, padx=10)
        
        self.extract_progress = ttk.Progressbar(action_frame, mode='determinate', maximum=100)
        self.extract_progress.pack(fill='x', pady=5)
        self.extract_status = ttk.Label(action_frame, text="Ready")
        self.extract_status.pack()

        self.extract_viewer_frame = ttk.LabelFrame(frame, text="Extraction Results", padding=5)
        self.extract_viewer_frame.pack(fill='both', expand=True, padx=10, pady=5)

        # Canvas with zoom toggle button
        canvas_header = ttk.Frame(self.extract_viewer_frame)
        canvas_header.pack(fill='x', pady=(0, 5))
        
        self.extract_zoom_enabled = tk.BooleanVar(value=False)
        self.extract_zoom_toggle_btn = ttk.Checkbutton(
            canvas_header, 
            text="Enable Zoom (scroll wheel)", 
            variable=self.extract_zoom_enabled,
            command=self.toggle_extract_zoom
        )
        self.extract_zoom_toggle_btn.pack(side='left')
        
        ttk.Label(canvas_header, text="| Pan: Middle mouse button").pack(side='left', padx=10)

        # Simple canvas - no scrollbars
        self.extract_canvas = tk.Canvas(self.extract_viewer_frame, bg="#333")
        self.extract_canvas.pack(fill='both', expand=True)

        # Zoom and pan state
        self.extract_zoom = 1.0
        self.extract_image_on_canvas = None
        self.extract_drag_start_x = 0
        self.extract_drag_start_y = 0
        self.extract_image_x = 0
        self.extract_image_y = 0

        # Bind middle mouse button for panning
        self.extract_canvas.bind("<ButtonPress-2>", self.start_drag_extract)
        self.extract_canvas.bind("<B2-Motion>", self.drag_extract)
        
        # Bind canvas resize to refit image
        self.extract_canvas.bind("<Configure>", self.on_extract_canvas_resize)
        
        nav_frame = ttk.Frame(frame)
        nav_frame.pack(fill='x', padx=10, pady=5)
        ttk.Button(nav_frame, text="<< Prev", command=self.prev_extract_image).pack(side='left', expand=True)
        self.extract_count_lbl = ttk.Label(nav_frame, text="0/0")
        self.extract_count_lbl.pack(side='left', padx=10)
        ttk.Button(nav_frame, text="Next >>", command=self.next_extract_image).pack(side='left', expand=True)

        self.on_method_change()

    def browse_video_file(self):
        f = filedialog.askopenfilename(filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv")])
        if f: self.video_input_path.set(f)

    def browse_video_folder(self):
        f = filedialog.askdirectory()
        if f: self.video_input_path.set(f)

    def browse_gpx(self):
        f = filedialog.askopenfilename(filetypes=[("GPX", "*.gpx")])
        if f: self.gpx_input_path.set(f)

    def run_extraction_thread(self):
        if not self.video_input_path.get() or not self.meta_location.get():
            messagebox.showerror("Error", "Missing inputs (Video or Location).")
            return

        self.btn_extract.config(state='disabled')
        self.extract_progress.start(10)
        self.extract_status.config(text="Extracting...")
        threading.Thread(target=self.process_extraction, daemon=True).start()

    def process_extraction(self):
        try:
            video_source = self.video_input_path.get()
            is_dir = os.path.isdir(video_source)
            video_files = []
            
            if is_dir:
                for ext in [".mp4", ".avi", ".mov", ".mkv"]:
                    video_files.extend(list(Path(video_source).glob(f"*{ext}")))
            else:
                video_files.append(Path(video_source))

            self.extracted_frames = []
            total_videos = len(video_files)
            
            for video_idx, v_path in enumerate(video_files):
                v_name = v_path.name
                dataset_name = os.path.splitext(v_name)[0]
                
                out_dir = os.path.join("extracted_frames", dataset_name)
                os.makedirs(out_dir, exist_ok=True)
                
                # Define callback for this video
                def update_progress(current, total):
                    # Calculate overall progress across all videos
                    video_progress = (video_idx / total_videos) * 100
                    current_video_progress = (current / total) * (100 / total_videos)
                    overall_progress = video_progress + current_video_progress
                    
                    self.root.after(0, lambda p=overall_progress: 
                                self.extract_progress.configure(value=p))
                    self.root.after(0, lambda: 
                                self.extract_status.config(text=f"Video {video_idx+1}/{total_videos}: {current}/{total} frames"))
                
                frames = []
                if self.extract_method.get() == "Histogram":
                    frames, _ = extract_histogram_based(
                        str(v_path), out_dir, self.target_dist.get(), 
                        self.speed_kph.get(), self.hist_thresh.get(),
                        progress_callback=update_progress
                    )
                else:
                    gpx = self.gpx_input_path.get()
                    if os.path.isdir(gpx):
                        cand = os.path.join(gpx, os.path.splitext(v_name)[0]+".gpx")
                        if os.path.exists(cand): gpx = cand
                    if os.path.exists(gpx):
                        frames, _ = extract_gps_based(
                            str(v_path), gpx, out_dir, self.target_dist.get(),
                            progress_callback=update_progress
                        )

                if frames:
                    self.extracted_frames.extend(frames)
                    meta_dir = os.path.join("metadata", dataset_name)
                    os.makedirs(meta_dir, exist_ok=True)
                    
                    for fp in frames:
                        fn = os.path.splitext(os.path.basename(fp))[0]
                        meta_data = {
                            "dataset_name": dataset_name,
                            "location": self.meta_location.get(),
                            "time_of_day": self.meta_time.get(),
                            "weather": self.meta_weather.get(),
                            "lighting": self.meta_lighting.get(),
                            "road_type": self.meta_road_type.get(),
                            "difficult": False,
                            "augmented": False
                        }
                        with open(os.path.join(meta_dir, f"{fn}.json"),'w') as f:
                            json.dump(meta_data, f, indent=4)

            self.root.after(0, self.finish_extraction)
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
            self.root.after(0, self.reset_extraction_ui)

    def finish_extraction(self):
        self.extract_progress.stop()
        self.extract_status.config(text=f"Done! {len(self.extracted_frames)} frames.")
        self.btn_extract.config(state='normal')
        self.current_extract_idx = 0
        if self.extracted_frames:
            self.show_extract_image()
            messagebox.showinfo("Done", f"Extracted {len(self.extracted_frames)} frames.")
        else:
            messagebox.showwarning("Result", "No frames extracted.")
        self.refresh_folder_lists()

    def reset_extraction_ui(self):
        self.extract_progress.stop()
        self.btn_extract.config(state='normal')

    def toggle_extract_zoom(self):
        """Toggle zoom mode on/off"""
        if self.extract_zoom_enabled.get():
            # Enable zoom - bind mousewheel
            self.extract_canvas.bind("<MouseWheel>", self.zoom_extract_image)
            self.extract_canvas.bind("<Button-4>", self.zoom_extract_image)
            self.extract_canvas.bind("<Button-5>", self.zoom_extract_image)
        else:
            # Disable zoom - unbind mousewheel
            self.extract_canvas.unbind("<MouseWheel>")
            self.extract_canvas.unbind("<Button-4>")
            self.extract_canvas.unbind("<Button-5>")

    def show_extract_image(self):
        if not self.extracted_frames: 
            return
        
        path = self.extracted_frames[self.current_extract_idx]
        
        # Load original image
        self.extract_pil_original = Image.open(path)
        
        # Reset zoom when showing new image
        self.extract_zoom = 1.0
        
        # Fit image to canvas
        self.fit_extract_image_to_canvas()
        
        # Update counter
        self.extract_count_lbl.config(text=f"{self.current_extract_idx+1}/{len(self.extracted_frames)}")

    def fit_extract_image_to_canvas(self):
        """Fit the image to canvas while maintaining aspect ratio"""
        if not hasattr(self, 'extract_pil_original'):
            return
        
        # Get canvas dimensions
        canvas_width = self.extract_canvas.winfo_width()
        canvas_height = self.extract_canvas.winfo_height()
        
        # Skip if canvas not ready
        if canvas_width <= 1 or canvas_height <= 1:
            return
        
        # Get image dimensions
        img_width, img_height = self.extract_pil_original.size
        
        # Calculate scaling to fit canvas while maintaining aspect ratio
        width_ratio = canvas_width / img_width
        height_ratio = canvas_height / img_height
        
        # Use the smaller ratio to ensure image fits entirely
        fit_ratio = min(width_ratio, height_ratio) * 0.95  # 95% to leave some padding
        
        # Set zoom to fit ratio
        self.extract_zoom = fit_ratio
        
        # Center the image
        scaled_width = img_width * fit_ratio
        scaled_height = img_height * fit_ratio
        
        self.extract_image_x = (canvas_width - scaled_width) // 2
        self.extract_image_y = (canvas_height - scaled_height) // 2
        
        # Redraw
        self.redraw_extract_image()

    def on_extract_canvas_resize(self, event):
        """Re-fit image when canvas is resized"""
        if hasattr(self, 'extract_pil_original') and self.extract_zoom == 1.0:
            # Only auto-refit if we're not zoomed in
            self.fit_extract_image_to_canvas()

    def redraw_extract_image(self):
        """Redraw the extraction image at current zoom level"""
        if not hasattr(self, 'extract_pil_original'):
            return
        
        # Calculate new size based on zoom
        orig_w, orig_h = self.extract_pil_original.size
        new_w = int(orig_w * self.extract_zoom)
        new_h = int(orig_h * self.extract_zoom)
        
        # Resize image
        resized = self.extract_pil_original.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.extract_tk_image = ImageTk.PhotoImage(resized)
        
        # Clear canvas and draw image at current position
        self.extract_canvas.delete("all")
        self.extract_image_on_canvas = self.extract_canvas.create_image(
            self.extract_image_x, self.extract_image_y, 
            anchor='nw', 
            image=self.extract_tk_image
        )

    def zoom_extract_image(self, event):
        """Handle mouse wheel zoom for extraction viewer - zooms toward mouse cursor"""
        if not hasattr(self, 'extract_pil_original'):
            return
        
        # Get mouse position relative to canvas
        mouse_x = event.x
        mouse_y = event.y
        
        # Calculate position in image before zoom
        old_zoom = self.extract_zoom
        
        # Determine zoom direction
        if event.num == 5 or event.delta < 0:  # Zoom out
            self.extract_zoom *= 0.9
        elif event.num == 4 or event.delta > 0:  # Zoom in
            self.extract_zoom *= 1.1
        
        # Clamp zoom between 0.1x and 10x
        self.extract_zoom = max(0.1, min(10.0, self.extract_zoom))
        
        # Calculate zoom factor change
        zoom_factor = self.extract_zoom / old_zoom
        
        # Adjust image position to zoom toward mouse cursor
        # The point under the mouse should stay in the same place
        self.extract_image_x = mouse_x - (mouse_x - self.extract_image_x) * zoom_factor
        self.extract_image_y = mouse_y - (mouse_y - self.extract_image_y) * zoom_factor
        
        # Redraw at new zoom level
        self.redraw_extract_image()

    def start_drag_extract(self, event):
        """Start dragging the image with middle mouse button"""
        self.extract_drag_start_x = event.x
        self.extract_drag_start_y = event.y

    def drag_extract(self, event):
        """Drag the image around the canvas with middle mouse button"""
        if not hasattr(self, 'extract_pil_original'):
            return
        
        # Calculate how much the mouse moved
        dx = event.x - self.extract_drag_start_x
        dy = event.y - self.extract_drag_start_y
        
        # Update image position
        self.extract_image_x += dx
        self.extract_image_y += dy
        
        # Update drag start position for next movement
        self.extract_drag_start_x = event.x
        self.extract_drag_start_y = event.y
        
        # Redraw image at new position
        self.redraw_extract_image()

    def next_extract_image(self):
        if self.current_extract_idx < len(self.extracted_frames) - 1:
            self.current_extract_idx += 1
            self.show_extract_image()

    def prev_extract_image(self):
        if self.current_extract_idx > 0:
            self.current_extract_idx -= 1
            self.show_extract_image()

    # =========================================================================
    # TAB 2: ANONYMIZATION
    # =========================================================================
    def setup_anonymization_ui(self):
        frame = self.tab_anonymization
        
        ctrl_frame = ttk.Frame(frame, padding=10)
        ctrl_frame.pack(fill='x')
        
        ttk.Label(ctrl_frame, text="Folder:").pack(side='left')
        self.anon_folder_var = tk.StringVar()
        self.anon_combo = ttk.Combobox(ctrl_frame, textvariable=self.anon_folder_var, width=25)
        self.anon_combo.pack(side='left', padx=5)
        self.anon_combo.bind("<<ComboboxSelected>>", self.load_anon_folder)
        
        ttk.Label(ctrl_frame, text="Conf:").pack(side='left', padx=5)
        self.anon_conf = tk.DoubleVar(value=0.05)
        ttk.Scale(ctrl_frame, variable=self.anon_conf, from_=0.01, to=0.5,
                command=lambda v: self.lbl_anon.config(text=f"{float(v):.2f}")).pack(side='left')
        self.lbl_anon = ttk.Label(ctrl_frame, text="0.05")
        self.lbl_anon.pack(side='left', padx=2)
        
        self.btn_anon = ttk.Button(ctrl_frame, text="Auto-Detect", command=self.run_anonymization_thread)
        self.btn_anon.pack(side='left', padx=10)

        self.anon_status_frame = ttk.Frame(ctrl_frame)
        self.anon_status_frame.pack(side='left', fill='x', expand=True)
        self.anon_progress = ttk.Progressbar(self.anon_status_frame, mode='determinate', maximum=100)
        self.anon_progress.pack(side='left', fill='x', expand=True)
        self.anon_status_lbl = ttk.Label(self.anon_status_frame, text="Ready")
        self.anon_status_lbl.pack(side='left', padx=5)

        self.anon_viewer_frame = ttk.Frame(frame)
        self.anon_viewer_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Canvas header with zoom toggle
        canvas_header = ttk.Frame(self.anon_viewer_frame)
        canvas_header.pack(fill='x', pady=(0, 5))
        
        self.anon_zoom_enabled = tk.BooleanVar(value=False)
        self.anon_zoom_toggle_btn = ttk.Checkbutton(
            canvas_header, 
            text="Enable Zoom (scroll wheel)", 
            variable=self.anon_zoom_enabled,
            command=self.toggle_anon_zoom
        )
        self.anon_zoom_toggle_btn.pack(side='left')
        
        ttk.Label(canvas_header, text="| Pan: Middle mouse | Draw box: Left click & drag | Delete: Select box & press Delete").pack(side='left', padx=10)
        
        self.anon_canvas = tk.Canvas(self.anon_viewer_frame, bg="#333", cursor="cross")
        self.anon_canvas.pack(fill='both', expand=True)

        # Zoom and pan state
        self.anon_zoom = 1.0
        self.anon_drag_start_x = 0
        self.anon_drag_start_y = 0
        self.anon_image_x = 0
        self.anon_image_y = 0

        # Bind events
        self.anon_canvas.bind("<Button-1>", self.on_anon_click)
        self.anon_canvas.bind("<B1-Motion>", self.on_anon_drag)
        self.anon_canvas.bind("<ButtonRelease-1>", self.on_anon_release)
        
        # Middle mouse for panning
        self.anon_canvas.bind("<ButtonPress-2>", self.start_drag_anon)
        self.anon_canvas.bind("<B2-Motion>", self.drag_anon)
        
        # Canvas resize
        self.anon_canvas.bind("<Configure>", self.on_anon_canvas_resize)
        
        nav_frame = ttk.Frame(frame, padding=5)
        nav_frame.pack(fill='x')
        ttk.Button(nav_frame, text="<< Prev", command=self.prev_anon_image).pack(side='left', expand=True)
        self.anon_count_lbl = ttk.Label(nav_frame, text="0/0")
        self.anon_count_lbl.pack(side='left', padx=10)
        ttk.Button(nav_frame, text="Save & Next >>", command=self.save_and_next_anon).pack(side='left', expand=True)

    def toggle_anon_zoom(self):
        """Toggle zoom mode on/off"""
        if self.anon_zoom_enabled.get():
            # Enable zoom - bind mousewheel
            self.anon_canvas.bind("<MouseWheel>", self.zoom_anon_image)
            self.anon_canvas.bind("<Button-4>", self.zoom_anon_image)
            self.anon_canvas.bind("<Button-5>", self.zoom_anon_image)
        else:
            # Disable zoom - unbind mousewheel
            self.anon_canvas.unbind("<MouseWheel>")
            self.anon_canvas.unbind("<Button-4>")
            self.anon_canvas.unbind("<Button-5>")

    def zoom_anon_image(self, event):
        """Handle mouse wheel zoom - zooms toward mouse cursor"""
        if not hasattr(self, 'pil_anon_source'):
            return
        
        # Get mouse position relative to canvas
        mouse_x = event.x
        mouse_y = event.y
        
        # Calculate position before zoom
        old_zoom = self.anon_zoom
        
        # Determine zoom direction
        if event.num == 5 or event.delta < 0:  # Zoom out
            self.anon_zoom *= 0.9
        elif event.num == 4 or event.delta > 0:  # Zoom in
            self.anon_zoom *= 1.1
        
        # Clamp zoom between 0.1x and 10x
        self.anon_zoom = max(0.1, min(10.0, self.anon_zoom))
        
        # Calculate zoom factor change
        zoom_factor = self.anon_zoom / old_zoom
        
        # Adjust image position to zoom toward mouse cursor
        self.anon_image_x = mouse_x - (mouse_x - self.anon_image_x) * zoom_factor
        self.anon_image_y = mouse_y - (mouse_y - self.anon_image_y) * zoom_factor
        
        # Redraw at new zoom level
        self.redraw_anon_editor()

    def start_drag_anon(self, event):
        """Start panning with middle mouse button"""
        self.anon_drag_start_x = event.x
        self.anon_drag_start_y = event.y

    def drag_anon(self, event):
        """Pan the image with middle mouse button"""
        if not hasattr(self, 'pil_anon_source'):
            return
        
        # Calculate how much the mouse moved
        dx = event.x - self.anon_drag_start_x
        dy = event.y - self.anon_drag_start_y
        
        # Update image position
        self.anon_image_x += dx
        self.anon_image_y += dy
        
        # Update drag start position for next movement
        self.anon_drag_start_x = event.x
        self.anon_drag_start_y = event.y
        
        # Redraw image at new position
        self.redraw_anon_editor()

    def on_anon_canvas_resize(self, event):
        """Re-fit image when canvas is resized"""
        if hasattr(self, 'pil_anon_source') and self.anon_zoom == 1.0:
            # Only auto-refit if we're not zoomed in
            self.fit_anon_image_to_canvas()

    def fit_anon_image_to_canvas(self):
        """Fit the image to canvas while maintaining aspect ratio"""
        if not hasattr(self, 'pil_anon_source'):
            return
        
        # Get canvas dimensions
        canvas_width = self.anon_canvas.winfo_width()
        canvas_height = self.anon_canvas.winfo_height()
        
        # Skip if canvas not ready
        if canvas_width <= 1 or canvas_height <= 1:
            return
        
        # Get image dimensions
        img_width, img_height = self.pil_anon_source.size
        
        # Calculate scaling to fit canvas while maintaining aspect ratio
        width_ratio = canvas_width / img_width
        height_ratio = canvas_height / img_height
        
        # Use the smaller ratio to ensure image fits entirely
        fit_ratio = min(width_ratio, height_ratio) * 0.95  # 95% to leave some padding
        
        # Set zoom to fit ratio
        self.anon_zoom = fit_ratio
        
        # Center the image
        scaled_width = img_width * fit_ratio
        scaled_height = img_height * fit_ratio
        
        self.anon_image_x = (canvas_width - scaled_width) // 2
        self.anon_image_y = (canvas_height - scaled_height) // 2
        
        # Redraw
        self.redraw_anon_editor()

    def load_anon_folder(self, event=None):
        folder = self.anon_folder_var.get()
        if not folder: return
        src = os.path.join("extracted_frames", folder)
        if not os.path.exists(src): return
        self.anonymized_images = sorted([f for f in os.listdir(src) if f.lower().endswith(('.jpg','.png'))])
        self.current_img_idx = 0
        self.load_current_anon_image_data()

    def run_anonymization_thread(self):
        if not self.anon_folder_var.get(): return
        self.btn_anon.config(state='disabled')
        self.anon_progress.start(10)
        self.anon_status_lbl.config(text="Processing...")
        threading.Thread(target=self.process_anonymization, daemon=True).start()

    def process_anonymization(self):
        folder = self.anon_folder_var.get()
        inp = os.path.join("extracted_frames", folder)
        out = os.path.join("anonymized_frames", folder)
        model = "models/anon_model.pt"
        
        try:
            # Get list of all images
            all_images = [f for f in os.listdir(inp) if f.lower().endswith(('.jpg', '.png'))]
            total_images = len(all_images)
            
            # Define batch size to prevent memory issues
            BATCH_SIZE = 500  # Process 500 images at a time
            
            if total_images > BATCH_SIZE:
                # Need to process in batches
                num_batches = (total_images + BATCH_SIZE - 1) // BATCH_SIZE  # Ceiling division
                
                self.root.after(0, lambda: 
                    self.anon_status_lbl.config(text=f"Large folder detected ({total_images} images). Processing in {num_batches} batches..."))
                
                processed_count = 0
                
                for batch_num in range(num_batches):
                    start_idx = batch_num * BATCH_SIZE
                    end_idx = min(start_idx + BATCH_SIZE, total_images)
                    batch_images = all_images[start_idx:end_idx]
                    
                    # Create temporary batch folder
                    batch_inp = os.path.join(inp, f"_temp_batch_{batch_num}")
                    os.makedirs(batch_inp, exist_ok=True)
                    
                    # Copy batch images to temp folder
                    for img in batch_images:
                        import shutil
                        shutil.copy(os.path.join(inp, img), os.path.join(batch_inp, img))
                    
                    # Progress callback for this batch
                    def update_progress(current, total):
                        overall_current = processed_count + current
                        overall_progress = (overall_current / total_images) * 100
                        self.root.after(0, lambda p=overall_progress: self.anon_progress.configure(value=p))
                        self.root.after(0, lambda: 
                            self.anon_status_lbl.config(text=f"Batch {batch_num+1}/{num_batches}: {current}/{total} (Overall: {overall_current}/{total_images})"))
                    
                    # Process this batch
                    _, err = anonymize_images(batch_inp, out, model, self.anon_conf.get(), 
                                            progress_callback=update_progress)
                    
                    # Clean up temp batch folder
                    import shutil
                    shutil.rmtree(batch_inp)
                    
                    if err:
                        self.root.after(0, lambda e=err: messagebox.showerror("Error", f"Batch {batch_num+1} failed: {e}"))
                        return
                    
                    processed_count += len(batch_images)
                    
                    # Force garbage collection between batches to free memory
                    import gc
                    gc.collect()
                
                self.root.after(0, lambda: self.finish_anonymization_batch())
            else:
                # Small folder, process normally
                def update_progress(current, total):
                    progress = (current / total) * 100
                    self.root.after(0, lambda p=progress: self.anon_progress.configure(value=p))
                    self.root.after(0, lambda c=current, t=total: 
                                self.anon_status_lbl.config(text=f"Processing: {c}/{t}"))
                
                _, err = anonymize_images(inp, out, model, self.anon_conf.get(), 
                                        progress_callback=update_progress)
                if err: 
                    self.root.after(0, lambda: messagebox.showerror("Error", err))
                else: 
                    self.root.after(0, lambda: self.finish_anonymization_batch())
                    
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
        finally:
            self.root.after(0, self.reset_anon_ui)

    def reset_anon_ui(self):
        self.btn_anon.config(state='normal')
        self.anon_progress.stop()
        self.anon_status_lbl.config(text="Ready")

    def finish_anonymization_batch(self):
        self.load_anon_folder()
        self.refresh_folder_lists() 
        messagebox.showinfo("Done", "Auto-detection complete. Please review frames.")

    def load_current_anon_image_data(self):
        if not self.anonymized_images: return
        folder = self.anon_folder_var.get()
        fname = self.anonymized_images[self.current_img_idx]
        
        self.anon_boxes = []
        self.rect_map = {}
        self.selected_rect_id = None
        
        src = os.path.join("extracted_frames", folder, fname)
        self.pil_anon_source = Image.open(src)
        
        txt = os.path.join("anonymized_frames", folder, "annot_txt", os.path.splitext(fname)[0]+".txt")
        if os.path.exists(txt):
            with open(txt) as f:
                for line in f:
                    try: self.anon_boxes.append(list(map(float, line.split()[:4])))
                    except: pass
        
        # Reset zoom when loading new image
        self.anon_zoom = 1.0
        self.fit_anon_image_to_canvas()

    def redraw_anon_editor(self):
        if not hasattr(self, 'pil_anon_source'): 
            return
        
        self.anon_canvas.delete("all")
        
        # Get original image dimensions
        orig_w, orig_h = self.pil_anon_source.size
        
        # Calculate new size based on zoom
        new_w = int(orig_w * self.anon_zoom)
        new_h = int(orig_h * self.anon_zoom)
        
        # Resize image
        resized = self.pil_anon_source.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.tk_anon_img = ImageTk.PhotoImage(resized)
        
        # Draw image at current position
        self.anon_canvas.create_image(
            self.anon_image_x, self.anon_image_y, 
            anchor='nw', 
            image=self.tk_anon_img
        )
        
        # Draw boxes scaled and positioned relative to image
        self.rect_map = {}
        for i, b in enumerate(self.anon_boxes):
            self.draw_box(b, i)
            
        self.anon_count_lbl.config(text=f"{self.current_img_idx+1}/{len(self.anonymized_images)}")

    def draw_box(self, box, idx):
        """Draw a bounding box on the canvas"""
        x1, y1, x2, y2 = box
        
        # Scale box coordinates by zoom level and offset by image position
        sx1 = x1 * self.anon_zoom + self.anon_image_x
        sy1 = y1 * self.anon_zoom + self.anon_image_y
        sx2 = x2 * self.anon_zoom + self.anon_image_x
        sy2 = y2 * self.anon_zoom + self.anon_image_y
        
        # Draw rectangle outline
        rid = self.anon_canvas.create_rectangle(
            sx1, sy1, sx2, sy2, 
            outline="red", width=2, tags="box"
        )
        
        # Draw semi-transparent fill
        self.anon_canvas.create_rectangle(
            sx1, sy1, sx2, sy2, 
            fill="red", stipple="gray25", width=0, tags=f"f_{rid}"
        )
        
        self.rect_map[rid] = idx

    def on_anon_click(self, e):
        """Handle left mouse button click - select box or start drawing"""
        self.anon_canvas.focus_set()
        clicked_items = self.anon_canvas.find_overlapping(e.x-2, e.y-2, e.x+2, e.y+2)
        box_ids = [i for i in clicked_items if "box" in self.anon_canvas.gettags(i) or any(t.startswith("f_") for t in self.anon_canvas.gettags(i))]
        
        if box_ids:
            # Clicked on existing box - select it
            rid = box_ids[-1]
            tags = self.anon_canvas.gettags(rid)
            if tags[0].startswith("f_"): 
                rid = int(tags[0].split('_')[1])
            self.select_rect(rid)
            self.drawing_start = None
        else:
            # Clicked on empty space - start drawing new box
            self.selected_rect_id = None
            self.anon_canvas.delete("sel")
            self.drawing_start = (e.x, e.y)
            self.current_rect_id = self.anon_canvas.create_rectangle(
                e.x, e.y, e.x, e.y, 
                outline="cyan", width=2
            )

    def on_anon_drag(self, e):
        """Handle dragging while drawing new box"""
        if self.drawing_start: 
            self.anon_canvas.coords(
                self.current_rect_id, 
                self.drawing_start[0], self.drawing_start[1], 
                e.x, e.y
            )

    def on_anon_release(self, e):
        """Handle mouse release - finalize new box"""
        if self.drawing_start:
            x1, y1 = self.drawing_start
            x2, y2 = e.x, e.y
            
            # Normalize coordinates
            if x1 > x2: x1, x2 = x2, x1
            if y1 > y2: y1, y2 = y2, y1
            
            # Only create box if it's large enough
            if (x2 - x1) > 5 and (y2 - y1) > 5:
                # Convert canvas coordinates to image coordinates
                img_x1 = (x1 - self.anon_image_x) / self.anon_zoom
                img_y1 = (y1 - self.anon_image_y) / self.anon_zoom
                img_x2 = (x2 - self.anon_image_x) / self.anon_zoom
                img_y2 = (y2 - self.anon_image_y) / self.anon_zoom
                
                # Add to boxes list
                self.anon_boxes.append([img_x1, img_y1, img_x2, img_y2])
                self.redraw_anon_editor()
            else: 
                # Box too small, delete it
                self.anon_canvas.delete(self.current_rect_id)
            
            self.drawing_start = None

    def delete_anon_rect(self, e):
        """Delete selected rectangle (called by Delete key)"""
        if self.notebook.index(self.notebook.select()) == 1 and self.selected_rect_id:
            if self.selected_rect_id in self.rect_map:
                del self.anon_boxes[self.rect_map[self.selected_rect_id]]
                self.selected_rect_id = None
                self.redraw_anon_editor()

    def select_rect(self, rid):
        """Highlight selected rectangle"""
        self.selected_rect_id = rid
        self.anon_canvas.delete("sel")
        c = self.anon_canvas.coords(rid)
        if c:
            self.anon_canvas.create_rectangle(
                c[0]-2, c[1]-2, c[2]+2, c[3]+2, 
                outline="blue", width=3, tags="sel"
            )

    def save_and_next_anon(self):
        """Save current annotations and move to next image"""
        folder = self.anon_folder_var.get()
        fname = self.anonymized_images[self.current_img_idx]
        src = os.path.join("extracted_frames", folder, fname)
        out_dir = os.path.join("anonymized_frames", folder)
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(os.path.join(out_dir, "annot_txt"), exist_ok=True)
        
        # Save annotations
        with open(os.path.join(out_dir, "annot_txt", os.path.splitext(fname)[0]+".txt"), 'w') as f:
            for b in self.anon_boxes: 
                f.write(f"{b[0]} {b[1]} {b[2]} {b[3]}\n")
        
        # Apply blur to image
        img = cv2.imread(src)
        if img is not None:
            h, w = img.shape[:2]
            for b in self.anon_boxes:
                x1, y1, x2, y2 = map(int, b)
                x1, x2 = max(0, x1), min(w, x2)
                y1, y2 = max(0, y1), min(h, y2)
                if x2 > x1 and y2 > y1:
                    img[y1:y2, x1:x2] = cv2.GaussianBlur(img[y1:y2, x1:x2], (51, 51), 30)
            cv2.imwrite(os.path.join(out_dir, fname), img)
        
        # Move to next image
        if self.current_img_idx < len(self.anonymized_images) - 1:
            self.current_img_idx += 1
            self.load_current_anon_image_data()
        else:
            messagebox.showinfo("Done", "Review complete for this folder.")

    def prev_anon_image(self):
        if self.current_img_idx > 0:
            self.current_img_idx -= 1
            self.load_current_anon_image_data()

    def next_anon_image(self):
        if self.current_img_idx < len(self.anonymized_images) - 1:
            self.current_img_idx += 1
            self.load_current_anon_image_data()

    # =========================================================================
    # TAB 3: ANNOTATION
    # =========================================================================
    def setup_annotation_ui(self):
        frame = self.tab_annotation
        top = ttk.Frame(frame, padding=5)
        top.pack(fill='x')
        
        ttk.Label(top, text="Folder:").pack(side='left')
        self.anno_folder_var = tk.StringVar()
        self.anno_combo = ttk.Combobox(top, textvariable=self.anno_folder_var, width=25)
        self.anno_combo.pack(side='left', padx=5)
        self.anno_combo.bind("<<ComboboxSelected>>", self.load_annotation_folder)
        
        ttk.Button(top, text="Pre-Annotate", command=self.run_pre_annotation).pack(side='left', padx=5)
        
        self.anno_stat_f = ttk.Frame(top)
        self.anno_stat_f.pack(side='left', fill='x', expand=True)
        self.anno_prog = ttk.Progressbar(self.anno_stat_f, mode='determinate')
        self.anno_prog.pack(side='left', fill='x', expand=True)
        self.anno_lbl = ttk.Label(self.anno_stat_f, text="Ready")
        self.anno_lbl.pack(side='left', padx=5)
        
        main = ttk.PanedWindow(frame, orient='horizontal')
        main.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Left side - Canvas for annotation
        vf = ttk.LabelFrame(main, text="Annotation", padding=5)
        main.add(vf, weight=3)
        
        # Canvas header with controls
        canvas_header = ttk.Frame(vf)
        canvas_header.pack(fill='x', pady=(0, 5))
        
        self.anno_zoom_enabled = tk.BooleanVar(value=False)
        self.anno_zoom_toggle_btn = ttk.Checkbutton(
            canvas_header, 
            text="Enable Zoom (scroll wheel)", 
            variable=self.anno_zoom_enabled,
            command=self.toggle_anno_zoom
        )
        self.anno_zoom_toggle_btn.pack(side='left')
        
        ttk.Label(canvas_header, text="| Pan: Middle mouse | Draw: Left click & drag | Delete: Select & press Delete").pack(side='left', padx=10)
        
        self.anno_canvas = tk.Canvas(vf, bg="#333", cursor="cross")
        self.anno_canvas.pack(fill='both', expand=True)
        
        # Zoom and pan state
        self.anno_zoom = 1.0
        self.anno_drag_start_x = 0
        self.anno_drag_start_y = 0
        self.anno_image_x = 0
        self.anno_image_y = 0
        self.anno_boxes = []  # Format: [class_id, cx, cy, w, h]
        self.anno_rect_map = {}
        self.anno_selected_rect_id = None
        self.anno_drawing_start = None
        self.anno_current_rect_id = None
        
        # Bind events
        self.anno_canvas.bind("<Button-1>", self.on_anno_click)
        self.anno_canvas.bind("<B1-Motion>", self.on_anno_drag_draw)
        self.anno_canvas.bind("<ButtonRelease-1>", self.on_anno_release)
        
        # Middle mouse for panning
        self.anno_canvas.bind("<ButtonPress-2>", self.start_drag_anno_pan)
        self.anno_canvas.bind("<B2-Motion>", self.drag_anno_pan)
        
        # Canvas resize
        self.anno_canvas.bind("<Configure>", self.on_anno_canvas_resize)
        
        # Right side - Metadata panel
        inf = ttk.Frame(main, padding=10)
        main.add(inf, weight=1)
        
        # Class selector at the top of right panel
        ttk.Label(inf, text="Annotation Class", font=('TkDefaultFont', 10, 'bold')).pack(anchor='w', pady=(0,5))
        
        class_frame = ttk.Frame(inf)
        class_frame.pack(fill='x', pady=(0, 10))
        
        ttk.Label(class_frame, text="Class:").pack(anchor='w')
        self.anno_class_var = tk.StringVar(value="0: traffic_sign")
        
        # Create searchable combobox
        self.anno_class_combo = ttk.Combobox(class_frame, textvariable=self.anno_class_var)
        self.anno_class_combo['values'] = [f"{k}: {v}" for k, v in self.CLASSES.items()]
        self.anno_class_combo.pack(fill='x', pady=(2, 0))
        
        # Bind events for search functionality and class change
        self.anno_class_combo.bind('<KeyRelease>', self.filter_class_dropdown)
        self.anno_class_combo.bind('<<ComboboxSelected>>', self.on_class_selected)
        
        ttk.Label(class_frame, text="(Type to search, or select to change class of selected box)", 
                font=('TkDefaultFont', 8), foreground='gray').pack(anchor='w', pady=(2, 0))
        
        ttk.Separator(inf, orient='horizontal').pack(fill='x', pady=10)
        
        ttk.Label(inf, text="Metadata", font=('TkDefaultFont', 10, 'bold')).pack(anchor='w', pady=(0,5))
        
        grid_f = ttk.Frame(inf)
        grid_f.pack(fill='x')
        
        ttk.Label(grid_f, text="Location:").grid(row=0, column=0, sticky='w')
        self.var_anno_loc = tk.StringVar()
        e_loc = ttk.Entry(grid_f, textvariable=self.var_anno_loc, width=20)
        e_loc.grid(row=0, column=1, sticky='ew', pady=2)
        e_loc.bind("<FocusOut>", lambda e: self.save_metadata())
        
        ttk.Label(grid_f, text="Time:").grid(row=1, column=0, sticky='w')
        self.var_anno_time = tk.StringVar()
        cb_time = ttk.Combobox(grid_f, textvariable=self.var_anno_time, values=["Day","Night","Dawn/Dusk"], width=18)
        cb_time.grid(row=1, column=1, sticky='ew', pady=2)
        cb_time.bind("<<ComboboxSelected>>", lambda e: self.save_metadata())
        
        ttk.Label(grid_f, text="Weather:").grid(row=2, column=0, sticky='w')
        self.var_anno_weather = tk.StringVar()
        cb_weather = ttk.Combobox(grid_f, textvariable=self.var_anno_weather, values=["Clear", "Rainy", "Foggy", "Cloudy", "Other"], width=18)
        cb_weather.grid(row=2, column=1, sticky='ew', pady=2)
        cb_weather.bind("<<ComboboxSelected>>", lambda e: self.save_metadata())
        
        ttk.Label(grid_f, text="Light:").grid(row=3, column=0, sticky='w')
        self.var_anno_light = tk.StringVar()
        cb_light = ttk.Combobox(grid_f, textvariable=self.var_anno_light, values=["Normal", "Low-light (Tunnel)", "Low-light (Underpass)", "Low-light (Other)"], width=18)
        cb_light.grid(row=3, column=1, sticky='ew', pady=2)
        cb_light.bind("<<ComboboxSelected>>", lambda e: self.save_metadata())

        ttk.Label(grid_f, text="Road:").grid(row=4, column=0, sticky='w')
        self.var_anno_road = tk.StringVar()
        cb_road = ttk.Combobox(grid_f, textvariable=self.var_anno_road, 
                            values=["Highway", "Urban", "Rural", "Residential", "Other"], 
                            width=18)
        cb_road.grid(row=4, column=1, sticky='ew', pady=2)
        cb_road.bind("<<ComboboxSelected>>", lambda e: self.save_metadata())
        
        self.var_diff = tk.BooleanVar()
        self.chk_diff = ttk.Checkbutton(inf, text="Difficult", variable=self.var_diff, command=self.save_metadata)
        self.chk_diff.pack(anchor='w', pady=(10, 2))
        
        self.var_aug = tk.BooleanVar()
        self.chk_aug = ttk.Checkbutton(inf, text="Augmented", variable=self.var_aug, command=self.save_metadata)
        self.chk_aug.pack(anchor='w', pady=(0, 10))
        
        # Navigation
        nav = ttk.Frame(inf)
        nav.pack(fill='x', pady=5)
        ttk.Button(nav, text="<< Prev", command=self.prev_anno_image).pack(side='left', expand=True)
        ttk.Button(nav, text="Save & Next >>", command=self.save_and_next_anno).pack(side='left', expand=True)
        self.lbl_anno_idx = ttk.Label(inf, text="0/0")
        self.lbl_anno_idx.pack(pady=5)
        
        ttk.Separator(inf, orient='horizontal').pack(fill='x', pady=20)
        ttk.Label(inf, text="Your Name:").pack(anchor='w')
        self.entry_name = ttk.Entry(inf)
        self.entry_name.pack(fill='x', pady=5)
        ttk.Button(inf, text="Generate ZIP", command=self.generate_submission).pack(fill='x', pady=10)

    def filter_class_dropdown(self, event):
        """Filter class dropdown based on search text"""
        # Get current text
        value = self.anno_class_combo.get()
        
        # If empty, show all classes
        if value == '':
            self.anno_class_combo['values'] = [f"{k}: {v}" for k, v in self.CLASSES.items()]
        else:
            # Filter classes that match the search
            filtered = [f"{k}: {v}" for k, v in self.CLASSES.items() 
                    if value.lower() in f"{k}: {v}".lower()]
            self.anno_class_combo['values'] = filtered

    def on_class_selected(self, event):
        """Handle class selection - if a box is selected, change its class"""
        if self.anno_selected_rect_id and self.anno_selected_rect_id in self.anno_rect_map:
            # Get the new class
            class_str = self.anno_class_var.get()
            if ":" in class_str:
                class_id = int(class_str.split(":")[0])
                
                # Update the class of the selected box
                box_idx = self.anno_rect_map[self.anno_selected_rect_id]
                self.anno_boxes[box_idx][0] = class_id
                
                # Redraw to show updated class
                self.redraw_anno_canvas()
                
                # Re-select the same box
                self.select_anno_rect(self.anno_selected_rect_id)

    def toggle_anno_zoom(self):
        """Toggle zoom mode on/off"""
        if self.anno_zoom_enabled.get():
            self.anno_canvas.bind("<MouseWheel>", self.zoom_anno_image)
            self.anno_canvas.bind("<Button-4>", self.zoom_anno_image)
            self.anno_canvas.bind("<Button-5>", self.zoom_anno_image)
        else:
            self.anno_canvas.unbind("<MouseWheel>")
            self.anno_canvas.unbind("<Button-4>")
            self.anno_canvas.unbind("<Button-5>")

    def zoom_anno_image(self, event):
        """Handle mouse wheel zoom - zooms toward mouse cursor"""
        if not hasattr(self, 'pil_anno_source'):
            return
        
        mouse_x = event.x
        mouse_y = event.y
        old_zoom = self.anno_zoom
        
        if event.num == 5 or event.delta < 0:  # Zoom out
            self.anno_zoom *= 0.9
        elif event.num == 4 or event.delta > 0:  # Zoom in
            self.anno_zoom *= 1.1
        
        self.anno_zoom = max(0.1, min(10.0, self.anno_zoom))
        zoom_factor = self.anno_zoom / old_zoom
        
        self.anno_image_x = mouse_x - (mouse_x - self.anno_image_x) * zoom_factor
        self.anno_image_y = mouse_y - (mouse_y - self.anno_image_y) * zoom_factor
        
        self.redraw_anno_canvas()

    def start_drag_anno_pan(self, event):
        """Start panning with middle mouse button"""
        self.anno_drag_start_x = event.x
        self.anno_drag_start_y = event.y

    def drag_anno_pan(self, event):
        """Pan the image with middle mouse button"""
        if not hasattr(self, 'pil_anno_source'):
            return
        
        dx = event.x - self.anno_drag_start_x
        dy = event.y - self.anno_drag_start_y
        
        self.anno_image_x += dx
        self.anno_image_y += dy
        
        self.anno_drag_start_x = event.x
        self.anno_drag_start_y = event.y
        
        self.redraw_anno_canvas()

    def on_anno_canvas_resize(self, event):
        """Re-fit image when canvas is resized"""
        if hasattr(self, 'pil_anno_source') and self.anno_zoom == 1.0:
            self.fit_anno_image_to_canvas()

    def fit_anno_image_to_canvas(self):
        """Fit the image to canvas while maintaining aspect ratio"""
        if not hasattr(self, 'pil_anno_source'):
            return
        
        canvas_width = self.anno_canvas.winfo_width()
        canvas_height = self.anno_canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            return
        
        img_width, img_height = self.pil_anno_source.size
        width_ratio = canvas_width / img_width
        height_ratio = canvas_height / img_height
        fit_ratio = min(width_ratio, height_ratio) * 0.95
        
        self.anno_zoom = fit_ratio
        scaled_width = img_width * fit_ratio
        scaled_height = img_height * fit_ratio
        
        self.anno_image_x = (canvas_width - scaled_width) // 2
        self.anno_image_y = (canvas_height - scaled_height) // 2
        
        self.redraw_anno_canvas()

    def load_annotation_folder(self, e=None):
        folder = self.anno_folder_var.get()
        p = os.path.join("anonymized_frames", folder)
        if not os.path.exists(p): return
        self.annotation_images = sorted([f for f in os.listdir(p) if f.lower().endswith(('.jpg','.png'))])
        self.current_review_idx = 0
        self.show_anno_image()

    def run_pre_annotation(self):
        if not self.anno_folder_var.get(): return
        self.anno_prog['value'] = 0
        self.anno_lbl.config(text="Starting...")
        threading.Thread(target=self.process_detection, daemon=True).start()

    def process_detection(self):
        folder = self.anno_folder_var.get()
        inp = os.path.join("anonymized_frames", folder)
        out = os.path.join("annotations", folder)
        os.makedirs(out, exist_ok=True)
        try:
            model = YOLO("models/own_detect_model.pt")
            imgs = [f for f in os.listdir(inp) if f.endswith(('.jpg','.png'))]
            for i, f in enumerate(imgs):
                self.root.after(0, lambda v=(i+1)/len(imgs)*100: self.anno_prog.configure(value=v))
                res = model(os.path.join(inp, f), verbose=False, conf=0.25)
                lines = []
                for r in res:
                    for b in r.boxes:
                        lines.append(f"{int(b.cls[0])} {' '.join(map(str, b.xywhn[0].tolist()))}\n")
                if lines:
                    with open(os.path.join(out, os.path.splitext(f)[0]+".txt"), 'w') as tf: 
                        tf.writelines(lines)
            self.root.after(0, lambda: messagebox.showinfo("Done", "Pre-annotation complete!"))
            self.root.after(0, self.show_anno_image)
        except Exception as e: 
            print(e)
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
        finally: 
            self.root.after(0, lambda: self.anno_lbl.config(text="Ready"))

    def show_anno_image(self):
        if not self.annotation_images: 
            return
        
        f = self.anno_folder_var.get()
        name = self.annotation_images[self.current_review_idx]
        img_p = os.path.join("anonymized_frames", f, name)
        txt_p = os.path.join("annotations", f, os.path.splitext(name)[0]+".txt")
        meta_p = os.path.join("metadata", f, os.path.splitext(name)[0]+".json")
        
        # Load image
        self.pil_anno_source = Image.open(img_p)
        img_width, img_height = self.pil_anno_source.size
        
        # Load annotations (YOLO format: class cx cy w h - normalized 0-1)
        self.anno_boxes = []
        if os.path.exists(txt_p):
            with open(txt_p) as file:
                for line in file:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        cx = float(parts[1]) * img_width
                        cy = float(parts[2]) * img_height
                        w = float(parts[3]) * img_width
                        h = float(parts[4]) * img_height
                        self.anno_boxes.append([class_id, cx, cy, w, h])
        
        # Load metadata
        if os.path.exists(meta_p):
            with open(meta_p) as file:
                d = json.load(file)
                self.var_anno_loc.set(d.get('location',''))
                self.var_anno_time.set(d.get('time_of_day','Day'))
                self.var_anno_weather.set(d.get('weather','Clear'))
                self.var_anno_light.set(d.get('lighting','Normal'))
                self.var_anno_road.set(d.get('road_type', 'Highway'))
                self.var_diff.set(d.get('difficult', False))
                self.var_aug.set(d.get('augmented', False))
        
        # Reset zoom and fit to canvas
        self.anno_zoom = 1.0
        self.anno_selected_rect_id = None
        self.fit_anno_image_to_canvas()
        
        self.lbl_anno_idx.config(text=f"{self.current_review_idx+1}/{len(self.annotation_images)}")

    def redraw_anno_canvas(self):
        """Redraw the annotation canvas with image and boxes"""
        if not hasattr(self, 'pil_anno_source'):
            return
        
        self.anno_canvas.delete("all")
        
        # Get original image dimensions
        orig_w, orig_h = self.pil_anno_source.size
        
        # Calculate new size based on zoom
        new_w = int(orig_w * self.anno_zoom)
        new_h = int(orig_h * self.anno_zoom)
        
        # Resize image
        resized = self.pil_anno_source.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.tk_anno_img = ImageTk.PhotoImage(resized)
        
        # Draw image at current position
        self.anno_canvas.create_image(
            self.anno_image_x, self.anno_image_y, 
            anchor='nw', 
            image=self.tk_anno_img
        )
        
        # Draw boxes
        self.anno_rect_map = {}
        for i, box in enumerate(self.anno_boxes):
            self.draw_anno_box(box, i)

    def draw_anno_box(self, box, idx):
        """Draw a bounding box on the canvas"""
        class_id, cx, cy, w, h = box
        
        # Convert center format to corner format
        x1 = cx - w/2
        y1 = cy - h/2
        x2 = cx + w/2
        y2 = cy + h/2
        
        # Scale by zoom and offset by image position
        sx1 = x1 * self.anno_zoom + self.anno_image_x
        sy1 = y1 * self.anno_zoom + self.anno_image_y
        sx2 = x2 * self.anno_zoom + self.anno_image_x
        sy2 = y2 * self.anno_zoom + self.anno_image_y
        
        # Draw rectangle
        color = "lime"
        rid = self.anno_canvas.create_rectangle(
            sx1, sy1, sx2, sy2,
            outline=color, width=2, tags="box"
        )
        
        # Draw class label
        class_name = self.CLASSES.get(class_id, str(class_id))
        self.anno_canvas.create_text(
            sx1, sy1 - 5,
            text=class_name,
            anchor='sw',
            fill=color,
            font=('TkDefaultFont', 10, 'bold'),
            tags=f"label_{rid}"
        )
        
        self.anno_rect_map[rid] = idx

    def on_anno_click(self, event):
        """Handle left mouse button click - select box or start drawing"""
        self.anno_canvas.focus_set()
        clicked_items = self.anno_canvas.find_overlapping(event.x-2, event.y-2, event.x+2, event.y+2)
        
        # Check if we clicked on a box or its label
        box_ids = []
        for item in clicked_items:
            tags = self.anno_canvas.gettags(item)
            if "box" in tags:
                box_ids.append(item)
            elif any(t.startswith("label_") for t in tags):
                # Clicked on label, get the associated box
                label_tag = [t for t in tags if t.startswith("label_")][0]
                box_id = int(label_tag.split("_")[1])
                box_ids.append(box_id)
        
        if box_ids:
            # Clicked on existing box - select it
            rid = box_ids[-1]
            self.select_anno_rect(rid)
            self.anno_drawing_start = None
        else:
            # Clicked on empty space - start drawing new box
            self.anno_selected_rect_id = None
            self.anno_canvas.delete("sel")
            self.anno_drawing_start = (event.x, event.y)
            self.anno_current_rect_id = self.anno_canvas.create_rectangle(
                event.x, event.y, event.x, event.y,
                outline="cyan", width=2
            )

    def on_anno_drag_draw(self, event):
        """Handle dragging while drawing new box"""
        if self.anno_drawing_start:
            self.anno_canvas.coords(
                self.anno_current_rect_id,
                self.anno_drawing_start[0], self.anno_drawing_start[1],
                event.x, event.y
            )

    def on_anno_release(self, event):
        """Handle mouse release - finalize new box"""
        if self.anno_drawing_start:
            x1, y1 = self.anno_drawing_start
            x2, y2 = event.x, event.y
            
            # Normalize coordinates
            if x1 > x2: x1, x2 = x2, x1
            if y1 > y2: y1, y2 = y2, y1
            
            # Only create box if it's large enough
            if (x2 - x1) > 10 and (y2 - y1) > 10:
                # Convert canvas coordinates to image coordinates
                img_x1 = (x1 - self.anno_image_x) / self.anno_zoom
                img_y1 = (y1 - self.anno_image_y) / self.anno_zoom
                img_x2 = (x2 - self.anno_image_x) / self.anno_zoom
                img_y2 = (y2 - self.anno_image_y) / self.anno_zoom
                
                # Convert to center format (cx, cy, w, h)
                cx = (img_x1 + img_x2) / 2
                cy = (img_y1 + img_y2) / 2
                w = img_x2 - img_x1
                h = img_y2 - img_y1
                
                # Get selected class
                class_str = self.anno_class_var.get()
                class_id = int(class_str.split(":")[0]) if ":" in class_str else 0
                
                # Add to boxes list
                self.anno_boxes.append([class_id, cx, cy, w, h])
                self.redraw_anno_canvas()
            else:
                # Box too small, delete it
                self.anno_canvas.delete(self.anno_current_rect_id)
            
            self.anno_drawing_start = None

    def select_anno_rect(self, rid):
        """Highlight selected rectangle"""
        self.anno_selected_rect_id = rid
        self.anno_canvas.delete("sel")
        coords = self.anno_canvas.coords(rid)
        if coords:
            self.anno_canvas.create_rectangle(
                coords[0]-2, coords[1]-2, coords[2]+2, coords[3]+2,
                outline="blue", width=3, tags="sel"
            )

    def delete_anno_rect(self, e):
        """Delete selected rectangle (called by Delete key)"""
        if self.notebook.index(self.notebook.select()) == 2 and self.anno_selected_rect_id:
            if self.anno_selected_rect_id in self.anno_rect_map:
                del self.anno_boxes[self.anno_rect_map[self.anno_selected_rect_id]]
                self.anno_selected_rect_id = None
                self.redraw_anno_canvas()

    def save_and_next_anno(self):
        """Save current annotations and move to next image"""
        f = self.anno_folder_var.get()
        if not f or not self.annotation_images:
            return
        
        name = self.annotation_images[self.current_review_idx]
        out_dir = os.path.join("annotations", f)
        os.makedirs(out_dir, exist_ok=True)
        
        # Get image dimensions for normalization
        img_width, img_height = self.pil_anno_source.size
        
        # Save annotations in YOLO format (class cx cy w h - normalized 0-1)
        txt_path = os.path.join(out_dir, os.path.splitext(name)[0]+".txt")
        with open(txt_path, 'w') as f:
            for box in self.anno_boxes:
                class_id, cx, cy, w, h = box
                # Normalize to 0-1
                norm_cx = cx / img_width
                norm_cy = cy / img_height
                norm_w = w / img_width
                norm_h = h / img_height
                f.write(f"{class_id} {norm_cx} {norm_cy} {norm_w} {norm_h}\n")
        
        # Save metadata
        self.save_metadata()
        
        # Move to next image
        if self.current_review_idx < len(self.annotation_images) - 1:
            self.current_review_idx += 1
            self.show_anno_image()
        else:
            messagebox.showinfo("Done", "All images annotated!")

    def next_anno_image(self):
        # Save current before moving
        self.save_and_next_anno()

    def prev_anno_image(self):
        # Save current annotations first
        if self.annotation_images:
            f = self.anno_folder_var.get()
            name = self.annotation_images[self.current_review_idx]
            out_dir = os.path.join("annotations", f)
            os.makedirs(out_dir, exist_ok=True)
            
            img_width, img_height = self.pil_anno_source.size
            txt_path = os.path.join(out_dir, os.path.splitext(name)[0]+".txt")
            with open(txt_path, 'w') as file:
                for box in self.anno_boxes:
                    class_id, cx, cy, w, h = box
                    norm_cx = cx / img_width
                    norm_cy = cy / img_height
                    norm_w = w / img_width
                    norm_h = h / img_height
                    file.write(f"{class_id} {norm_cx} {norm_cy} {norm_w} {norm_h}\n")
            
            self.save_metadata()
        
        # Move to previous
        if self.current_review_idx > 0:
            self.current_review_idx -= 1
            self.show_anno_image()

    def save_metadata(self):
        f = self.anno_folder_var.get()
        if not f or not self.annotation_images: 
            return
        
        name = self.annotation_images[self.current_review_idx]
        meta_dir = os.path.join("metadata", f)
        os.makedirs(meta_dir, exist_ok=True)
        p = os.path.join(meta_dir, os.path.splitext(name)[0]+".json")
        
        # Read existing to preserve 'dataset_name'
        if os.path.exists(p):
            with open(p, 'r') as file: 
                d = json.load(file)
        else:
            d = {}
        
        d['location'] = self.var_anno_loc.get()
        d['time_of_day'] = self.var_anno_time.get()
        d['weather'] = self.var_anno_weather.get()
        d['lighting'] = self.var_anno_light.get()
        d['road_type'] = self.var_anno_road.get()
        d['difficult'] = self.var_diff.get()
        d['augmented'] = self.var_aug.get()
        
        with open(p, 'w') as file: 
            json.dump(d, file, indent=4)

    def generate_submission(self):
        name, f = self.entry_name.get().strip(), self.anno_folder_var.get()
        if not name or not f: 
            return
        
        path = filedialog.asksaveasfilename(
            defaultextension=".zip", 
            initialfile=f"{name.replace(' ','_')}_{f}.zip"
        )
        
        if path:
            with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
                # Add images
                for r, _, files in os.walk(os.path.join("anonymized_frames", f)):
                    for file in files: 
                        z.write(os.path.join(r, file), f"images/{file}")
                
                # Add annotations
                for r, _, files in os.walk(os.path.join("annotations", f)):
                    for file in files: 
                        z.write(os.path.join(r, file), f"labels/{file}")
                
                # Add metadata
                for r, _, files in os.walk(os.path.join("metadata", f)):
                    for file in files: 
                        z.write(os.path.join(r, file), f"metadata/{file}")
            
            messagebox.showinfo("Done", f"Submission saved to {path}")

    def refresh_folder_lists(self):
        if os.path.exists("extracted_frames"):
            self.anon_combo['values'] = [d for d in os.listdir("extracted_frames") 
                                        if os.path.isdir(os.path.join("extracted_frames", d))]
        if os.path.exists("anonymized_frames"):
            self.anno_combo['values'] = [d for d in os.listdir("anonymized_frames") 
                                        if os.path.isdir(os.path.join("anonymized_frames", d))]

if __name__ == "__main__":
    for d in ["extracted_frames", "anonymized_frames", "annotations", "metadata"]: os.makedirs(d, exist_ok=True)
    root = tk.Tk()
    app = TanawApp(root)
    app.refresh_folder_lists()
    root.mainloop()