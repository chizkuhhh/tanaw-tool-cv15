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

        self.tab_extraction = ttk.Frame(self.notebook)
        self.tab_anonymization = ttk.Frame(self.notebook)
        self.tab_annotation = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_extraction, text="1. Extraction")
        self.notebook.add(self.tab_anonymization, text="2. Anonymization (Edit)")
        self.notebook.add(self.tab_annotation, text="3. Annotation")

        # --- Setup UI ---
        self.setup_extraction_ui()
        self.setup_anonymization_ui()
        self.setup_annotation_ui()

        # --- Bind Keys ---
        self.root.bind('<Left>', self.handle_keypress)
        self.root.bind('<Right>', self.handle_keypress)
        self.root.bind('<Delete>', self.delete_anon_rect)
        self.root.bind('<BackSpace>', self.delete_anon_rect)

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

    # =========================================================================
    # TAB 1: EXTRACTION
    # =========================================================================
    def setup_extraction_ui(self):
        frame = self.tab_extraction
        
        top_frame = ttk.Frame(frame)
        top_frame.pack(side='top', fill='x')

        config_frame = ttk.LabelFrame(top_frame, text="Configuration", padding=10)
        config_frame.pack(fill='x', padx=10, pady=5)

        ttk.Label(config_frame, text="Method:").grid(row=0, column=0, sticky='w')
        self.extract_method = tk.StringVar(value="Histogram")
        ttk.Radiobutton(config_frame, text="Histogram", variable=self.extract_method, value="Histogram").grid(row=0, column=1, sticky='w')
        ttk.Radiobutton(config_frame, text="GPS", variable=self.extract_method, value="GPS").grid(row=0, column=2, sticky='w')

        ttk.Label(config_frame, text="Input:").grid(row=1, column=0, sticky='w', pady=5)
        self.video_input_path = tk.StringVar()
        ttk.Entry(config_frame, textvariable=self.video_input_path, width=40).grid(row=1, column=1, columnspan=2)
        ttk.Button(config_frame, text="File", command=self.browse_video_file).grid(row=1, column=3, padx=2)
        ttk.Button(config_frame, text="Folder", command=self.browse_video_folder).grid(row=1, column=4, padx=2)

        ttk.Label(config_frame, text="GPX:").grid(row=2, column=0, sticky='w', pady=5)
        self.gpx_input_path = tk.StringVar()
        ttk.Entry(config_frame, textvariable=self.gpx_input_path, width=40).grid(row=2, column=1, columnspan=2)
        ttk.Button(config_frame, text="Browse", command=self.browse_gpx).grid(row=2, column=3, padx=2)

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

        ttk.Label(meta_frame, text="Light:").grid(row=0, column=6)
        self.meta_lighting = tk.StringVar(value="Normal")
        ttk.Combobox(meta_frame, textvariable=self.meta_lighting, values=["Normal", "Low-light (Tunnel)", "Low-light (Underpass)", "Low-light (Other)"], width=8).grid(row=0, column=7)

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
        
        self.extract_progress = ttk.Progressbar(action_frame, mode='indeterminate')
        self.extract_progress.pack(fill='x', pady=5)
        self.extract_status = ttk.Label(action_frame, text="Ready")
        self.extract_status.pack()

        self.extract_viewer_frame = ttk.LabelFrame(frame, text="Extraction Results", padding=5)
        self.extract_viewer_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.extract_image_label = ttk.Label(self.extract_viewer_frame, anchor="center")
        self.extract_image_label.pack(fill='both', expand=True)
        
        nav_frame = ttk.Frame(frame)
        nav_frame.pack(fill='x', padx=10, pady=5)
        ttk.Button(nav_frame, text="<< Prev", command=self.prev_extract_image).pack(side='left', expand=True)
        self.extract_count_lbl = ttk.Label(nav_frame, text="0/0")
        self.extract_count_lbl.pack(side='left', padx=10)
        ttk.Button(nav_frame, text="Next >>", command=self.next_extract_image).pack(side='left', expand=True)

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
            
            for v_path in video_files:
                v_name = v_path.name
                # Use folder name as dataset_name
                dataset_name = os.path.splitext(v_name)[0]
                
                out_dir = os.path.join("extracted_frames", dataset_name)
                os.makedirs(out_dir, exist_ok=True)
                frames = []
                
                if self.extract_method.get() == "Histogram":
                    frames, _ = extract_histogram_based(
                        str(v_path), out_dir, self.target_dist.get(), self.speed_kph.get(), self.hist_thresh.get()
                    )
                else:
                    gpx = self.gpx_input_path.get()
                    if os.path.isdir(gpx):
                        cand = os.path.join(gpx, os.path.splitext(v_name)[0]+".gpx")
                        if os.path.exists(cand): gpx = cand
                    if os.path.exists(gpx):
                         frames, _ = extract_gps_based(str(v_path), gpx, out_dir, self.target_dist.get())

                if frames:
                    self.extracted_frames.extend(frames)
                    meta_dir = os.path.join("metadata", dataset_name)
                    os.makedirs(meta_dir, exist_ok=True)
                    
                    for fp in frames:
                        fn = os.path.splitext(os.path.basename(fp))[0]
                        # UPDATED METADATA FORMAT
                        meta_data = {
                            "dataset_name": dataset_name,
                            "location": self.meta_location.get(),
                            "time_of_day": self.meta_time.get(),
                            "weather": self.meta_weather.get(),
                            "lighting": self.meta_lighting.get(),
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

    def show_extract_image(self):
        if not self.extracted_frames: return
        path = self.extracted_frames[self.current_extract_idx]
        img = Image.open(path)
        img.thumbnail((1000, 600))
        tk_img = ImageTk.PhotoImage(img)
        self.extract_image_label.config(image=tk_img)
        self.extract_image_label.image = tk_img
        self.extract_count_lbl.config(text=f"{self.current_extract_idx+1}/{len(self.extracted_frames)}")

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
        self.anon_progress = ttk.Progressbar(self.anon_status_frame, mode='indeterminate')
        self.anon_progress.pack(side='left', fill='x', expand=True)
        self.anon_status_lbl = ttk.Label(self.anon_status_frame, text="Ready")
        self.anon_status_lbl.pack(side='left', padx=5)

        self.anon_viewer_frame = ttk.Frame(frame)
        self.anon_viewer_frame.pack(fill='both', expand=True, padx=10, pady=5)
        self.anon_canvas = tk.Canvas(self.anon_viewer_frame, bg="#333", cursor="cross")
        self.anon_canvas.pack(fill='both', expand=True)

        self.anon_canvas.bind("<Button-1>", self.on_anon_click)
        self.anon_canvas.bind("<B1-Motion>", self.on_anon_drag)
        self.anon_canvas.bind("<ButtonRelease-1>", self.on_anon_release)
        
        nav_frame = ttk.Frame(frame, padding=5)
        nav_frame.pack(fill='x')
        ttk.Button(nav_frame, text="<< Prev", command=self.prev_anon_image).pack(side='left', expand=True)
        self.anon_count_lbl = ttk.Label(nav_frame, text="0/0")
        self.anon_count_lbl.pack(side='left', padx=10)
        ttk.Button(nav_frame, text="Save & Next >>", command=self.save_and_next_anon).pack(side='left', expand=True)

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
            _, err = anonymize_images(inp, out, model, self.anon_conf.get())
            if err: self.root.after(0, lambda: messagebox.showerror("Error", err))
            else: self.root.after(0, lambda: self.finish_anonymization_batch())
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
        
        self.redraw_anon_editor()

    def redraw_anon_editor(self):
        if not hasattr(self, 'pil_anon_source'): return
        
        self.anon_canvas.delete("all")
        w, h = self.pil_anon_source.size
        cw = self.anon_canvas.winfo_width() or 800
        ch = self.anon_canvas.winfo_height() or 600
        
        self.anon_scale = min(cw/w, ch/h, 1.0)
        nw, nh = int(w*self.anon_scale), int(h*self.anon_scale)
        self.anon_offset_x = (cw - nw)//2
        self.anon_offset_y = (ch - nh)//2
        
        self.tk_anon_img = ImageTk.PhotoImage(self.pil_anon_source.resize((nw,nh), Image.Resampling.LANCZOS))
        self.anon_canvas.create_image(self.anon_offset_x, self.anon_offset_y, anchor='nw', image=self.tk_anon_img)
        
        self.rect_map = {}
        for i, b in enumerate(self.anon_boxes):
            self.draw_box(b, i)
            
        self.anon_count_lbl.config(text=f"{self.current_img_idx+1}/{len(self.anonymized_images)}")

    def draw_box(self, box, idx):
        x1, y1, x2, y2 = box
        sx, sy, offx, offy = self.anon_scale, self.anon_scale, self.anon_offset_x, self.anon_offset_y
        rid = self.anon_canvas.create_rectangle(
            x1*sx+offx, y1*sy+offy, x2*sx+offx, y2*sy+offy, 
            outline="red", width=2, tags="box"
        )
        self.anon_canvas.create_rectangle(
            x1*sx+offx, y1*sy+offy, x2*sx+offx, y2*sy+offy, 
            fill="red", stipple="gray25", width=0, tags=f"f_{rid}"
        )
        self.rect_map[rid] = idx

    def on_anon_click(self, e):
        self.anon_canvas.focus_set()
        clicked_items = self.anon_canvas.find_overlapping(e.x-2, e.y-2, e.x+2, e.y+2)
        box_ids = [i for i in clicked_items if "box" in self.anon_canvas.gettags(i) or any(t.startswith("f_") for t in self.anon_canvas.gettags(i))]
        
        if box_ids:
            rid = box_ids[-1]
            tags = self.anon_canvas.gettags(rid)
            if tags[0].startswith("f_"): rid = int(tags[0].split('_')[1])
            self.select_rect(rid)
        else:
            self.selected_rect_id = None
            self.anon_canvas.delete("sel")
            self.drawing_start = (e.x, e.y)
            self.current_rect_id = self.anon_canvas.create_rectangle(e.x, e.y, e.x, e.y, outline="cyan", width=2)

    def on_anon_drag(self, e):
        if self.drawing_start: 
            self.anon_canvas.coords(self.current_rect_id, self.drawing_start[0], self.drawing_start[1], e.x, e.y)

    def on_anon_release(self, e):
        if self.drawing_start:
            x1, y1 = self.drawing_start
            x2, y2 = e.x, e.y
            if x1>x2: x1,x2=x2,x1
            if y1>y2: y1,y2=y2,y1
            
            if (x2-x1)>5 and (y2-y1)>5:
                sx, offx, offy = self.anon_scale, self.anon_offset_x, self.anon_offset_y
                self.anon_boxes.append([(x1-offx)/sx, (y1-offy)/sx, (x2-offx)/sx, (y2-offy)/sx])
                self.redraw_anon_editor()
            else: 
                self.anon_canvas.delete(self.current_rect_id)
            self.drawing_start = None

    def delete_anon_rect(self, e):
        if self.notebook.index(self.notebook.select())==1 and self.selected_rect_id:
            if self.selected_rect_id in self.rect_map:
                del self.anon_boxes[self.rect_map[self.selected_rect_id]]
                self.selected_rect_id = None
                self.redraw_anon_editor()

    def select_rect(self, rid):
        self.selected_rect_id = rid
        self.anon_canvas.delete("sel")
        c = self.anon_canvas.coords(rid)
        if c:
            self.anon_canvas.create_rectangle(c[0]-2, c[1]-2, c[2]+2, c[3]+2, outline="blue", width=3, tags="sel")

    def save_and_next_anon(self):
        folder = self.anon_folder_var.get()
        fname = self.anonymized_images[self.current_img_idx]
        src = os.path.join("extracted_frames", folder, fname)
        out_dir = os.path.join("anonymized_frames", folder)
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(os.path.join(out_dir, "annot_txt"), exist_ok=True)
        
        with open(os.path.join(out_dir, "annot_txt", os.path.splitext(fname)[0]+".txt"), 'w') as f:
            for b in self.anon_boxes: f.write(f"{b[0]} {b[1]} {b[2]} {b[3]}\n")
            
        img = cv2.imread(src)
        if img is not None:
            h, w = img.shape[:2]
            for b in self.anon_boxes:
                x1, y1, x2, y2 = map(int, b)
                x1,x2 = max(0,x1),min(w,x2)
                y1,y2 = max(0,y1),min(h,y2)
                if x2>x1 and y2>y1:
                    img[y1:y2, x1:x2] = cv2.GaussianBlur(img[y1:y2, x1:x2], (51,51), 30)
            cv2.imwrite(os.path.join(out_dir, fname), img)
            
        if self.current_img_idx < len(self.anonymized_images)-1:
            self.current_img_idx += 1
            self.load_current_anon_image_data()
        else:
            messagebox.showinfo("Done", "Review complete for this folder.")

    def prev_anon_image(self):
        if self.current_img_idx > 0:
            self.current_img_idx -= 1
            self.load_current_anon_image_data()
    
    def next_anon_image(self):
        if self.current_img_idx < len(self.anonymized_images)-1:
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
        
        ttk.Button(top, text="1. Pre-Annotate", command=self.run_pre_annotation).pack(side='left', padx=5)
        
        self.anno_stat_f = ttk.Frame(top)
        self.anno_stat_f.pack(side='left', fill='x', expand=True)
        self.anno_prog = ttk.Progressbar(self.anno_stat_f, mode='determinate')
        self.anno_prog.pack(side='left', fill='x', expand=True)
        self.anno_lbl = ttk.Label(self.anno_stat_f, text="Ready")
        self.anno_lbl.pack(side='left', padx=5)

        ttk.Button(top, text="2. LabelImg", command=self.launch_labelimg).pack(side='left', padx=5)
        
        main = ttk.PanedWindow(frame, orient='horizontal')
        main.pack(fill='both', expand=True, padx=5, pady=5)
        
        vf = ttk.LabelFrame(main, text="Review", padding=5)
        main.add(vf, weight=3)
        self.anno_canvas = tk.Label(vf)
        self.anno_canvas.pack(fill='both', expand=True)
        
        inf = ttk.Frame(main, padding=10)
        main.add(inf, weight=1)
        
        # --- Metadata Editing Area ---
        ttk.Label(inf, text="Metadata (Editable)", style="Header.TLabel").pack(anchor='w', pady=(0,5))
        
        # Grid layout for neatness
        grid_f = ttk.Frame(inf)
        grid_f.pack(fill='x')
        
        ttk.Label(grid_f, text="Location:").grid(row=0, column=0, sticky='w')
        self.var_anno_loc = tk.StringVar()
        # Bind FocusOut to save
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
        
        # --- Checkboxes ---
        self.var_diff = tk.BooleanVar()
        self.chk_diff = ttk.Checkbutton(inf, text="Difficult", variable=self.var_diff, command=self.save_metadata)
        self.chk_diff.pack(anchor='w', pady=(10, 2))
        
        self.var_aug = tk.BooleanVar()
        self.chk_aug = ttk.Checkbutton(inf, text="Augmented", variable=self.var_aug, command=self.save_metadata)
        self.chk_aug.pack(anchor='w', pady=(0, 10))
        
        # Nav
        nav = ttk.Frame(inf)
        nav.pack(fill='x', pady=5)
        ttk.Button(nav, text="<", command=self.prev_anno_image).pack(side='left', expand=True)
        ttk.Button(nav, text=">", command=self.next_anno_image).pack(side='left', expand=True)
        self.lbl_anno_idx = ttk.Label(inf, text="0/0")
        self.lbl_anno_idx.pack(pady=5)
        
        ttk.Separator(inf, orient='horizontal').pack(fill='x', pady=20)
        ttk.Label(inf, text="Your Name:").pack(anchor='w')
        self.entry_name = ttk.Entry(inf)
        self.entry_name.pack(fill='x', pady=5)
        ttk.Button(inf, text="Generate ZIP", command=self.generate_submission).pack(fill='x', pady=10)

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
        threading.Thread(target=self.process_detection).start()

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
                    with open(os.path.join(out, os.path.splitext(f)[0]+".txt"), 'w') as tf: tf.writelines(lines)
            self.root.after(0, lambda: messagebox.showinfo("Done", "Complete"))
            self.root.after(0, self.show_anno_image)
        except Exception as e: print(e)
        finally: self.root.after(0, lambda: self.anno_lbl.config(text="Ready"))

    def show_anno_image(self):
        if not self.annotation_images: return
        f = self.anno_folder_var.get()
        name = self.annotation_images[self.current_review_idx]
        img_p = os.path.join("anonymized_frames", f, name)
        txt_p = os.path.join("annotations", f, os.path.splitext(name)[0]+".txt")
        meta_p = os.path.join("metadata", f, os.path.splitext(name)[0]+".json")
        
        # Load Metadata into UI
        if os.path.exists(meta_p):
            with open(meta_p) as file:
                d = json.load(file)
                self.var_anno_loc.set(d.get('location',''))
                self.var_anno_time.set(d.get('time_of_day','Day'))
                self.var_anno_weather.set(d.get('weather','Clear'))
                self.var_anno_light.set(d.get('lighting','Normal'))
                self.var_diff.set(d.get('difficult', False))
                self.var_aug.set(d.get('augmented', False))
        
        img = cv2.imread(img_p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h,w = img.shape[:2]
        if os.path.exists(txt_p):
            with open(txt_p) as file:
                for line in file:
                    p = line.split()
                    c, cx, cy, bw, bh = int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])
                    x1, y1 = int((cx-bw/2)*w), int((cy-bh/2)*h)
                    x2, y2 = int((cx+bw/2)*w), int((cy+bh/2)*h)
                    cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
                    cv2.putText(img, self.CLASSES.get(c, str(c)), (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        
        pil = Image.fromarray(img)
        pil.thumbnail((900, 600))
        tkimg = ImageTk.PhotoImage(pil)
        self.anno_canvas.config(image=tkimg)
        self.anno_canvas.image = tkimg
        self.lbl_anno_idx.config(text=f"{self.current_review_idx+1}/{len(self.annotation_images)}")

    def next_anno_image(self):
        if self.current_review_idx < len(self.annotation_images)-1:
            self.current_review_idx += 1
            self.show_anno_image()

    def prev_anno_image(self):
        if self.current_review_idx > 0:
            self.current_review_idx -= 1
            self.show_anno_image()
            
    def save_metadata(self):
        f = self.anno_folder_var.get()
        if not f or not self.annotation_images: return
        name = self.annotation_images[self.current_review_idx]
        p = os.path.join("metadata", f, os.path.splitext(name)[0]+".json")
        
        # Read existing to preserve 'dataset_name'
        if os.path.exists(p):
            with open(p, 'r') as file: d = json.load(file)
        else:
            d = {}
            
        d['location'] = self.var_anno_loc.get()
        d['time_of_day'] = self.var_anno_time.get()
        d['weather'] = self.var_anno_weather.get()
        d['lighting'] = self.var_anno_light.get()
        d['difficult'] = self.var_diff.get()
        d['augmented'] = self.var_aug.get()
        
        with open(p, 'w') as file: json.dump(d, file, indent=4)

    def launch_labelimg(self):
        f = self.anno_folder_var.get()
        if not f: return
        try:
            subprocess.Popen(['labelImg', os.path.abspath(os.path.join("anonymized_frames", f)),
                              os.path.abspath(os.path.join("annotations", f, "classes.txt")),
                              os.path.abspath(os.path.join("annotations", f))])
        except Exception as e: messagebox.showerror("Error", str(e))

    def generate_submission(self):
        name, f = self.entry_name.get().strip(), self.anno_folder_var.get()
        if not name or not f: return
        path = filedialog.asksaveasfilename(defaultextension=".zip", initialfile=f"{name.replace(' ','_')}_{f}.zip")
        if path:
            with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
                for r,_,files in os.walk(os.path.join("anonymized_frames", f)):
                    for file in files: z.write(os.path.join(r,file), f"images/{file}")
                for r,_,files in os.walk(os.path.join("annotations", f)):
                    for file in files: z.write(os.path.join(r,file), f"labels/{file}")
                for r,_,files in os.walk(os.path.join("metadata", f)):
                    for file in files: z.write(os.path.join(r,file), f"metadata/{file}")
            messagebox.showinfo("Done", "Saved")

    def refresh_folder_lists(self):
        if os.path.exists("anonymized_frames"):
            self.anon_combo['values'] = [d for d in os.listdir("extracted_frames") if os.path.isdir(os.path.join("extracted_frames",d))]
            self.anno_combo['values'] = [d for d in os.listdir("anonymized_frames") if os.path.isdir(os.path.join("anonymized_frames",d))]

if __name__ == "__main__":
    for d in ["extracted_frames", "anonymized_frames", "annotations", "metadata"]: os.makedirs(d, exist_ok=True)
    root = tk.Tk()
    app = TanawApp(root)
    app.refresh_folder_lists()
    root.mainloop()