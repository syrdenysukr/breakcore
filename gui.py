import tkinter as tk
from tkinter import filedialog, ttk
import test as trackingboxes # Import your main script
import threading
import cv2
import sv_ttk
import math

def run_processing(config, input_path, output_path):
    """Function to run the video processing in a separate thread."""
    try:
        # We will need to modify run_video_processing to accept a progress callback
        # For now, the progress bar will just run in indeterminate mode.
        progress_bar.start()
        trackingboxes.run_video_processing(config, input_path, output_path, progress_callback=update_progress)
        status_label.config(text="Готово!", foreground="green")
    except Exception as e:
        status_label.config(text=f"Ошибка: {e}", foreground="red")
    finally:
        start_button.config(state=tk.NORMAL)
        progress_bar.stop()
        progress_var.set(0)


def update_progress(value):
    """Callback function to update the progress bar from the processing thread."""
    progress_var.set(value)


def start_processing_thread():
    """Starts the video processing in a new thread to keep the GUI responsive."""
    input_path = input_path_var.get()
    output_path = output_path_var.get()
    
    config = trackingboxes.DEFAULT_CONFIG.copy()
    
    config.update({
        "SHAPE": shape_var.get(),
        "MAX_TRACKERS": int(max_trackers_var.get()),
        "OBJ_LIFESPAN_MIN": float(lifespan_min_var.get()),
        "OBJ_LIFESPAN_MAX": float(lifespan_max_var.get()),
        "OBJ_SIZE_MIN": int(size_min_var.get()),
        "OBJ_SIZE_MAX": int(size_max_var.get()),
        "STAR_POINTS": int(star_points_var.get()),
        "LINE_THICKNESS": int(line_thickness_var.get()),
        "THRESHOLD": float(threshold_var.get()), # New parameter
    })

    config['feature_params'] = dict(maxCorners=config['MAX_TRACKERS'], qualityLevel=0.3, minDistance=8, blockSize=7)
    config['lk_params'] = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    status_label.config(text="В обработке...", foreground="orange")
    start_button.config(state=tk.DISABLED)
    
    processing_thread = threading.Thread(target=run_processing, args=(config, input_path, output_path))
    processing_thread.start()


def select_input_file():
    filepath = filedialog.askopenfilename(title="Выберите исходное видео", filetypes=(("MP4 files", "*.mp4"), ("All files", "*.*")))
    if filepath:
        input_path_var.set(filepath)

def select_output_file():
    filepath = filedialog.asksaveasfilename(title="Сохранить итоговое видео как...", defaultextension=".mp4", filetypes=(("MP4 files", "*.mp4"), ("All files", "*.*")))
    if filepath:
        output_path_var.set(filepath)

def draw_shape_on_canvas(canvas, shape, size, points, color="white"):
    """Helper function to draw a shape on a given canvas."""
    canvas.delete("all")
    width = canvas.winfo_width()
    height = canvas.winfo_height()
    if width < 2 or height < 2: # Canvas not ready yet
        return
        
    center_x, center_y = width / 2, height / 2
    radius = size / 2

    if shape == "square":
        canvas.create_rectangle(center_x - radius, center_y - radius, center_x + radius, center_y + radius, outline=color, width=1)
    elif shape == "star":
        angle = -math.pi / 2
        step = 2 * math.pi / (points * 2)
        outer_radius = radius
        inner_radius = radius / 2
        
        star_points = []
        for i in range(points * 2):
            r = outer_radius if i % 2 == 0 else inner_radius
            star_points.append((center_x + r * math.cos(angle), center_y + r * math.sin(angle)))
            angle += step
        canvas.create_polygon(star_points, outline=color, fill="", width=1)

def update_previews(*args):
    """Updates the min/max size preview canvases."""
    shape = shape_var.get()
    try:
        min_size = int(size_min_var.get())
        max_size = int(size_max_var.get())
        points = int(star_points_var.get())
    except (ValueError, TclError):
        return # Ignore errors from empty/invalid entry during typing

    draw_shape_on_canvas(min_canvas, shape, min_size, points)
    draw_shape_on_canvas(max_canvas, shape, max_size, points)
    
def update_slider_from_entry(*args):
    """Updates slider when entry is changed."""
    try:
        val = float(threshold_entry.get())
        if 0.0 <= val <= 1.0:
            threshold_var.set(val)
    except (ValueError, tk.TclError):
        pass # Ignore invalid input

def update_entry_from_slider(val):
    """Updates entry when slider is moved."""
    threshold_var.set(f"{float(val):.2f}")


# --- GUI Setup ---
root = tk.Tk()
root.title("Настройки @winterchroma_trackingboxes")
sv_ttk.set_theme("dark")

# --- Main container ---
main_frame = ttk.Frame(root, padding="15")
main_frame.pack(fill="both", expand=True)
main_frame.columnconfigure(0, weight=3) # Settings column
main_frame.columnconfigure(1, weight=1) # Preview column

# --- Left Column ---
left_frame = ttk.Frame(main_frame)
left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

# --- Right Column ---
right_frame = ttk.Frame(main_frame)
right_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
right_frame.grid(row=0, column=1, sticky="nsew")


# --- Default values ---
defaults = trackingboxes.DEFAULT_CONFIG
if "THRESHOLD" not in defaults:
    defaults["THRESHOLD"] = 0.7 # Add default if not present

# --- Variables ---
input_path_var = tk.StringVar(value="исходники/мск.mp4")
output_path_var = tk.StringVar(value="результ/output.mp4")
shape_var = tk.StringVar(value=defaults["SHAPE"])
star_points_var = tk.StringVar(value=defaults["STAR_POINTS"])
max_trackers_var = tk.StringVar(value=defaults["MAX_TRACKERS"])
lifespan_min_var = tk.StringVar(value=defaults["OBJ_LIFESPAN_MIN"])
lifespan_max_var = tk.StringVar(value=defaults["OBJ_LIFESPAN_MAX"])
size_min_var = tk.StringVar(value=defaults["OBJ_SIZE_MIN"])
size_max_var = tk.StringVar(value=defaults["OBJ_SIZE_MAX"])
line_thickness_var = tk.StringVar(value=defaults["LINE_THICKNESS"])
threshold_var = tk.DoubleVar(value=defaults["THRESHOLD"])
progress_var = tk.DoubleVar(value=0)

# Bind preview update to changes
shape_var.trace_add("write", update_previews)
size_min_var.trace_add("write", update_previews)
size_max_var.trace_add("write", update_previews)
star_points_var.trace_add("write", update_previews)
threshold_var.trace_add("write", update_slider_from_entry)


# --- LEFT COLUMN WIDGETS ---

# --- Input/Output Paths ---
path_frame = ttk.LabelFrame(left_frame, text="Файлы", padding="10")
path_frame.pack(fill="x", expand=True, pady=5)
ttk.Label(path_frame, text="Исходное видео:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
ttk.Entry(path_frame, textvariable=input_path_var, width=50).grid(row=0, column=1, sticky="ew", padx=5, pady=2)
ttk.Button(path_frame, text="Обзор...", command=select_input_file).grid(row=0, column=2, padx=5, pady=2)
ttk.Label(path_frame, text="Итоговое видео:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
ttk.Entry(path_frame, textvariable=output_path_var, width=50).grid(row=1, column=1, sticky="ew", padx=5, pady=2)
ttk.Button(path_frame, text="Обзор...", command=select_output_file).grid(row=1, column=2, padx=5, pady=2)
path_frame.columnconfigure(1, weight=1)

# --- Shape & Tracking Settings (Combined) ---
settings_grid = ttk.LabelFrame(left_frame, text="Параметры", padding="10")
settings_grid.pack(fill="x", expand=True, pady=5)
settings_grid.columnconfigure(1, weight=1)
settings_grid.columnconfigure(3, weight=1)

# Shape
ttk.Label(settings_grid, text="Форма:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
shape_menu = ttk.Combobox(settings_grid, textvariable=shape_var, values=["star", "square"], state="readonly")
shape_menu.grid(row=0, column=1, sticky="ew", padx=5, pady=2)

# Star Points
ttk.Label(settings_grid, text="Вершин у звезды:").grid(row=0, column=2, sticky="w", padx=5, pady=2)
ttk.Entry(settings_grid, textvariable=star_points_var, width=8).grid(row=0, column=3, sticky="ew", padx=5, pady=2)

# Line Thickness
ttk.Label(settings_grid, text="Толщина линий (px):").grid(row=1, column=0, sticky="w", padx=5, pady=2)
ttk.Entry(settings_grid, textvariable=line_thickness_var).grid(row=1, column=1, sticky="ew", padx=5, pady=2)

# Max Trackers
ttk.Label(settings_grid, text="Макс. объектов:").grid(row=2, column=0, sticky="w", padx=5, pady=2)
ttk.Entry(settings_grid, textvariable=max_trackers_var).grid(row=2, column=1, sticky="ew", padx=5, pady=2)

# Min/Max Size
ttk.Label(settings_grid, text="Мин. размер (px):").grid(row=3, column=0, sticky="w", padx=5, pady=2)
ttk.Entry(settings_grid, textvariable=size_min_var).grid(row=3, column=1, sticky="ew", padx=5, pady=2)
ttk.Label(settings_grid, text="Макс. размер (px):").grid(row=3, column=2, sticky="w", padx=5, pady=2)
ttk.Entry(settings_grid, textvariable=size_max_var).grid(row=3, column=3, sticky="ew", padx=5, pady=2)

# Min/Max Lifespan
ttk.Label(settings_grid, text="Мин. время жизни (с):").grid(row=4, column=0, sticky="w", padx=5, pady=2)
ttk.Entry(settings_grid, textvariable=lifespan_min_var).grid(row=4, column=1, sticky="ew", padx=5, pady=2)
ttk.Label(settings_grid, text="Макс. время жизни (с):").grid(row=4, column=2, sticky="w", padx=5, pady=2)
ttk.Entry(settings_grid, textvariable=lifespan_max_var).grid(row=4, column=3, sticky="ew", padx=5, pady=2)


# --- RIGHT COLUMN WIDGETS ---

# --- Preview Frame ---
preview_frame = ttk.LabelFrame(right_frame, text="Предпросмотр размера", padding="10")
preview_frame.pack(fill="both", expand=True, pady=5)
preview_frame.columnconfigure(0, weight=1)
preview_frame.columnconfigure(1, weight=1)

ttk.Label(preview_frame, text="Мин. размер").grid(row=0, column=0)
min_canvas = tk.Canvas(preview_frame, width=100, height=100, bg="#2B2B2B", highlightthickness=0)
min_canvas.grid(row=1, column=0, padx=5, pady=5)

ttk.Label(preview_frame, text="Макс. размер").grid(row=0, column=1)
max_canvas = tk.Canvas(preview_frame, width=100, height=100, bg="#2B2B2B", highlightthickness=0)
max_canvas.grid(row=1, column=1, padx=5, pady=5)

# --- Threshold Slider ---
threshold_frame = ttk.LabelFrame(right_frame, text="Порог обнаружения (Threshold)", padding="10")
threshold_frame.pack(fill="x", expand=False, pady=5)
threshold_frame.columnconfigure(0, weight=1)

threshold_slider = ttk.Scale(threshold_frame, from_=0.0, to=1.0, orient="horizontal", variable=threshold_var, command=update_entry_from_slider)
threshold_slider.grid(row=0, column=0, sticky="ew", padx=5)
threshold_entry = ttk.Entry(threshold_frame, textvariable=threshold_var, width=5)
threshold_entry.grid(row=0, column=1, padx=5)


# --- BOTTOM FRAME WIDGETS ---
bottom_frame = ttk.Frame(root, padding=(15, 0, 15, 15))
bottom_frame.pack(fill="x", expand=False)

# --- Progress Bar ---
progress_bar = ttk.Progressbar(bottom_frame, orient="horizontal", mode="determinate", variable=progress_var, maximum=100)
progress_bar.pack(fill="x", expand=True, pady=(0, 5))

# --- Control Buttons ---
control_frame = ttk.Frame(bottom_frame)
control_frame.pack(fill="x", expand=True)

start_button = ttk.Button(control_frame, text="Запустить обработку", command=start_processing_thread)
start_button.pack(side="right")

status_label = ttk.Label(control_frame, text="Нажмите 'Запустить'", anchor="w")
status_label.pack(side="left", fill="x", expand=True)


# --- Finalize ---
root.after(250, update_previews) # Initial draw of previews
root.mainloop()
