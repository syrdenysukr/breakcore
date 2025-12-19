import tkinter as tk
from tkinter import filedialog, ttk
import test as trackingboxes
import threading
import cv2
import sv_ttk
import math
import os
import subprocess
import platform

# --- Функция запуска ---
def run_processing(config, input_path, output_path):
    try:
        progress_bar['mode'] = 'determinate'
        trackingboxes.run_video_processing(config, input_path, output_path, progress_callback=update_progress)
        status_label.config(text="Готово! Видео сохранено.", foreground="#88ff88") # light green
        open_btn.config(state=tk.NORMAL)
    except Exception as e:
        status_label.config(text=f"Ошибка: {e}", foreground="#ff8888")
    finally:
        start_button.config(state=tk.NORMAL)
        progress_bar.stop()

def update_progress(value=0, **kwargs):
    if isinstance(value, (int, float)):
        progress_var.set(value)

def start_processing_thread():
    input_path = input_path_var.get()
    output_path = output_path_var.get()
    
    if not os.path.exists(input_path):
        status_label.config(text="Ошибка: Исходный файл не найден!", foreground="#ff8888")
        return

    config = trackingboxes.DEFAULT_CONFIG.copy()
    
    # Обработка слов
    raw_words = words_var.get()
    word_list = [w.strip() for w in raw_words.split(',') if w.strip()]
    
    if not word_list:
        status_label.config(text="Ошибка: Введите хотя бы одно слово!", foreground="#ff8888")
        return

    try:
        config.update({
            "SHAPE": shape_var.get(),
            "MAX_TRACKERS": int(max_trackers_var.get()),
            "OBJ_LIFESPAN_MIN": float(lifespan_min_var.get()),
            "OBJ_LIFESPAN_MAX": float(lifespan_max_var.get()),
            "OBJ_SIZE_MIN": int(size_min_var.get()),
            "OBJ_SIZE_MAX": int(size_max_var.get()),
            "STAR_POINTS": int(star_points_var.get()),
            "LINE_THICKNESS": int(line_thickness_var.get()),
            "THRESHOLD": float(threshold_var.get()),
            "WORDS": word_list, # Передаем новый список слов
        })
    except ValueError:
        status_label.config(text="Ошибка: Проверьте числовые поля!", foreground="#ff8888")
        return

    config['feature_params'] = dict(
        maxCorners=config['MAX_TRACKERS'],
        qualityLevel=1.0 - config['THRESHOLD'] + 0.01,
        minDistance=8,
        blockSize=7
    )
    config['lk_params'] = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    status_label.config(text="Обработка видео... Пожалуйста, подождите.", foreground="orange")
    start_button.config(state=tk.DISABLED)
    open_btn.config(state=tk.DISABLED)
    progress_var.set(0)
    
    processing_thread = threading.Thread(target=run_processing, args=(config, input_path, output_path))
    processing_thread.start()

def open_result_file():
    path = output_path_var.get()
    if os.path.exists(path):
        if platform.system() == 'Windows':
            os.startfile(path)
        elif platform.system() == 'Darwin':
            subprocess.call(('open', path))
        else:
            subprocess.call(('xdg-open', path))

def select_input_file():
    filepath = filedialog.askopenfilename(title="Выберите исходное видео", filetypes=(("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*")))
    if filepath:
        input_path_var.set(filepath)

def select_output_file():
    filepath = filedialog.asksaveasfilename(title="Сохранить как...", defaultextension=".mp4", filetypes=(("MP4 files", "*.mp4"), ("All files", "*.*")))
    if filepath:
        output_path_var.set(filepath)

# --- Рендер превью фигур ---
def draw_shape_on_canvas(canvas, shape, size, points, color="#e0e0e0"):
    canvas.delete("all")
    width = canvas.winfo_width()
    height = canvas.winfo_height()
    if width < 2: return
        
    cx, cy = width / 2, height / 2
    r = size / 2

    if shape == "square":
        canvas.create_rectangle(cx - r, cy - r, cx + r, cy + r, outline=color, width=2)
    elif shape == "star":
        angle = -math.pi / 2
        step = 2 * math.pi / (points * 2)
        star_pts = []
        for i in range(points * 2):
            curr_r = r if i % 2 == 0 else r / 2.5
            star_pts.append((cx + curr_r * math.cos(angle), cy + curr_r * math.sin(angle)))
            angle += step
        canvas.create_polygon(star_pts, outline=color, fill="", width=2)

def update_previews(*args):
    try:
        min_s = int(size_min_var.get())
        max_s = int(size_max_var.get())
        pts = int(star_points_var.get())
        shp = shape_var.get()
        draw_shape_on_canvas(min_canvas, shp, min_s, pts)
        draw_shape_on_canvas(max_canvas, shp, max_s, pts)
    except: pass

def update_entry_from_slider(val):
    threshold_var.set(f"{float(val):.2f}")

# --- GUI SETUP ---
root = tk.Tk()
root.title("Breakcore Visualizer GUI")
root.geometry("700x650") # Чуть увеличил высоту
sv_ttk.set_theme("dark")

# Данные
defaults = trackingboxes.DEFAULT_CONFIG
input_path_var = tk.StringVar(value=os.path.join("исходники", "мск.mp4"))
output_path_var = tk.StringVar(value=os.path.join("результ", "output.mp4"))
shape_var = tk.StringVar(value=defaults["SHAPE"])
star_points_var = tk.StringVar(value=defaults["STAR_POINTS"])
max_trackers_var = tk.StringVar(value=defaults["MAX_TRACKERS"])
lifespan_min_var = tk.StringVar(value=defaults["OBJ_LIFESPAN_MIN"])
lifespan_max_var = tk.StringVar(value=defaults["OBJ_LIFESPAN_MAX"])
size_min_var = tk.StringVar(value=defaults["OBJ_SIZE_MIN"])
size_max_var = tk.StringVar(value=defaults["OBJ_SIZE_MAX"])
line_thickness_var = tk.StringVar(value=defaults["LINE_THICKNESS"])
threshold_var = tk.DoubleVar(value=defaults.get("THRESHOLD", 0.7))
words_var = tk.StringVar(value=", ".join(defaults["WORDS"])) # Новая переменная для слов
progress_var = tk.DoubleVar(value=0)

# Триггеры обновлений
for var in (shape_var, size_min_var, size_max_var, star_points_var):
    var.trace_add("write", update_previews)

# --- ВЕРХНЯЯ ЧАСТЬ (Файлы) ---
file_frame = ttk.LabelFrame(root, text="Файлы", padding=10)
file_frame.pack(fill="x", padx=10, pady=5)

ttk.Label(file_frame, text="Вход:").grid(row=0, column=0, sticky="w")
ttk.Entry(file_frame, textvariable=input_path_var).grid(row=0, column=1, sticky="ew", padx=5)
ttk.Button(file_frame, text="📂", width=3, command=select_input_file).grid(row=0, column=2)

ttk.Label(file_frame, text="Выход:").grid(row=1, column=0, sticky="w")
ttk.Entry(file_frame, textvariable=output_path_var).grid(row=1, column=1, sticky="ew", padx=5)
ttk.Button(file_frame, text="📂", width=3, command=select_output_file).grid(row=1, column=2)
file_frame.columnconfigure(1, weight=1)

# --- ЦЕНТРАЛЬНАЯ ЧАСТЬ (Настройки) ---
notebook = ttk.Notebook(root)
notebook.pack(fill="both", expand=True, padx=10, pady=5)

# Вкладка 1: Визуал
visual_tab = ttk.Frame(notebook, padding=10)
notebook.add(visual_tab, text="Визуал")

# Левая колонка визуала
v_left = ttk.Frame(visual_tab)
v_left.pack(side="left", fill="both", expand=True)

ttk.Label(v_left, text="Текст (через запятую):").pack(anchor="w", pady=(0,2))
ttk.Entry(v_left, textvariable=words_var).pack(fill="x", pady=(0,10))

ttk.Label(v_left, text="Фигура:").pack(anchor="w", pady=(0,2))
ttk.Combobox(v_left, textvariable=shape_var, values=["star", "square"], state="readonly").pack(fill="x", pady=(0,10))

ttk.Label(v_left, text="Лучей звезды:").pack(anchor="w", pady=(0,2))
ttk.Entry(v_left, textvariable=star_points_var).pack(fill="x", pady=(0,10))

ttk.Label(v_left, text="Толщина линий:").pack(anchor="w", pady=(0,2))
ttk.Entry(v_left, textvariable=line_thickness_var).pack(fill="x", pady=(0,10))

# Правая колонка визуала (Превью)
v_right = ttk.LabelFrame(visual_tab, text="Предпросмотр размера", padding=10)
v_right.pack(side="right", fill="both", expand=True, padx=(10,0))

v_right.columnconfigure(0, weight=1)
v_right.columnconfigure(1, weight=1)

ttk.Label(v_right, text="Min").grid(row=0, column=0)
ttk.Label(v_right, text="Max").grid(row=0, column=1)

min_canvas = tk.Canvas(v_right, height=100, bg="#2b2b2b", highlightthickness=0)
min_canvas.grid(row=1, column=0, sticky="ew", padx=2)
max_canvas = tk.Canvas(v_right, height=100, bg="#2b2b2b", highlightthickness=0)
max_canvas.grid(row=1, column=1, sticky="ew", padx=2)

ttk.Label(v_right, text="Размер (px):").grid(row=2, column=0, columnspan=2, pady=(10,2))
s_frame = ttk.Frame(v_right)
s_frame.grid(row=3, column=0, columnspan=2)
ttk.Entry(s_frame, textvariable=size_min_var, width=5).pack(side="left", padx=2)
ttk.Label(s_frame, text="-").pack(side="left")
ttk.Entry(s_frame, textvariable=size_max_var, width=5).pack(side="left", padx=2)


# Вкладка 2: Поведение (Трекинг)
logic_tab = ttk.Frame(notebook, padding=10)
notebook.add(logic_tab, text="Настройки трекера")

l_grid = ttk.Frame(logic_tab)
l_grid.pack(fill="x")
l_grid.columnconfigure(1, weight=1)

ttk.Label(l_grid, text="Макс. объектов:").grid(row=0, column=0, sticky="w", pady=5)
ttk.Entry(l_grid, textvariable=max_trackers_var).grid(row=0, column=1, sticky="ew", padx=10)

ttk.Label(l_grid, text="Время жизни (сек):").grid(row=1, column=0, sticky="w", pady=5)
l_lifespan = ttk.Frame(l_grid)
l_lifespan.grid(row=1, column=1, sticky="ew", padx=10)
ttk.Entry(l_lifespan, textvariable=lifespan_min_var, width=8).pack(side="left")
ttk.Label(l_lifespan, text=" - ").pack(side="left")
ttk.Entry(l_lifespan, textvariable=lifespan_max_var, width=8).pack(side="left")

ttk.Label(l_grid, text="Чувствительность (Threshold):").grid(row=2, column=0, sticky="w", pady=(20, 5))
l_thresh = ttk.Frame(l_grid)
l_thresh.grid(row=2, column=1, sticky="ew", padx=10, pady=(20, 5))
ttk.Scale(l_thresh, from_=0.0, to=1.0, variable=threshold_var, command=update_entry_from_slider).pack(side="left", fill="x", expand=True)
ttk.Entry(l_thresh, textvariable=threshold_var, width=5).pack(side="left", padx=(5,0))
ttk.Label(l_grid, text="(Больше = меньше мусора)").grid(row=3, column=1, sticky="w", padx=10, pady=0)


# --- НИЖНЯЯ ЧАСТЬ (Контроль) ---
bottom_frame = ttk.Frame(root, padding=15)
bottom_frame.pack(fill="x", side="bottom")

# Статус
status_label = ttk.Label(bottom_frame, text="Готов к работе", font=("Segoe UI", 9))
status_label.pack(anchor="w", pady=(0, 5))

# Прогресс
progress_bar = ttk.Progressbar(bottom_frame, variable=progress_var, mode='determinate')
progress_bar.pack(fill="x", pady=(0, 10))

# Кнопки
btn_frame = ttk.Frame(bottom_frame)
btn_frame.pack(fill="x")

open_btn = ttk.Button(btn_frame, text="Открыть результат", state=tk.DISABLED, command=open_result_file)
open_btn.pack(side="left")

start_button = ttk.Button(btn_frame, text="ЗАПУСТИТЬ РЕНДЕР", style="Accent.TButton", command=start_processing_thread)
start_button.pack(side="right")

# Инициализация
root.after(100, update_previews)
root.mainloop()