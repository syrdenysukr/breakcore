import cv2
import numpy as np
import moviepy.editor as mpe
import random
import os
import math

# --- Defaults ---
DEFAULT_CONFIG = {
    "MAX_TRACKERS": 15,
    "REDETECTION_INTERVAL": 30,
    "WORDS": ["CAN", "YOU", "SEE", "ME", "?"],
    "OBJ_LIFESPAN_MIN": 1.0,
    "OBJ_LIFESPAN_MAX": 3.0,
    "OBJ_SIZE_MIN": 30,
    "OBJ_SIZE_MAX": 70,
    "STAR_POINTS": 5,
    "LINE_THICKNESS": 1,
    "SHAPE": "star"
    # LINE_COLOR убран, так как для звезд цвет будет динамическим
}

# --- Состояние трекинга (глобальные переменные, так как moviepy.fl не позволяет легко передавать состояние) ---
tracked_objects = []
prev_gray = None
frame_count = 0
last_time = -1

def get_input(prompt, default, value_type=str, options=None):
    while True:
        if options:
            prompt_str = f"{prompt} ({'/'.join(options)}) (по умолчанию: {default}): "
        else:
            prompt_str = f"{prompt} (по умолчанию: {default}): "
            
        user_input = input(prompt_str)
        if user_input == "":
            return default
        
        if options and user_input.lower() not in options:
            print(f"Ошибка: выберите один из вариантов: {', '.join(options)}")
            continue

        try:
            return value_type(user_input.lower()) if options else value_type(user_input)
        except ValueError:
            print("Ошибка: неверный тип данных. Попробуйте еще раз.")

class TrackedObject:
    def __init__(self, point, creation_time, config):
        self.id = random.randint(1000, 9999)
        self.point = point
        self.text = random.choice(config["WORDS"])
        self.creation_time = creation_time
        self.lifespan = random.uniform(config["OBJ_LIFESPAN_MIN"], config["OBJ_LIFESPAN_MAX"])
        self.size = random.randint(config["OBJ_SIZE_MIN"], config["OBJ_SIZE_MAX"])
        # Добавляем случайную фазу для мерцания, чтобы звезды не мерцали синхронно
        self.shimmer_phase = random.uniform(0, 2 * np.pi)

    def is_alive(self, current_time):
        return (current_time - self.creation_time) < self.lifespan

def draw_star(img, center, size, thickness, star_points, current_time, creation_time, shimmer_phase):
    x, y = center
    outer_radius = size // 2
    inner_radius = outer_radius // 2
    
    # Расчет эффекта мерцания
    age = current_time - creation_time
    # Синусоида заставляет цвет пульсировать. Частоту можно изменить.
    shimmer = (math.sin(age * 4 + shimmer_phase) + 1) / 2 # значение от 0 до 1
    
    points = []
    angle = np.pi / star_points

    for i in range(2 * star_points):
        r = outer_radius if i % 2 == 0 else inner_radius
        current_angle = i * angle - np.pi / 2
        px = int(x + r * np.cos(current_angle))
        py = int(y + r * np.sin(current_angle))
        points.append((px, py))

    # Рисуем каждый сегмент линии разным цветом для эффекта градиента
    num_points = len(points)
    for i in range(num_points):
        p1 = points[i]
        p2 = points[(i + 1) % num_points] # трюк для соединения последней точки с первой
        
        # Базовый серый цвет, который мерцает
        base_gray = 120 + shimmer * 80 # от 120 до 200
        # Добавляем эффект градиента для каждой линии
        gradient_offset = (i / num_points) * 55 # от 0 до 55
        line_gray = int(base_gray + gradient_offset)
        
        color = (line_gray, line_gray, line_gray)
        cv2.line(img, p1, p2, color, thickness, lineType=cv2.LINE_AA)

def draw_square(img, center, size, color, thickness):
    x, y = center
    half_size = size // 2
    pt1 = (x - half_size, y - half_size)
    pt2 = (x + half_size, y + half_size)
    cv2.rectangle(img, pt1, pt2, color, thickness, lineType=cv2.LINE_AA)

def process_frame_with_tracking(frame, t, config):
    global prev_gray, tracked_objects, frame_count, last_time

    if t < last_time:
        prev_gray = None
        tracked_objects = []
        frame_count = 0
    last_time = t

    current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    output_frame = frame.copy()

    tracked_objects = [obj for obj in tracked_objects if obj.is_alive(t)]

    if len(tracked_objects) > 0:
        old_points = np.float32([obj.point for obj in tracked_objects]).reshape(-1, 1, 2)
        new_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, current_gray, old_points, None, **config['lk_params'])
        good_new_points = new_points[status == 1]
        survived_objects = [obj for i, obj in enumerate(tracked_objects) if status[i] == 1]

        for i, obj in enumerate(survived_objects):
            obj.point = tuple(good_new_points[i].ravel())
        tracked_objects = survived_objects

    if len(tracked_objects) < config['MAX_TRACKERS'] // 2 or frame_count % config['REDETECTION_INTERVAL'] == 0:
        mask = np.ones_like(current_gray)
        for obj in tracked_objects:
            x, y = map(int, obj.point)
            cv2.circle(mask, (x, y), 15, 0, -1)

        new_features = cv2.goodFeaturesToTrack(current_gray, mask=mask, **config['feature_params'])
        
        if new_features is not None:
            for point in new_features:
                if len(tracked_objects) < config['MAX_TRACKERS']:
                    tracked_objects.append(TrackedObject(tuple(point.ravel()), t, config))

    if tracked_objects:
        for obj in tracked_objects:
            x, y = map(int, obj.point)
            
            # --- Логика отрисовки ---
            text_color = (255, 255, 255) # Белый текст по умолчанию
            if config['SHAPE'] == 'star':
                draw_star(output_frame, (x, y), obj.size, config['LINE_THICKNESS'], config['STAR_POINTS'], t, obj.creation_time, obj.shimmer_phase)
                # Для звезд текст тоже может мерцать
                shimmer_val = (math.sin((t - obj.creation_time) * 4 + obj.shimmer_phase) + 1) / 2
                text_gray = int(155 + shimmer_val * 100)
                text_color = (text_gray, text_gray, text_gray)

            elif config['SHAPE'] == 'square':
                # Квадраты используют один цвет, зададим его
                square_color = (200, 200, 200) # Светло-серый
                draw_square(output_frame, (x,y), obj.size, square_color, config['LINE_THICKNESS'])
                text_color = square_color

            # --- Отрисовка текста ---
            text_x = x - obj.size // 2
            text_y = y - obj.size // 2 - 5
            cv2.putText(output_frame, obj.text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, lineType=cv2.LINE_AA)

        if len(tracked_objects) > 1:
            # Соединительные линии сделаем немного тусклее
            line_color = (100, 100, 100)
            num_lines = len(tracked_objects) // 2
            temp_list = random.sample(tracked_objects, len(tracked_objects))
            for i in range(num_lines):
                obj1 = temp_list[i*2]
                obj2 = temp_list[i*2 + 1]
                pt1 = tuple(map(int, obj1.point))
                pt2 = tuple(map(int, obj2.point))
                cv2.line(output_frame, pt1, pt2, line_color, config['LINE_THICKNESS'], lineType=cv2.LINE_AA)

    prev_gray = current_gray.copy()
    frame_count += 1
    
    return output_frame

def run_video_processing(config, input_video_path, output_video_path):
    """
    Runs the video processing with the given configuration.
    """
    print("--------------------------\n")
    print("загрузка видео ща")
    try:
        clip = mpe.VideoFileClip(input_video_path)
    except Exception as e:
        print(f"Ошибка при загрузке видео: {e}")
        print("Проверьте, что путь к файлу указан верно и файл существует.")
        return # Use return instead of exit

    print(f"рисую {config['SHAPE']}s!")
    
    # Сброс состояния трекера перед каждым запуском
    global prev_gray, tracked_objects, frame_count, last_time
    prev_gray = None
    tracked_objects = []
    frame_count = 0
    last_time = -1

    processing_function = lambda gf, t: process_frame_with_tracking(gf(t)[:,:,::-1], t, config)[:,:,::-1]
    final_clip = clip.fl(processing_function)

    print(f"результат сохранен в {output_video_path}...")
    final_clip.write_videofile(output_video_path, codec='libx264', audio_codec='aac', logger=None)
    print("Готово!")

def main():
    config = DEFAULT_CONFIG.copy()

    print("--- Настройка параметров ---")
    
    default_input = os.path.join("исходники", "мск.mp4")
    input_video_path = get_input("Путь к исходному видео", default_input)

    default_output = os.path.join("результ", "interactive_output.mp4")
    output_video_path = get_input("Путь к итоговому видео", default_output)

    config["SHAPE"] = get_input("Выберите фигуру", config["SHAPE"], str, options=["star", "square"])

    config["MAX_TRACKERS"] = get_input("Макс. число объектов", config["MAX_TRACKERS"], int)
    config["OBJ_LIFESPAN_MIN"] = get_input("Мин. время жизни объекта (сек)", config["OBJ_LIFESPAN_MIN"], float)
    config["OBJ_LIFESPAN_MAX"] = get_input("Макс. время жизни объекта (сек)", config["OBJ_LIFESPAN_MAX"], float)
    config["OBJ_SIZE_MIN"] = get_input("Мин. размер объекта (пикс)", config["OBJ_SIZE_MIN"], int)
    config["OBJ_SIZE_MAX"] = get_input("Макс. размер объекта (пикс)", config["OBJ_SIZE_MAX"], int)
    
    if config["SHAPE"] == 'star':
        config["STAR_POINTS"] = get_input("Кол-во вершин у звезды", config["STAR_POINTS"], int)
    
    config["LINE_THICKNESS"] = get_input("Толщина линий", config["LINE_THICKNESS"], int)

    config['feature_params'] = dict(
        maxCorners=config['MAX_TRACKERS'],
        qualityLevel=0.3,
        minDistance=8,
        blockSize=7
    )
    config['lk_params'] = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )
    
    run_video_processing(config, input_video_path, output_video_path)

if __name__ == '__main__':
    main()


