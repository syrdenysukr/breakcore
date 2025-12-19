import cv2
import numpy as np
import moviepy.editor as mpe
import random
import os
import math
from proglog import ProgressBarLogger  # Нужно для связи прогресс-бара

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
    "SHAPE": "star",
    "THRESHOLD": 0.7 
}

# --- Глобальное состояние ---
tracked_objects = []
prev_gray = None
frame_count = 0
last_time = -1

# --- Класс для прогресс-бара ---
class TkLogger(ProgressBarLogger):
    def __init__(self, callback):
        super().__init__()
        self.callback = callback

    def bars_callback(self, bar, attr, value, old_value=None):
        # MoviePy использует бар с именем 't' для времени
        if bar == 't' and self.callback:
            total = self.bars[bar]['total']
            if total > 0:
                percentage = (value / total) * 100
                self.callback(percentage)

class TrackedObject:
    def __init__(self, point, creation_time, config):
        self.id = random.randint(1000, 9999)
        self.point = point
        self.text = random.choice(config["WORDS"])
        self.creation_time = creation_time
        self.lifespan = random.uniform(config["OBJ_LIFESPAN_MIN"], config["OBJ_LIFESPAN_MAX"])
        self.size = random.randint(config["OBJ_SIZE_MIN"], config["OBJ_SIZE_MAX"])
        self.shimmer_phase = random.uniform(0, 2 * np.pi)

    def is_alive(self, current_time):
        return (current_time - self.creation_time) < self.lifespan

def draw_star(img, center, size, thickness, star_points, current_time, creation_time, shimmer_phase):
    x, y = center
    outer_radius = size // 2
    inner_radius = outer_radius // 2
    
    age = current_time - creation_time
    shimmer = (math.sin(age * 4 + shimmer_phase) + 1) / 2
    
    points = []
    angle = np.pi / star_points

    for i in range(2 * star_points):
        r = outer_radius if i % 2 == 0 else inner_radius
        current_angle = i * angle - np.pi / 2
        px = int(x + r * np.cos(current_angle))
        py = int(y + r * np.sin(current_angle))
        points.append((px, py))

    num_points = len(points)
    for i in range(num_points):
        p1 = points[i]
        p2 = points[(i + 1) % num_points]
        
        base_gray = 120 + shimmer * 80
        gradient_offset = (i / num_points) * 55
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

    # Условие переобнаружения с учетом THRESHOLD (qualityLevel)
    if len(tracked_objects) < config['MAX_TRACKERS'] // 2 or frame_count % config['REDETECTION_INTERVAL'] == 0:
        mask = np.ones_like(current_gray)
        for obj in tracked_objects:
            x, y = map(int, obj.point)
            cv2.circle(mask, (x, y), 15, 0, -1)
            
        # Используем параметр THRESHOLD как qualityLevel, но инвертируем или адаптируем если нужно
        # Обычно qualityLevel это от 0.01 до 1.0. 
        # В GUI у тебя THRESHOLD. Давай используем его напрямую в feature_params
        
        # Обновляем параметры, если они были переданы через config
        local_feature_params = config['feature_params'].copy()
        
        new_features = cv2.goodFeaturesToTrack(current_gray, mask=mask, **local_feature_params)
        
        if new_features is not None:
            for point in new_features:
                if len(tracked_objects) < config['MAX_TRACKERS']:
                    tracked_objects.append(TrackedObject(tuple(point.ravel()), t, config))

    if tracked_objects:
        for obj in tracked_objects:
            x, y = map(int, obj.point)
            
            text_color = (255, 255, 255)
            if config['SHAPE'] == 'star':
                draw_star(output_frame, (x, y), obj.size, config['LINE_THICKNESS'], config['STAR_POINTS'], t, obj.creation_time, obj.shimmer_phase)
                shimmer_val = (math.sin((t - obj.creation_time) * 4 + obj.shimmer_phase) + 1) / 2
                text_gray = int(155 + shimmer_val * 100)
                text_color = (text_gray, text_gray, text_gray)

            elif config['SHAPE'] == 'square':
                square_color = (200, 200, 200)
                draw_square(output_frame, (x,y), obj.size, square_color, config['LINE_THICKNESS'])
                text_color = square_color

            text_x = x - obj.size // 2
            text_y = y - obj.size // 2 - 5
            cv2.putText(output_frame, obj.text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, lineType=cv2.LINE_AA)

        if len(tracked_objects) > 1:
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

def run_video_processing(config, input_video_path, output_video_path, progress_callback=None):
    print("--------------------------\n")
    print("загрузка видео...")
    try:
        clip = mpe.VideoFileClip(input_video_path)
    except Exception as e:
        print(f"Ошибка при загрузке видео: {e}")
        raise e

    print(f"рисую {config['SHAPE']}s!")
    
    global prev_gray, tracked_objects, frame_count, last_time
    prev_gray = None
    tracked_objects = []
    frame_count = 0
    last_time = -1

    # Подготовка логгера
    logger = 'bar'
    if progress_callback:
        logger = TkLogger(progress_callback)

    processing_function = lambda gf, t: process_frame_with_tracking(gf(t)[:,:,::-1], t, config)[:,:,::-1]
    final_clip = clip.fl(processing_function)

    print(f"результат будет сохранен в {output_video_path}...")
    final_clip.write_videofile(output_video_path, codec='libx264', audio_codec='aac', logger=logger)
    print("Готово!")

if __name__ == '__main__':
    # Для теста без GUI
    pass