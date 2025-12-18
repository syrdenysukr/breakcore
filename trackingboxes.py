import cv2
import numpy as np
import moviepy.editor as mpe
import random
import os

# --- Defaults ---
DEFAULT_CONFIG = {
    "MAX_TRACKERS": 15,
    "REDETECTION_INTERVAL": 30,
    "WORDS": ["CAN", "YOU", "SEE", "ME", "?"],
    "BOX_LIFESPAN_MIN": 1.0,
    "BOX_LIFESPAN_MAX": 3.0,
    "BOX_SIZE_MIN": 30,
    "BOX_SIZE_MAX": 70,
    "STAR_POINTS": 5,
    "LINE_COLOR": (255, 255, 255),
    "LINE_THICKNESS": 1
}

# --- Tracking State (остаются глобальными, т.к. moviepy.fl не позволяет легко передавать состояние) ---
tracked_objects = []
prev_gray = None
frame_count = 0
last_time = -1

def get_input(prompt, default, value_type=str):
    while True:
        user_input = input(f"{prompt} (по умолчанию: {default}): ")
        if user_input == "":
            return default
        try:
            return value_type(user_input)
        except ValueError:
            print("Ошибка: неверный тип данных. Попробуйте еще раз.")

class TrackedObject:
    def __init__(self, point, creation_time, config):
        self.id = random.randint(1000, 9999)
        self.point = point
        self.text = random.choice(config["WORDS"])
        self.creation_time = creation_time
        self.lifespan = random.uniform(config["BOX_LIFESPAN_MIN"], config["BOX_LIFESPAN_MAX"])
        self.size = random.randint(config["BOX_SIZE_MIN"], config["BOX_SIZE_MAX"])

    def is_alive(self, current_time):
        return (current_time - self.creation_time) < self.lifespan

def draw_star(img, center, outer_radius, inner_radius, num_points, color, thickness):
    x, y = center
    points = []
    angle = np.pi / num_points

    for i in range(2 * num_points):
        r = outer_radius if i % 2 == 0 else inner_radius
        current_angle = i * angle - np.pi / 2
        px = int(x + r * np.cos(current_angle))
        py = int(y + r * np.sin(current_angle))
        points.append((px, py))

    pts = np.array(points, np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(img, [pts], True, color, thickness)

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
            outer_radius = obj.size // 2
            inner_radius = outer_radius // 2
            draw_star(output_frame, (x, y), outer_radius, inner_radius, config['STAR_POINTS'], config['LINE_COLOR'], config['LINE_THICKNESS'])
            cv2.putText(output_frame, obj.text, (x - outer_radius, y - outer_radius - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, config['LINE_COLOR'], 1)

        if len(tracked_objects) > 1:
            num_lines = len(tracked_objects) // 2
            temp_list = random.sample(tracked_objects, len(tracked_objects))
            for i in range(num_lines):
                obj1 = temp_list[i*2]
                obj2 = temp_list[i*2 + 1]
                pt1 = tuple(map(int, obj1.point))
                pt2 = tuple(map(int, obj2.point))
                cv2.line(output_frame, pt1, pt2, config['LINE_COLOR'], config['LINE_THICKNESS'])

    prev_gray = current_gray.copy()
    frame_count += 1
    
    return output_frame

def main():
    config = DEFAULT_CONFIG.copy()

    print("--- Настройка параметров ---")
    
    default_input = os.path.join("исходники", "мск.mp4")
    input_video_path = get_input("Путь к исходному видео", default_input)

    default_output = os.path.join("результ", "interactive_output.mp4")
    output_video_path = get_input("Путь к итоговому видео", default_output)

    config["MAX_TRACKERS"] = get_input("Макс. число трекеров", config["MAX_TRACKERS"], int)
    config["BOX_LIFESPAN_MIN"] = get_input("Мин. время жизни звезды (сек)", config["BOX_LIFESPAN_MIN"], float)
    config["BOX_LIFESPAN_MAX"] = get_input("Макс. время жизни звезды (сек)", config["BOX_LIFESPAN_MAX"], float)
    config["BOX_SIZE_MIN"] = get_input("Мин. размер звезды (пикс)", config["BOX_SIZE_MIN"], int)
    config["BOX_SIZE_MAX"] = get_input("Макс. размер звезды (пикс)", config["BOX_SIZE_MAX"], int)
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

    print("--------------------------\n")
    print("загрузка видео ща")
    try:
        clip = mpe.VideoFileClip(input_video_path)
    except Exception as e:
        print(f"Ошибка при загрузке видео: {e}")
        print("Проверьте, что путь к файлу указан верно и файл существует.")
        exit()

    print("рисую звезды!")
    
    processing_function = lambda gf, t: process_frame_with_tracking(gf(t)[:,:,::-1], t, config)[:,:,::-1]
    final_clip = clip.fl(processing_function)

    print(f"результат сохранен в {output_video_path}...")
    final_clip.write_videofile(output_video_path, codec='libx264', audio_codec='aac', logger='bar')

if __name__ == '__main__':
    main()
