import os
import threading
from collections import deque, Counter
import pygame            # ← 已经导入 pygame
import cv2
import dlib
import numpy as np
import joblib

# ========= 配置 =========
model_path = "random_forest_emotion_model_with_geo.pkl"
shape_predictor_path = "shape_predictor_68_face_landmarks.dat"
classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
EFFECT_DIR = r"D:\face_expression_zhang\effects"  # 效果集
FILTER_MAP = {
    "angry":     "warm",     # 红橙调，带一点曝光
    "disgust":   "green",    # 绿色偏移
    "fear":      "purple",   # 冷紫色
    "happy":     "bright",   # 提亮 + 对比度
    "neutral":   "none",     # 不加滤镜
    "sad":       "cool",     # 冰蓝调
    "surprise":  "cartoon"   # 卡通漫画
}
current_filter = "none"
BUFFER_SIZE = 12
CHANGE_THRESHOLD = 8       # 避免一直切换
history = deque(maxlen=BUFFER_SIZE)
current_effect = None

# ========= 初始化 pygame 音频 =========
# 采样率 / 声道数 使用默认值即可，如需更改可在 init() 里指定
pygame.mixer.init()

# ========= 加载模型和人脸检测器 =========
clf = joblib.load(model_path)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor_path)

# ========= 资源加载（图片 + 声音）=========
def apply_filter(frame, mode="none"):
    if mode == "warm":         # 红橙调
        overlay = np.full_like(frame, (0, 40, 120))   # BGR
        return cv2.addWeighted(frame, 0.7, overlay, 0.3, 10)

    elif mode == "green":      # 绿色偏移
        overlay = np.full_like(frame, (0, 80, 0))
        return cv2.addWeighted(frame, 0.7, overlay, 0.3, 5)

    elif mode == "purple":     # 紫蓝冷调
        overlay = np.full_like(frame, (80, 0, 80))
        return cv2.addWeighted(frame, 0.75, overlay, 0.25, 0)

    elif mode == "bright":     # 提亮&增强对比度
        return cv2.convertScaleAbs(frame, alpha=1.4, beta=25)

    elif mode == "cool":       # 冰蓝调
        overlay = np.full_like(frame, (120, 30, 0))
        return cv2.addWeighted(frame, 0.75, overlay, 0.25, 0)

    elif mode == "cartoon":    # 卡通化
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur  = cv2.medianBlur(gray, 7)
        edges = cv2.adaptiveThreshold(
                    blur, 255,
                    cv2.ADAPTIVE_THRESH_MEAN_C,
                    cv2.THRESH_BINARY, 9, 9)
        color = cv2.bilateralFilter(frame, 9, 150, 150)
        return cv2.bitwise_and(color, color, mask=edges)

    else:                      # "none" 或未知
        return frame
def load_effect_assets(effect_dir):
    """
    返回:
        image_dict[emotion] -> BGRA png (cv2)
        sound_dict[emotion] -> pygame.mixer.Sound 对象
    """
    image_dict, sound_dict = {}, {}
    for emotion in classes:
        img_path = os.path.join(effect_dir, f"{emotion}.png")
        wav_path = os.path.join(effect_dir, f"{emotion}.wav")

        if os.path.exists(img_path):
            image_dict[emotion] = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        if os.path.exists(wav_path):
            # 预加载为 Sound 对象，后面播放更快
            sound_dict[emotion] = pygame.mixer.Sound(wav_path)

    return image_dict, sound_dict

effect_images, effect_sounds = load_effect_assets(EFFECT_DIR)

# ========= 播放音效（异步）=========
def play_sound_async(sound: pygame.mixer.Sound):
    """在新线程里播放，避免阻塞主循环"""
    def _play():
        channel = sound.play()
        # 等待声音播完再结束线程，防止被提前 GC
        while channel.get_busy():
            pygame.time.wait(10)
    threading.Thread(target=_play, daemon=True).start()

# ========= 叠加 PNG（含透明通道）=========
def overlay_image_alpha(img, overlay, x, y, overlay_size=None):
    if overlay_size is not None:
        overlay = cv2.resize(overlay, overlay_size, interpolation=cv2.INTER_AREA)
    h, w = overlay.shape[:2]
    if x < 0 or y < 0 or x + w > img.shape[1] or y + h > img.shape[0]:
        return  # 防止越界
    b, g, r, a = cv2.split(overlay)
    mask = a / 255.0
    overlay_rgb = cv2.merge((b, g, r))
    roi = img[y:y+h, x:x+w]
    blended = (1.0 - mask[..., None]) * roi + mask[..., None] * overlay_rgb
    img[y:y+h, x:x+w] = blended.astype(np.uint8)

# ========= 特征提取函数 =========
def preprocess(coords):
    # ……（此处保持你的原实现不变）
    selected_indices = np.array([*range(17, 27), *range(27, 36),
                                 *range(36, 48), *range(48, 68)])
    selected_coords = coords[selected_indices]
    left_eye_center = np.mean(coords[36:42], axis=0)
    right_eye_center = np.mean(coords[42:48], axis=0)
    eye_center = (left_eye_center + right_eye_center) / 2
    eye_distance = np.linalg.norm(left_eye_center - right_eye_center)
    centered = selected_coords - eye_center
    normalized = centered / eye_distance if eye_distance != 0 else centered

    # ------- 几何特征 -------
    def polyline_length(points):
        return np.sum(np.linalg.norm(points[1:] - points[:-1], axis=1))
    left_eyebrow = coords[17:22]
    right_eyebrow = coords[22:27]
    eyebrow_length = polyline_length(left_eyebrow) + polyline_length(right_eyebrow)
    nose_length = np.linalg.norm(coords[33] - coords[27])
    mouth_width = np.linalg.norm(coords[54] - coords[48])
    upper_lip = np.mean(coords[50:53], axis=0)
    lower_lip = np.mean(coords[56:59], axis=0)
    mouth_height = np.linalg.norm(upper_lip - lower_lip)
    left_eye_width = np.linalg.norm(coords[39] - coords[36])
    right_eye_width = np.linalg.norm(coords[45] - coords[42])
    eye_width = (left_eye_width + right_eye_width) / 2
    left_eye_height = (np.linalg.norm(coords[37] - coords[41]) +
                       np.linalg.norm(coords[38] - coords[40])) / 2
    right_eye_height = (np.linalg.norm(coords[43] - coords[47]) +
                        np.linalg.norm(coords[44] - coords[46])) / 2
    eye_height = (left_eye_height + right_eye_height) / 2
    left_eyebrow_curve = np.linalg.norm(coords[19] - (coords[17] + coords[21]) / 2)
    right_eyebrow_curve = np.linalg.norm(coords[24] - (coords[22] + coords[26]) / 2)
    inner_mouth_height = np.linalg.norm(coords[62] - coords[66])
    nose_width = np.linalg.norm(coords[31] - coords[35])
    left_eye_openness = np.mean([np.linalg.norm(coords[37]-coords[41]),
                                 np.linalg.norm(coords[38]-coords[40])])
    right_eye_openness = np.mean([np.linalg.norm(coords[43]-coords[47]),
                                  np.linalg.norm(coords[44]-coords[46])])

    geo_feats = np.array([
        eyebrow_length, nose_length, mouth_width, mouth_height, eye_width,
        eye_height, left_eyebrow_curve, right_eyebrow_curve,
        inner_mouth_height, nose_width, left_eye_openness,
        right_eye_openness
    ]) / eye_distance if eye_distance != 0 else np.zeros(12)

    return np.concatenate([normalized.flatten(), geo_feats])

# ========= 摄像头循环 =========
cap = cv2.VideoCapture(0)
print("按 'q' 退出")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        shape = predictor(gray, face)
        coords = np.array([[p.x, p.y] for p in shape.parts()])  # (68, 2)

        feature = preprocess(coords).reshape(1, -1)
        pred = clf.predict(feature)[0]
        label = classes[pred]

        history.append(label)
        most_common_label, freq = Counter(history).most_common(1)[0]

        if freq >= CHANGE_THRESHOLD and most_common_label != current_effect:
            current_effect = most_common_label
            current_filter = FILTER_MAP.get(current_effect, "none")  # ← 新增
            if current_effect in effect_sounds:
                play_sound_async(effect_sounds[current_effect])
            print(f"[INFO] 切换特效 -> {current_effect} | 滤镜 -> {current_filter}")

        # ---- 画框 + 文字 ----
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        # ---- 叠加 PNG 特效 ----
        if current_effect in effect_images:
            overlay_img = effect_images[current_effect]
            face_width = x2 - x1
            overlay_w = face_width
            overlay_h = int(overlay_img.shape[0] * (overlay_w / overlay_img.shape[1]))
            overlay_x = x1
            overlay_y = y1 - overlay_h   # 放在脸上方
            overlay_image_alpha(frame, overlay_img, overlay_x, overlay_y,
                                (overlay_w, overlay_h))

    frame = apply_filter(frame, current_filter)
    cv2.imshow("Real-time Emotion Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()