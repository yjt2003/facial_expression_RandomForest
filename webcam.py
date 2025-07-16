import cv2
import dlib
import numpy as np
import joblib
import os
import threading
from collections import deque, Counter
from playsound import playsound

# ========= 配置 ========= #
model_path = "random_forest_emotion_model.pkl"
shape_predictor_path = "shape_predictor_68_face_landmarks.dat"
classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
EFFECT_DIR = "D:\\face_expression_zhang\\effects"  # 效果集----------

# 缓冲配置（防抖）
BUFFER_SIZE = 12
CHANGE_THRESHOLD = 8     #避免一直切换-----------
history = deque(maxlen=BUFFER_SIZE)
current_effect = None

# ========== 加载模型 ==========
clf = joblib.load(model_path)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor_path)

# ========== 加载贴图和音效 ==========
def load_effect_assets(effect_dir):
    image_dict, sound_dict = {}, {}
    for emotion in classes:
        img_path = os.path.join(effect_dir, f"{emotion}.png")
        wav_path = os.path.join(effect_dir, f"{emotion}.wav")
        if os.path.exists(img_path):
            image_dict[emotion] = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # BGRA
        if os.path.exists(wav_path):
            sound_dict[emotion] = wav_path
    return image_dict, sound_dict

effect_images, effect_sounds = load_effect_assets(EFFECT_DIR)

# ========== 播放音效（异步） ==========
def play_sound_async(path):
    threading.Thread(target=playsound, args=(path,), daemon=True).start()

# ========== 图像叠加 ==========
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

# ========== 特征预处理 ==========
def preprocess(coords):
    coords = coords - np.mean(coords, axis=0)
    max_dist = np.linalg.norm(coords, axis=1).max()
    coords = coords / max_dist if max_dist != 0 else coords
    return coords.flatten()

# ========== 打开摄像头 ==========
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        shape = predictor(gray, face)
        coords = np.array([[p.x, p.y] for p in shape.parts()])
        processed = preprocess(coords).reshape(1, -1)

        pred = clf.predict(processed)[0]
        label = classes[pred]

        # 更新历史记录
        history.append(label)
        most_common_label, freq = Counter(history).most_common(1)[0]

        # 是否切换特效
        if freq >= CHANGE_THRESHOLD and most_common_label != current_effect:
            current_effect = most_common_label
            if current_effect in effect_sounds:
                play_sound_async(effect_sounds[current_effect])
            print(f"[INFO] 切换特效 -> {current_effect}")

        # 绘制矩形 + 标签
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # 添加贴图
        if current_effect in effect_images:
            overlay_img = effect_images[current_effect]
            face_width = x2 - x1
            overlay_w = face_width
            overlay_h = int(overlay_img.shape[0] * (overlay_w / overlay_img.shape[1]))
            overlay_x = x1
            overlay_y = y1 - overlay_h  # 放在脸上方
            overlay_image_alpha(frame, overlay_img, overlay_x, overlay_y, (overlay_w, overlay_h))

    cv2.imshow("Emotion Recognition - Random Forest", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
