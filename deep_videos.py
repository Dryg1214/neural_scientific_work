"""
import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from moviepy.editor import VideoFileClip
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

import numpy as np

# Загрузка предобученной модели
model = resnet50(pretrained=True)

model_path = 'ffpp_c40.pth'
device = torch.device('cpu')  # Устанавливаем устройство на CPU
model = torch.load(model_path, map_location=device)



# Определение функции классификации кадра
def classify_frame(frame):
    # Предобработка изображения
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Преобразование кадра в изображение
    image = transform(frame).unsqueeze(0)

    # Предсказание с помощью модели
    with torch.no_grad():
        prediction = model(image)

    # Определение класса (0 - реальное изображение, 1 - Deepfake)
    _, predicted_class = torch.max(prediction, 1)

    return predicted_class.item()

# Путь к исходному видео и выходному видео
input_video_path = 'deepfakevideo.mp4'
output_video_path = 'outputdeepvideo.mp4'


# Создание функции для обработки каждого кадра видео
def process_frame(frame):
    # Преобразование кадра в массив NumPy
    frame_array = np.array(frame)

    # Классификация текущего кадра
    class_id = classify_frame(frame_array)

    # Определение текста для надписи
    label = 'Real' if class_id == 0 else 'Fake'

    # Добавление надписи на кадр
    frame_with_text = frame_array.copy()
    cv2.putText(frame_with_text, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame_with_text

# Загрузка исходного видео
clip = VideoFileClip(input_video_path)


# Создание нового видео с обработанными кадрами
processed_frames = [process_frame(frame) for frame in clip.iter_frames()]
processed_clip = ImageSequenceClip(processed_frames, fps=clip.fps)

# Сохранение обработанного видео
processed_clip.write_videofile(output_video_path, codec='libx264', audio=False)

# Очистка ресурсов
clip.reader.close()
clip.audio.reader.close_proc()
"""

import torch
import torchvision.transforms as transforms
import cv2
import torchvision.models as models

# Загрузка предобученной модели
#model_path = 'resnet50-19c8e357.pth'
model_path = 'gandetection_resnet50nodown_stylegan2.pth'


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_state_dict = torch.load(model_path, map_location=device)

# Создание экземпляра модели
model = models.resnet50()  # Замените resnet50 на вашу модель
model.load_state_dict(model_state_dict)
model.to(device)
model.eval()

# Задание преобразований для входных изображений модели
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Загрузка видео
video_path = 'deepfakevideo.mp4'
cap = cv2.VideoCapture(video_path)

# Создание выходного видео
output_path = 'output_video.mp4'
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Обработка кадров видео
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Преобразование кадра в тензор
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = transform(frame).unsqueeze(0).to(device)
    
    # Подача тензора через модель
    threshold = 0.5  # Пороговое значение для классификации

# ...

    with torch.no_grad():
        output = model(tensor)

    max_value, max_index = torch.max(output, dim=1)
    is_fake = True if max_value.item() > threshold else False

    # Добавление надписи к кадру
    label = 'Fake' if is_fake else 'Real'
    frame = cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Запись кадра в выходное видео
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    out.write(frame)
    
    # Отображение обработанного кадра
    cv2.imshow('Processed Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
cap.release()
out.release()
cv2.destroyAllWindows()