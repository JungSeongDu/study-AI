"""
from PIL import Image
import os
import numpy as np


def create_color_dataset(output_folder='dataset', num_samples_per_class=100):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Define colors and corresponding labels
    colors = {'red': [255, 0, 0], 'green': [0, 255, 0], 'blue': [0, 0, 255]}

    for label, color in colors.items():
        label_folder = os.path.join(output_folder, label)
        if not os.path.exists(label_folder):
            os.makedirs(label_folder)

        for i in range(num_samples_per_class):
            image_data = np.ones((32, 32, 3), dtype=np.uint8) * np.array(color, dtype=np.uint8)
            image = Image.fromarray(image_data)
            image_path = os.path.join(label_folder, f"{i}_{color}.png")
            image.save(image_path)

create_color_dataset()
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 데이터셋을 저장할 폴더 경로
dataset_folder = "dataset"

# 클래스 레이블 정의
class_labels = ["blue", "green", "red"]  # 폴더 정렬 주의!!!!

# 데이터셋 불러오기 및 전처리
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    dataset_folder,
    target_size=(32, 32),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    dataset_folder,
    target_size=(32, 32),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# CNN 모델 정의
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 모델 학습
history = model.fit(train_generator, epochs=10, validation_data=val_generator)

# 정확도 계산 및 출력
test_loss, test_accuracy = model.evaluate(val_generator)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# 훈련 과정 시각화
plt.figure(figsize=(12, 4))

# 훈련 데이터셋의 정확도
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# 훈련 데이터셋의 손실
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# 테스트 이미지 생성 및 예측
test_image = cv2.imread("0_[0, 0, 255].png")  # 테스트 이미지 파일 경로 [빨강,초록,파랑]
if test_image is not None:
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)  # BGR을 RGB로 변환
    test_image = cv2.resize(test_image, (32, 32))  # 이미지 크기 조정
    test_image = np.expand_dims(test_image, axis=0)  # 배치 차원 추가
    test_image = test_image / 255.0  # 이미지 정규화
    prediction = model.predict(test_image)
    predicted_class = class_labels[np.argmax(prediction)]
    print("Predicted Color:", predicted_class)
else:
    print("Error: Unable to read the test image.")
