# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 16:37:52 2024

@author: sungd
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping  # EarlyStopping 임포트

# 데이터셋을 저장할 폴더 경로
dataset_folder = "mood_1"

# 클래스 레이블 정의
class_labels = sorted(os.listdir(dataset_folder))

# 데이터셋 불러오기 및 전처리
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    dataset_folder,
    target_size=(100, 100),
    batch_size=100,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    dataset_folder,
    target_size=(100, 100),
    batch_size=100,
    class_mode='categorical',
    subset='validation'
)

# CNN 모델 정의
model = Sequential([
    Conv2D(100, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(len(class_labels), activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Early Stopping 콜백 정의
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# 모델 학습 with Early Stopping
history = model.fit(train_generator, epochs=10, validation_data=val_generator, callbacks=[early_stopping])

# 정확도 계산 및 출력
test_loss, test_accuracy = model.evaluate(val_generator)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# 훈련 과정 시각화
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

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
test_image = cv2.imread("./testcolor/use-lighting-to-change-room-mood-1240x828.jpe")
if test_image is not None:
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    test_image = cv2.resize(test_image, (100, 100))
    test_image = np.expand_dims(test_image, axis=0)
    test_image = test_image / 255.0
    prediction = model.predict(test_image)
    predicted_class = class_labels[np.argmax(prediction)]
    print("Predicted Color >>>>>>> :", predicted_class, '!!!!!!!')
else:
    print("Error: Unable to read the test image.")
