# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 20:18:44 2023

@author: sungd
"""

# 필요한 라이브러리 임포트
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# CIFAR-10 데이터 로드 및 전처리
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 픽셀값 정규화 [0,255] --> [0,1]
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# 레이블을 원-핫 인코딩
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 간단한 CNN 모델 정의
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10개의 클래스에 대한 출력
])

# 모델 컴파일
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 모델 요약
model.summary()

# 모델 학습
history = model.fit(x_train, y_train, epochs=10, validation_split=0.2)

# 테스트 데이터에서 성능 평가
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\nTest Accuracy: {test_acc * 100:.2f}%")

# 학습 과정 시각화
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
