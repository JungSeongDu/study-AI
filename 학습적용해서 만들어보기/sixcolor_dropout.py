# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 09:10:37 2024

@author: sungd
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 21:03:40 2023

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
    Dense(32, activation='relu',kernel_initializer='he_normal'),
    Dropout(0.2),  # Add dropout with a dropout rate of 0.5
                   # 30%의 뉴런을 무작위로 비활성화하
    Dense(len(class_labels), activation='softmax')
])
                        #매개변수 갱신방법
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 모델 학습
history = model.fit(train_generator, epochs=15, validation_data=val_generator)

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

test_image = cv2.imread("./testcolor/test_19.jpg")
if test_image is not None:
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)  # BGR을 RGB로 변환
    test_image = cv2.resize(test_image, (100, 100))  # 이미지 크기 조정
    test_image = np.expand_dims(test_image, axis=0)  # 배치 차원 추가
    test_image = test_image / 255.0  # 이미지 정규화
    prediction = model.predict(test_image)
    predicted_class = class_labels[np.argmax(prediction)]
    print("Predicted Color >>>>>>> :", predicted_class, '!!!!!!!')
else:
    print("Error: Unable to read the test image.")

# accuracy: 0.8142 - val_loss: 0.2935 - val_accuracy: 0.9583
#모델의 훈련 과정 중에 각 에폭(epoch)에서의 정확도(accuracy)와 손실(loss) 
#그리고 검증 데이터셋에 대한 정확도와 손실

"""

accuracy (정확도): 모델이 훈련 데이터셋에 대해 올바르게 예측한 비율입니다. 
예를 들어, accuracy가 0.8615이라면 86.15%의 훈련 데이터에 대해 정확하게 예측했다는 의미입니다.

loss (손실): 모델이 훈련 데이터셋에 대해 얼마나 정확하게 예측하지 못했는지를 나타내는 지표입니다. 
손실이 낮을수록 모델이 훈련 데이터에 더 잘 적합되었다고 볼 수 있습니다.

val_accuracy (검증 정확도): 모델이 검증 데이터셋에 대해 얼마나 정확하게 예측하는지를 나타냅니다. 
훈련 데이터와는 다른 데이터에 대한 성능을 평가하는데 사용됩니다. 예를 들어, val_accuracy가 0.9583이라면 95.83%의 검증 데이터에 대해 정확하게 예측했다는 의미입니다.

val_loss (검증 손실): 모델이 검증 데이터셋에 대해 얼마나 정확하게 예측하지 못했는지를 나타내는 지표입니다. 
검증 손실이 낮을수록 모델이 검증 데이터에 더 잘 적합되었다고 볼 수 있습니다.

"""

"""

Training Loss (훈련 손실): 모델이 훈련 데이터에 대해 얼마나 정확하게 예측하지 못했는지를 나타내는 지표입니다. 
훈련 손실이 감소하면, 모델은 훈련 데이터에 더 잘 적합되고 있는 것이라고 볼 수 있습니다. 하지만, 훈련 손실만 줄어든다고 해서 항상 모델이 더 좋아진다고 말할 수는 없습니다. 
과적합(overfitting)의 가능성이 있으므로 주의가 필요합니다.

Validation Loss (검증 손실): 모델이 훈련 데이터 이외의 새로운 데이터에 대해 얼마나 정확하게 예측하는지를 나타냅니다. 
검증 손실이 감소하는 것은 모델이 훈련 데이터 이외의 데이터에도 잘 일반화되고 있다는 것을 의미합니다. 
모델이 검증 데이터에 대해 일반화되지 않고 과적합되고 있다면, 검증 손실은 오히려 증가할 수 있습니다.

"""