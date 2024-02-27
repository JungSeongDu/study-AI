import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical

# 데이터셋을 저장할 폴더 생성
def create_dataset_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

# 단일 색상 이미지 데이터 생성 및 저장
def generate_and_save_color_images(color, num_samples=100, size=(32, 32), save_folder="dataset"):
    create_dataset_folder(save_folder)
    
    images = []
    labels = []
    for i in range(num_samples):
        image = np.ones((size[0], size[1], 3)) * np.array(color)
        images.append(image)
        labels.append(color)
        
        # 이미지 저장
        image_filename = os.path.join(save_folder, f"{i}_{color}.png")
        plt.imsave(image_filename, image)

    return np.array(images), np.array(labels)

# 빨간색, 녹색, 파란색 이미지 생성 및 저장
red_images, red_labels = generate_and_save_color_images([1, 0, 0], save_folder="dataset/red")
green_images, green_labels = generate_and_save_color_images([0, 1, 0], save_folder="dataset/green")
blue_images, blue_labels = generate_and_save_color_images([0, 0, 1], save_folder="dataset/blue")

# 데이터셋 생성
images = np.concatenate([red_images, green_images, blue_images], axis=0)
labels = np.concatenate([red_labels, green_labels, blue_labels], axis=0)

# 레이블을 one-hot 인코딩으로 변환
labels_one_hot = to_categorical(labels)

# 데이터셋 확인
print("Images shape:", images.shape)
print("Labels one-hot shape:", labels_one_hot.shape)

# 데이터셋 시각화
plt.figure(figsize=(10, 3))
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i])
    plt.title(f"Label: {labels[i]}")
    plt.axis("off")

plt.show()
