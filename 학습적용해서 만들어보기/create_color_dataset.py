from PIL import Image
import os
import numpy as np

def create_color_dataset(output_folder='dataset', num_samples_per_class=200):
    # 출력 폴더가 없으면 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 색상 및 해당 레이블 정의
    colors = {
        'red': [255, 0, 0], 
        'green': [0, 255, 0], 
        'blue': [0, 0, 255],
        'black': [0, 0, 0], 
        'white': [255, 255, 255], 
        'brown': [139, 69, 19]
    }

    # 각 색상에 대해 여러 톤의 이미지 생성
    for label, color in colors.items():
        label_folder = os.path.join(output_folder, label)
        if not os.path.exists(label_folder):
            os.makedirs(label_folder)

        for i in range(num_samples_per_class):
            # 순차적으로 톤을 증가시키기
            tone_variation = np.array([i - num_samples_per_class // 15, i - num_samples_per_class // 15, i - num_samples_per_class // 15])

            # 색상에 순차적으로 톤을 더하여 새로운 컬러 생성
            toned_color = np.clip(np.array(color) + tone_variation, 0, 255).astype(np.uint8)

            image_data = np.ones((32, 32, 3), dtype=np.uint8) * toned_color
            image = Image.fromarray(image_data)
            
            # 이미지 저장
            image_path = os.path.join(label_folder, f"{i}_{label}_{tone_variation}.png")
            image.save(image_path)

create_color_dataset()
