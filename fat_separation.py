import os
import numpy as np
from PIL import Image

# 원본 이미지 경로 설정
img_path = 'test.png'

# 마스크 이미지 경로 설정
mask_path = 'maskedtest.png'

# 이미지 열기 (컬러 이미지로)
image = Image.open(img_path).convert("RGB")

# 마스크 이미지 열기 (그레이스케일로 변환)
mask_image = Image.open(mask_path).convert("L")   

# 임계값 설정 (지방만 흰색으로 나타나도록 높게 설정)
threshold = 130  # 필요에 따라 조정하세요

# 이미지를 NumPy 배열로 변환
image_np = np.array(image)  # shape: (height, width, 3)
mask_np = np.array(mask_image)  # shape: (height, width)

# 마스크 적용: 마스크 영역이 255인 부분만 남기기
# 마스크를 3차원으로 확장하여 컬러 채널에 적용
mask_np_expanded = np.expand_dims(mask_np == 255, axis=2)
masked_np = np.where(mask_np_expanded, image_np, 0).astype(np.uint8)

# 원본 이미지의 파일 이름 추출
filename = os.path.basename(img_path)  # 예: '1.jpg'

# NumPy 배열을 다시 이미지로 변환
masked_image = Image.fromarray(masked_np)

# 저장할 디렉토리 경로 설정
save_data_dir = 'fat_separation/data/'

# 저장 경로 생성 (디렉토리가 존재하지 않으면 생성)
os.makedirs(save_data_dir, exist_ok=True)

# 최종 저장 경로 설정
save_data_path = os.path.join(save_data_dir, filename)

# 결과 이미지 저장
masked_image.save(save_data_path)

print(f"마스크 이미지가 '{save_data_path}'로 저장되었습니다.")

# 그레이스케일 이미지로 변환하여 임계값 적용
gray_image = image.convert("L")
gray_image_np = np.array(gray_image)

# 임계값 적용: threshold 이상이면 255(흰색), 아니면 0(검은색)
binary_np = np.where(gray_image_np >= threshold, 255, 0).astype(np.uint8)

# 마스크를 사용하여 배경을 검은색으로 설정
binary_np[mask_np == 0] = 0

# NumPy 배열을 다시 이미지로 변환
binary_image = Image.fromarray(binary_np)

# 저장할 디렉토리 경로 설정
save_label_dir = 'fat_separation/label/'

# 저장 경로 생성 (디렉토리가 존재하지 않으면 생성)
os.makedirs(save_label_dir, exist_ok=True)

# 최종 저장 경로 설정
save_label_path = os.path.join(save_label_dir, filename)

# 결과 이미지 저장
binary_image.save(save_label_path)

print(f"이진화된 이미지가 '{save_label_path}'로 저장되었습니다.")