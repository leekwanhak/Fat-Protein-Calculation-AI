import cv2
import numpy as np

def calculate_masked_pixels(image_path):
    """
    이미지에서 검은색(0, 0, 0)을 제외한 모든 픽셀을 1로 변환하고,
    해당하는 픽셀 수를 계산하는 함수입니다.
    
    Parameters:
        image_path (str): 이미지 파일 경로
        
    Returns:
        int: 검은색이 아닌 부분의 픽셀 수
    """
    # 이미지 불러오기
    image = cv2.imread(image_path)
    
    # 검은색과 일치하는 픽셀은 0, 나머지는 1로 변환
    binary_mask = np.where(np.all(image == [0, 0, 0], axis=-1), 0, 1)
    
    # 검은색이 아닌 부분의 픽셀 수 계산
    non_black_pixels = np.sum(binary_mask)
    
    return non_black_pixels

def merge_rectangles(rect1, rect2):
    """두 개의 직사각형을 감싸는 하나의 큰 직사각형을 생성"""
    # 두 직사각형의 꼭지점들을 모두 가져옴
    box1 = cv2.boxPoints(rect1)
    box2 = cv2.boxPoints(rect2)
    
    # 모든 점들을 하나의 배열로 합침
    all_points = np.vstack((box1, box2))
    
    # 모든 점들을 포함하는 최소 직사각형 찾기
    merged_rect = cv2.minAreaRect(all_points)
    
    return merged_rect

def detect_and_merge_scale_marker(image_path):
    # 이미지 읽기
    image = cv2.imread(image_path)
    # 그레이스케일로 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 노이즈 제거를 위한 가우시안 블러 적용
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Canny 엣지 검출
    edges = cv2.Canny(blurred, 80, 200)
    # 윤곽선 찾기
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 결과 이미지 생성
    result = image.copy()
    
    # 유효한 직사각형들을 저장할 리스트
    valid_rectangles = []
    
    # 각 윤곽선에 대해 처리
    for contour in contours:
        # 윤곽선의 면적 계산
        area = cv2.contourArea(contour)
        # 작은 노이즈 제거
        if area < 100:
            continue
        
        # 윤곽선을 감싸는 최소 직사각형 찾기
        rect = cv2.minAreaRect(contour)
        valid_rectangles.append(rect)
    
    # 직사각형 병합
    if len(valid_rectangles) >= 2:
        merged_rect = merge_rectangles(valid_rectangles[0], valid_rectangles[1])
        for i in range(2, len(valid_rectangles)):
            merged_rect = merge_rectangles(merged_rect, valid_rectangles[i])
        
        # 병합된 직사각형 그리기
        box = cv2.boxPoints(merged_rect)
        box = np.intp(box)
        cv2.drawContours(result, [box], 0, (0, 255, 0), 2)
        
    elif len(valid_rectangles) == 1:
        box = cv2.boxPoints(valid_rectangles[0])
        box = np.intp(box)
        cv2.drawContours(result, [box], 0, (0, 255, 0), 2)
    
    return result, edges, valid_rectangles

def calculate_pixels_per_cm2(reference_image_path):
    """
    스케일 미터 마커가 포함된 이미지를 사용하여 1cm²당 픽셀 수를 계산합니다.
    """
    # 스케일 마커 검출 및 병합
    result_image, edges, valid_rectangles = detect_and_merge_scale_marker(reference_image_path)
    
    # 시각화 (결과 이미지 표시)
    cv2.imshow('Edges', edges)
    cv2.imshow('Merged Scale Marker', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 참조 영역의 픽셀 면적 계산
    reference_area_pixels = sum(cv2.contourArea(cv2.boxPoints(rect)) for rect in valid_rectangles)
    
    # 참조 객체의 실제 크기: 25 cm²
    reference_area_cm2 = 75
    
    # 1cm²당 픽셀 수 계산
    pixels_per_cm2 = reference_area_pixels / reference_area_cm2
    return pixels_per_cm2

def calculate_fat_percentage(fat_mask_pixels, beef_mask_pixels):
    """
    지방의 비율을 계산합니다.
    """
    fat_percentage = (fat_mask_pixels / beef_mask_pixels) * 100
    return fat_percentage

def calculate_mass(area_cm2, thickness_cm=3.81, density_g_per_cm3=0.9):
    """
    지방의 질량을 계산합니다.
    지방의 면적, 두께, 밀도를 곱하여 계산.
    """
    mass = area_cm2 * thickness_cm * density_g_per_cm3
    return mass

def calculate_protein_mass(beef_mass, fat_mass, water_percentage=0.725, mineral_percentage=0.01, vitamin_percentage=0.005):
    """
    고기의 전체 질량에서 지방, 수분, 무기질, 비타민 및 기타 성분을 뺀 단백질 질량을 계산합니다.
    
    Parameters:
        beef_mass (float): 고기의 전체 질량 (g)
        fat_mass (float): 지방 질량 (g)
        water_percentage (float): 수분 비율 (기본값 72.5%)
        mineral_percentage (float): 무기질 비율 (기본값 1%)
        vitamin_percentage (float): 비타민 및 기타 성분 비율 (기본값 0.5%)
    
    Returns:
        float: 단백질 질량 (g)
    """
    # 수분, 무기질, 비타민 및 기타 성분의 질량을 계산
    water_mass = beef_mass * water_percentage
    mineral_mass = beef_mass * mineral_percentage
    vitamin_mass = beef_mass * vitamin_percentage

    # 단백질 질량 = 전체 질량 - (지방 + 수분 + 무기질 + 비타민 및 기타 성분)
    protein_mass = beef_mass - (fat_mass + water_mass + mineral_mass + vitamin_mass)
    
    return protein_mass

# 1. 참조 이미지 경로
reference_image_path = './testpng/test.png'  # 스케일 미터 마커가 있는 이미지

# 2. 1cm²당 픽셀 수 계산
pixels_per_cm2 = calculate_pixels_per_cm2(reference_image_path)
print(f'1cm²당 픽셀 수: {pixels_per_cm2:.2f}')

# 3. 지방 마스크 픽셀 수와 소고기 마스크 픽셀 수 계산
image_path_first = './testpng/maskedtest.png'  # 마스킹된 이미지 파일 경로
image_path_second = './testpng/fat_test.png'  # 마스킹된 이미지 파일 경로
masked_pixels_first = calculate_masked_pixels(image_path_first)
masked_pixels_second = calculate_masked_pixels(image_path_second)
print(f'첫번째 U-net 마스킹된 픽셀 수: {masked_pixels_first}')
print(f'두번째 U-net 마스킹된 픽셀 수: {masked_pixels_second}')

# 4. 지방 비율 계산
fat_percentage = calculate_fat_percentage(masked_pixels_second, masked_pixels_first)
print(f'지방 비율: {fat_percentage:.2f}%')

# 5. 전체 고기 질량
beef_area_cm2 = masked_pixels_first / pixels_per_cm2
beef_mass = calculate_mass(beef_area_cm2)
print(f'전체 고기 질량: {beef_mass:.2f}g')

# 6. 지방 면적 및 지방 질량 계산
fat_area_cm2 = masked_pixels_second / pixels_per_cm2
print(f'지방 면적: {fat_area_cm2:.2f}cm²')

fat_mass = calculate_mass(fat_area_cm2)
print(f'지방 질량: {fat_mass:.2f}g')

# 7. 수분, 무기질, 비타민 및 기타 성분 질량 계산
print(f'수분 질량: {beef_mass * 0.725:.2f}g')
print(f'무기질 질량: {beef_mass * 0.01:.2f}g')
print(f'비타민 및 기타 성분 질량: {beef_mass * 0.005:.2f}g')

# 8. 단백질 질량 계산
protein_mass = calculate_protein_mass(beef_mass, fat_mass)
print(f'단백질 질량: {protein_mass:.2f}g')
