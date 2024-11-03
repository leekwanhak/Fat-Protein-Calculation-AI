import cv2
import numpy as np

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

def detect_rectangular_object(image_path):
    # 이미지 읽기
    image = cv2.imread(image_path)
    # 그레이스케일로 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 노이즈 제거를 위한 가우시안 블러 적용
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Canny 엣지 검출
    edges = cv2.Canny(blurred, 100, 150)
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
    
    # 직사각형이 2개 이상인 경우 병합
    if len(valid_rectangles) >= 2:
        # 첫 번째와 두 번째 직사각형 병합
        merged_rect = merge_rectangles(valid_rectangles[0], valid_rectangles[1])
        # 나머지 직사각형들도 순차적으로 병합
        for i in range(2, len(valid_rectangles)):
            merged_rect = merge_rectangles(merged_rect, valid_rectangles[i])
            
        # 병합된 직사각형 그리기
        box = cv2.boxPoints(merged_rect)
        box = np.int0(box)
        cv2.drawContours(result, [box], 0, (0, 255, 0), 2)
    
    # 직사각형이 1개인 경우 해당 직사각형만 그리기
    elif len(valid_rectangles) == 1:
        box = cv2.boxPoints(valid_rectangles[0])
        box = np.int0(box)
        cv2.drawContours(result, [box], 0, (0, 255, 0), 2)
    
    return result, edges

def main():
    image_path = 'test.png'  # 이미지 경로를 지정해주세요
    result_image, edges = detect_rectangular_object(image_path)
    
    # 결과 표시
    cv2.imshow('Original with Detections', result_image)
    cv2.imshow('Edges', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()