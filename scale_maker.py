import cv2
import numpy as np

def merge_rectangles(rect1, rect2):
    # 두 개의 직사각형을 병합하는 함수 (구현 필요)
    pass

def is_valid_rectangle(rect):
    # 직사각형의 유효성을 확인하는 함수
    (center, (width, height), angle) = rect
    if width > 10 and height > 10:  # 너비와 높이가 10보다 큰지 확인
        return True
    return False

def detect_rectangular_object(image_path):
    # 이미지 읽기
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # 윤곽선 찾기
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    valid_rectangles = []
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        if is_valid_rectangle(rect):  # 유효한 직사각형인지 확인하는 함수
            valid_rectangles.append(rect)

    result = image.copy()
    # 유효한 직사각형을 이미지에 그리기 (구현 필요)
    for rect in valid_rectangles:
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(result, [box], 0, (0, 0, 255), 2)

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
