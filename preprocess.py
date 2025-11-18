import cv2
import easyocr
import numpy as np
import matplotlib.pyplot as plt

def preprocess_for_ocr(image_path):
    """
    OCR 성능 향상을 위한 전처리 파이프라인
    """
    # 1. 이미지 읽기 (그레이스케일로)
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"이미지를 읽을 수 없습니다. 경로를 확인하세요: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. 이미지 스케일링 (확대)
    # 보간법(interpolation)으로 INTER_CUBIC을 사용하여 확대 시 퀄리티 유지
    scale_factor = 2.0
    width = int(gray.shape[1] * scale_factor)
    height = int(gray.shape[0] * scale_factor)
    dim = (width, height)
    
    resized_gray = cv2.resize(gray, dim, interpolation=cv2.INTER_CUBIC)

    # 3. 적응형 이진화 (Adaptive Thresholding)
    # blockSize: 픽셀 주변을 얼마만큼 볼 것인지 (홀수여야 함)
    # C: 계산된 평균(또는 가우시안 평균)에서 뺄 값 (양수면 어두워짐)
    # 이 두 파라미터를 조절하며 최적의 값을 찾아야 합니다.
    
    # 배경이 어둡고 글자가 밝은 경우 (Dark Mode)
    # cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=15, C=2
    
    # 배경이 밝고 글자가 어두운 경우 (Light Mode) - 일반적인 경우
    # cv2.THRESH_BINARY_INV는 흑/백 반전
    processed_image = cv2.adaptiveThreshold(
        resized_gray, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, # MEAN_C 보다 GASSUIAN_C가 경계선을 부드럽게 잡아줄 때가 많음
        cv2.THRESH_BINARY_INV, # 배경이 밝고(255) 글자가 어둡게(0) 되도록 반전
        blockSize=11, 
        C=2
    )

    # --- 실험해볼 모폴로지 연산 (선택 사항) ---
    # 커널: 연산을 적용할 영역 크기
    kernel = np.ones((2, 2), np.uint8)

    # 4-A. '홀' -> '혼' 문제 해결 시도 (획을 두껍게)
    # 얇은 'ㄹ' 획이 끊어지는 것을 방지
    # processed_image = cv2.dilate(processed_image, kernel, iterations=1)

    # 4-B. '$' -> '8' 문제 해결 시도 (획을 얇게)
    # 붙어있는 'S'와 '|'를 분리 시도
    # processed_image = cv2.erode(processed_image, kernel, iterations=1)

    return processed_image

# --- 실행 부분 ---
if __name__ == "__main__":
    # 1. EasyOCR 리더기 초기화 (한번만 실행)
    reader = easyocr.Reader(['ko', 'en'])
    
    # 2. 전처리할 이미지 경로 (상대 경로 사용)
    IMAGE_PATH = 'toss_photo/dallor_dark_mode.jpg' # <-- 여기에 실제 이미지 경로를 입력하세요.

    # 3. 원본 이미지로 OCR 실행
    print("--- [1] 원본 이미지 OCR 결과 ---")
    try:
        original_result = reader.readtext(IMAGE_PATH)
        for (bbox, text, prob) in original_result:
            print(f'Text: {text}, Confidence: {prob:.4f}')
    except Exception as e:
        print(f"원본 이미지 처리 중 오류: {e}")

    # 4. 전처리된 이미지로 OCR 실행
    print("\n--- [2] 전처리된 이미지 OCR 결과 ---")
    try:
        preprocessed_img = preprocess_for_ocr(IMAGE_PATH)
        
        # 참고: EasyOCR은 이미지 경로뿐만 아니라 numpy 배열(이미지 자체)도 입력으로 받습니다.
        preprocessed_result = reader.readtext(preprocessed_img)
        
        for (bbox, text, prob) in preprocessed_result:
            print(f'Text: {text}, Confidence: {prob:.4f}')

        # 5. (선택) 전처리된 이미지 저장해서 눈으로 확인
        cv2.imwrite('preprocessed_output.png', preprocessed_img)
        print("\n전처리된 이미지가 'preprocessed_output.png'로 저장되었습니다.")
        
    except Exception as e:
        print(f"전처리 이미지 처리 중 오류: {e}")