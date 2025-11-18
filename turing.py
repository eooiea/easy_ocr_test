import cv2
import numpy as np
import easyocr

# --- 1. OCR 리더 로딩 (한 번만 실행) ---
print("EasyOCR 모델을 로딩 중입니다... (잠시 기다려주세요)")
reader = easyocr.Reader(['ko', 'en']) 
print("모델 로딩 완료.")

# --- 2. 이미지 로드 ---
IMAGE_PATH = 'toss_photo/pc_dark_mode.png' # <-- 원본 이미지
img_gray = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)

if img_gray is None:
    print(f"오류: '{IMAGE_PATH}' 파일을 찾을 수 없습니다.")
    exit()

# --- 3. (수정) 전처리 파라미터 고정 ---
# (이 값들은 예시이며, light_mode3.jpg에 맞게 조절이 필요합니다.)

# [새로 추가] 이미지 확대 배율
# 1.0 = 원본 크기, 2.0 = 2배 확대
# 2배 확대 시, ksize와 blockSize도 2배로 키우는 것을 권장합니다.
scale_factor = 2.3 

ksize = 1       # Gaussian Blur 커널 크기 (0이 아닌 홀수)
blockSize = 43  # Adaptive Threshold 영역 크기 (11 -> 21로 2배 증가 예시)
C = -10       # Adaptive Threshold 민감도

# --- 4. (수정) 전처리 실행 ---
print(f"전처리 실행 (확대={scale_factor}x, ksize={ksize}, blockSize={blockSize}, C={C})...")

# 1단계: [추가] 이미지 확대 (글씨 키우기)
# (0, 0)은 dsize를 scale_factor에 따라 자동 계산하라는 의미
# interpolation=cv2.INTER_CUBIC : 이미지를 확대할 때 부드럽게 처리
img_resized = cv2.resize(
    img_gray, 
    (0, 0), 
    fx=scale_factor, 
    fy=scale_factor, 
    interpolation=cv2.INTER_CUBIC
)

# 2단계: Gaussian Blur (흐림 처리) - 확대된 이미지에 적용
img_blur = cv2.GaussianBlur(img_resized, (ksize, ksize), 0)

# 3단계: Adaptive Threshold (적응형 이진화)
img_binary = cv2.adaptiveThreshold(
    img_blur,
    255, # 최대값
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, # 방식
    cv2.THRESH_BINARY,              # 라이트 모드용 옵션
    blockSize,
    C
)

# (디버깅용) 전처리된 이미지를 확인하고 싶다면 아래 주석을 푸세요.
cv2.imwrite('preprocessed_final_scaled.png', img_binary)

# --- 5. OCR 실행 및 결과 출력 ---
print("\n--- OCR 실행 ---")
try:
    results = reader.readtext(img_binary)
    
    # 결과 출력
    if not results:
        print("인식된 텍스트가 없습니다.")
        
    for (bbox, text, prob) in results:
        print(f"인식된 텍스트: {text} (신뢰도: {prob:.4f})")
    
    print("--- OCR 완료 ---")

except Exception as e:
    print(f"OCR 실행 중 오류 발생: {e}")