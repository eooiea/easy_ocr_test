import cv2
from ocr_logic import get_portfolio_from_image # 핵심 로직 임포트

print("--- 디버깅 테스트 시작 ---")

# (1) 테스트할 실제 스크린샷 이미지 경로
# (★중요★) 본인의 스크린샷 파일 경로로 변경하세요
test_image_path = "C:/Coding/easy_ocr_test/toss_photo/dallor_dark_mode.jpg" 

image = cv2.imread(test_image_path)

if image is None:
    print(f"이미지 로드 실패! 경로 확인: {test_image_path}")
else:
    # (2) 핵심 함수 테스트
    print(f"'{test_image_path}' 파일로 파싱 테스트 시작...")
    
    portfolio_result = get_portfolio_from_image(image)
    
    print("\n--- 테스트 종료 ---")
    print("최종 파싱 결과 (딕셔너리):")
    
    # (3) 결과 예쁘게 출력
    import json
    print(json.dumps(portfolio_result, indent=2, ensure_ascii=False))