import easyocr # 숫자 인식을 위한 EasyOCR 추가
import numpy as np
import re
import json
import cv2 # 이미지를 자르기(Crop) 위해 필요
from paddleocr import PaddleOCR

# --- 1. (로드) OCR 리더기 준비 ---

# 1-1. PaddleOCR (구조 파악 및 한글 인식용)
try:
    paddle_reader = PaddleOCR(lang='korean', use_angle_cls=False, show_log=False, det_limit_side_len=700)
    print("기본 PaddleOCR 리더기 로드 완료.")
except Exception as e:
    print(f"PaddleOCR 리더기 로드 실패: {e}")
    paddle_reader = None

# 1-2. EasyOCR (숫자 정밀 인식용)
try:
    # gpu=True가 가능하면 True로 설정하세요. 숫자와 화폐단위('원', '$') 인식을 위해 'ko', 'en' 모두 사용
    easy_reader = easyocr.Reader(['ko', 'en'], gpu=True) 
    print("보조 EasyOCR 리더기 로드 완료.")
except Exception as e:
    print(f"EasyOCR 리더기 로드 실패: {e}")
    easy_reader = None

# --- 2. (정의) 유틸리티 함수들 ---

def print_pretty_dict(data_dict):
    """딕셔너리를 JSON 형식으로 예쁘게 출력합니다."""
    try:
        print(json.dumps(data_dict, indent=2, ensure_ascii=False))
    except TypeError as e:
        print(f"출력 중 오류 발생: {e}\n원본 데이터: {data_dict}")

def get_y_center(bbox):
    y1 = bbox[0][1]
    y2 = bbox[2][1]
    return (y1 + y2) / 2

def is_stock_name(text):
    year  = re.search(r'\d+년', text)
    has_valid_chars = re.search(r'([가-힣]{2,}|[A-Za-z]{2,})', text)
    has_forbidden_currency = re.search(r'\d+\s*(?:원|\$)', text)

    if year or (has_valid_chars and not has_forbidden_currency):
        return True
    return False

def is_total_price(text):
    text_stripped = text.strip()
    # PaddleOCR이 읽은 결과로 1차 필터링
    pattern_won = re.compile(r'^[\d,]+원$')
    if pattern_won.search(text_stripped):
        return True
    pattern_dollar = re.compile(r'^\$[\d,]+(\.\d+)?$')
    if pattern_dollar.search(text_stripped):
        return True
    return False

def clean_price_text(price_text):
    text_stripped = price_text.strip()
    currency_code = None 
    number_text = text_stripped

    if text_stripped.endswith('원'):
        currency_code = 'KRW'
        number_text = text_stripped.replace('원', '')
    elif text_stripped.startswith('$'): # EasyOCR은 $가 뒤에 붙을수도 있음 대비
        currency_code = 'USD'
        number_text = text_stripped.replace('$', '')
        if '.' not in number_text and number_text.isdigit():
            # 1230 -> 12.30 (문자열로 변환하여 아래 로직에 넘김)
            number_text = str(float(number_text) / 100)
        else:
            # 소수점이 이미 있거나(12.30) 숫자가 아니면 그대로 둠
            number_text = number_text
    else:
        # EasyOCR이 숫자만 읽고 화폐단위를 놓친 경우 기본 처리 (상황에 따라 수정 가능)
        return None

    try:
        number_text = number_text.replace(',', '').replace(' ', '') # 공백 제거 추가
        if '.' in number_text:
            number_value = float(number_text)
        else:
            number_value = int(number_text)
        return (number_value, currency_code)
    except ValueError:
        return None

def clean_stock_text_list(text_list):
    pattern_ = re.compile(r'^\d+(주|개)$')
    pattern_valid_char = re.compile(r'[가-힣a-zA-Z0-9]')
    
    cleaned_list = []
    for item in text_list:
        item = item.replace('+', '').replace('-', '').replace(':', '')
        if pattern_.fullmatch(item.strip()):
            continue
        if not pattern_valid_char.search(item):
            continue
        cleaned_list.append(item)
    return cleaned_list

def run_easyocr_on_crop(image_cv, bbox):
    """
    원본 이미지와 PaddleOCR의 bbox를 받아, 해당 영역을 Crop한 뒤 EasyOCR을 수행합니다.
    """
    if easy_reader is None:
        return None
        
    try:
        # PaddleOCR bbox: [[x1, y1], [x2, y1], [x2, y2], [x1, y2]] (float일 수 있음)
        # 좌표 정수 변환 및 min/max 계산
        xs = [pt[0] for pt in bbox]
        ys = [pt[1] for pt in bbox]
        
        x_min = int(min(xs))
        x_max = int(max(xs))
        y_min = int(min(ys))
        y_max = int(max(ys))
        
        # 이미지 범위 체크 및 패딩 추가 (OCR 인식률 향상을 위해 주변 여백 줌)
        h, w = image_cv.shape[:2]
        padding = 10
        
        x_min = max(0, x_min - padding)
        x_max = min(w, x_max + padding)
        y_min = max(0, y_min - 5)
        y_max = min(h, y_max + 5)
        
        # 이미지 자르기 (ROI: Region of Interest)
        roi = image_cv[y_min:y_max, x_min:x_max]
        
        # EasyOCR 실행 (detail=0: 텍스트만 리스트로 반환)
        # allowlist를 사용하여 숫자와 화폐 관련 문자만 인식하게 제한하면 정확도가 더 올라갈 수 있음
        # 예: allowlist='0123456789,.원$' 
        result = easy_reader.readtext(roi, detail=0)
        
        if result:
            # 결과가 리스트로 나오므로 합침 (공백 없이 합치거나 상황에 맞게 조절)
            return "".join(result)
        else:
            return None
            
    except Exception as e:
        print(f"Crop EasyOCR 오류: {e}")
        return None

# --- 3. (핵심) 메인 로직 함수 ---

def get_portfolio_from_image(image_cv: np.ndarray):
    if paddle_reader is None:
        return {"error": "PaddleOCR 리더기가 로드되지 않았습니다."}

    print("Step 1: PaddleOCR로 전체 레이아웃 파악 중...")
    
    try:
        raw_results = paddle_reader.ocr(image_cv, cls=False)
        results = raw_results[0] if raw_results else []
        if results is None: results = []
    except Exception as e:
        return {"error": f"PaddleOCR 실행 중 오류: {e}"}
        
    initial_candidates = [] 
    price_candidates = []  

    # (2) 후보군 분류
    for line in results:
        bbox = line[0]
        text_info = line[1]
        text = text_info[0]
        y_center = get_y_center(bbox)
        
        if is_stock_name(text):
            initial_candidates.append({
                'text': text.strip(),
                'y_center': y_center,
                'x': int(bbox[0][0]),
                'bbox' : bbox
            })
            
        elif is_total_price(text):
            # 일단 PaddleOCR 결과를 저장해두지만, 아래에서 EasyOCR로 덮어씌울 예정
            price_candidates.append({
                'text': text.strip(),
                'y_center': y_center,
                'x': int(bbox[0][0]),
                'bbox' : bbox
            })

    # =========================================================================
    # [추가됨] EasyOCR로 가격 정보 재인식 (Refinement)
    # =========================================================================
    if easy_reader:
        print(f"Step 2: 가격 후보 {len(price_candidates)}개에 대해 EasyOCR 정밀 인식 수행...")
        for candidate in price_candidates:
            original_text = candidate['text']
            new_text = run_easyocr_on_crop(image_cv, candidate['bbox'])
            
            if new_text:
                # EasyOCR 결과가 있으면 교체
                print(f"   [보정] '{original_text}' -> '{new_text}'")
                candidate['text'] = new_text
            else:
                print(f"   [유지] '{original_text}' (EasyOCR 인식 실패)")
    # =========================================================================
    
    price_candidates = [price for price in price_candidates if is_total_price(price['text'])]

    candidate_x_coords = [s['x'] for s in initial_candidates]
    if not candidate_x_coords: 
        return {"error": "1차 필터링 후 유효한 종목명 후보가 없습니다."}

    X_MEDIAN = np.median(candidate_x_coords) 
    X_TOLERANCE_LEFT = 10
    X_THRESHOLD = X_MEDIAN - X_TOLERANCE_LEFT
    
    stock_candidates = []
    for candidate in initial_candidates:
        if candidate['x'] >= X_THRESHOLD:
            stock_candidates.append(candidate)

    # (3) Y축 기반 매칭
    portfolio = {}
    Y_TOLERANCE = 20 

    for price in price_candidates:
        best_match_stock = []
        for stock in stock_candidates:
            y_diff = abs(stock['y_center'] - price['y_center'])
            if y_diff <= Y_TOLERANCE:
                best_match_stock.extend(stock['text'].split())

        if best_match_stock:
            # 여기서 EasyOCR로 보정된 텍스트가 파싱됨
            cleaned_price = clean_price_text(price['text'])

            if cleaned_price is not None:
                stock_name = " ".join(clean_stock_text_list(best_match_stock))
                if stock_name.strip():
                    portfolio[stock_name] = cleaned_price

    print(f"최종 {len(portfolio)}개 항목 매칭 성공")
    return portfolio

# --- 4. 테스트 코드 (동일) ---
if __name__ == "__main__":
    import os
    import sys

    TARGET_IMAGE_PATH = "toss_photo\dallor_dark_mode.jpg" 

    print(f"--- 테스트 시작: {TARGET_IMAGE_PATH} ---")

    if not os.path.exists(TARGET_IMAGE_PATH):
        print(f"[오류] 파일을 찾을 수 없습니다: {TARGET_IMAGE_PATH}")
        sys.exit()

    image_cv = cv2.imread(TARGET_IMAGE_PATH)

    if image_cv is None:
        print("[오류] 이미지를 읽는 데 실패했습니다.")
        sys.exit()

    final_result = get_portfolio_from_image(image_cv)

    print("\n" + "="*30)
    print("       [최종 분석 결과]       ")
    print("="*30)
    
    if "error" in final_result:
        print(f"분석 실패: {final_result['error']}")
    elif not final_result:
        print("결과 없음")
    else:
        print_pretty_dict(final_result)
        
    print("="*30)