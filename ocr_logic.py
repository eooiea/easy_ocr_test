import easyocr
import cv2
import numpy as np
import re # 정규표현식(Regex)을 위한 라이브러리
import json

# --- 1. (로드) 기본 EasyOCR 리더기 준비 ---
# 이 스크립트가 임포트될 때 한 번만 실행됩니다.
# (나중에 모델을 바꾸려면 이 부분만 수정하면 됩니다.)
try:
    reader = easyocr.Reader(['ko', 'en']) 
    print("기본 EasyOCR 리더기 로드 완료.")
except Exception as e:
    print(f"EasyOCR 리더기 로드 실패: {e}")
    reader = None





# --- 2. (정의) 유틸리티 함수들 ---


def get_y_center(bbox):
    """EasyOCR 바운딩 박스(bbox)의 Y축 중심 좌표를 반환합니다."""
    # bbox는 [[x1, y1], [x2, y1], [x2, y2], [x1, y2]] 형태입니다.
    y1 = bbox[0][1] # 상단 Y
    y2 = bbox[2][1] # 하단 Y
    return (y1 + y2) / 2


def is_stock_name(text):
    """
    이 텍스트가 '종목명'일 가능성이 있는지 정규표현식으로 확인합니다.
    (★해커톤에서 계속 수정해야 할 부분 1★)
    """
    # 조건: 
    # 1. 최소 2글자 이상
    # 2. 한글, 영어, 숫자가 포함될 수 있음
    # 3. '원', '%', '+' 등 가격/수익률 관련 문자가 없음
    # 4. '평가금액', '총자산' 등 헤더/타이틀이 아님 (-> 이것은 나중에 Y좌표로 거름)
    
    # [가-힣A-Za-z0-9] -> 한글, 영어, 숫자
    # {2,} -> 2글자 이상
    if re.search(r'[가-힣A-Za-z0-9]{2,}', text) and not re.search(r'[%$원]', text): # %나 원이 포함되지 않음
        return True
    return False

def is_total_price(text):
    """
    이 텍스트가 '보유 주식의 총 가격'일 가능성이 있는지 정규표현식으로 확인합니다.
    (원화, 달러, 소수점을 포함하는 일반적인 가격 형식 포함)
    """
    
    text_stripped = text.strip()
    
    # 1. 원화 (KRW) 패턴 검사: [숫자, 쉼표]가 있고 '원'으로 끝나는 패턴
    # 예: '1,234,500원', '500원'
    pattern_won = re.compile(r'[\d,]+원$')
    if pattern_won.search(text_stripped):
        return True
    
    # 2. 달러/외화 패턴 검사: '$' 기호로 시작하고 [숫자, 쉼표, 소수점]이 오는 패턴
    # 예: '$178.50', '$1,200'
    pattern_dollar = re.compile(r'^\$[\d,]+(\.\d+)?$')
    if pattern_dollar.search(text_stripped):
        return True
        
    return False

def clean_price_text(price_text):
    """'1,234,500원', '$1,234.50' 같은 텍스트를 숫자(int/float)로 변환합니다."""
    try:
        # 1. 모든 통화 기호와 쉼표를 제거
        cleaned = price_text.replace('원', '').replace(',', '').replace('$', '')
        cleaned = cleaned.strip()

        # 2. 소수점(.)이 남아있다면 float으로 변환, 아니면 int로 변환
        if '.' in cleaned:
            return float(cleaned)
        else:
            return int(cleaned)
            
    except ValueError:
        # 숫자로 변환 실패 시
        return None

def clean_stock_text_list(text_list):
    """
    제거 대상:
    1. (숫자 + "주") 형태의 문자열 (예: "2주", "15주")
    2. 한글, 영어, 숫자가 전혀 포함되지 않은 순수 기호 문자열 (예: "-", "--", "▶")
    """
    
    # 1. (숫자 + "주") 패턴: ^(시작) \d+(숫자 1개 이상) 주$(끝)
    pattern_ = re.compile(r'^\d+(주|개)$')
    
    # 2. (유효 문자) 패턴: 한글(가-힣), 영어(a-z, A-Z), 숫자(0-9) 중 하나라도 포함
    pattern_valid_char = re.compile(r'[가-힣a-zA-Z0-9]')
    
    cleaned_list = []
    for item in text_list:
        # (1) (숫자 + "주") 패턴에 일치하는지 확인
        if pattern_.fullmatch(item.strip()):
            # "2주" 등은 제거 (continue)
            continue
            
        # (2) 유효 문자가 하나도 없는지 확인 (순수 기호인지)
        if not pattern_valid_char.search(item):
            # "-", "---" 등은 제거 (continue)
            continue
            
        # 위 두 조건에 걸리지 않은 항목만 새 리스트에 추가
        cleaned_list.append(item)
        
    return cleaned_list




# --- 3. (핵심) 메인 로직 함수 ---

def get_portfolio_from_image(image_cv: np.ndarray):
    """
    OpenCV(Numpy) 이미지를 입력받아 OCR 및 Y축 파싱을 수행하고
    {종목명: 가격} 딕셔너리를 반환하는 핵심 로직 함수
    """
    if reader is None:
        return {"error": "EasyOCR 리더기가 로드되지 않았습니다."}

    print("핵심 로직: OCR 실행 중...")
    
    # (1) EasyOCR 실행
    # detail=1 : 바운딩 박스(bbox) 좌표를 함께 받음
    # paragraph=False : 텍스트를 문단이 아닌 개별 라인으로 받음
    try:
        results = reader.readtext(image_cv, detail=1, paragraph=False)
    except Exception as e:
        return {"error": f"EasyOCR readtext 실행 중 오류: {e}"}
        
    print(f"OCR 결과 (Raw): {len(results)}개의 텍스트 감지")
    
    # (2) 후보군 분류 (Y축 파싱 준비)
    initial_candidates = []  # { 'text': '삼성전자', 'y_center': 150.5 }
    price_candidates = []  # { 'text': '1,234,500원', 'y_center': 151.0, 'x': 500 }

    for (bbox, text, prob) in results:
        y_center = get_y_center(bbox)
        
        # '종목명' 후보 필터링
        if is_stock_name(text):
            initial_candidates.append({
                'text': text.strip(),
                'y_center': y_center,
                'x': bbox[0][0],
                'bbox': bbox
            })
            
        # '총 가격' 후보 필터링
        elif is_total_price(text):
            price_candidates.append({
                'text': text.strip(),
                'y_center': y_center,
                'x': bbox[0][0] # X좌표 (나중에 정렬용으로 쓸 수 있음)
            })
            
    candidate_x_coords = [s['x'] for s in initial_candidates]

    if not candidate_x_coords: 
        return {"error": "1차 필터링 후 유효한 종목명 후보가 없습니다."}

    #X좌표 기준선 설정 (티커 제거용)
    X_MEDIAN = np.median(candidate_x_coords) 
    # 허용 오차: 중앙값보다 왼쪽으로 10픽셀 이상 벗어나면 노이즈로 간주
    X_TOLERANCE_LEFT = 5
    X_THRESHOLD = X_MEDIAN - X_TOLERANCE_LEFT
    stock_candidates = []
    for candidate in initial_candidates:
        if candidate['x'] >= X_THRESHOLD:
            stock_candidates.append(candidate)

    print(f"파싱: 종목명 후보 {len(stock_candidates)}개, 가격 후보 {len(price_candidates)}개")
    # (3) Y축 기반 매칭 (Spatial Join)
    portfolio = {}
    
    # Y축 허용 오차 (픽셀 단위)
    # 두 텍스트의 Y중심이 이 값 이내로 차이나면 '같은 줄'로 간주
    Y_TOLERANCE = 20 # (★해커톤에서 튜닝해야 할 값★)

    for price in price_candidates:
        best_match_stock = []
        for stock in stock_candidates:
            y_diff = abs(stock['y_center'] - price['y_center'])
            if y_diff <= Y_TOLERANCE:
                best_match_stock.extend(stock['text'].split())

        if best_match_stock:
            cleaned_price = clean_price_text(price['text'])

            if cleaned_price is not None:
                stock_name = " ".join(clean_stock_text_list(best_match_stock))
                portfolio[stock_name] = cleaned_price

    print(f"파싱 완료: 최종 {len(portfolio)}개 항목 매칭 성공")
    
    if not portfolio and (stock_candidates or price_candidates):
        print("경고: 후보는 있으나 매칭된 항목이 없습니다. Y_TOLERANCE 값을 조절해보세요.")
        
    return portfolio