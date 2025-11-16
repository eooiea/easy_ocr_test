# main.py
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
import numpy as np
import cv2
import io

# --- 1. ocr_logic.py 에서 핵심 함수 임포트 ---
# (같은 폴더에 있으므로 바로 임포트 가능)
try:
    from ocr_logic import get_portfolio_from_image, print_pretty_dict
except ImportError:
    print("오류: ocr_logic.py 파일을 찾을 수 없거나, 내부 오류가 있습니다.")
    exit()

# --- 2. FastAPI 앱 생성 ---
app = FastAPI(
    title="Stock Portfolio OCR API",
    description="주식창 스크린샷을 받아 종목명과 총 가격을 JSON으로 반환합니다."
)

# --- 3. API 엔드포인트 정의 ---

@app.get("/")
async def root():
    """
    API 루트 엔드포인트 - API 정보를 반환합니다.
    """
    return {
        "message": "Stock Portfolio OCR API",
        "description": "주식창 스크린샷을 받아 종목명과 총 가격을 JSON으로 반환합니다.",
        "endpoints": {
            "POST /ocr/process-image": "이미지를 업로드하여 OCR 처리",
            "GET /docs": "Swagger UI API 문서",
            "GET /redoc": "ReDoc API 문서"
        }
    }

# (이전에 친구와 약속한 주소: POST /ocr/process-image)
@app.post("/ocr/process-image")
async def process_ocr_request(
    # (이전에 약속한 Key: 'image_file')
    image_file: UploadFile = File(...) 
):
    """
    사용자로부터 업로드된 스크린샷(바이트 스트림)을 받아,
    OCR 및 파싱을 수행하고 결과를 JSON으로 반환합니다.
    """
    
    print(f"API: '{image_file.filename}' 이미지 수신 완료.")

    # --- 4. (핵심) 바이트 스트림 -> OpenCV 변환 ---
    # (1) UploadFile 객체에서 바이트 스트림을 읽습니다.
    image_bytes = await image_file.read()

    # (2) 바이트 데이터를 Numpy 배열로 변환합니다.
    image_np = np.frombuffer(image_bytes, dtype=np.uint8)
    
    # (3) Numpy 배열을 OpenCV 이미지(BGR 형식)로 디코딩합니다.
    image_cv = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    if image_cv is None:
        # 이미지 파일이 아니거나 손상된 경우 400 에러 반환
        raise HTTPException(
            status_code=400, 
            detail="업로드한 파일이 유효한 이미지 파일이 아닙니다."
        )

    # --- 5. ocr_logic.py의 핵심 함수 호출 ---
    print("API: OCR 로직 실행...")
    try:
        portfolio_result = get_portfolio_from_image(image_cv)
        
        print("API: 파싱 결과 (서버 로그):")
        print_pretty_dict(portfolio_result) # 서버 터미널에 예쁘게 출력
        
        # --- 6. 친구에게 JSON 결과 반환 ---
        return portfolio_result
        
    except Exception as e:
        # ocr_logic 실행 중 알 수 없는 오류 발생 시 500 에러 반환
        print(f"API: 치명적 오류 발생: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"OCR 처리 중 서버 오류 발생: {str(e)}"
        )

# --- 7. (선택) 서버 실행 (터미널에서 uvicorn 명령어를 직접 쳐도 됨) ---
if __name__ == "__main__":
    print("FastAPI 서버를 http://127.0.0.1:8000 에서 시작합니다.")
    uvicorn.run(app, host="127.0.0.1", port=8000)