import os
from google.oauth2 import service_account
from googleapiclient.discovery import build
from langchain_google_vertexai import ChatVertexAI

# ==========================================
# [설정 입력 구간] 본인의 환경에 맞게 수정하세요
# ==========================================
# 1. PC에 다운로드 받아둔 서비스 계정 JSON 파일의 이름 (또는 전체 경로)
KEY_PATH = "gcp_key.json"  

# 2. 구글 시트 주소에서 /d/ 와 /edit 사이에 있는 영문+숫자 ID 문자열
SPREADSHEET_ID = "1UtP25H4mcyPONVjrY2YWnx3AOn45YZe6TPZSxWhBCgM" 

PROJECT_ID = "stockreportcheck"
# ==========================================

print("=== 🔍 연결 진단 테스트를 시작합니다 ===\n")

# 0. JSON 키 파일 존재 여부 확인
if not os.path.exists(KEY_PATH):
    print(f"❌ [에러] '{KEY_PATH}' 파일을 찾을 수 없습니다. 파일 이름이나 경로를 확인해 주세요.")
    exit()

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = KEY_PATH
print(f"✅ 1. 인증 키 파일 로드 완료: {KEY_PATH}\n")

# 1. 구글 스프레드시트 연결 테스트
print("--- 📊 [테스트 1] 구글 스프레드시트 연결 확인 ---")
try:
    creds = service_account.Credentials.from_service_account_file(
        KEY_PATH, scopes=["https://www.googleapis.com/auth/spreadsheets"]
    )
    service = build("sheets", "v4", credentials=creds)
    sheet_metadata = service.spreadsheets().get(spreadsheetId=SPREADSHEET_ID).execute()
    sheet_title = sheet_metadata.get('properties', {}).get('title')
    print(f"✅ 시트 연결 성공! (찾은 문서 제목: '{sheet_title}')\n")
except Exception as e:
    print(f"❌ 시트 연결 실패!\n   원인: {e}\n")

# 2. Vertex AI 모델별 연결 테스트
print("--- 🧠 [테스트 2] Vertex AI 모델 호출 확인 ---")
# 2026년 현재 활성화된 모델과 구버전 모델을 순서대로 테스트합니다.
test_models = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-1.5-pro-002"]

for model_name in test_models:
    print(f"▶️ '{model_name}' 모델 연결 시도 중...")
    try:
        llm = ChatVertexAI(
            model_name=model_name,
            project=PROJECT_ID,
            location="us-central1",
            max_retries=1 # 실패 시 기다리지 않고 바로 넘어가도록 설정
        )
        response = llm.invoke("연결 테스트입니다. '정상'이라고 두 글자로만 답해주세요.")
        print(f"   ✅ [{model_name}] 성공! AI 응답: {response.content.strip()}")
        print(f"   👉 본 코드(News_Find_main.py)의 model_name에 '{model_name}'을 사용하시면 됩니다!\n")
        break # 하나라도 성공하면 테스트 종료
    except Exception as e:
        # 에러 메시지 핵심만 출력
        error_msg = str(e).splitlines()[0] if str(e) else str(e)
        print(f"   ❌ [{model_name}] 실패: {error_msg}\n")

print("=== 🏁 진단 테스트 종료 ===")