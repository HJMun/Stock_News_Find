import os
import io
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import google.auth
from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import HumanMessage
from datetime import datetime

# ==========================================
# [프로젝트] Stock_News_Find
# ==========================================
# 로컬(내 PC)에서 테스트할 때만 gcp_key.json을 찾고, 
# GitHub Actions 환경에서는 구글 공식 모듈이 알아서 세팅한 경로를 사용합니다.
if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gcp_key.json"

# 드라이브 파일 읽기 및 스프레드시트 쓰기 권한
SCOPES = [
    'https://www.googleapis.com/auth/drive',
    'https://www.googleapis.com/auth/spreadsheets'
]

credentials, project = google.auth.default(scopes=SCOPES)
drive_service = build('drive', 'v3', credentials=credentials)
sheets_service = build('sheets', 'v4', credentials=credentials)

# ==========================================
# 환경 변수 (GitHub Secrets에서 가져옴)
# ==========================================
DRIVE_FOLDER_ID = os.environ.get("DRIVE_FOLDER_ID", "여기에_폴더_ID_입력")
SPREADSHEET_ID = os.environ.get("SPREADSHEET_ID", "여기에_시트_ID_입력")

# Vertex AI LLM 초기화
llm = ChatVertexAI(
    model_name="gemini-1.5-pro",
    location="us-central1",
    temperature=0.2
)

def get_files_from_folder(folder_id):
    """지정된 'Stock_News_Find' 폴더에서 뉴스 파일 목록 가져오기"""
    query = f"'{folder_id}' in parents and trashed=false"
    results = drive_service.files().list(q=query, fields="files(id, name, mimeType)").execute()
    return results.get('files', [])

def read_file_content(file_id, mime_type):
    """파일 종류에 따라 텍스트 내용 추출"""
    try:
        if mime_type == 'application/vnd.google-apps.document':
            request = drive_service.files().export_media(fileId=file_id, mimeType='text/plain')
        elif mime_type.startswith('text/'):
            request = drive_service.files().get_media(fileId=file_id)
        else:
            return None
            
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
        
        return fh.getvalue().decode('utf-8')
    except Exception as e:
        print(f"파일 읽기 에러 ({file_id}): {e}")
        return None

def analyze_news_with_vertex(title, content):
    """Vertex AI를 이용한 주식 뉴스 심층 분석"""
    if not content or len(content) < 50:
        return "내용이 너무 짧거나 파싱할 수 없는 포맷입니다."
    
    safe_content = content[:4000] # 토큰 제한 방지
    
    prompt = f"""
    당신은 기업의 재무제표와 밸류에이션을 깊이 있게 다루는 전문 주식 애널리스트입니다.
    아래의 '주식 뉴스(또는 리포트)'를 읽고, 이 이슈가 기업의 펀더멘털에 미칠 영향을 분석해주세요.
    
    [뉴스 제목]: {title}
    [뉴스 본문]: {safe_content}
    
    다음 3가지 항목을 반드시 포함하여 작성해주세요:
    1. 핵심 내용 요약: 기사의 주요 팩트 2~3줄 요약
    2. 호재/악재 판별: 단기 및 중장기적 관점에서의 호재/악재/중립 여부와 그 근거
    3. 재무 지표 및 실적 영향: 이 뉴스가 기업의 부채비율, PER, PBR, ROE 변화에 어떤 논리적 영향을 미칠지 서술하고, 향후 실적 추정치 조정