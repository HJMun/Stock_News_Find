import os
import io
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import google.auth
from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import HumanMessage
from datetime import datetime
import pypdf

# ==========================================
# [프로젝트] Stock_News_Find
# ==========================================
if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gcp_key.json"

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
DRIVE_FOLDER_ID = os.environ.get("DRIVE_FOLDER_ID", "여기에_원본_폴더_ID_입력")
SPREADSHEET_ID = os.environ.get("SPREADSHEET_ID", "여기에_시트_ID_입력")
# ✨ 완료 폴더 변수명을 OLD_FOLDER_ID로 변경
OLD_FOLDER_ID = os.environ.get("OLD_FOLDER_ID", "여기에_old_폴더_ID_입력")

# Vertex AI LLM 초기화
llm = ChatVertexAI(
    model_name="gemini-2.5-flash",
    location="us-central1",
    temperature=0.2
)

def get_files_from_folder(folder_id):
    """지정된 폴더에서 파일 목록 가져오기"""
    query = f"'{folder_id}' in parents and trashed=false"
    results = drive_service.files().list(q=query, fields="files(id, name, mimeType, parents)").execute()
    return results.get('files', [])

def read_file_content(file_id, mime_type):
    """파일 종류(문서, 텍스트, PDF)에 따라 텍스트 내용 추출"""
    try:
        if mime_type == 'application/vnd.google-apps.document':
            request = drive_service.files().export_media(fileId=file_id, mimeType='text/plain')
        elif mime_type.startswith('text/'):
            request = drive_service.files().get_media(fileId=file_id)
        elif mime_type == 'application/pdf':
            request = drive_service.files().get_media(fileId=file_id)
        else:
            return None
            
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            
        if mime_type == 'application/pdf':
            fh.seek(0)
            pdf_reader = pypdf.PdfReader(fh)
            text_content = ""
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text_content += extracted + "\n"
            return text_content
        else:
            return fh.getvalue().decode('utf-8')
            
    except Exception as e:
        print(f"파일 읽기 에러 ({file_id}): {e}")
        return None

def analyze_news_with_vertex(title, content):
    """Vertex AI를 이용한 뉴스 및 증권사 리포트 심층 분석"""
    if not content or len(content) < 50:
        return "내용이 너무 짧거나 파싱할 수 없는 포맷입니다."
    
    safe_content = content[:50000] 
    
    prompt = f"""
    당신은 기업의 재무제표와 밸류에이션을 깊이 있게 다루는 전문 주식 애널리스트입니다.
    아래의 '주식 뉴스(또는 증권사 리포트)'를 읽고, 이 이슈가 기업의 펀더멘털에 미칠 영향을 분석해주세요.
    
    [문서 제목]: {title}
    [문서 본문]: {safe_content}
    
    다음 3가지 항목을 반드시 포함하여 작성해주세요:
    1. 핵심 내용 요약: 문서의 주요 팩트 2~3줄 요약
    2. 호재/악재 판별: 단기 및 중장기적 관점에서의 호재/악재/중립 여부와 그 근거 (리포트의 경우 목표가/투자의견 변경 사유 포함)
    3. 재무 지표 및 실적 영향: 이 내용이 기업의 부채비율, PER, PBR, ROE 변화에 어떤 논리적 영향을 미칠지 서술하고, 향후 실적 추정치 조정의 모멘텀이 될 수 있는지 중점적으로 심층 분석할 것.
    """
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()
    except Exception as e:
        print(f"Vertex AI 분석 에러: {e}")
        return "AI 분석 실패"

def append_to_spreadsheet(spreadsheet_id, data_row):
    """스프레드시트에 분석 결과 추가"""
    try:
        body = {'values': [data_row]}
        sheets_service.spreadsheets().values().append(
            spreadsheetId=spreadsheet_id,
            range="A1", 
            valueInputOption="USER_ENTERED",
            body=body
        ).execute()
        return True
    except Exception as e:
        print(f"스프레드시트 기록 에러: {e}")
        return False

def move_file_to_old(file_id, current_parents, old_folder_id):
    """✨ 파일을 분석 완료(old) 폴더로 자동 이동"""
    try:
        previous_parents = ",".join(current_parents)
        drive_service.files().update(
            fileId=file_id,
            addParents=old_folder_id,
            removeParents=previous_parents,
            fields='id, parents'
        ).execute()
        print(f"  -> 📂 완료된 파일 이동 성공 (old 폴더로 이동됨)")
    except Exception as e:
        print(f"  -> ⚠️ 파일 이동 에러: {e}")

def main():
    print(f"🔍 [Stock_News_Find] 구글 드라이브 폴더 스캔 시작...")
    files = get_files_from_folder(DRIVE_FOLDER_ID)
    
    if not files:
        print("폴더에 새로 읽을 파일이 없습니다.")
        return

    for f in files:
        file_id = f['id']
        file_name = f['name']
        mime_type = f['mimeType']
        parents = f.get('parents', [])
        
        if 'folder' in mime_type or 'spreadsheet' in mime_type:
            continue

        print(f"\n▶️ 문서 확인 중: {file_name}")
        content = read_file_content(file_id, mime_type)
        
        if content:
            print(f"  -> 🧠 AI 주식 심층 분석 진행 중...")
            analysis_result = analyze_news_with_vertex(file_name, content)
            
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
            data_row = [now_str, file_name, analysis_result]
            
            print(f"  -> 💾 [Stock_News_Find] 스프레드시트에 결과 저장 중...")
            is_saved = append_to_spreadsheet(SPREADSHEET_ID, data_row)
            
            # ✨ 저장이 성공적으로 끝난 경우에만 파일 이동 실행
            if is_saved:
                move_file_to_old(file_id, parents, OLD_FOLDER_ID)
        else:
            print(f"  -> ⚠️ 텍스트를 추출할 수 없는 포맷이거나 내용이 비어 건너뜁니다.")

if __name__ == "__main__":
    main()