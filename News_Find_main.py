import os
import io
import re
import base64
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
OLD_FOLDER_ID = os.environ.get("OLD_FOLDER_ID", "여기에_old_폴더_ID_입력")

# Vertex AI LLM 초기화
llm = ChatVertexAI(
    model_name="gemini-2.5-flash",
    location="us-central1",
    temperature=0.2
)

def parse_filename_metadata(filename):
    """파일 제목에서 (날짜, 파일명 기준 종목명, 깔끔한 제목) 3가지를 분리합니다."""
    name_without_ext = os.path.splitext(filename)[0]
    parts = name_without_ext.split('_')

    # 1. 날짜 추출
    report_date = datetime.now().strftime("%Y-%m-%d")
    match_4yr = re.search(r'(20\d{2})[.\-_]?(0[1-9]|1[0-2])[.\-_]?(0[1-9]|[12]\d|3[01])', name_without_ext)
    match_6digit = re.search(r'(?:_|^|\D)(\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])(?=\D|$)', name_without_ext)
    match_2yr_sep = re.search(r'(\d{2})[.\-](0[1-9]|1[0-2])[.\-](0[1-9]|[12]\d|3[01])', name_without_ext)

    if match_4yr:
        report_date = f"{match_4yr.group(1)}-{match_4yr.group(2)}-{match_4yr.group(3)}"
    elif match_6digit:
        report_date = f"20{match_6digit.group(1)}-{match_6digit.group(2)}-{match_6digit.group(3)}"
    elif match_2yr_sep:
        report_date = f"20{match_2yr_sep.group(1)}-{match_2yr_sep.group(2)}-{match_2yr_sep.group(3)}"

    # 2. 종목명 및 순수 제목 분리
    if len(parts) >= 3:
        stock_from_file = parts[0].strip()
        clean_title = "_".join(parts[1:-1]).strip()
    elif len(parts) == 2:
        stock_from_file = None
        clean_title = parts[0].strip()
    else:
        stock_from_file = None
        clean_title = name_without_ext.strip()

    return report_date, stock_from_file, clean_title

def get_files_from_folder(folder_id):
    """지정된 폴더에서 파일 목록 가져오기"""
    query = f"'{folder_id}' in parents and trashed=false"
    results = drive_service.files().list(q=query, fields="files(id, name, mimeType, parents)").execute()
    return results.get('files', [])

def read_file_content(file_id, mime_type):
    """
    파일 내용 추출 (텍스트 추출 실패 시 PDF 원본 바이트를 그대로 반환하여 AI가 직접 읽게 함)
    반환값: (text_content, pdf_raw_bytes)
    """
    try:
        if mime_type == 'application/vnd.google-apps.document':
            request = drive_service.files().export_media(fileId=file_id, mimeType='text/plain')
        elif mime_type.startswith('text/') or mime_type == 'application/pdf':
            request = drive_service.files().get_media(fileId=file_id)
        else:
            return None, None
            
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            
        raw_bytes = fh.getvalue()

        if mime_type == 'application/pdf':
            text_content = ""
            try:
                fh.seek(0)
                pdf_reader = pypdf.PdfReader(fh)
                for page in pdf_reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text_content += extracted + "\n"
            except Exception as pdf_err:
                print(f"  -> ℹ️ 일반 텍스트 추출 불가 PDF ({pdf_err}) -> AI 직접 읽기 모드로 전환합니다.")
                
            # 추출된 글자가 50자 이상이면 텍스트 사용, 아니면 PDF 원본 바이트를 통째로 전달!
            if len(text_content.strip()) >= 50:
                return text_content, None
            else:
                print("  -> 🔍 이미지형/인쇄형 PDF 감지! Gemini가 PDF 원본을 직접 눈으로 읽고 분석합니다.")
                return None, raw_bytes
        else:
            return raw_bytes.decode('utf-8', errors='ignore'), None
            
    except Exception as e:
        print(f"파일 읽기 에러 ({file_id}): {e}")
        return None, None

def analyze_news_with_vertex(title, content, pdf_bytes=None):
    """
    Vertex AI를 이용한 리포트 심층 분석 (텍스트 또는 PDF 원본 직접 분석 모두 지원)
    """
    base_instruction = f"""
    당신은 기업의 재무제표와 밸류에이션을 깊이 있게 다루는 전문 주식 애널리스트입니다.
    제공된 '주식 뉴스(또는 증권사 리포트)'를 읽고, 이 이슈가 기업의 펀더멘털에 미칠 영향을 분석해주세요.
    
    [문서 제목]: {title}
    
    답변의 맨 첫 번째 줄에는 반드시 이 문서가 다루는 핵심 대상을 아래 형식에 맞춰 한 줄로만 적어주세요:
    - 개별 기업 리포트인 경우: [분류]: 종목명 (예: [분류]: 에코프로비엠)
    - 산업/섹터 리포트인 경우: [분류]: 섹터명 (관련주: 종목명들) (예: [분류]: 탠덤 셀·유리기판 (관련주: 주성엔지니어링, 와이엠티, 씨앤지하이테크))
    - 전반적인 시황/거시경제 리포트인 경우: [분류]: 시황/매크로
    
    그 다음 줄부터 아래 3가지 항목을 반드시 포함하여 본문을 작성해주세요:
    1. 핵심 내용 요약: 문서의 주요 팩트 2~3줄 요약
    2. 호재/악재 판별: 단기 및 중장기적 관점에서의 호재/악재/중립 여부와 그 근거 (리포트의 경우 목표가/투자의견 변경 사유 포함)
    3. 재무 지표 및 실적 영향: 이 내용이 관련 기업들의 부채비율, PER, PBR, ROE 변화에 어떤 논리적 영향을 미칠지 서술하고, 향후 실적 추정치 조정의 모멘텀이 될 수 있는지 중점적으로 심층 분석할 것.
    """

    try:
        # 1) 일반 텍스트가 잘 뽑힌 경우
        if content:
            safe_content = content[:50000]
            prompt = base_instruction + f"\n\n[문서 본문]:\n{safe_content}"
            message = HumanMessage(content=prompt)
        # 2) 네이버 인쇄 PDF 등 글자 추출이 안 되어 PDF 원본을 통째로 넣는 경우
        elif pdf_bytes:
            pdf_b64 = base64.b64encode(pdf_bytes).decode("utf-8")
            message = HumanMessage(
                content=[
                    {"type": "text", "text": base_instruction},
                    {
                        "type": "media",
                        "mime_type": "application/pdf",
                        "data": pdf_b64
                    }
                ]
            )
        else:
            return "분류 불가", "내용이 비어 있습니다."

        response = llm.invoke([message])
        full_text = response.content.strip()
        
        lines = full_text.splitlines()
        ai_stock_label = "시황/기타"
        body_lines = []
        found_label = False
        
        for line in lines:
            if not found_label and "[분류]:" in line:
                ai_stock_label = line.split("[분류]:")[1].strip()
                ai_stock_label = ai_stock_label.replace("*", "").strip()
                found_label = True
            else:
                body_lines.append(line)
                
        clean_analysis = "\n".join(body_lines).strip()
        return ai_stock_label, clean_analysis
        
    except Exception as e:
        print(f"Vertex AI 분석 에러: {e}")
        return "오류", "AI 분석 실패"

def append_to_spreadsheet(spreadsheet_id, data_row):
    """스프레드시트에 분석 결과 추가 (A:날짜, B:종목/섹터, C:제목, D:내용)"""
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
        content, pdf_bytes = read_file_content(file_id, mime_type)
        
        if content or pdf_bytes:
            print(f"  -> 🧠 AI 주식 심층 분석 및 종목/섹터 판별 진행 중...")
            report_date, stock_from_file, clean_title = parse_filename_metadata(file_name)
            ai_stock_label, analysis_result = analyze_news_with_vertex(file_name, content, pdf_bytes)
            
            final_stock_name = stock_from_file if stock_from_file else ai_stock_label
            data_row = [report_date, final_stock_name, clean_title, analysis_result]
            
            print(f"  -> 💾 시트 저장 중 | 날짜: {report_date} | 분류: {final_stock_name} | 제목: {clean_title}")
            is_saved = append_to_spreadsheet(SPREADSHEET_ID, data_row)
            
            if is_saved:
                move_file_to_old(file_id, parents, OLD_FOLDER_ID)
        else:
            print(f"  -> ⚠️ 텍스트를 추출할 수 없는 포맷이거나 내용이 비어 건너뜁니다.")

if __name__ == "__main__":
    main()