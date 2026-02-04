import pandas as pd

def summarize_csv(file_path):
    """CSV 파일을 읽어 컬럼 정보, 샘플 데이터, 통계 요약을 문자열로 반환"""
    try:
        df = pd.read_csv(file_path)
        summary = [
            f"- 파일명: {file_path.split('/')[-1]}",
            f"- 데이터 크기: {df.shape}",
            f"- 컬럼 목록: {', '.join(df.columns)}",
            f"- 데이터 타입:\n{df.dtypes.to_string()}",
            f"- 결측치 수:\n{df.isnull().sum().to_string()}",
            f"- 상위 3개 행:\n{df.head(3).to_string()}"
        ]
        return "\n".join(summary)
    except Exception as e:
        return f"파일을 읽는 중 오류 발생: {e}"