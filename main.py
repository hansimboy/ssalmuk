import os
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai
from src.utils import summarize_csv

# 환경 설정
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-1.5-pro')

def run_agent():
    print("🔍 [1/3] 데이터를 분석하고 있습니다...")
    # 파일 경로 정의
    train_path = "data/raw/train.csv"
    test_path = "data/raw/test.csv"
    sub_path = "data/raw/sample_submission.csv"

    # 데이터 요약 (src/utils.py의 함수 사용)
    train_info = summarize_csv(train_path)
    test_info = summarize_csv(test_path)
    sample_info = summarize_csv(sub_path)

    # 에이전트에게 보낼 프롬프트 구성
    prompt = f"""
    당신은 Kaggle 전문 데이터 과학자 에이전트입니다.
    제공된 데이터 요약 정보를 바탕으로, 테스트 데이터에 대한 예측을 수행하고 'submission.csv'를 생성하는 완벽한 Python 코드를 작성하세요.

    [데이터 요약]
    {train_info}
    {test_info}
    {sample_info}

    [필수 요구사항]
    1. 데이터 로드: '{train_path}', '{test_path}'를 사용하세요.
    2. 데이터 전처리: 결측치(Null) 처리와 범주형 변수 인코딩을 포함하세요.
    3. 모델링: 데이터 특성에 맞는 적절한 머신러닝 모델(XGBoost, RandomForest 등)을 사용하세요.
    4. 저장: 최종 결과물은 반드시 'data/submissions/submission.csv' 경로에 저장하세요.
    5. 형식: 설명 없이 오직 Python 코드만 출력하세요. 마크다운 기호(```python)는 제거하세요.
    """

    print("🤖 [2/3] Gemini가 해결 전략을 수립하고 코드를 생성 중입니다...")
    response = model.generate_content(prompt)
    
    # 생성된 코드 정제 (마크다운 등 불필요한 텍스트 제거)
    clean_code = response.text.replace("```python", "").replace("```", "").strip()

    # 결과 저장
    script_name = "generated_scripts/solution_v1.py"
    with open(script_name, "w", encoding="utf-8") as f:
        f.write(clean_code)
    
    print(f"✅ [3/3] 에이전트가 코드를 완성했습니다: {script_name}")
    print("\n--- 생성된 코드의 일부 ---")
    print(clean_code[:300] + "...")

if __name__ == "__main__":
    run_agent()