import os
import pandas as pd
from transformers import pipeline

# CSV 파일 경로 설정
csv_file = r"#"

# 저장할 폴더 및 파일 경로 설정
output_folder = r"data/output"  # 결과 CSV 파일이 저장될 폴더
output_file = os.path.join(output_folder, "information_security_courses.csv")

# CSV 파일 로드 및 확인
try:
    data = pd.read_csv(csv_file, sep=',', encoding='utf-8-sig', on_bad_lines='skip')
    print("데이터가 성공적으로 로드되었습니다:")
    print(data.head())  # 데이터의 첫 몇 줄 출력하여 확인
except FileNotFoundError:
    print(f"파일을 찾을 수 없습니다: {csv_file}")
    exit()
except pd.errors.ParserError:
    print("CSV 파일을 구문 분석하는 중 오류가 발생했습니다.")
    exit()

# 직무 분류 파이프라인 초기화
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", multi_label=True)

# 직무 분류 함수 정의 (상위 3개 직무 반환)
def classify_job(content):
    candidate_labels = [
        "게임개발자", "게임 기획자", "VR/AR 개발자", "웹 개발자", "앱 개발자", 
        "데이터 엔지니어", "데브옵스 엔지니어", "네트워크 엔지니어", "보안 엔지니어",
        "사이버 침해 대응", "디지털 포렌식", "클라우드 보안", "IoT 보안", 
        "정보보안 전문가", "IoT", "데이터 분석", "인공지능", "스마트 팩토리", 
        "자동화 시스템", "모바일 앱 개발", "스마트 기술", "블록체인"
    ]
    result = classifier(content, candidate_labels)
    # 상위 3개의 일치율이 높은 직무를 반환
    top_jobs = result['labels'][:3]
    top_score = result['scores'][0]
    return top_jobs, top_score

# 최종 결과를 저장할 데이터프레임 생성
output_data = pd.DataFrame(columns=['과목코드', '과목명', '학점', '직무1', '직무2', '직무3', '일치율'])

# 과목코드별로 그룹화하여 결과 집계
for code, group in data.groupby('과목코드'):
    course_name = group['과목명'].iloc[0]
    top_score = 0
    matched_jobs = [None, None, None]
    
    for _, row in group.iterrows():
        # 주차별 학습 내용을 바탕으로 직무 분류
        jobs, score = classify_job(row['주차별 학습내용'])
        # 상위 3개의 직무와 최고 일치율을 저장
        matched_jobs = jobs
        top_score = f"{score:.3%}"

    # 결과 데이터를 output_data에 추가
    new_row = {
        '과목코드': code,
        '과목명': course_name,
        '학점': 3,  # 예제에 따라 학점을 3으로 설정
        '직무1': matched_jobs[0],
        '직무2': matched_jobs[1],
        '직무3': matched_jobs[2],
        '일치율': top_score
    }
    output_data = pd.concat([output_data, pd.DataFrame([new_row])], ignore_index=True)

# 결과 CSV 파일로 저장
output_data.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"결과가 {output_file}에 저장되었습니다.")

