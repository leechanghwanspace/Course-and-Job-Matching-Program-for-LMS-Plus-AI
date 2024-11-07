# Course and Job Matching Program for LMS Plus AI

### 설명
이 프로젝트는 **LMS Plus AI** 시스템에서 교과목과 관련된 직무를 자동으로 매칭하고 분석하는 프로그램입니다.<br>
추출된 데이터를 바탕으로 상위 3개의 관련 직무와 일치율을 계산하여 CSV 파일로 저장합니다.

### 주요 기능
- 교과목과 관련 직무의 일치율 분석
- 상위 3개의 직무와 일치율을 CSV 파일로 저장

### 저장할 폴더 및 파일 경로 설정
output_folder = r"data/output"  
output_file = os.path.join(output_folder, "저장할 파일명.csv")