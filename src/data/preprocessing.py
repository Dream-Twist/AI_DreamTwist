'''
File Name : preprocessing
Description : 주피터 노트북 파일 파이썬 파일로 분리, 동화 머신러닝 데이터 로드 및 전처리
Author : 원경혜

History
Date        Author      Status      Description
2024.07.29  원경혜      Created      파일 분리
'''

## 학습 데이터 로드
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, '../../data/raw', 'train_data.txt')

train_texts = []
with open(data_path, "r") as file:
    for i in file:
        train_texts.append(i.strip())

# len(train_texts)


## 슬라이딩 윈도우 설정
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')

## 슬라이딩 윈도우 파라미터
window_size = 10  # 윈도우 크기 (문장 수)
sliding_step = 5  # 슬라이딩 간격 (문장 수)

def sliding_window(sentences, window_size, step):
    """ 문장 리스트에 슬라이딩 윈도우를 적용하는 함수 """
    windows = []
    for start in range(0, len(sentences) - window_size + 1, step):
        window = sentences[start:start + window_size]
        windows.append(window)
    
    # 마지막 윈도우 추가 (중복 방지)
    if len(sentences) % step != 0:
        last_window = sentences[-window_size:]
        if last_window not in windows:
            windows.append(last_window)
    return windows

# 동화 데이터에 슬라이딩 윈도우 적용
all_story_windows = []

for story in train_texts:
    # 동화에서 문장 분리
    sentences = sent_tokenize(story)

    # 슬라이딩 윈도우 적용
    story_windows = sliding_window(sentences, window_size, sliding_step)
    
    all_story_windows.append(story_windows)


# 예를 들어 문장 수가 충분하지 않은 경우를 확인
for i, story in enumerate(train_texts):
    sentences = sent_tokenize(story)
    if len(sentences) < window_size:
        print(f"Story {i} has less than {window_size} sentences and cannot have any windows.")

# 결과를 저장할 리스트
all_new_story = []

# 전체 all_story_windows에 대해 반복
for story_window in all_story_windows:
    # 문장들을 저장할 리스트
    sentences_list = []
    
    for sentences in story_window:
        # 각 문장을 sentences_list에 추가
        sentences_list.extend(sentences)
    
    # 문장들을 하나의 문자열로 결합 (문장 사이에만 공백을 추가)
    new_story = ' '.join(sentences_list)

    # 줄바꿈 문자 제거 및 앞뒤 공백 제거
    new_story = new_story.replace('\n', ' ').strip()
    
    # 결과 리스트에 추가
    all_new_story.append(new_story)

# print(len(all_new_story))
# all_new_story[397]

# 새로운 전처리된 텍스트 데이터로 업데이트
train_texts = all_new_story

# 전처리 데이터 저장
preprocessed_data_path = os.path.join(current_dir, '../../data/processed', 'preprocessed_train_data.txt')
with open(preprocessed_data_path, 'w') as file:
    for story in train_texts:
        file.write(story + "\n")