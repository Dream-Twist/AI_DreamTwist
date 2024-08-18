'''
File Name : dataset_setup
Description : 주피터 노트북 파일 파이썬 파일로 분리, 동화 머신러닝 데이터셋 설정
Author : 원경혜

History
Date        Author      Status      Description
2024.07.29  원경혜      Created      파일 분리
'''

## 토크나이저 추가 및 데이터셋 설정
import os
import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast, DataCollatorForLanguageModeling

# GPU 환경에서 사용 가능하도록 변경
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# SKT에서 개발한 한국어 GPT-2 모델, (한국어 텍스트의 생성, 분류, 번역) 등 다양한 자연어 처리 작업에 사용할 수 있는 사전 훈련된 모델
model_name = 'skt/kogpt2-base-v2'
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)

# 토크나이저 로드  (토크나이저는 텍스트를 입력으로 받아서 모델이 처리할 수 있는 형식으로 변환하고, 반대로 모델의 출력을 해석할 수 있는 역할)
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)

# 해당 토큰들은 추가적인 토큰으로, 추가하거나 빼거나 하시면 됩니다!
tokenizer.add_special_tokens({'pad_token': '[PAD]'})   # 패딩 토큰 (일정한 길이로 맞추기 위해 사용)
tokenizer.add_special_tokens({'bos_token': '<BOS>'})   # 시작 토큰 (더욱더 동화스럽게 만들기 위해 시작 구문 추가) - ex) 옛날 옛날에~
tokenizer.add_special_tokens({'eos_token': '<EOS>'})   # 종료 토큰 (더욱더 동화스럽게 만들기 위해 끝 맺음 추가) - ex) 행복하게 살았답니다~
tokenizer.add_special_tokens({'sep_token': '<SEP>'})   # 문장 경계 토큰 (새로운 장면이나 시간이 흐른 것을 알리는 문구)

# 모델의 임베딩 테이블 크기를 토크나이저 설정에 맞게 재조정
model.resize_token_embeddings(len(tokenizer))

# 전처리 된 데이터 로드
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, '../../data/processed', 'preprocessed_train_data.txt')

# 꿈틀 데이터셋 클래스 정의
class DreamTwistDataset(torch.utils.data.Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.encodings = tokenizer(texts, padding=True, truncation=True, max_length=max_length)

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]).to(device) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

# 데이터셋 인스턴스 생성
max_length = 512  # 문장 최대 길이로 설정해주시면 됩니다.

# 꿈틀 트레인 데이터셋 설정
train_dataset = DreamTwistDataset(data_path, tokenizer, max_length)

# 데이터 콜레이터 설정
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=max_length
)