'''
File Name : story_train
Description : 주피터 노트북 파일 파이썬 파일로 분리, 동화 머신러닝 훈련
Author : 원경혜

History
Date        Author      Status      Description
2024.07.29  원경혜      Created      파일 분리
'''

import os
from transformers import Trainer, TrainingArguments
import wandb

# dataset_setup.py에서 model, tokenizer, dataset, data collator 가져오기
from src.features.dataset_setup import model, tokenizer, train_dataset, data_collator

# logging을 위한 wandb 연결
wandb.login()

# 트레이닝 인자 설정 (여러분들이 인공지능 수업에서 배운 노하우를 활용하여 에포크, 배치사이즈, 스탭 등 자유롭게 조절하여 학습 인자 셋팅)
training_args = TrainingArguments(
    # output_dir='./results',
    num_train_epochs=50 ,
    per_device_train_batch_size=1,  # 배치 사이즈 조정
    save_steps=10_000,
    save_total_limit=2,
    eval_strategy='epoch',
    # logging_dir='./logs',
    logging_steps=100,
    logging_first_step=True,
    learning_rate=5e-5,
    overwrite_output_dir=True,
)

# 트레이너 객체 생성 및 훈련
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=train_dataset,
    data_collator=data_collator,
)

# 모델 훈련
trainer.train()

# 모델 및 토크나이저 저장
current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, '../../models/story_generator')

model.save_pretrained(output_dir)

tokenizer.save_pretrained(output_dir)