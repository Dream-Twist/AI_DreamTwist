'''
File Name : story.entity
Description : 동화 생성 관련
Author : 이유민

History
Date        Author      Status      Description
2024.07.27  이유민      Created     
2024.07.27  이유민      Modified    동화 생성 함수 추가
2024.07.27  이유민      Modified    동화 문단 나누기 함수 추가
2024.07.29  원경혜      Modified    import에서 re, pipeline 삭제
'''

import torch
import os
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel

# 모델과 토크나이저 로드
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, '../../models/story_generator')

model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 동화 생성 함수
def makeStory(prompt, model, tokenizer, max_length=512, min_length=100, num_beams=5, temperature=0.7, top_k=50, top_p=0.95):
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    attention_mask = torch.ones_like(input_ids).to(device)
    num_return_sequences = 1

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids.to(model.device),
            attention_mask=attention_mask.to(model.device),
            max_length=max_length,
            min_length=min_length,
            num_return_sequences=num_return_sequences,
            no_repeat_ngram_size=4,
            num_beams = num_beams,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text
