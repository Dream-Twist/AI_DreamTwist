'''
File Name : utils.helpers
Description : 유틸리티 함수
Author : 이유민

History
Date        Author      Status      Description
2024.07.27  이유민      Created     
2024.07.27  이유민      Modified    동화 자르기 추가
2024.07.29  이유민      Modified    문단 나누기 추가
2024.07.29  이유민      Modified    마지막 문장 확인 추가
'''

import re

# 맞춤법 검사에 사용될 동화 자르기 함수
def splitText(text, chunk_size=490):
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

# 생성된 동화 문단 나누는 함수
def splitParagraphs(story, num_paragraphs = 10):
    # 문장 단위로 분할
    sentences = re.split(r'(?<=[.!?]) +', story)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    
    if len(sentences) == 0:
        return [""]

    # 문단 수가 문장 수보다 크면 모든 문장을 단일 문단으로 반환
    if len(sentences) <= num_paragraphs:
        return [" ".join(sentences)]

    # 각 문단에 들어갈 문장 수 계산
    avg_sentences_per_paragraph = len(sentences) // num_paragraphs
    extra_sentences = len(sentences) % num_paragraphs

    paragraphs = []
    current_paragraph = []
    sentence_counter = 0

    for sentence in sentences:
        current_paragraph.append(sentence)
        sentence_counter += 1
        
        # 문장 수가 평균 문장 수에 도달하거나 추가 문장이 필요한 경우
        if sentence_counter >= avg_sentences_per_paragraph + (1 if extra_sentences > 0 else 0):
            paragraphs.append(' '.join(current_paragraph))
            current_paragraph = []
            sentence_counter = 0
            extra_sentences -= 1
    
    # 남은 문장 추가
    if current_paragraph:
        paragraphs.append(' '.join(current_paragraph))
    
    return paragraphs

# 마지막 문장 종료되지 않았을 경우 제거하는 함수
def truncateSentence(story):
    sentence_endings = r'[.!?\'"]'  # 문장 구분자 정의

    if not re.search(sentence_endings, story[-1]):
        sentences = re.split(sentence_endings, story)

        if len(sentences) > 1:
            # 마지막 문장 이전까지 반환
            result = story[:story.rfind(sentences[-2]) + len(sentences[-2]) + 1].strip()
            
            # 마지막 문장의 이전 문장이 " 혹은 '가 포함될 경우
            if sentences[len(sentences)-1][0] == '\"' or '\'':
                    result += sentences[len(sentences)-1][0]
            return result
        else:
            # 문장 구분자가 없는 경우, 원래 텍스트 반환
            return story.strip()
    else:
        # 마지막 문장이 정상적으로 종료된 경우, 원래 텍스트 반환
        return story.strip()