'''
File Name : src.api.story
Description : 동화 생성 관련
Author : 이유민

History
Date        Author      Status      Description
2024.07.28  이유민      Created     
2024.07.28  이유민      Modified    동화 생성 추가
2024.08.03  이유민      Modified    주제 및 제목 추가
'''
import json
from flask import request, jsonify
from flask_restx import Namespace, Resource, fields
from models.py_hanspell.hanspell.spell_checker import check  # 맞춤법 검사
from src.models.story_generator import makeStory, model, tokenizer    # 동화 관련
from src.utils.helpers import splitText, splitParagraphs, truncateSentence, generateThemeAndTitle

story_api = Namespace('Story', description='AI 동화 스토리 생성')

@story_api.route('')
class Speller(Resource):
    @story_api.expect(story_api.model('Story', {
        'prompt': fields.String(required=True, description='동화의 첫 문장')
    }))
    @story_api.response(201, 'AI 동화 스토리 생성 성공', story_api.model('Response', {
        'story': fields.List(fields.String, description='생성된 동화 내용')
    }))
    @story_api.response(400, '잘못된 요청', story_api.model('ErrorResponse', {
        'error': fields.String(description='잘못된 요청입니다.')
    }))
    @story_api.response(401, '인증 실패', story_api.model('ErrorResponse', {
        'error': fields.String(description='인증에 실패했습니다.')
    }))
    def post(self):
        '''AI 동화 스토리를 생성하는 API입니다'''
        data = story_api.payload
        prompt = data.get('prompt')

        if not prompt:
            return {"error": "잘못된 요청입니다."}, 400

        # AI 스토리 생성
        generated_story = generate_story_endpoint(prompt)

        return generated_story, 201

def generate_story_endpoint(prompt):
    story = makeStory(prompt, model, tokenizer) # 동화 생성

    # 맞춤법 검사
    try:
        splitTexts = splitText(story)
        corrected_text = [
                (result := check(text)).checked if hasattr(result, 'checked') else text
                for text in splitTexts
            ]
        corrected_text = ''.join(corrected_text)
    except:
        corrected_text = story  # 맞춤법 검사 오류날 경우
    
    # 생성된 스토리 바탕으로 주제와 제목 정하기
    themeAndTitle = generateThemeAndTitle(story)

    # 마지막 문장 종료되었는지 확인
    story = truncateSentence(corrected_text)

    # 문단 분리
    story = splitParagraphs(story, num_paragraphs = 6)

    return {"theme": themeAndTitle["theme"], "title": themeAndTitle["title"], "story": story}
