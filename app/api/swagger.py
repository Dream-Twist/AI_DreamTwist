'''
File Name : src.api.swagger
Description : api 및 swagger 관련 코드
Author : 이유민

History
Date        Author      Status      Description
2024.07.28  이유민      Created     
2024.07.28  이유민      Modified    swagger 코드 추가
2024.07.28  이유민      Modified    story api 추가
'''

from flask_restx import Api

api = Api(
    version='1.0',
    title='꿈틀(DreamTwist) - AI',
    description='DreamTwist-AI API 문서',
    doc="/api-docs"
)

from .story import story_api

api.add_namespace(story_api, path='/story')
