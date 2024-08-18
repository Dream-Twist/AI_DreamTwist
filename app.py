from flask import Flask, request, jsonify
from app.api.swagger import api

app = Flask(__name__)

@app.route('/')
def DreamTwist():
    return 'DreamTwist'

api.init_app(app)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
