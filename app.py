from flask import Flask, request, jsonify
import pipeline as pipeline
import os

app = Flask(__name__)


@app.route('/config/model')
def download_model():
    os.popen(f'wget {os.environ["MODEL_DOWNLOAD_URL"]}').read()


@app.route('/predict', methods=["POST", "GET"])
def call_model():
    req = request.args
    print(req.get('action'))
    prediction, predictions_list = pipeline.predict(req.get('action'))

    result_dict = {
        "player": req.get('player'),
        "action": req.get('action'),
        "prediction": prediction,
        "predictions_list": predictions_list
    }
    return jsonify(result_dict)
