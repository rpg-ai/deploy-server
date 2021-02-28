from flask import Flask, request, jsonify
import pipeline as pipeline
import os

app = Flask(__name__)


@app.route('/predict', methods=["POST", "GET"])
def predict():
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


@app.route('/')
def home():
    model_dir = os.listdir('.')
    model_name = "text_classification.joblib"

    if model_name in model_dir:
        model_status = True
    else:
        model_status = False

    result_dict = {
        "model_status": model_status,
        "used_model": model_name
    }
    return jsonify(result_dict)
