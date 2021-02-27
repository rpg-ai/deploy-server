from flask import Flask, request, jsonify
import pipeline as pipeline
app = Flask(__name__)


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
