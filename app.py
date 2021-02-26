from flask import Flask, request, jsonify
from joblib import load
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)


@app.route('/', methods=["GET"])
def call_model():
    req = request.args
    predict = pipeline.predict([req.get('action')])

    result_dict = {
        "player": req.get('player'),
        "action": req.get('action'),
        "prediction": predict[0],#{"Acrobatics": "32.54%"},
        "other_predicts": [{"Athletics": "12.9%"}, {"Survival": "2.41%"}]
    }
    return jsonify(result_dict)


if __name__ == '__main__':
    app.run()


# load the classifier model
pipeline = load("text_classification.joblib")