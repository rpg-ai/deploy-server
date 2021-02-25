from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route('/', methods=["GET"])
def call_model():
    req = request.args
    result_dict = {
        "player": req.get('player'),
        "action": req.get('action'),
        "prediction": {"Acrobatics": "32.54%"},
        "other_predicts": [{"Athletics": "12.9%"}, {"Survival": "2.41%"}]
    }
    return jsonify(result_dict)


if __name__ == '__main__':
    app.run()
