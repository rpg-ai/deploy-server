from flask import Flask, request, jsonify
import os
from nlp_classfier import NLP_Classifier
import pickle
import numpy as np

app = Flask(__name__)
with app.app_context():
    global nlp
    global dp
    global skill_model
    global skill_tfidf
    global path_dir_models

    nlp = NLP_Classifier()
    path_dir_models = 'temp'

    filename = os.path.join(path_dir_models, 'model.sav')
    skill_model = pickle.load(open(filename, 'rb'))
    skill_model.probability = True

    filename = os.path.join(path_dir_models, 'tfidf.pickle')
    skill_tfidf = pickle.load(open(filename, 'rb'))


@app.route('/')
def home():
    result_dict = {
        "api_status": "maintenance",
        "used_model": None
    }
    return jsonify(result_dict)


@app.route('/predict', methods=["POST", "GET"])
def predict():
    content = request.args
    action_list = []
    action = ''.join(content.get('action'))
    action_list.append(action)

    print(action_list)

    bow_tfidf = nlp.use_TDIDF_Vec_model_in_memory(action_list, skill_tfidf)
    pred_skills = skill_model.predict_proba(bow_tfidf)  #[:, 1]

    print(pred_skills)

    best_n = np.argsort(pred_skills, axis=1)[:, -3:]
    classes = skill_model.classes_

    predictions_list = [{classes[best_n[0, 2]]: f"{round(pred_skills[0, best_n[0, 2]] * 100, 2)}%"},
                        {classes[best_n[0, 1]]: f"{round(pred_skills[0, best_n[0, 1]] * 100, 2)}%"},
                        {classes[best_n[0, 0]]: f"{round(pred_skills[0, best_n[0, 0]] * 100, 2)}%"}]

    result_dict = {
        "player": content.get('player'),
        "action": content.get('action'),
        "predictions_list": predictions_list
    }
    return jsonify(result_dict)


"""if __name__ == "__main__":
    app.run(debug=True)"""
