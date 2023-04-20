import os
import numpy as np

from flask import Flask, request, jsonify
from src.inference import TextModel


app = Flask(__name__)
model = TextModel()


@app.route("/api/v1/healthcheck", methods = ["GET"])
def healthcheck():
    return jsonify({"status": "ok"})


@app.route("/api/v1/readiness_check", methods = ["GET"])
def readiness_check():
    # TODO: edit code start
    probs = model(["when I was at school you weren't there yet"])[0]
    assert model.labels[np.argmax(probs)] == "POS"
    # TODO: edit code end
    return jsonify({"status": "ok"})


@app.route("/api/v1/predict", methods = ["POST"])
def predict():
    payload = request.json
    probs = model(payload["prompts"])
    label_probs = [dict(zip(model.labels, x.tolist())) for x in probs]
    return jsonify(label_probs)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=(os.environ['LOG_LEVEL'] == 'DEBUG'))
