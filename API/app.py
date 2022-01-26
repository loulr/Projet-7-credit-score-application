import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import re

app = Flask(__name__)

# Load objects
model = pickle.load(open('pickles/final_model.pkl', 'rb'))
df = pd.read_csv('data/outputs/data_final.csv')
df = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    id_client = [int(x) for x in request.form.values()]

    features = np.array(df[df['SK_ID_CURR'] == id_client[0]].drop(['TARGET', 'TARGET_1_PROBA', 'SK_ID_CURR'], axis=1))

    prediction = model.predict_proba(features)[:, 1]

    # print("Features", features)
    print("prediction:", prediction)
    output = round(prediction[0] * 100, 2)
    print(output)

    if output <= 50:
        return render_template('index.html',
                               prediction_text='{}% chances of repayment default: The client is likely to repay the '
                                               'Credit'.format(
                                   output))
    else:
        return render_template('index.html',
                               prediction_text='{}% chances of repayment default: The client is NOT likely to repay '
                                               'the Credit'.format(
                                   output))


@app.route('/predict_api')
def results():
    id_client = int(request.args.get('id_client'))

    features = np.array(df[df['SK_ID_CURR'] == id_client].drop(['TARGET', 'TARGET_1_PROBA', 'SK_ID_CURR'], axis=1))

    prediction = model.predict_proba(features)[:, 1]

    output = round(prediction[0] * 100, 2)

    return jsonify('Prediction', output)


if __name__ == "__main__":
    app.run(debug=False)
