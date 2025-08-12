import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler



app = Flask(__name__)


# import ridge regressor and standard scaler pickle

model = pickle.load(open(r'basic_EDA\flask_app\ridge.pkl', 'rb'))
scaler = pickle.load(open(r'basic_EDA\flask_app\scaler.pkl', 'rb'))




@app.route("/")
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        data = {
            'Temperature': float(request.form['temperature']),
            'RH': float(request.form['rh']),
            'Ws': float(request.form['ws']),
            'Rain': float(request.form['rain']),
            'FFMC': float(request.form['ffmc']),
            'DMC': float(request.form['dmc']),
            'ISI': float(request.form['isi']),
            'Classes': int(request.form['classes']),
            'Region': int(request.form['region'])
        }

        input_data = pd.DataFrame([data])
        input_scaled = scaler.transform(input_data)

        pred = model.predict(input_scaled)[0]

        return render_template('index.html', prediction=pred)

    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0") 