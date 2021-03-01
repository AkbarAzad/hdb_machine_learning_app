from flask import Flask, request, url_for, redirect, render_template, jsonify
import pandas as pd
import numpy as np
import pickle


app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods = ['POST'])
def predict():
    model = pickle.load(open("model.pkl", 'rb'))
    columns = ['floor_area_sqm', 'month', 'town_num', 'flat_type_num']
    int_features = [float(x) for x in request.form.values()]
    final = np.array([int_features])
    prediction = model.predict(final)
    prediction = prediction[0][0]
    print(prediction)
    
    return render_template("predict.html", prediction = prediction)
    
if __name__ == "__main__":
 app.run(debug=True)