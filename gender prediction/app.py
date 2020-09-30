import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from gender_classification import get_features2

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    name = request.form.get("Name")
    print(name)
    prediction = model.classify(get_features2(name))
    print(prediction)
    return render_template('index.html', prediction_text = 'The gender of {} will be {}'.format(name,prediction))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    name = request.get_json(force=True)["Name"]
    print(name)
    prediction = model.classify(get_features2(name ))

    output = prediction
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True,port = 5010)