from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('trained_model.sav')

# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for the prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve the input data from the HTML form
    age = int(request.form['age'])
    sex = request.form['sex']
    cp = request.form['cp']
    trestbps = int(request.form['trestbps'])
    chol = int(request.form['chol'])
    fbs = request.form['fbs']
    restecg = request.form['restecg']
    thalach = int(request.form['thalach'])
    exang = request.form['exang']
    oldpeak = float(request.form['oldpeak'])

    # Convert the input data into a numpy array
    sex_value = 1 if sex == 'Male' else 0
    cp_value = {
        'Typical Angina': 0,
        'Atypical Angina': 1,
        'Non-anginal Pain': 2,
        'Asymptomatic': 3
    }[cp]
    fbs_value = 1 if fbs == 'True' else 0
    restecg_value = {
        'Normal': 0,
        'ST-T Abnormality': 1,
        'Left Ventricular Hypertrophy': 2
    }[restecg]
    data = np.array([[age, sex_value, cp_value, trestbps, chol, fbs_value, restecg_value, thalach, exang, oldpeak]])

    # Use the model to make a prediction
    prediction = model.predict(data)[0]

    # Determine the prediction result
    result = 'The patient is likely to have heart disease.' if prediction else 'The patient is not likely to have heart disease.'

    # Render the result on a new HTML page
    return render_template('result.html', prediction_result=result)

if __name__ == '__main__':
    app.run(debug=True)
