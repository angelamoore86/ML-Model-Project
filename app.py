import joblib
from flask import Flask, request, render_template

app = Flask(__name__)

mpg_model = joblib.load('mpg_model.joblib')
diabetes_model = joblib.load('diabetes_model.joblib')


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        model_type = request.form.get('model_type')
        if model_type == 'mpg':
            return render_template('mpg_form.html')
        elif model_type == 'diabetes':
            return render_template('diabetes_form.html')
    return render_template('index.html')


@app.route('/mpg_form')
def show_mpg_form():
    return render_template('mpg_form.html')


@app.route('/diabetes_form')
def show_diabetes_form():
    return render_template('diabetes_form.html')


@app.route('/predict_mpg', methods=['POST'])
def predict_mpg():
    cylinders = float(request.form['cylinders'])
    horsepower = float(request.form['horsepower'])
    weight = float(request.form['weight'])
    age = float(request.form['age'])
    japan = float(request.form.get('japan', 0))
    usa = float(request.form.get('usa', 0))

    input_data = [cylinders, horsepower, weight, age, japan, usa]
    prediction = round(mpg_model.predict([input_data])[0], 2)

    return render_template('results.html', model_type='mpg', prediction=prediction)


@app.route('/predict_diabetes', methods=['POST'])
def predict_diabetes():
    pregnancies = float(request.form['pregnancies'])
    glucose = float(request.form['glucose'])
    bp = float(request.form['bp'])
    bmi = float(request.form['bmi'])
    dpf = float(request.form['dpf'])
    age = float(request.form['age'])

    input_data = [pregnancies, glucose, bp, bmi, dpf, age]
    prediction = diabetes_model.predict([input_data])[0]

    return render_template('results.html', model_type='diabetes',  prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)

