import pickle
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

salary_model = pickle.load(open("models/salary_model.pkl", "rb"))
income_model = pickle.load(open("models/income_model.pkl", "rb"))
advertise_model = pickle.load(open("models/advertise_model.pkl", "rb"))

with open("models/advertise_corr.pkl", "rb") as f:
    advertise_corr = pickle.load(f)

with open("models/advertise_features.pkl", "rb") as f:
    advertise_features = pickle.load(f)

# HTML PAGES
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/salary')
def salary_page():
    return render_template('salary_model.html')

@app.route('/income')
def income_page():
    return render_template('income_model.html')

@app.route('/sales')
def sales_page():
    return render_template('sales_model.html', correlations=advertise_corr)

# API ROUTES
@app.route("/predict_salary", methods=["POST"])
def predict_salary():
    data = request.json
    years_exp = [[data["YearsExperience"]]]
    prediction = salary_model.predict(years_exp)
    return jsonify({"predicted_salary": float(prediction[0])})

@app.route("/predict_income", methods=["POST"])
def predict_income():
    data = request.json
    features = [[data["age"], data["experience"]]]
    pred = income_model.predict(features)
    return jsonify({"predicted_income": float(pred[0])})

@app.route("/predict_sales", methods=["POST"])
def predict_sales():
    data = request.json
    try:
        features = [[data[feat] for feat in advertise_features]]
    except KeyError as e:
        return jsonify({"error": f"Missing required feature: {e.args[0]}"}), 400
    pred = advertise_model.predict(features)
    return jsonify({"predicted_sales": float(pred[0])})

if __name__=="__main__":
    app.run(debug=True)